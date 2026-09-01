from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from batch_color.baseline import A0_BASELINE, a0_compatible
from batch_color.color import oklab_to_srgb, srgb_to_oklab
from batch_color.image_io import load_mask, load_srgb, make_proxy, save_mask, save_srgb
from batch_color.masking import get_background_mask
from batch_color.planning import compile_shadow_plan
from batch_color.runtime import a0_runtime_compatibility, runtime_identity
from batch_color.safety import atomic_json, atomic_output, file_hash
from batch_color.semantic import SemanticMasks, build_semantic_masks
from batch_color.sku import SKUInput, scan_sku, validate_inputs_unchanged
from batch_color.transfer import _bounded_luminance_curve
from batch_color.workflow import (
    CANDIDATE_DIRECTORY,
    ensure_sku_profile,
    load_sku_profile,
    profile_confirmed_garment,
)


@dataclass(frozen=True)
class RegionStats:
    pixels: int
    l_quantiles: tuple[float, ...]
    a_median: float
    b_median: float
    clipped_percent: float


@dataclass(frozen=True)
class RegionPlan:
    name: str
    mask: Image.Image
    source: RegionStats
    target: RegionStats
    strength: float
    luminance_cap: float
    chroma_cap: float


def region_stats(image: Image.Image, mask: Image.Image, *, proxy_edge: int = 1024) -> RegionStats:
    proxy = make_proxy(image, max_edge=proxy_edge)
    region = mask.resize(proxy.size, Image.Resampling.BILINEAR)
    weights = np.asarray(region, dtype=np.uint8)
    core = weights >= 128
    if int(np.count_nonzero(core)) < 128:
        raise ValueError("Region has too few confident pixels for color statistics")
    rgb_u8 = np.asarray(proxy.convert("RGB"), dtype=np.uint8)
    rgb = rgb_u8.astype(np.float32) / 255.0
    pixels = srgb_to_oklab(rgb)[core]
    q = np.quantile(pixels[:, 0], [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    clipped = np.any((rgb_u8[core] <= 1) | (rgb_u8[core] >= 254), axis=1)
    return RegionStats(
        pixels=int(len(pixels)),
        l_quantiles=tuple(round(float(value), 7) for value in q),
        a_median=round(float(np.median(pixels[:, 1])), 7),
        b_median=round(float(np.median(pixels[:, 2])), 7),
        clipped_percent=round(100.0 * float(np.mean(clipped)), 6),
    )


def optional_region_stats(image: Image.Image, mask: Image.Image) -> RegionStats | None:
    try:
        return region_stats(image, mask)
    except ValueError:
        return None


def region_distance(left: RegionStats, right: RegionStats) -> float:
    l_delta = np.asarray(left.l_quantiles) - np.asarray(right.l_quantiles)
    chroma_delta = np.array(
        [left.a_median - right.a_median, left.b_median - right.b_median]
    )
    return float(100.0 * np.sqrt(np.mean(np.square(l_delta)) + np.sum(np.square(chroma_delta))))


def region_style_distance(left: RegionStats, right: RegionStats) -> float:
    """Compare colour style while discounting composition-specific deep shadows/highlights."""
    left_iqr = left.l_quantiles[4] - left.l_quantiles[2]
    right_iqr = right.l_quantiles[4] - right.l_quantiles[2]
    features = np.array(
        [
            left.l_quantiles[3] - right.l_quantiles[3],
            0.45 * (left_iqr - right_iqr),
            left.a_median - right.a_median,
            left.b_median - right.b_median,
        ],
        dtype=np.float64,
    )
    return float(100.0 * np.sqrt(np.sum(np.square(features))))


def normalized_garment_signature(
    garment: RegionStats, background: RegionStats
) -> RegionStats:
    """Remove first-order scene illumination while retaining garment identity."""
    background_l = background.l_quantiles[3]
    return RegionStats(
        pixels=garment.pixels,
        l_quantiles=tuple(round(value - background_l, 7) for value in garment.l_quantiles),
        a_median=round(garment.a_median - background.a_median, 7),
        b_median=round(garment.b_median - background.b_median, 7),
        clipped_percent=garment.clipped_percent,
    )


def scene_style_target(
    subject: RegionStats,
    source_background: RegionStats,
    reference_background: RegionStats,
    *,
    luminance_scale: float,
    chroma_scale: float,
) -> RegionStats:
    """Transfer scene direction without replacing a person's skin or hair identity."""
    l_delta = np.clip(
        np.asarray(reference_background.l_quantiles)
        - np.asarray(source_background.l_quantiles),
        -0.10,
        0.10,
    ) * luminance_scale
    delta_a = float(
        np.clip(reference_background.a_median - source_background.a_median, -0.035, 0.035)
        * chroma_scale
    )
    delta_b = float(
        np.clip(reference_background.b_median - source_background.b_median, -0.035, 0.035)
        * chroma_scale
    )
    return RegionStats(
        pixels=subject.pixels,
        l_quantiles=tuple(
            round(float(value + delta), 7)
            for value, delta in zip(subject.l_quantiles, l_delta)
        ),
        a_median=round(subject.a_median + delta_a, 7),
        b_median=round(subject.b_median + delta_b, 7),
        clipped_percent=subject.clipped_percent,
    )


def _pairwise_mean(stats: list[RegionStats]) -> float:
    values = [region_distance(stats[i], stats[j]) for i in range(len(stats)) for j in range(i + 1, len(stats))]
    return 0.0 if not values else float(np.mean(values))


def _pairwise_summary(stats: list[RegionStats]) -> dict[str, float]:
    values = [
        region_distance(stats[i], stats[j])
        for i in range(len(stats))
        for j in range(i + 1, len(stats))
    ]
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": round(float(np.mean(values)), 6),
        "p95": round(float(np.quantile(values, 0.95)), 6),
        "max": round(float(np.max(values)), 6),
    }


def _pairwise_style_summary(stats: list[RegionStats]) -> dict[str, float]:
    values = [
        region_style_distance(stats[i], stats[j])
        for i in range(len(stats))
        for j in range(i + 1, len(stats))
    ]
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": round(float(np.mean(values)), 6),
        "p95": round(float(np.quantile(values, 0.95)), 6),
        "max": round(float(np.max(values)), 6),
    }


def background_style_target(source: RegionStats, reference: RegionStats) -> RegionStats:
    """Match scene midtone/colour while preserving composition-specific shadow structure."""
    source_median = source.l_quantiles[3]
    reference_median = reference.l_quantiles[3]
    source_iqr = max(source.l_quantiles[4] - source.l_quantiles[2], 0.01)
    reference_iqr = max(reference.l_quantiles[4] - reference.l_quantiles[2], 0.01)
    contrast = float(np.clip(reference_iqr / source_iqr, 0.82, 1.18))
    values = tuple(
        round(
            float(
                np.clip(
                    reference_median + (value - source_median) * contrast,
                    0.0,
                    1.0,
                )
            ),
            7,
        )
        for value in source.l_quantiles
    )
    return RegionStats(
        pixels=source.pixels,
        l_quantiles=values,
        a_median=reference.a_median,
        b_median=reference.b_median,
        clipped_percent=source.clipped_percent,
    )


def choose_anchor(records: list[dict[str, object]]) -> tuple[int, list[dict[str, object]]]:
    if not records:
        raise ValueError("Cannot choose an anchor from an empty SKU")
    stats = [
        record["garment_signature"]
        if "garment_signature" in record
        else record["garment_stats"]
        for record in records
    ]
    if not all(isinstance(item, RegionStats) for item in stats):
        raise ValueError("Every target needs garment statistics before anchor selection")
    ranking = []
    for index, candidate in enumerate(stats):
        distances = [region_distance(candidate, other) for other in stats]
        centrality = float(np.median(distances))
        clipping_penalty = min(candidate.clipped_percent, 20.0) * 0.08
        pixel_reward = min(np.log10(max(candidate.pixels, 1)), 6.0) * 0.03
        score = centrality + clipping_penalty - pixel_reward
        ranking.append(
            {
                "index": index,
                "file": records[index]["path"],
                "score": round(score, 6),
                "centrality": round(centrality, 6),
                "garment_clipped_percent": candidate.clipped_percent,
                "garment_pixels_proxy": candidate.pixels,
            }
        )
    ranking.sort(key=lambda item: (item["score"], str(item["file"]).casefold()))
    return int(ranking[0]["index"]), ranking


def apply_region_plans(
    image: Image.Image, plans: list[RegionPlan], *, tile_rows: int = 256
) -> tuple[Image.Image, dict[str, object]]:
    exact_identity = all(
        plan.source.l_quantiles == plan.target.l_quantiles
        and plan.source.a_median == plan.target.a_median
        and plan.source.b_median == plan.target.b_median
        for plan in plans
    )
    if not plans or exact_identity:
        return image.copy(), {
            "no_op": True,
            "no_op_reason": "no_plans" if not plans else "exact_transform_identity",
            "editable_pixels": 0,
            "newly_clipped_pixels": 0,
            "newly_clipped_percent_of_editable": 0.0,
            "unauthorized_changed_pixels": 0,
            "plans": [],
        }
    source_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
    output_u8 = np.empty_like(source_u8)
    curves = []
    for plan in plans:
        x, y = _bounded_luminance_curve(plan.source.l_quantiles, plan.target.l_quantiles)
        raw_a = plan.target.a_median - plan.source.a_median
        raw_b = plan.target.b_median - plan.source.b_median
        curves.append(
            (
                x,
                y,
                float(np.clip(raw_a, -plan.chroma_cap, plan.chroma_cap)),
                float(np.clip(raw_b, -plan.chroma_cap, plan.chroma_cap)),
                raw_a,
                raw_b,
            )
        )

    editable_pixels = 0
    newly_clipped = 0
    unauthorized_changed = 0
    for row_start in range(0, image.height, tile_rows):
        row_end = min(row_start + tile_rows, image.height)
        rgb_u8 = source_u8[row_start:row_end]
        rgb = rgb_u8.astype(np.float32) / 255.0
        lab = srgb_to_oklab(rgb)
        delta = np.zeros_like(lab)
        total_weight = np.zeros(lab.shape[:2], dtype=np.float32)
        for plan, (curve_x, curve_y, delta_a, delta_b, _raw_a, _raw_b) in zip(plans, curves):
            weight = np.asarray(plan.mask.crop((0, row_start, image.width, row_end)), dtype=np.float32) / 255.0
            if not np.any(weight > 0):
                continue
            target_l = np.interp(lab[..., 0], curve_x, curve_y).astype(np.float32)
            delta_l = np.clip(target_l - lab[..., 0], -plan.luminance_cap, plan.luminance_cap)
            region_delta = np.empty_like(lab)
            region_delta[..., 0] = delta_l
            region_delta[..., 1] = delta_a
            region_delta[..., 2] = delta_b
            effective = weight * float(plan.strength)
            delta += effective[..., None] * region_delta
            total_weight += effective
        overlap = np.maximum(total_weight, 1.0)
        corrected_lab = lab + delta / overlap[..., None]
        corrected_rgb = oklab_to_srgb(corrected_lab)
        editable = total_weight > (1.0 / 255.0)
        editable_pixels += int(np.count_nonzero(editable))
        raw_clipped = np.any((corrected_rgb < -1e-6) | (corrected_rgb > 1.0 + 1e-6), axis=-1)
        source_not_clipped = ~np.any((rgb_u8 <= 1) | (rgb_u8 >= 254), axis=-1)
        newly_clipped += int(np.count_nonzero(raw_clipped & editable & source_not_clipped))
        encoded = np.round(np.clip(corrected_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        encoded[~editable] = rgb_u8[~editable]
        unauthorized_changed += int(np.count_nonzero(np.any(encoded[~editable] != rgb_u8[~editable], axis=-1)))
        output_u8[row_start:row_end] = encoded

    output = Image.fromarray(output_u8, mode="RGB")
    return output, {
        "no_op": bool(np.array_equal(output_u8, source_u8)),
        "no_op_reason": "encoded_pixels_unchanged" if np.array_equal(output_u8, source_u8) else None,
        "editable_pixels": editable_pixels,
        "newly_clipped_pixels": newly_clipped,
        "newly_clipped_percent_of_editable": round(
            0.0 if editable_pixels == 0 else 100.0 * newly_clipped / editable_pixels, 6
        ),
        "unauthorized_changed_pixels": unauthorized_changed,
        "plans": [
            {
                "region": plan.name,
                "strength": plan.strength,
                "luminance_cap": plan.luminance_cap,
                "chroma_cap": plan.chroma_cap,
                "requested_luminance_delta_median": round(
                    plan.target.l_quantiles[3] - plan.source.l_quantiles[3], 7
                ),
                "requested_chroma_delta": [round(raw_a, 7), round(raw_b, 7)],
                "chroma_cap_hit": bool(
                    abs(raw_a) > plan.chroma_cap or abs(raw_b) > plan.chroma_cap
                ),
            }
            for plan, (_x, _y, _delta_a, _delta_b, raw_a, raw_b) in zip(plans, curves)
        ],
    }


def _save_preview(image: Image.Image, path: Path, *, quality: int = 92) -> None:
    with atomic_output(path) as staged:
        image.convert("RGB").save(staged, format="JPEG", quality=quality, subsampling=0, optimize=True)


def _thumb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = Image.new("RGB", size, "white")
    copy = image.convert("RGB").copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    result.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return result


def _save_contact_sheet(
    rows: list[tuple[str, Image.Image, Image.Image]], reference: Image.Image, path: Path
) -> None:
    cell = (300, 330)
    header = 42
    sheet = Image.new("RGB", (cell[0] * 3, header + cell[1] * len(rows)), (242, 242, 242))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for col, title in enumerate(("SOURCE", "CORRECTED", "SCENE REFERENCE")):
        draw.text((col * cell[0] + 10, 14), title, fill="black", font=font)
    ref_thumb = _thumb(reference, (cell[0], cell[1] - 24))
    for row, (name, source, corrected) in enumerate(rows):
        y = header + row * cell[1]
        sheet.paste(_thumb(source, (cell[0], cell[1] - 24)), (0, y))
        sheet.paste(_thumb(corrected, (cell[0], cell[1] - 24)), (cell[0], y))
        sheet.paste(ref_thumb, (cell[0] * 2, y))
        draw.text((8, y + cell[1] - 20), name, fill="black", font=font)
    _save_preview(sheet, path, quality=90)


def _mask_bbox(mask: Image.Image, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    small = mask.copy()
    small.thumbnail((512, 512), Image.Resampling.BILINEAR)
    bbox = small.point(lambda value: 255 if value >= 96 else 0).getbbox()
    if not bbox:
        return fallback
    sx, sy = mask.width / small.width, mask.height / small.height
    x0, y0, x1, y1 = (int(bbox[0] * sx), int(bbox[1] * sy), int(bbox[2] * sx), int(bbox[3] * sy))
    pad_x, pad_y = int((x1 - x0) * 0.08), int((y1 - y0) * 0.08)
    return max(0, x0 - pad_x), max(0, y0 - pad_y), min(mask.width, x1 + pad_x), min(mask.height, y1 + pad_y)


def _save_detail_sheet(
    source: Image.Image, corrected: Image.Image, garment: Image.Image, skin: Image.Image, path: Path
) -> None:
    full = (0, 0, source.width, source.height)
    garment_box = _mask_bbox(garment, full)
    skin_box = _mask_bbox(skin, (0, 0, source.width, max(1, source.height // 3)))
    panels = []
    for title, box in (("SKIN", skin_box), ("GARMENT", garment_box)):
        panels.append((title + " SOURCE", source.crop(box)))
        panels.append((title + " CORRECTED", corrected.crop(box)))
    cell = (420, 360)
    sheet = Image.new("RGB", (cell[0] * 2, cell[1] * 2), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (title, image) in enumerate(panels):
        x, y = (index % 2) * cell[0], (index // 2) * cell[1]
        sheet.paste(_thumb(image, (cell[0], cell[1] - 24)), (x, y))
        draw.text((x + 8, y + cell[1] - 19), title, fill="black", font=font)
    _save_preview(sheet, path, quality=94)


def _mask_paths(root: Path, stem: str) -> dict[str, Path]:
    directory = root / stem
    names = (
        "background",
        "background_core",
        "background_transition",
        "garment",
        "garment_core",
        "garment_transition",
        "skin",
        "skin_core",
        "skin_transition",
        "hair",
        "hair_core",
        "accessory_protect",
        "unknown_person",
        "conflicts",
    )
    return {name: directory / f"{name}.png" for name in names}


def _save_semantic_masks(masks: SemanticMasks, paths: dict[str, Path]) -> None:
    for name, path in paths.items():
        save_mask(getattr(masks, name), path)
    evidence_path = next(iter(paths.values())).parent / "probability-evidence.npz"
    with atomic_output(evidence_path) as staged:
        with staged.open("wb") as handle:
            np.savez_compressed(handle, **masks.probabilities)


def _load_semantic_masks(paths: dict[str, Path], size: tuple[int, int]) -> dict[str, Image.Image]:
    return {name: load_mask(path, size) for name, path in paths.items()}


def artifact_metrics(
    source: Image.Image,
    corrected: Image.Image,
    masks: dict[str, Image.Image],
    *,
    proxy_edge: int = 1024,
) -> dict[str, float]:
    """Measure abrupt correction jumps and tone collapse without judging aesthetics."""
    source_proxy = make_proxy(source, max_edge=proxy_edge).convert("RGB")
    corrected_proxy = corrected.resize(source_proxy.size, Image.Resampling.LANCZOS).convert("RGB")
    source_lab = srgb_to_oklab(np.asarray(source_proxy, dtype=np.float32) / 255.0)
    corrected_lab = srgb_to_oklab(np.asarray(corrected_proxy, dtype=np.float32) / 255.0)
    correction = corrected_lab - source_lab
    magnitude = np.linalg.norm(correction, axis=-1)
    horizontal_jump = np.linalg.norm(correction[:, 1:] - correction[:, :-1], axis=-1)
    vertical_jump = np.linalg.norm(correction[1:] - correction[:-1], axis=-1)
    active_h = np.maximum(magnitude[:, 1:], magnitude[:, :-1]) > 1e-4
    active_v = np.maximum(magnitude[1:], magnitude[:-1]) > 1e-4
    jumps = np.concatenate([horizontal_jump[active_h], vertical_jump[active_v]])

    background = masks["background_core"].resize(source_proxy.size, Image.Resampling.BILINEAR)
    bg = np.asarray(background, dtype=np.uint8) >= 192
    source_l = source_lab[..., 0]
    corrected_l = corrected_lab[..., 0]
    source_grad = np.abs(source_l[:, 1:] - source_l[:, :-1])
    corrected_grad = np.abs(corrected_l[:, 1:] - corrected_l[:, :-1])
    bg_pairs = bg[:, 1:] & bg[:, :-1]
    informative = bg_pairs & (source_grad > 1.0 / 1024.0) & (source_grad < 0.04)
    collapse = informative & (corrected_grad < 0.25 * source_grad)
    return {
        "correction_jump_p99": round(
            0.0 if jumps.size == 0 else float(np.quantile(jumps, 0.99)), 7
        ),
        "correction_jump_max": round(0.0 if jumps.size == 0 else float(np.max(jumps)), 7),
        "background_tone_collapse_percent": round(
            0.0
            if not np.any(informative)
            else 100.0 * float(np.count_nonzero(collapse)) / float(np.count_nonzero(informative)),
            6,
        ),
    }


def _two_region_masks(
    input_path: str | Path,
    image: Image.Image,
    *,
    mask_backend: str,
) -> tuple[dict[str, Image.Image], dict[str, object]]:
    result = get_background_mask(
        input_path, image, backend=mask_backend, quality="accurate"
    )
    background = np.asarray(result.background_mask.convert("L"), dtype=np.float32) / 255.0
    person = 1.0 - background
    background_core = np.where(background >= 0.86, background, 0.0)
    person_core = np.where(person >= 0.86, person, 0.0)
    transition = np.where(
        (background > 0.05) & (background < 0.95),
        1.0 - np.abs(2.0 * background - 1.0),
        0.0,
    )

    def as_mask(values: np.ndarray) -> Image.Image:
        return Image.fromarray(
            np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L"
        )

    masks = {
        "background": as_mask(background),
        "background_core": as_mask(background_core),
        "person": as_mask(person),
        "person_core": as_mask(person_core),
        "transition": as_mask(transition),
    }
    person_pixels = int(np.count_nonzero(person_core >= 0.86))
    background_pixels = int(np.count_nonzero(background_core >= 0.86))
    if min(person_pixels, background_pixels) < 256:
        raise ValueError(f"Two-region mask quality gate failed for {Path(input_path).name}")
    return masks, {
        "backend": result.backend,
        "person_core_pixels": person_pixels,
        "background_core_pixels": background_pixels,
        "transition_pixels": int(np.count_nonzero(transition > 0)),
        "warnings": [],
        "human_review_required": True,
    }


def _save_two_region_masks(masks: dict[str, Image.Image], root: Path) -> dict[str, str]:
    paths = {}
    for name, mask in masks.items():
        path = root / f"{name}.png"
        save_mask(mask, path)
        paths[name] = str(path)
    return paths


_FLAT_OUTPUT_ENTRIES = (
    CANDIDATE_DIRECTORY,
    "蒙版",
    "报告",
    "预览",
    "整套对照.jpg",
    "summary.json",
    "input-manifest.json",
    "sku-profile.json",
    "run-identity.json",
    "execution-plan.json",
    "review-status.json",
)


def _relative_artifact(path: str | Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def _publish_flat_sku_output(
    staging: Path,
    final: Path,
    *,
    replace_output: bool,
) -> None:
    """Publish owned artifacts while preserving unrelated legacy directories."""
    if final.is_symlink():
        raise ValueError(f"Refusing symlink SKU output: {final}")
    unexpected = sorted(path.name for path in staging.iterdir() if path.name not in _FLAT_OUTPUT_ENTRIES)
    if unexpected:
        raise RuntimeError(f"Unexpected staged SKU artifacts: {unexpected}")
    missing = [name for name in _FLAT_OUTPUT_ENTRIES if not (staging / name).exists()]
    if missing:
        raise RuntimeError(f"Missing staged SKU artifacts: {missing}")

    existed = final.exists()
    if existed and not final.is_dir():
        raise ValueError(f"SKU output is not a directory: {final}")
    conflicts = [name for name in _FLAT_OUTPUT_ENTRIES if (final / name).exists()]
    if conflicts and not replace_output:
        raise FileExistsError(
            f"SKU output already contains current artifacts; use --replace-output: {final}"
        )

    final.mkdir(parents=True, exist_ok=True)
    backup = final.parent / f".{final.name}.backup-{uuid.uuid4().hex}"
    backup.mkdir()
    published: list[str] = []
    backed_up: list[str] = []
    history: Path | None = None
    try:
        for name in _FLAT_OUTPUT_ENTRIES:
            destination = final / name
            if destination.exists():
                destination.rename(backup / name)
                backed_up.append(name)
            (staging / name).rename(destination)
            published.append(name)
        # Keep compact, versioned evidence from the verified backup. Large
        # candidates/masks/previews are deliberately not duplicated.
        if backed_up:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            history = final / "历史" / f"{stamp}-{uuid.uuid4().hex[:8]}"
            history.mkdir(parents=True, exist_ok=False)
            for name in (
                "summary.json",
                "input-manifest.json",
                "sku-profile.json",
                "run-identity.json",
                "execution-plan.json",
                "review-status.json",
                "整套对照.jpg",
                "报告",
            ):
                source = backup / name
                if source.is_dir():
                    shutil.copytree(source, history / name)
                elif source.is_file():
                    shutil.copy2(source, history / name)
        staging.rmdir()
    except BaseException:
        if history is not None:
            shutil.rmtree(history, ignore_errors=True)
        for name in reversed(published):
            destination = final / name
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink(missing_ok=True)
        for name in backed_up:
            (backup / name).rename(final / name)
        if not existed and final.exists() and not any(final.iterdir()):
            final.rmdir()
        shutil.rmtree(backup, ignore_errors=True)
        raise
    else:
        shutil.rmtree(backup)


def run_sku_pilot(
    *,
    dataset_root: str | Path,
    sku: str,
    output_root: str | Path,
    run_name: str,
    garment_kind: str,
    garment_hint: str,
    background_strength: float = 0.68,
    garment_strength: float = 0.58,
    skin_strength: float = 0.22,
    hair_strength: float = 0.12,
    mask_backend: str = "vision",
    parser_backend: str = "mediapipe",
    pose_backend: str = "vision",
    garment_anchor: str | None = None,
    set_color_tolerance: float = 2.0,
) -> tuple[Path, dict[str, object]]:
    manifest = scan_sku(dataset_root, sku)
    output_base = Path(output_root).resolve()
    source_directory = Path(manifest.directory).resolve()
    if output_base == source_directory or output_base.is_relative_to(source_directory):
        raise ValueError("Output root must not be inside the source SKU directory")
    final = output_base / sku / run_name
    if final.exists():
        raise FileExistsError(f"Pilot output already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{run_name}.processing-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        atomic_json(staging / "input-manifest.json", manifest.as_dict())
        corrected_dir = staging / "校色成品"
        masks_root = staging / "蒙版"
        reports_dir = staging / "报告"
        details_dir = staging / "局部检查"
        previews_dir = staging / "预览"

        reference, reference_info = load_srgb(manifest.scene_reference)
        reference_masks = build_semantic_masks(
            manifest.scene_reference,
            reference,
            garment_kind="none",
            garment_hint="none",
            mask_backend=mask_backend,
            parser_backend=parser_backend,
            pose_backend=pose_backend,
        )
        reference_paths = _mask_paths(masks_root / "指定场景", "regions")
        _save_semantic_masks(reference_masks, reference_paths)
        reference_stats = {
            "background": region_stats(reference, reference_masks.background_core),
            "skin": optional_region_stats(reference, reference_masks.skin_core),
            "hair": optional_region_stats(reference, reference_masks.hair_core),
        }

        records: list[dict[str, object]] = []
        for target_path in manifest.targets:
            path = Path(target_path)
            source, source_info = load_srgb(path)
            masks = build_semantic_masks(
                path,
                source,
                garment_kind=garment_kind,
                garment_hint=garment_hint,
                mask_backend=mask_backend,
                parser_backend=parser_backend,
                pose_backend=pose_backend,
            )
            paths = _mask_paths(masks_root, path.stem)
            _save_semantic_masks(masks, paths)
            records.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "source_info": asdict(source_info),
                    "mask_paths": {name: str(value) for name, value in paths.items()},
                    "mask_backend": masks.backend,
                    "mask_diagnostics": masks.diagnostics,
                    "garment_stats": region_stats(source, masks.garment_core),
                    "background_stats": region_stats(source, masks.background_core),
                    "skin_stats": optional_region_stats(source, masks.skin_core),
                    "hair_stats": optional_region_stats(source, masks.hair_core),
                }
            )
            records[-1]["garment_signature"] = normalized_garment_signature(
                records[-1]["garment_stats"], records[-1]["background_stats"]
            )
            validate_inputs_unchanged(manifest)

        auto_anchor_index, anchor_ranking = choose_anchor(records)
        anchor_index = auto_anchor_index
        anchor_selection = "auto_normalized_medoid"
        if garment_anchor:
            requested = garment_anchor.casefold()
            matches = [
                index
                for index, record in enumerate(records)
                if str(record["name"]).casefold() == requested
                or Path(str(record["name"])).stem.casefold() == Path(requested).stem.casefold()
            ]
            if len(matches) != 1:
                raise ValueError(f"Garment anchor does not uniquely match an SKU target: {garment_anchor}")
            anchor_index = matches[0]
            anchor_selection = "user_fixed"
        anchor_stats = records[anchor_index]["garment_stats"]
        anchor_name = str(records[anchor_index]["name"])
        before_garment_pairwise = _pairwise_mean([record["garment_stats"] for record in records])
        before_background_summary = _pairwise_summary(
            [record["background_stats"] for record in records]
        )
        before_skin_summary = _pairwise_summary(
            [record["skin_stats"] for record in records if record["skin_stats"] is not None]
        )

        item_reports = []
        contact_rows: list[tuple[str, Image.Image, Image.Image]] = []
        after_garment_stats = []
        after_background_stats = []
        after_skin_stats = []
        for index, record in enumerate(records):
            path = Path(str(record["path"]))
            source, _ = load_srgb(path)
            paths = {name: Path(value) for name, value in record["mask_paths"].items()}
            masks = _load_semantic_masks(paths, source.size)
            plans = [
                RegionPlan("background", masks["background"], record["background_stats"], reference_stats["background"],
                           background_strength, 0.10, 0.035),
                RegionPlan("garment", masks["garment"], record["garment_stats"], anchor_stats,
                           0.0 if index == anchor_index else garment_strength, 0.075, 0.025),
            ]
            skin_target = None
            if record["skin_stats"] is not None:
                skin_target = scene_style_target(
                    record["skin_stats"],
                    record["background_stats"],
                    reference_stats["background"],
                    luminance_scale=0.70,
                    chroma_scale=0.55,
                )
                plans.append(
                    RegionPlan(
                        "skin_scene_direction",
                        masks["skin"],
                        record["skin_stats"],
                        skin_target,
                        skin_strength,
                        0.035,
                        0.012,
                    )
                )
            hair_target = None
            if record["hair_stats"] is not None:
                hair_target = scene_style_target(
                    record["hair_stats"],
                    record["background_stats"],
                    reference_stats["background"],
                    luminance_scale=0.55,
                    chroma_scale=0.45,
                )
                plans.append(
                    RegionPlan(
                        "hair_scene_direction",
                        masks["hair"],
                        record["hair_stats"],
                        hair_target,
                        hair_strength,
                        0.025,
                        0.010,
                    )
                )
            corrected, transform = apply_region_plans(source, plans)
            output_path = corrected_dir / path.name.lower()
            final_output_path = final / "校色成品" / path.name.lower()
            verification = save_srgb(corrected, output_path)
            after = {
                "background": region_stats(corrected, masks["background_core"]),
                "garment": region_stats(corrected, masks["garment_core"]),
                "skin": optional_region_stats(corrected, masks["skin_core"]),
                "hair": optional_region_stats(corrected, masks["hair_core"]),
            }
            after_garment_stats.append(after["garment"])
            after_background_stats.append(after["background"])
            if after["skin"] is not None:
                after_skin_stats.append(after["skin"])
            artifacts = artifact_metrics(source, corrected, masks)
            distances = {
                "background_before": region_distance(record["background_stats"], reference_stats["background"]),
                "background_after": region_distance(after["background"], reference_stats["background"]),
                "garment_before": region_distance(record["garment_stats"], anchor_stats),
                "garment_after": region_distance(after["garment"], anchor_stats),
            }
            if record["skin_stats"] is not None and skin_target is not None and after["skin"] is not None:
                distances.update(
                    skin_before=region_distance(record["skin_stats"], skin_target),
                    skin_after=region_distance(after["skin"], skin_target),
                )
            flags = []
            flags.extend(
                f"mask_warning:{warning}"
                for warning in record["mask_diagnostics"].get("warnings", [])
            )
            if transform["unauthorized_changed_pixels"] != 0:
                flags.append("unauthorized_pixels_changed")
            if transform["newly_clipped_percent_of_editable"] > 0.20:
                flags.append("new_clipping_above_0.20_percent")
            if distances["background_after"] > distances["background_before"] + 0.05:
                flags.append("background_distance_worsened")
            if distances["garment_after"] > distances["garment_before"] + 0.05:
                flags.append("garment_distance_worsened")
            if artifacts["correction_jump_p99"] > 0.035:
                flags.append("abrupt_correction_transition")
            if artifacts["background_tone_collapse_percent"] > 3.0:
                flags.append("background_tone_collapse_risk")
            item_report = {
                "input": str(path),
                "input_sha256": manifest.input_hashes[str(path)],
                "output": str(final_output_path),
                "output_sha256": verification["sha256"],
                "anchor": index == anchor_index,
                "distances": {key: round(float(value), 6) for key, value in distances.items()},
                "transform": transform,
                "artifact_metrics": artifacts,
                "identity_policy": "scene_direction_only_not_reference_person_identity",
                "automatic_flags": flags,
                "automatic_checks_passed": not flags,
                "accepted": False,
                "status": "review",
                "source_stats": {
                    "background": asdict(record["background_stats"]),
                    "garment": asdict(record["garment_stats"]),
                    "skin": asdict(record["skin_stats"]) if record["skin_stats"] else None,
                    "hair": asdict(record["hair_stats"]) if record["hair_stats"] else None,
                },
                "output_stats": {name: asdict(value) if value else None for name, value in after.items()},
                "export_verification": verification,
                "mask_diagnostics": record["mask_diagnostics"],
            }
            atomic_json(reports_dir / f"{path.name}.json", item_report)
            _save_detail_sheet(source, corrected, masks["garment"], masks["skin"], details_dir / f"{path.stem}.jpg")
            source_preview = make_proxy(source, max_edge=1000)
            corrected_preview = make_proxy(corrected, max_edge=1000)
            _save_preview(ImageOps.expand(source_preview, border=2, fill="white"), previews_dir / f"{path.stem}-source.jpg")
            _save_preview(ImageOps.expand(corrected_preview, border=2, fill="white"), previews_dir / f"{path.stem}-corrected.jpg")
            contact_rows.append((path.name, source_preview, corrected_preview))
            item_reports.append(item_report)
            validate_inputs_unchanged(manifest)

        after_garment_pairwise = _pairwise_mean(after_garment_stats)
        after_garment_summary = _pairwise_summary(after_garment_stats)
        after_background_summary = _pairwise_summary(after_background_stats)
        after_skin_summary = _pairwise_summary(after_skin_stats)
        garment_set_passed = (
            after_garment_pairwise <= before_garment_pairwise + 1e-6
            and after_garment_summary["max"] <= set_color_tolerance
        )
        background_set_passed = (
            after_background_summary["mean"] <= before_background_summary["mean"] + 1e-6
            and after_background_summary["max"] <= set_color_tolerance
        )
        skin_set_passed = (
            len(after_skin_stats) < 2
            or (
                after_skin_summary["mean"] <= before_skin_summary["mean"] + 1e-6
                and after_skin_summary["max"] <= max(set_color_tolerance, 3.5)
            )
        )
        summary = {
            "schema_version": 2,
            "pipeline": "sku-dual-anchor-semantic-pilot-v2",
            "sku": sku,
            "run_name": run_name,
            "status": "review",
            "accepted": False,
            "source_directory": manifest.directory,
            "scene_reference": manifest.scene_reference,
            "target_count": len(records),
            "anchor": anchor_name,
            "anchor_selection": anchor_selection,
            "auto_anchor": str(records[auto_anchor_index]["name"]),
            "anchor_ranking": anchor_ranking,
            "configuration": {
                "garment_kind": garment_kind,
                "garment_hint": garment_hint,
                "background_strength": background_strength,
                "garment_strength": garment_strength,
                "skin_strength": skin_strength,
                "hair_strength": hair_strength,
                "mask_backend": mask_backend,
                "parser_backend": parser_backend,
                "pose_backend": pose_backend,
                "set_color_tolerance": set_color_tolerance,
                "spatial_surface": False,
                "reference_scope": "this_sku_only",
                "person_transfer": "scene_direction_only",
            },
            "reference_info": asdict(reference_info),
            "reference_stats": {name: asdict(value) if value else None for name, value in reference_stats.items()},
            "set_consistency": {
                "garment_pairwise_mean_before": round(before_garment_pairwise, 6),
                "garment_pairwise_mean_after": round(after_garment_pairwise, 6),
                "garment_after": after_garment_summary,
                "background_before": before_background_summary,
                "background_after": after_background_summary,
                "skin_before": before_skin_summary,
                "skin_after": after_skin_summary,
                "garment_passed": garment_set_passed,
                "background_passed": background_set_passed,
                "skin_passed": skin_set_passed,
                "all_passed": garment_set_passed and background_set_passed and skin_set_passed,
            },
            "automatic_checks_passed": all(item["automatic_checks_passed"] for item in item_reports)
            and garment_set_passed
            and background_set_passed
            and skin_set_passed,
            "human_review_required": True,
            "items": item_reports,
        }
        _save_contact_sheet(contact_rows, make_proxy(reference, max_edge=1000), staging / "整套对照.jpg")
        atomic_json(staging / "summary.json", summary)
        validate_inputs_unchanged(manifest)
        staging.rename(final)
        return final, summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_sku_simple_pilot(
    *,
    dataset_root: str | Path,
    sku: str,
    output_root: str | Path,
    run_name: str,
    background_strength: float = A0_BASELINE.background_strength,
    person_strength: float = A0_BASELINE.person_strength,
    mask_backend: str = "vision",
    set_color_tolerance: float = A0_BASELINE.set_color_tolerance,
    replace_output: bool = False,
) -> tuple[Path, dict[str, object]]:
    """Two-region A/B baseline: one transform for background and one for the whole person."""
    manifest = scan_sku(dataset_root, sku)
    output_base = Path(output_root).resolve()
    source_directory = Path(manifest.directory).resolve()
    if output_base == source_directory or output_base.is_relative_to(source_directory):
        raise ValueError("Output root must not be inside the source SKU directory")
    final = output_base / sku
    if final.resolve() == source_directory:
        raise ValueError("Output SKU directory must not overwrite the source SKU directory")
    output_base.mkdir(parents=True, exist_ok=True)
    if final.is_symlink():
        raise ValueError(f"Refusing symlink SKU output: {final}")
    profile_path = final / "sku-profile.json"
    existing_profile = (
        load_sku_profile(profile_path, manifest) if profile_path.is_file() else None
    )
    existing_artifacts = [name for name in _FLAT_OUTPUT_ENTRIES if (final / name).exists()]
    if existing_artifacts and not replace_output:
        raise FileExistsError(
            f"SKU output already contains current artifacts; use --replace-output: {final}"
        )
    staging = output_base / f".{sku}.{run_name}.processing-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        atomic_json(staging / "input-manifest.json", manifest.as_dict())
        corrected_dir = staging / CANDIDATE_DIRECTORY
        masks_root = staging / "蒙版"
        reports_dir = staging / "报告"
        previews_dir = staging / "预览"

        reference, reference_info = load_srgb(
            manifest.scene_reference, alpha_policy="drop_near_opaque"
        )
        reference_masks, reference_diagnostics = _two_region_masks(
            manifest.scene_reference, reference, mask_backend=mask_backend
        )
        reference_mask_paths_absolute = _save_two_region_masks(
            reference_masks, masks_root / "指定场景" / "regions"
        )
        reference_mask_paths = {
            name: _relative_artifact(path, staging)
            for name, path in reference_mask_paths_absolute.items()
        }
        reference_stats = {
            "background": region_stats(reference, reference_masks["background_core"]),
            "person": region_stats(reference, reference_masks["person_core"]),
        }

        records: list[dict[str, object]] = []
        for target_path in manifest.targets:
            path = Path(target_path)
            source, source_info = load_srgb(path, alpha_policy="drop_near_opaque")
            masks, diagnostics = _two_region_masks(path, source, mask_backend=mask_backend)
            mask_paths = _save_two_region_masks(masks, masks_root / path.stem)
            source_background = region_stats(source, masks["background_core"])
            source_person = region_stats(source, masks["person_core"])
            records.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "source_info": asdict(source_info),
                    "mask_paths": mask_paths,
                    "mask_paths_relative": {
                        name: _relative_artifact(mask_path, staging)
                        for name, mask_path in mask_paths.items()
                    },
                    "mask_diagnostics": diagnostics,
                    "background_stats": source_background,
                    "person_stats": source_person,
                    "garment_signature": normalized_garment_signature(
                        source_person, source_background
                    ),
                }
            )
            validate_inputs_unchanged(manifest)

        auto_anchor_index, anchor_ranking = choose_anchor(records)
        anchor_index = auto_anchor_index
        anchor_selection = "auto_normalized_whole_person_medoid"
        confirmed_anchor = profile_confirmed_garment(existing_profile)
        if confirmed_anchor:
            matches = [
                index
                for index, record in enumerate(records)
                if Path(str(record["name"])).name.casefold()
                == Path(confirmed_anchor).name.casefold()
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Confirmed garment anchor no longer uniquely matches an SKU target"
                )
            anchor_index = matches[0]
            anchor_selection = "confirmed_sku_profile_whole_person_anchor"
        anchor_record = records[anchor_index]
        sku_profile = ensure_sku_profile(
            manifest,
            existing_path=profile_path,
            staged_path=staging / "sku-profile.json",
            auto_garment_candidate=str(anchor_record["name"]),
        )
        identity = runtime_identity()
        runtime_compatibility = a0_runtime_compatibility(
            background_strength=background_strength,
            person_strength=person_strength,
            set_color_tolerance=set_color_tolerance,
            identity=identity,
        )
        atomic_json(staging / "run-identity.json", identity)
        execution_plan = compile_shadow_plan(sku_profile, runtime_compatibility)
        atomic_json(staging / "execution-plan.json", execution_plan)
        execution_plan_sha256 = file_hash(staging / "execution-plan.json")
        person_target = scene_style_target(
            anchor_record["person_stats"],
            anchor_record["background_stats"],
            reference_stats["background"],
            luminance_scale=A0_BASELINE.person_scene_luminance_scale,
            chroma_scale=A0_BASELINE.person_scene_chroma_scale,
        )
        before_background_stats = [record["background_stats"] for record in records]
        before_person_stats = [record["person_stats"] for record in records]
        after_background_stats = []
        after_person_stats = []
        contact_rows: list[tuple[str, Image.Image, Image.Image]] = []
        item_reports = []
        for record in records:
            path = Path(str(record["path"]))
            source, _ = load_srgb(path, alpha_policy="drop_near_opaque")
            masks = {
                name: load_mask(mask_path, source.size)
                for name, mask_path in record["mask_paths"].items()
            }
            source_background = record["background_stats"]
            source_person = record["person_stats"]
            background_target = background_style_target(
                source_background, reference_stats["background"]
            )
            plans = [
                RegionPlan(
                    "background",
                    masks["background"],
                    source_background,
                    background_target,
                    background_strength,
                    A0_BASELINE.background_luminance_cap,
                    A0_BASELINE.background_chroma_cap,
                ),
                RegionPlan(
                    "whole_person_including_garment",
                    masks["person"],
                    source_person,
                    person_target,
                    person_strength,
                    A0_BASELINE.person_luminance_cap,
                    A0_BASELINE.person_chroma_cap,
                ),
            ]
            corrected, transform = apply_region_plans(source, plans)
            output_path = corrected_dir / path.name.lower()
            final_output_path = Path(CANDIDATE_DIRECTORY) / path.name.lower()
            verification = save_srgb(corrected, output_path)
            after_background = region_stats(corrected, masks["background_core"])
            after_person = region_stats(corrected, masks["person_core"])
            after_background_stats.append(after_background)
            after_person_stats.append(after_person)
            distances = {
                "background_before": region_style_distance(
                    source_background, reference_stats["background"]
                ),
                "background_after": region_style_distance(
                    after_background, reference_stats["background"]
                ),
                "person_before": region_style_distance(source_person, person_target),
                "person_after": region_style_distance(after_person, person_target),
            }
            artifacts = artifact_metrics(source, corrected, masks)
            flags = []
            if transform["unauthorized_changed_pixels"] != 0:
                flags.append("unauthorized_pixels_changed")
            if transform["newly_clipped_percent_of_editable"] > 0.20:
                flags.append("new_clipping_above_0.20_percent")
            if distances["background_after"] > distances["background_before"] + 0.05:
                flags.append("background_distance_worsened")
            if distances["person_after"] > distances["person_before"] + 0.05:
                flags.append("person_distance_worsened")
            if artifacts["correction_jump_p99"] > 0.035:
                flags.append("abrupt_correction_transition")
            if artifacts["background_tone_collapse_percent"] > 25.0:
                flags.append("background_tone_collapse_risk")
            item_report = {
                "input": str(path),
                "input_sha256": manifest.input_hashes[str(path)],
                "output": final_output_path.as_posix(),
                "output_sha256": verification["sha256"],
                "pipeline": "person-background-two-anchor-v2",
                "baseline": {
                    "id": A0_BASELINE.baseline_id,
                    "fingerprint": A0_BASELINE.fingerprint,
                    "compatible": a0_compatible(
                        background_strength=background_strength,
                        person_strength=person_strength,
                        set_color_tolerance=set_color_tolerance,
                        runtime_compatibility=runtime_compatibility,
                    ),
                    "runtime_compatibility": runtime_compatibility,
                },
                "run_identity_sha256": identity["identity_sha256"],
                "mask_paths": record["mask_paths_relative"],
                "mask_diagnostics": record["mask_diagnostics"],
                "distances": {key: round(float(value), 6) for key, value in distances.items()},
                "transform": transform,
                "artifact_metrics": artifacts,
                "automatic_flags": flags,
                "baseline_diagnostics_passed": not flags,
                "automatic_checks_passed": not flags,
                "accepted": False,
                "status": "candidate",
                "source_info": record["source_info"],
                "export_verification": verification,
            }
            atomic_json(reports_dir / f"{path.name}.json", item_report)
            source_preview = make_proxy(source, max_edge=1000)
            corrected_preview = make_proxy(corrected, max_edge=1000)
            _save_preview(source_preview, previews_dir / f"{path.stem}-source.jpg")
            _save_preview(corrected_preview, previews_dir / f"{path.stem}-corrected.jpg")
            contact_rows.append((path.name, source_preview, corrected_preview))
            item_reports.append(item_report)
            validate_inputs_unchanged(manifest)

        before_background = _pairwise_style_summary(before_background_stats)
        before_person = _pairwise_style_summary(before_person_stats)
        after_background = _pairwise_style_summary(after_background_stats)
        after_person = _pairwise_style_summary(after_person_stats)
        background_set_passed = (
            after_background["mean"] <= before_background["mean"] + 1e-6
            and after_background["max"] <= set_color_tolerance
        )
        person_set_passed = (
            after_person["mean"] <= before_person["mean"] + 1e-6
            and after_person["max"] <= set_color_tolerance
        )
        summary = {
            "schema_version": 4,
            "pipeline": "person-background-two-anchor-v2",
            "baseline": {
                "id": A0_BASELINE.baseline_id,
                "fingerprint": A0_BASELINE.fingerprint,
                "compatible": a0_compatible(
                    background_strength=background_strength,
                    person_strength=person_strength,
                    set_color_tolerance=set_color_tolerance,
                    runtime_compatibility=runtime_compatibility,
                ),
                "runtime_compatibility": runtime_compatibility,
                "contract": A0_BASELINE.as_dict(),
            },
            "run_identity_sha256": identity["identity_sha256"],
            "execution_plan_sha256": execution_plan_sha256,
            "execution_plan_fingerprint": execution_plan["plan_sha256"],
            "sku": sku,
            "run_name": run_name,
            "status": "candidate",
            "accepted": False,
            "capability": {
                "mode": "look-consistency",
                "product_truth_applied_to_colour_target": False,
                "precise_sku_colour_fidelity_claimed": False,
                "statement": "A0 aligns background and whole-person look; it is not physical product-colour truth.",
            },
            "sku_profile_fingerprint": sku_profile["profile_fingerprint"],
            "product_truth_status": sku_profile["product_truth"]["status"],
            "reference_scope": "this_sku_only",
            "scene_reference": manifest.scene_reference,
            "person_anchor": str(anchor_record["name"]),
            "person_anchor_selection": anchor_selection,
            "auto_person_anchor": str(records[auto_anchor_index]["name"]),
            "person_anchor_ranking": anchor_ranking,
            "reference_info": asdict(reference_info),
            "reference_mask_paths": reference_mask_paths,
            "reference_mask_diagnostics": reference_diagnostics,
            "target_count": len(item_reports),
            "configuration": {
                "background_strength": background_strength,
                "person_strength": person_strength,
                "mask_backend": mask_backend,
                "set_color_tolerance": set_color_tolerance,
                "background_target": "this_sku_designated_scene",
                "person_target": "this_sku_whole_person_medoid_plus_scene_direction",
                "garment_policy": "same_transform_as_whole_person",
                "product_image_policy": "hashed_profile_evidence_only_not_a0_colour_target",
                "path_policy": "relative_to_sku_output",
                "output_layout": "workflow-sku-v2",
            },
            "set_consistency": {
                "background_before": before_background,
                "background_after": after_background,
                "person_before": before_person,
                "person_after": after_person,
                "background_passed": background_set_passed,
                "person_passed": person_set_passed,
                "all_passed": background_set_passed and person_set_passed,
            },
            "baseline_diagnostics_passed": all(
                item["automatic_checks_passed"] for item in item_reports
            )
            and background_set_passed
            and person_set_passed,
            # Deprecated compatibility alias.  It is a risk diagnostic, never
            # an approval decision.
            "automatic_checks_passed": all(
                item["automatic_checks_passed"] for item in item_reports
            )
            and background_set_passed
            and person_set_passed,
            "human_review_required": True,
            "items": item_reports,
        }
        _save_contact_sheet(
            contact_rows, make_proxy(reference, max_edge=1000), staging / "整套对照.jpg"
        )
        atomic_json(staging / "summary.json", summary)
        atomic_json(
            staging / "review-status.json",
            {
                "status": "candidate",
                "accepted": False,
                "latest_review": None,
                "human_review_required": True,
            },
        )
        validate_inputs_unchanged(manifest)
        _publish_flat_sku_output(staging, final, replace_output=replace_output)
        return final, summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
