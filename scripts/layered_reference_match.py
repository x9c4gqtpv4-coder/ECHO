#!/usr/bin/env python3
"""Create one review-gated layered colour-match candidate.

The command keeps background, skin, hair and garment transforms independent,
applies bounded OKLab corrections, and renders all regions from the original
source pixels in one pass.  It never marks the result approved automatically.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Always execute the current working-tree engine.  A stale non-editable package
# in the virtual environment must never silently override reviewed mask fixes.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from batch_color.image_io import load_mask, load_srgb, save_mask, save_srgb
from batch_color.runtime import runtime_identity
from batch_color.safety import atomic_json, file_hash
from batch_color.semantic import build_semantic_masks
from batch_color.sku_pipeline import (
    RegionPlan,
    apply_region_plans,
    artifact_metrics,
    background_style_target,
    optional_region_stats,
    region_distance,
    region_stats,
)


MASK_NAMES = (
    "background",
    "background_core",
    "background_transition",
    "skin",
    "skin_core",
    "skin_transition",
    "hair",
    "hair_core",
    "garment",
    "garment_core",
    "garment_transition",
    "accessory_protect",
    "unknown_person",
    "conflicts",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-mask-dir", type=Path)
    parser.add_argument("--reference-mask-dir", type=Path)
    parser.add_argument("--background-strength", type=float, default=0.78)
    parser.add_argument(
        "--skin-strength",
        type=float,
        default=None,
        help="Legacy override that sets both face and body skin strength.",
    )
    parser.add_argument("--face-strength", type=float, default=0.78)
    parser.add_argument("--body-skin-strength", type=float, default=0.52)
    parser.add_argument("--hair-strength", type=float, default=0.28)
    parser.add_argument("--garment-strength", type=float, default=0.78)
    return parser


def _load_masks(directory: Path, size: tuple[int, int]) -> dict[str, Image.Image]:
    result = {name: load_mask(directory / f"{name}.png", size) for name in MASK_NAMES}
    for name in ("face_skin", "body_skin"):
        path = directory / f"{name}.png"
        if path.is_file():
            result[name] = load_mask(path, size)
    return result


def _probability_mask(values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    proxy = Image.fromarray(
        np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L"
    )
    return proxy.resize(size, Image.Resampling.BILINEAR)


def _save_masks(masks: object, directory: Path) -> dict[str, Image.Image]:
    directory.mkdir(parents=True, exist_ok=True)
    result: dict[str, Image.Image] = {}
    for name in MASK_NAMES:
        mask = getattr(masks, name)
        save_mask(mask, directory / f"{name}.png")
        result[name] = mask
    return result


def _get_masks(
    image_path: Path,
    image: Image.Image,
    existing: Path | None,
    output: Path,
) -> tuple[dict[str, Image.Image], dict[str, object]]:
    if existing:
        return _load_masks(existing, image.size), {"source": "reused", "path": str(existing)}
    built = build_semantic_masks(
        image_path,
        image,
        garment_kind="set",
        garment_hint="any",
        mask_backend="vision",
        parser_backend="mediapipe",
        pose_backend="vision",
        proxy_edge=768,
    )
    result = _save_masks(built, output)
    probabilities = built.probabilities
    face = np.clip(
        probabilities["face_edit_authorization"]
        * (1.0 - probabilities["face_feature_protection"]),
        0.0,
        1.0,
    )
    body = np.clip(
        probabilities["body_edit_authorization"] * (1.0 - face),
        0.0,
        1.0,
    )
    for name, values in (("face_skin", face), ("body_skin", body)):
        mask = _probability_mask(values, image.size)
        save_mask(mask, output / f"{name}.png")
        result[name] = mask
    return result, {
        "source": "generated",
        "diagnostics": built.diagnostics,
        "fatal_flags": list(built.diagnostics.get("fatal_flags", [])),
        "warning_flags": list(built.diagnostics.get("warnings", [])),
    }


def _thumb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = Image.new("RGB", size, "white")
    copy = image.convert("RGB").copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    result.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return result


def _save_triptych(source: Image.Image, corrected: Image.Image, reference: Image.Image, path: Path) -> None:
    cell = (560, 760)
    header = 42
    sheet = Image.new("RGB", (cell[0] * 3, cell[1] + header), (244, 244, 244))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(
        (("SOURCE", source), ("LAYERED CORRECTED", corrected), ("REFERENCE", reference))
    ):
        draw.text((index * cell[0] + 12, 15), label, fill="black", font=font)
        sheet.paste(_thumb(image, cell), (index * cell[0], header))
    save_srgb(sheet, path, quality=94)


def _upper_skin_box(mask: Image.Image) -> tuple[int, int, int, int]:
    data = np.asarray(mask.convert("L"), dtype=np.uint8).copy()
    data[int(mask.height * 0.48) :] = 0
    ys, xs = np.nonzero(data >= 96)
    if not len(xs):
        return (0, 0, mask.width, max(1, int(mask.height * 0.45)))
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    px = max(24, int((x1 - x0) * 0.18))
    py = max(24, int((y1 - y0) * 0.18))
    return max(0, x0 - px), max(0, y0 - py), min(mask.width, x1 + px), min(mask.height, y1 + py)


def _save_face_detail(
    source: Image.Image,
    corrected: Image.Image,
    reference: Image.Image,
    source_skin: Image.Image,
    reference_skin: Image.Image,
    path: Path,
) -> None:
    source_box = _upper_skin_box(source_skin)
    reference_box = _upper_skin_box(reference_skin)
    panels = (
        ("SOURCE FACE", source.crop(source_box)),
        ("CORRECTED FACE", corrected.crop(source_box)),
        ("REFERENCE FACE", reference.crop(reference_box)),
    )
    cell = (520, 520)
    header = 32
    sheet = Image.new("RGB", (cell[0] * 3, cell[1] + header), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(panels):
        draw.text((index * cell[0] + 10, 10), label, fill="black", font=font)
        sheet.paste(_thumb(image, cell), (index * cell[0], header))
    save_srgb(sheet, path, quality=96)


def main() -> int:
    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source, source_info = load_srgb(args.source)
    reference, reference_info = load_srgb(args.reference)
    source_masks, source_mask_report = _get_masks(
        args.source, source, args.source_mask_dir, args.output_dir / "masks" / "source"
    )
    reference_masks, reference_mask_report = _get_masks(
        args.reference, reference, args.reference_mask_dir, args.output_dir / "masks" / "reference"
    )

    source_stat_masks = {
        name: source_masks[f"{name}_core"]
        for name in ("background", "skin", "hair", "garment")
    }
    reference_stat_masks = {
        name: reference_masks[f"{name}_core"]
        for name in ("background", "skin", "hair", "garment")
    }
    split_skin = all(
        name in masks
        for masks in (source_masks, reference_masks)
        for name in ("face_skin", "body_skin")
    )
    if split_skin:
        split_pairs = {
            "face_skin": (source_masks["face_skin"], reference_masks["face_skin"]),
            "body_skin": (source_masks["body_skin"], reference_masks["body_skin"]),
        }
        for name, (source_mask, reference_mask) in split_pairs.items():
            if optional_region_stats(source, source_mask) is None or optional_region_stats(
                reference, reference_mask
            ) is None:
                split_skin = False
                break
            source_stat_masks[name] = source_mask
            reference_stat_masks[name] = reference_mask

    stat_names = ["background", "hair", "garment"]
    stat_names += ["face_skin", "body_skin"] if split_skin else ["skin"]
    source_stats = {
        name: region_stats(source, source_stat_masks[name]) for name in stat_names
    }
    reference_stats = {
        name: region_stats(reference, reference_stat_masks[name]) for name in stat_names
    }
    targets = dict(reference_stats)
    targets["background"] = background_style_target(
        source_stats["background"], reference_stats["background"]
    )
    plans = [
        RegionPlan(
            "background",
            source_masks["background"],
            source_stats["background"],
            targets["background"],
            args.background_strength,
            0.060,
            0.018,
        ),
        RegionPlan(
            "hair",
            source_masks["hair"],
            source_stats["hair"],
            targets["hair"],
            args.hair_strength,
            0.035,
            0.016,
        ),
        RegionPlan(
            "garment",
            source_masks["garment"],
            source_stats["garment"],
            targets["garment"],
            args.garment_strength,
            0.035,
            0.018,
        ),
    ]
    if args.skin_strength is not None:
        face_strength = body_skin_strength = args.skin_strength
    else:
        face_strength = args.face_strength
        body_skin_strength = args.body_skin_strength
    if split_skin:
        plans.extend(
            [
                RegionPlan(
                    "face_skin",
                    source_masks["face_skin"],
                    source_stats["face_skin"],
                    targets["face_skin"],
                    face_strength,
                    0.065,
                    0.018,
                ),
                RegionPlan(
                    "body_skin",
                    source_masks["body_skin"],
                    source_stats["body_skin"],
                    targets["body_skin"],
                    body_skin_strength,
                    0.055,
                    0.016,
                ),
            ]
        )
    else:
        plans.append(
            RegionPlan(
                "skin",
                source_masks["skin"],
                source_stats["skin"],
                targets["skin"],
                face_strength,
                0.080,
                0.018,
            )
        )
    corrected, render = apply_region_plans(source, plans)
    corrected_path = args.output_dir / "corrected_layered.png"
    triptych_path = args.output_dir / "source_corrected_reference.jpg"
    face_path = args.output_dir / "face_detail.jpg"
    encoded = save_srgb(corrected, corrected_path)
    _save_triptych(source, corrected, reference, triptych_path)
    _save_face_detail(
        source,
        corrected,
        reference,
        source_masks["skin"],
        reference_masks["skin"],
        face_path,
    )

    corrected_stats = {
        name: region_stats(corrected, source_stat_masks[name]) for name in stat_names
    }
    regions = {}
    for name in source_stats:
        before = region_distance(source_stats[name], targets[name])
        after = region_distance(corrected_stats[name], targets[name])
        regions[name] = {
            "source": asdict(source_stats[name]),
            "reference": asdict(reference_stats[name]),
            "bounded_target": asdict(targets[name]),
            "corrected": asdict(corrected_stats[name]),
            "distance_before": round(before, 6),
            "distance_after": round(after, 6),
            "distance_improvement": round(before - after, 6),
        }
    quality_reasons = []
    if render["unauthorized_changed_pixels"]:
        quality_reasons.append("pixels_changed_outside_authorized_masks")
    if render["newly_clipped_percent_of_editable"] > 0.10:
        quality_reasons.append("new_clipping_above_0.10_percent")
    if any(item["distance_after"] > item["distance_before"] for item in regions.values()):
        quality_reasons.append("one_or_more_region_distances_increased")
    report = {
        "schema": "layered-reference-match/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": {
            "runtime_identity": runtime_identity(),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_hash(Path(__file__).resolve()),
        },
        "status": "candidate_review_required",
        "accepted": False,
        "automatic_checks_passed": not quality_reasons,
        "review_reasons": quality_reasons or ["human_visual_review_required"],
        "contract": {
            "render_passes": 1,
            "regions": stat_names,
            "skin_statistics": "face_and_body_separate" if split_skin else "combined_fallback",
            "identity_guards": [
                "hair_luminance_bounded",
                "garment_chroma_bounded",
                "continuous_garment_interior",
                "face_features_protected",
                "soft_masks",
            ],
        },
        "inputs": {
            "source": {"path": str(args.source), "sha256": file_hash(args.source), "info": asdict(source_info)},
            "reference": {"path": str(args.reference), "sha256": file_hash(args.reference), "info": asdict(reference_info)},
        },
        "mask_reports": {"source": source_mask_report, "reference": reference_mask_report},
        "render": render,
        "artifact_metrics": artifact_metrics(source, corrected, source_masks),
        "regions": regions,
        "output": {
            "corrected": {"path": str(corrected_path), **encoded},
            "triptych": str(triptych_path),
            "face_detail": str(face_path),
        },
    }
    atomic_json(args.output_dir / "layered-match-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
