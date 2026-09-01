from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from batch_color.color import oklab_to_srgb, srgb_to_oklab
from batch_color.image_io import image_to_float
from batch_color.profile import (
    ColorProfile,
    RegionStatistics,
    SurfaceStatistics,
    analyse_background,
    analyse_background_surface,
    evaluate_surface,
    validate_profile,
    evidence_status,
    reference_evidence_verified,
)
from batch_color.segmentation import estimate_studio_background_mask


@dataclass(frozen=True)
class TransferReport:
    path: str
    mask_backend: str
    strength: float
    background_distance_before: float
    background_distance_after: float
    background_improvement_percent: float
    spatial_distance_before: float
    spatial_distance_after: float
    spatial_improvement_percent: float
    gamut_clipped_percent: float
    source_background: dict[str, object]
    target_background: dict[str, object]
    output_background: dict[str, object]
    accepted: bool
    baseline_checks_passed: bool
    status: str
    mode: str
    no_op: bool
    review_reasons: list[str]
    curve_min_slope: float
    curve_max_slope: float
    surface_enabled: bool
    surface_diagnostics: dict[str, object]
    reference_evidence: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _control_points(source_l: np.ndarray, target_l: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_l = np.asarray(source_l, dtype=np.float32)
    target_l = np.asarray(target_l, dtype=np.float32)

    low_slope = (target_l[1] - target_l[0]) / max(float(source_l[1] - source_l[0]), 1e-4)
    high_slope = (target_l[-1] - target_l[-2]) / max(
        float(source_l[-1] - source_l[-2]), 1e-4
    )
    low_y = target_l[0] - np.clip(low_slope, 0.25, 2.5) * source_l[0]
    high_y = target_l[-1] + np.clip(high_slope, 0.25, 2.5) * (1.0 - source_l[-1])

    x = np.concatenate(([0.0], source_l, [1.0])).astype(np.float32)
    y = np.concatenate(([low_y], target_l, [high_y])).astype(np.float32)
    y = np.clip(y, 0.0, 1.0)
    y = np.maximum.accumulate(y)

    keep = np.concatenate(([True], np.diff(x) > 1e-5))
    return x[keep], y[keep]


def _map_luminance(values: np.ndarray, source_l: list[float], target_l: list[float]) -> np.ndarray:
    x, y = _bounded_luminance_curve(source_l, target_l)
    return np.interp(values, x, y).astype(np.float32)


def _bounded_luminance_curve(source_l: list[float], target_l: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Smooth positive slopes BEFORE integration, with no L-dependent protection.

    At a fixed pixel/mask, convex blending with identity preserves monotonicity.
    This is not a guarantee about spatial mask edges or post-gamut artifacts.
    """
    grid = np.linspace(0.0, 1.0, 4097)
    if np.allclose(source_l, target_l, atol=1e-7, rtol=0):
        return grid, grid.copy()
    if np.ptp(source_l) < 1e-4:
        # A flat source cannot identify seven independent quantile segments.
        # Preserve black/white and match the median with a bounded simple map.
        return grid, _subject_luminance(grid, source_l[3], target_l[3]).astype(float)
    x, y = _control_points(np.asarray(source_l), np.asarray(target_l))
    raw = np.interp(grid, x, y)
    slopes = np.clip(np.diff(raw) * 4096, 0.35, 2.5)
    axis = np.arange(-64, 65, dtype=float)
    kernel = np.exp(-0.5 * (axis / 24.0) ** 2)
    kernel /= kernel.sum()
    slopes = np.convolve(np.pad(slopes, (64, 64), mode="edge"), kernel, mode="valid")
    integrated = np.concatenate(([0.0], np.cumsum(slopes) / 4096))
    integrated /= max(1.0, float(integrated[-1]))
    offset = np.clip(np.median(raw - integrated), 0.0, max(0.0, 1.0 - integrated[-1]))
    return grid, integrated + offset


def _subject_luminance(values: np.ndarray, source_median: float, target_median: float) -> np.ndarray:
    """Endpoint-anchored exposure-like map, NOT a semantic skin target.

    f(L)=kL/(1+(k-1)L): f(0)=0, f(1)=1, f'>0; k in [0.5,2]
    bounds slopes to [0.5,2]. No additive black lift or luminance-gated blend.
    """
    source = np.clip(source_median, 1e-4, 1 - 1e-4)
    target = np.clip(target_median, 1e-4, 1 - 1e-4)
    scale = np.clip((target / (1 - target)) / (source / (1 - source)), 0.5, 2.0)
    return (scale * values / (1 + (scale - 1) * values)).astype(np.float32)


def _region_distance(left: RegionStatistics, right: RegionStatistics) -> float:
    l_delta = np.asarray(left.l_quantiles) - np.asarray(right.l_quantiles)
    chroma_delta = np.array(
        [left.a_median - right.a_median, left.b_median - right.b_median]
    )
    return float(100.0 * np.sqrt(np.mean(np.square(l_delta)) + np.sum(np.square(chroma_delta))))


def _surface_distance(left: SurfaceStatistics, right: SurfaceStatistics) -> float:
    axis = np.linspace(-1.0, 1.0, 21, dtype=np.float32)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    delta = evaluate_surface(left, x, y) - evaluate_surface(right, x, y)
    return float(100.0 * np.sqrt(np.mean(np.sum(np.square(delta), axis=-1))))


def apply_profile(
    image: Image.Image,
    profile: ColorProfile,
    *,
    strength: float = 0.85,
    tile_rows: int = 512,
    use_spatial_surface: bool = True,
    background_mask: Image.Image | None = None,
    mask_backend: str = "heuristic-color",
    mode: str = "background",
    protected_mask: Image.Image | None = None,
) -> tuple[Image.Image, TransferReport, Image.Image]:
    validate_profile(profile)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be finite and between 0.0 and 1.0")
    if mode not in {"background", "both"}:
        raise ValueError("mode must be background or both (experimental)")
    if tile_rows <= 0:
        raise ValueError("tile_rows must be positive")

    target_background = profile.background
    target_surface = profile.background_surface
    if background_mask is None:
        source_background = analyse_background(image)
        source_surface = analyse_background_surface(image)
        background_mask = estimate_studio_background_mask(
            image,
            source_background,
            source_surface,
        )
        mask_backend = "heuristic-color"
    else:
        background_mask = background_mask.convert("L")
        if background_mask.size != image.size:
            raise ValueError("Background mask must match canonical image geometry")
    if protected_mask is not None and protected_mask.size != image.size:
        raise ValueError("Protection mask must match canonical image geometry")
    protection_u8 = np.zeros((image.height, image.width), dtype=np.uint8)
    if protected_mask is not None:
        protection_u8 = np.asarray(protected_mask.convert("L"), dtype=np.uint8)
        background_mask = Image.fromarray(np.minimum(np.asarray(background_mask), 255 - protection_u8))
    # The same explicit source domain drives estimation and baseline measurement.
    # These measurements are deliberately NOT called independent quality approval.
    source_background = analyse_background(image, background_mask)
    source_surface = analyse_background_surface(image, background_mask)
    requested_surface = use_spatial_surface
    verified_reference = reference_evidence_verified(profile)
    use_spatial_surface = (use_spatial_surface and verified_reference
                           and source_surface.trusted and target_surface.trusted)
    background_mask_u8 = np.asarray(background_mask, dtype=np.uint8)
    source_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
    output_u8 = np.empty_like(source_u8)
    clipped_pixels = 0
    total_pixels = source_u8.shape[0] * source_u8.shape[1]

    delta_a = target_background.a_median - source_background.a_median
    delta_b = target_background.b_median - source_background.b_median
    curve_x, curve_y = _bounded_luminance_curve(source_background.l_quantiles, target_background.l_quantiles)

    for row_start in range(0, source_u8.shape[0], tile_rows):
        row_end = min(row_start + tile_rows, source_u8.shape[0])
        rgb = source_u8[row_start:row_end].astype(np.float32) / 255.0
        lab = srgb_to_oklab(rgb)
        mapped = lab.copy()
        target_l = np.interp(lab[..., 0], curve_x, curve_y).astype(np.float32)
        background_weight = (
            background_mask_u8[row_start:row_end].astype(np.float32) / 255.0
        )
        global_background_mapped = lab.copy()
        global_background_mapped[..., 0] = target_l
        global_background_mapped[..., 1] += delta_a
        global_background_mapped[..., 2] += delta_b
        subject_mapped = lab.copy()
        subject_mapped[..., 0] = _subject_luminance(
            lab[..., 0], source_background.l_quantiles[3], target_background.l_quantiles[3]
        )
        # Avoid assigning a nonzero background-derived chroma to pure black/white.
        shadow_weight = np.clip(lab[..., 0] / 0.25, 0, 1)
        highlight_weight = np.clip((1 - lab[..., 0]) / 0.15, 0, 1)
        color_weight = (shadow_weight ** 2 * (3 - 2 * shadow_weight)
                        * highlight_weight ** 2 * (3 - 2 * highlight_weight))
        subject_mapped[..., 1] += color_weight * delta_a
        subject_mapped[..., 2] += color_weight * delta_b

        y_values = 2.0 * np.arange(row_start, row_end, dtype=np.float32) / max(
            source_u8.shape[0] - 1, 1
        ) - 1.0
        x_values = 2.0 * np.arange(source_u8.shape[1], dtype=np.float32) / max(
            source_u8.shape[1] - 1, 1
        ) - 1.0
        grid_y, grid_x = np.meshgrid(y_values, x_values, indexing="ij")
        local_delta = evaluate_surface(target_surface, grid_x, grid_y) - evaluate_surface(
            source_surface, grid_x, grid_y
        )
        local_delta[..., 0] = np.clip(local_delta[..., 0], -0.12, 0.12)
        local_delta[..., 1] = np.clip(local_delta[..., 1], -0.035, 0.035)
        local_delta[..., 2] = np.clip(local_delta[..., 2], -0.035, 0.035)
        background_mapped = lab + local_delta if use_spatial_surface else global_background_mapped
        person_mapped = subject_mapped if mode == "both" else lab
        mapped = (background_weight[..., None] * background_mapped
                  + (1.0 - background_weight[..., None]) * person_mapped)
        editable = 1.0 - protection_u8[row_start:row_end].astype(np.float32) / 255.0
        corrected_lab = lab + float(strength) * editable[..., None] * (mapped - lab)
        if not np.all(np.isfinite(corrected_lab)):
            raise ValueError("Color transform produced non-finite Oklab pixels")
        corrected_rgb = oklab_to_srgb(corrected_lab)
        if not np.all(np.isfinite(corrected_rgb)):
            raise ValueError("Color transform produced non-finite RGB pixels; refusing quantization")
        out_of_gamut = np.any((corrected_rgb < -1e-6) | (corrected_rgb > 1.0 + 1e-6), axis=-1)
        out_of_gamut &= editable > 0
        clipped_pixels += int(np.count_nonzero(out_of_gamut))
        output_u8[row_start:row_end] = np.round(
            np.clip(corrected_rgb, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        identity = protection_u8[row_start:row_end] == 255
        if mode == "background":
            identity = identity | (background_mask_u8[row_start:row_end] == 0)
        if strength == 0:
            identity = np.ones_like(identity)
        output_u8[row_start:row_end][identity] = source_u8[row_start:row_end][identity]

    output = Image.fromarray(output_u8, mode="RGB")
    output_background = analyse_background(output, background_mask)
    output_surface = analyse_background_surface(output, background_mask)
    before = _region_distance(source_background, target_background)
    after = _region_distance(output_background, target_background)
    improvement = 0.0 if before < 1e-6 else 100.0 * (before - after) / before
    spatial_before = _surface_distance(source_surface, target_surface)
    spatial_after = _surface_distance(output_surface, target_surface)
    spatial_improvement = (
        0.0
        if spatial_before < 1e-6
        else 100.0 * (spatial_before - spatial_after) / spatial_before
    )

    no_op = bool(np.array_equal(output_u8, source_u8))
    baseline_passed = (
        (after < before or (no_op and before < 1e-4))
        and clipped_pixels / total_pixels < 0.02
        and (not use_spatial_surface or spatial_after < spatial_before or (no_op and spatial_before < 1e-4))
    )
    reasons = ["automatic_artifact_validation_not_complete", "human_review_required"]
    if not verified_reference:
        reasons.append("reference_evidence_unverified_global_only_rebuild_bundle")
    if use_spatial_surface:
        reasons.append("spatial_correspondence_requires_human_review")
    if requested_surface and not use_spatial_surface:
        reasons.append("spatial_surface_disabled_untrusted_source_or_reference")
    if mode == "both":
        reasons.append("person_adjustment_is_background_driven_not_semantic_matching")
    if profile.background_sampling != "masked-core":
        reasons.append("legacy_reference_border_statistics_rebuild_profile_recommended")
    if not baseline_passed:
        reasons.append("background_baseline_checks_failed")
    slopes = np.diff(curve_y) / np.diff(curve_x)
    report = TransferReport(
        path="spatial-surface" if use_spatial_surface else "global-monotone",
        mask_backend=mask_backend,
        strength=round(float(strength), 4),
        background_distance_before=round(before, 4),
        background_distance_after=round(after, 4),
        background_improvement_percent=round(improvement, 2),
        spatial_distance_before=round(spatial_before, 4),
        spatial_distance_after=round(spatial_after, 4),
        spatial_improvement_percent=round(spatial_improvement, 2),
        gamut_clipped_percent=round(100.0 * clipped_pixels / total_pixels, 4),
        source_background=asdict(source_background),
        target_background=asdict(target_background),
        output_background=asdict(output_background),
        accepted=False,
        baseline_checks_passed=bool(baseline_passed),
        status="review",
        mode=mode,
        no_op=no_op,
        review_reasons=reasons,
        curve_min_slope=float(slopes.min()),
        curve_max_slope=float(slopes.max()),
        surface_enabled=bool(use_spatial_surface),
        surface_diagnostics={"source": asdict(source_surface), "reference": asdict(target_surface)},
        reference_evidence=evidence_status(profile),
    )
    # Report serialization must be valid BEFORE callers write a candidate.
    import json
    json.dumps(report.as_dict(), allow_nan=False)
    return output, report, background_mask


def select_profile_path(
    image: Image.Image,
    profile: ColorProfile,
    *,
    strength: float = 0.85,
    path: str = "auto",
    background_mask: Image.Image | None = None,
    mask_backend: str = "heuristic-color",
    mode: str = "background",
    protected_mask: Image.Image | None = None,
) -> tuple[Image.Image, TransferReport, Image.Image]:
    if path == "surface":
        return apply_profile(
            image,
            profile,
            strength=strength,
            use_spatial_surface=True,
            background_mask=background_mask,
            mask_backend=mask_backend,
            mode=mode,
            protected_mask=protected_mask,
        )
    if path == "global":
        return apply_profile(
            image,
            profile,
            strength=strength,
            use_spatial_surface=False,
            background_mask=background_mask,
            mask_backend=mask_backend,
            mode=mode,
            protected_mask=protected_mask,
        )
    if path != "auto":
        raise ValueError("path must be one of: auto, global, surface")

    surface_candidate = apply_profile(
        image,
        profile,
        strength=strength,
        use_spatial_surface=True,
        background_mask=background_mask,
        mask_backend=mask_backend,
        mode=mode,
        protected_mask=protected_mask,
    )

    if not surface_candidate[1].surface_enabled:
        return surface_candidate  # Already computed a conservative global fallback.

    global_candidate = apply_profile(
        image,
        profile,
        strength=strength,
        use_spatial_surface=False,
        background_mask=background_mask,
        mask_backend=mask_backend,
        mode=mode,
        protected_mask=protected_mask,
    )

    if surface_candidate[1].baseline_checks_passed != global_candidate[1].baseline_checks_passed:
        return surface_candidate if surface_candidate[1].baseline_checks_passed else global_candidate

    if not surface_candidate[1].baseline_checks_passed:
        # Failed candidates must not select the more complex extrapolating path.
        return global_candidate

    surface_score = (
        surface_candidate[1].background_distance_after
        + surface_candidate[1].spatial_distance_after
        + 5.0 * surface_candidate[1].gamut_clipped_percent
    )
    global_score = (
        global_candidate[1].background_distance_after
        + global_candidate[1].spatial_distance_after
        + 5.0 * global_candidate[1].gamut_clipped_percent
    )
    return surface_candidate if surface_score < global_score else global_candidate


def mean_oklab_change(source: Image.Image, output: Image.Image) -> float:
    source_lab = srgb_to_oklab(image_to_float(source))
    output_lab = srgb_to_oklab(image_to_float(output))
    return float(100.0 * np.mean(np.linalg.norm(output_lab - source_lab, axis=-1)))
