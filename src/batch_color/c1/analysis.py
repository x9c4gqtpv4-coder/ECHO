from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from batch_color.c1.schema import C1_ANALYZER_ID, C1_CONFIG, C1AnalyzerConfig
from batch_color.color import linear_rgb_to_oklab, srgb_to_linear


_LUMINANCE_QUANTILES = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90])
_EXPLICIT_NEUTRAL_EVIDENCE = {"human_confirmed", "same_entity"}
_NEUTRAL_EVIDENCE = {"automatic", *_EXPLICIT_NEUTRAL_EVIDENCE}
_EXPLICIT_COMPARISON_EVIDENCE = {"human_confirmed", "same_surface"}
_COMPARISON_EVIDENCE = {"automatic", *_EXPLICIT_COMPARISON_EVIDENCE}


@dataclass(frozen=True)
class _PreparedEvidence:
    rgb: np.ndarray
    linear_rgb: np.ndarray
    oklab: np.ndarray
    luminance: np.ndarray
    region: np.ndarray
    tonal_valid: np.ndarray
    neutral_valid: np.ndarray
    region_pixels: int
    tonal_pixels: int
    neutral_pixels: int
    clipped_fraction: float
    proxy_size: tuple[int, int]


def _proxy(image: Image.Image, max_edge: int) -> Image.Image:
    result = image.convert("RGB").copy()
    result.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return result


def _resized_mask(mask: Image.Image | None, image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones((size[1], size[0]), dtype=bool)
    if mask.size != image.size:
        raise ValueError("C1 masks must match the corresponding canonical image")
    resized = mask.convert("L").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8) >= 128


def _prepare(
    image: Image.Image,
    region_mask: Image.Image | None,
    neutral_mask: Image.Image | None,
    *,
    explicit_neutral: bool,
    config: C1AnalyzerConfig,
) -> _PreparedEvidence:
    proxy = _proxy(image, config.max_edge)
    rgb = np.asarray(proxy, dtype=np.float32) / 255.0
    linear = srgb_to_linear(rgb)
    oklab = linear_rgb_to_oklab(linear)
    luminance = (
        0.2126729 * linear[..., 0]
        + 0.7151522 * linear[..., 1]
        + 0.0721750 * linear[..., 2]
    )
    region = _resized_mask(region_mask, image, proxy.size)
    clipped = np.any(
        (rgb <= config.clip_low_srgb) | (rgb >= config.clip_high_srgb), axis=-1
    )
    tonal_valid = region & ~clipped & (luminance > config.min_linear_luminance)
    neutral_domain = region & _resized_mask(neutral_mask, image, proxy.size)
    neutral_valid = neutral_domain & ~clipped
    if not explicit_neutral:
        chroma = np.sqrt(np.square(oklab[..., 1]) + np.square(oklab[..., 2]))
        neutral_valid &= chroma <= config.neutral_oklab_chroma_max
        neutral_valid &= oklab[..., 0] >= config.neutral_lightness_low
        neutral_valid &= oklab[..., 0] <= config.neutral_lightness_high
    region_pixels = int(np.count_nonzero(region))
    return _PreparedEvidence(
        rgb=rgb,
        linear_rgb=linear,
        oklab=oklab,
        luminance=luminance,
        region=region,
        tonal_valid=tonal_valid,
        neutral_valid=neutral_valid,
        region_pixels=region_pixels,
        tonal_pixels=int(np.count_nonzero(tonal_valid)),
        neutral_pixels=int(np.count_nonzero(neutral_valid)),
        clipped_fraction=(
            1.0 if region_pixels == 0 else float(np.count_nonzero(clipped & region) / region_pixels)
        ),
        proxy_size=proxy.size,
    )


def _finite(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("C1 analysis produced a non-finite value")
    return value


def _rounded(values: np.ndarray | list[float], digits: int = 7) -> list[float]:
    return [round(_finite(value), digits) for value in values]


def _median_absolute_deviation(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = np.median(values)
    return _finite(np.median(np.abs(values - median)))


def _block_medians(
    evidence: _PreparedEvidence,
    values: np.ndarray,
    valid: np.ndarray,
    config: C1AnalyzerConfig,
) -> dict[tuple[int, int], float]:
    height, width = valid.shape
    result: dict[tuple[int, int], float] = {}
    for row in range(config.grid_size):
        y0 = row * height // config.grid_size
        y1 = (row + 1) * height // config.grid_size
        for column in range(config.grid_size):
            x0 = column * width // config.grid_size
            x1 = (column + 1) * width // config.grid_size
            cell = valid[y0:y1, x0:x1]
            if int(np.count_nonzero(cell)) < config.min_spatial_cell_pixels:
                continue
            result[(row, column)] = _finite(np.median(values[y0:y1, x0:x1][cell]))
    return result


def _exposure_analysis(
    source: _PreparedEvidence,
    target: _PreparedEvidence,
    config: C1AnalyzerConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    sufficient = (
        source.tonal_pixels >= config.min_region_pixels
        and target.tonal_pixels >= config.min_region_pixels
    )
    if not sufficient:
        empty = {
            "status": "insufficient_region_evidence",
            "relative_exposure_like_stops": None,
            "quantile_stops": [],
            "fit_mad_stops": None,
            "claim_boundary": "display-referred relative gain, not camera EV",
        }
        return empty, {"status": "unavailable"}

    source_q = np.quantile(source.luminance[source.tonal_valid], _LUMINANCE_QUANTILES)
    target_q = np.quantile(target.luminance[target.tonal_valid], _LUMINANCE_QUANTILES)
    stops = np.log2((target_q + 1.0e-6) / (source_q + 1.0e-6))
    exposure = _finite(np.median(stops))
    residual = stops - exposure
    fit_mad = _median_absolute_deviation(stops)
    fit_span = _finite(np.ptp(stops))

    source_blocks = _block_medians(source, source.luminance, source.tonal_valid, config)
    target_blocks = _block_medians(target, target.luminance, target.tonal_valid, config)
    common = sorted(source_blocks.keys() & target_blocks.keys())
    block_stops = np.asarray(
        [
            np.log2((target_blocks[key] + 1.0e-6) / (source_blocks[key] + 1.0e-6))
            for key in common
        ],
        dtype=np.float64,
    )
    spatial_mad = _median_absolute_deviation(block_stops)
    spatial = {
        "status": "available" if len(common) >= 3 else "insufficient_common_blocks",
        "common_blocks": len(common),
        "block_relative_stops": _rounded(block_stops, 5),
        "dispersion_mad_stops": round(spatial_mad, 6) if len(common) else None,
    }
    parallel = (
        fit_mad <= config.exposure_parallel_mad_stops
        and fit_span <= config.exposure_fit_review_stops
    )
    stable = len(common) >= 3 and spatial_mad <= config.spatial_exposure_review_stops
    analysis_status = "valid" if parallel and stable else "compound_or_composition_unstable"
    exposure_report = {
        "status": analysis_status,
        "relative_exposure_like_stops": round(exposure, 6),
        "quantile_probabilities": _rounded(_LUMINANCE_QUANTILES, 2),
        "source_linear_luminance_quantiles": _rounded(source_q),
        "target_linear_luminance_quantiles": _rounded(target_q),
        "quantile_stops": _rounded(stops, 6),
        "fit_mad_stops": round(fit_mad, 6),
        "fit_span_stops": round(fit_span, 6),
        "fit_span_supported": bool(fit_span <= config.exposure_fit_review_stops),
        "parallel_gain_supported": bool(parallel),
        "claim_boundary": "display-referred relative gain, not camera EV",
    }
    contrast_log2_ratio = np.log2(
        ((target_q[-1] + 1.0e-6) / (target_q[0] + 1.0e-6))
        / ((source_q[-1] + 1.0e-6) / (source_q[0] + 1.0e-6))
    )
    tone_report = {
        "status": "descriptive_only",
        "shadow_residual_stops": round(_finite(np.mean(residual[:2])), 6),
        "midtone_residual_stops": round(_finite(residual[2]), 6),
        "highlight_residual_stops": round(_finite(np.mean(residual[3:])), 6),
        "contrast_log2_ratio": round(_finite(contrast_log2_ratio), 6),
        "display_tone_difference_only": True,
    }
    return {**exposure_report, "tone": tone_report}, spatial


def _xyz_and_chromaticity(linear_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    red, green, blue = linear_rgb[:, 0], linear_rgb[:, 1], linear_rgb[:, 2]
    x_value = 0.4124564 * red + 0.3575761 * green + 0.1804375 * blue
    y_value = 0.2126729 * red + 0.7151522 * green + 0.0721750 * blue
    z_value = 0.0193339 * red + 0.1191920 * green + 0.9503041 * blue
    xyz = np.stack([x_value, y_value, z_value], axis=-1)
    total = np.sum(xyz, axis=-1)
    valid = total > 1.0e-8
    xy = np.zeros((len(xyz), 2), dtype=np.float64)
    xy[valid] = xyz[valid, :2] / total[valid, None]
    denominator = x_value + 15.0 * y_value + 3.0 * z_value
    uv = np.zeros((len(xyz), 2), dtype=np.float64)
    uv_valid = denominator > 1.0e-8
    uv[uv_valid, 0] = 4.0 * x_value[uv_valid] / denominator[uv_valid]
    uv[uv_valid, 1] = 9.0 * y_value[uv_valid] / denominator[uv_valid]
    return xy[valid & uv_valid], uv[valid & uv_valid], valid & uv_valid


def _mccamy_cct(xy: np.ndarray) -> float | None:
    x_value, y_value = map(float, xy)
    denominator = y_value - 0.1858
    if abs(denominator) < 1.0e-7:
        return None
    n = (x_value - 0.3320) / denominator
    cct = -449.0 * n**3 + 3525.0 * n**2 - 6823.3 * n + 5520.33
    if not np.isfinite(cct) or not 1667.0 <= cct <= 25000.0:
        return None
    return float(cct)


def _whitepoint_estimate(evidence: _PreparedEvidence, *, explicit: bool) -> dict[str, object] | None:
    pixels = evidence.linear_rgb[evidence.neutral_valid]
    labs = evidence.oklab[evidence.neutral_valid]
    if len(pixels) == 0:
        return None
    xy, uv, valid = _xyz_and_chromaticity(pixels)
    labs = labs[valid]
    if len(xy) == 0:
        return None
    median_xy = np.median(xy, axis=0)
    median_uv = np.median(uv, axis=0)
    median_ab = np.median(labs[:, 1:3], axis=0)
    trimmed = np.mean(
        np.clip(
            labs[:, 1:3],
            np.quantile(labs[:, 1:3], 0.10, axis=0),
            np.quantile(labs[:, 1:3], 0.90, axis=0),
        ),
        axis=0,
    )
    chroma = np.linalg.norm(labs[:, 1:3], axis=1)
    return {
        "xy": _rounded(median_xy),
        "uv_prime": _rounded(median_uv),
        "oklab_ab_median": _rounded(median_ab),
        "oklab_ab_trimmed_mean": _rounded(trimmed),
        "neutral_oklab_chroma_median": round(_finite(np.median(chroma)), 7),
        # A correlated colour temperature is not emitted until a validated
        # Planckian-locus + Duv implementation exists.  McCamy alone is unsafe
        # for deciding whether an arbitrary sampled surface is actually neutral.
        "apparent_cct_kelvin": None,
        "apparent_mired": None,
        "cct_reason": "disabled_until_validated_planckian_locus_and_duv",
    }


def _neutral_spatial_dispersion(
    source: _PreparedEvidence,
    target: _PreparedEvidence,
    config: C1AnalyzerConfig,
) -> dict[str, object]:
    source_a = _block_medians(source, source.oklab[..., 1], source.neutral_valid, config)
    source_b = _block_medians(source, source.oklab[..., 2], source.neutral_valid, config)
    target_a = _block_medians(target, target.oklab[..., 1], target.neutral_valid, config)
    target_b = _block_medians(target, target.oklab[..., 2], target.neutral_valid, config)
    common = sorted(source_a.keys() & source_b.keys() & target_a.keys() & target_b.keys())
    shifts = np.asarray(
        [
            [target_a[key] - source_a[key], target_b[key] - source_b[key]]
            for key in common
        ],
        dtype=np.float64,
    )
    if len(shifts) == 0:
        return {"status": "insufficient_common_blocks", "common_blocks": 0, "dispersion": None}
    center = np.median(shifts, axis=0)
    distances = np.linalg.norm(shifts - center, axis=1)
    return {
        "status": "available" if len(common) >= 3 else "insufficient_common_blocks",
        "common_blocks": len(common),
        "shift_center_oklab_ab": _rounded(center),
        "dispersion": round(_finite(np.median(distances)), 7),
    }


def _direction(value: float, *, positive: str, negative: str, deadband: float) -> str:
    if value > deadband:
        return positive
    if value < -deadband:
        return negative
    return "stable"


def _whitepoint_analysis(
    source: _PreparedEvidence,
    target: _PreparedEvidence,
    neutral_evidence: str,
    config: C1AnalyzerConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    explicit = neutral_evidence in _EXPLICIT_NEUTRAL_EVIDENCE
    sufficient = (
        source.neutral_pixels >= config.min_neutral_pixels
        and target.neutral_pixels >= config.min_neutral_pixels
    )
    spatial = _neutral_spatial_dispersion(source, target, config) if sufficient else {
        "status": "unavailable",
        "common_blocks": 0,
        "dispersion": None,
    }
    if not sufficient:
        return {
            "status": "insufficient_neutral_evidence",
            "evidence_level": neutral_evidence,
            "source": None,
            "target": None,
            "neutral_axis_shift_oklab_ab": None,
            "warm_cool_direction": "unknown",
            "tint_direction": "unknown",
            "eligible_for_future_transform": False,
        }, spatial

    source_estimate = _whitepoint_estimate(source, explicit=explicit)
    target_estimate = _whitepoint_estimate(target, explicit=explicit)
    if source_estimate is None or target_estimate is None:
        return {
            "status": "insufficient_neutral_evidence",
            "evidence_level": neutral_evidence,
            "source": source_estimate,
            "target": target_estimate,
            "neutral_axis_shift_oklab_ab": None,
            "warm_cool_direction": "unknown",
            "tint_direction": "unknown",
            "eligible_for_future_transform": False,
        }, spatial

    source_ab = np.asarray(source_estimate["oklab_ab_median"], dtype=np.float64)
    target_ab = np.asarray(target_estimate["oklab_ab_median"], dtype=np.float64)
    delta_ab = target_ab - source_ab
    source_uv = np.asarray(source_estimate["uv_prime"], dtype=np.float64)
    target_uv = np.asarray(target_estimate["uv_prime"], dtype=np.float64)
    mixed = (
        spatial["status"] == "available"
        and spatial["dispersion"] is not None
        and float(spatial["dispersion"]) > config.neutral_spatial_review_oklab
    )
    neutral_plausible = bool(
        float(source_estimate["neutral_oklab_chroma_median"])
        <= config.explicit_neutral_oklab_chroma_max
        and float(target_estimate["neutral_oklab_chroma_median"])
        <= config.explicit_neutral_oklab_chroma_max
    )
    status = "valid_explicit_neutral" if explicit and not mixed else "hypothesis_only"
    if explicit and not neutral_plausible:
        status = "explicit_surface_not_neutral"
    if mixed:
        status = "mixed_illumination_or_surface_instability"
    source_mired = source_estimate["apparent_mired"]
    target_mired = target_estimate["apparent_mired"]
    return {
        "status": status,
        "evidence_level": neutral_evidence,
        "source": source_estimate,
        "target": target_estimate,
        "neutral_axis_shift_oklab_ab": _rounded(delta_ab),
        "uv_prime_shift": _rounded(target_uv - source_uv),
        "mired_shift": (
            None
            if source_mired is None or target_mired is None
            else round(float(target_mired) - float(source_mired), 4)
        ),
        "duv": None,
        "duv_reason": "not_implemented_without_a_validated_planckian_locus_method",
        "neutral_surface_plausible": neutral_plausible,
        "warm_cool_direction": _direction(
            float(delta_ab[1]),
            positive="target_warmer",
            negative="target_cooler",
            deadband=config.direction_deadband_oklab,
        ),
        "tint_direction": _direction(
            float(delta_ab[0]),
            positive="target_more_magenta_red",
            negative="target_more_green",
            deadband=config.direction_deadband_oklab,
        ),
        "eligible_for_future_transform": bool(
            explicit
            and neutral_plausible
            and not mixed
            and spatial["status"] == "available"
        ),
        "claim_boundary": "apparent display-referred neutral-axis evidence, not physical illuminant",
    }, spatial


def _evidence_summary(evidence: _PreparedEvidence) -> dict[str, object]:
    return {
        "proxy_size": list(evidence.proxy_size),
        "region_pixels": evidence.region_pixels,
        "tonal_pixels": evidence.tonal_pixels,
        "neutral_pixels": evidence.neutral_pixels,
        "neutral_fraction_of_region": round(
            0.0 if evidence.region_pixels == 0 else evidence.neutral_pixels / evidence.region_pixels,
            7,
        ),
        "clipped_fraction_of_region": round(evidence.clipped_fraction, 7),
    }


def analyse_relative_illumination(
    source_image: Image.Image,
    reference_image: Image.Image,
    *,
    source_region_mask: Image.Image | None = None,
    reference_region_mask: Image.Image | None = None,
    source_neutral_mask: Image.Image | None = None,
    reference_neutral_mask: Image.Image | None = None,
    neutral_evidence: str = "automatic",
    comparison_evidence: str = "automatic",
    region_name: str = "scene",
    config: C1AnalyzerConfig = C1_CONFIG,
) -> dict[str, object]:
    """Compare source and reference without rendering or approving pixels.

    Results are hypotheses about display-referred relative differences.  C1 v1
    intentionally has no candidate renderer and no authority over A0.
    """

    config.validate()
    if neutral_evidence not in _NEUTRAL_EVIDENCE:
        raise ValueError("neutral_evidence must be automatic, human_confirmed or same_entity")
    if comparison_evidence not in _COMPARISON_EVIDENCE:
        raise ValueError(
            "comparison_evidence must be automatic, human_confirmed or same_surface"
        )
    if not region_name.strip():
        raise ValueError("region_name must not be empty")
    provided_neutral = source_neutral_mask is not None or reference_neutral_mask is not None
    if provided_neutral and (source_neutral_mask is None or reference_neutral_mask is None):
        raise ValueError("Source and reference neutral masks must be supplied together")
    if neutral_evidence in _EXPLICIT_NEUTRAL_EVIDENCE and not provided_neutral:
        raise ValueError("Explicit neutral evidence requires both neutral masks")
    if neutral_evidence == "automatic" and provided_neutral:
        raise ValueError("Neutral masks require human_confirmed or same_entity evidence")
    provided_regions = source_region_mask is not None or reference_region_mask is not None
    if provided_regions and (source_region_mask is None or reference_region_mask is None):
        raise ValueError("Source and reference region masks must be supplied together")
    if comparison_evidence in _EXPLICIT_COMPARISON_EVIDENCE and not provided_regions:
        raise ValueError("Explicit comparison evidence requires both region masks")

    explicit = neutral_evidence in _EXPLICIT_NEUTRAL_EVIDENCE
    source = _prepare(
        source_image,
        source_region_mask,
        source_neutral_mask,
        explicit_neutral=explicit,
        config=config,
    )
    target = _prepare(
        reference_image,
        reference_region_mask,
        reference_neutral_mask,
        explicit_neutral=explicit,
        config=config,
    )
    exposure, exposure_spatial = _exposure_analysis(source, target, config)
    tone = exposure.pop("tone", {"status": "unavailable"})
    whitepoint, neutral_spatial = _whitepoint_analysis(
        source, target, neutral_evidence, config
    )

    reasons: list[str] = []
    if source.tonal_pixels < config.min_region_pixels or target.tonal_pixels < config.min_region_pixels:
        reasons.append("MASK_OR_REGION_EVIDENCE_INSUFFICIENT")
    if max(source.clipped_fraction, target.clipped_fraction) > config.heavy_clipping_fraction:
        reasons.append("HEAVY_CLIPPING")
    if whitepoint["status"] == "insufficient_neutral_evidence":
        reasons.append("INSUFFICIENT_NEUTRAL_EVIDENCE")
    elif whitepoint["status"] == "hypothesis_only":
        reasons.append("AUTOMATIC_NEUTRAL_HYPOTHESIS_ONLY")
    elif whitepoint["status"] == "mixed_illumination_or_surface_instability":
        reasons.append("MIXED_ILLUMINATION_OR_SURFACE_INSTABILITY")
    elif whitepoint["status"] == "explicit_surface_not_neutral":
        reasons.append("EXPLICIT_SURFACE_NOT_NEUTRAL")
    if exposure["status"] == "compound_or_composition_unstable":
        reasons.append("COMPOSITION_OR_TONE_UNSTABLE")
    if comparison_evidence not in _EXPLICIT_COMPARISON_EVIDENCE:
        reasons.append("COMPARABLE_SURFACE_NOT_CONFIRMED")
    analysis_status = "valid" if not reasons else "limited"
    exposure_eligible = (
        exposure["status"] == "valid"
        and "HEAVY_CLIPPING" not in reasons
        and exposure_spatial["status"] == "available"
        and comparison_evidence in _EXPLICIT_COMPARISON_EVIDENCE
    )
    return {
        "schema_version": config.schema_version,
        "analyzer": C1_ANALYZER_ID,
        "mode": "shadow_read_only",
        "status": "review",
        "accepted": False,
        "analysis_status": analysis_status,
        "pixel_output_changed": False,
        "region_name": region_name,
        "whitepoint": whitepoint,
        "exposure": exposure,
        "tone": tone,
        "illumination": {
            "mixed_or_surface_instability": (
                whitepoint["status"] == "mixed_illumination_or_surface_instability"
            ),
            "neutral_spatial": neutral_spatial,
            "exposure_spatial": exposure_spatial,
        },
        "evidence": {
            "neutral_evidence_level": neutral_evidence,
            "comparison_evidence_level": comparison_evidence,
            "source": _evidence_summary(source),
            "reference": _evidence_summary(target),
            "holdout_prediction_gain": "not_applicable_observer_has_no_renderer",
        },
        "applicability": {
            "a0": "not_evaluated_observer_does_not_change_or_veto_a0",
            "c1_exposure_future_candidate": bool(exposure_eligible),
            "c1_whitepoint_future_candidate": bool(
                whitepoint["eligible_for_future_transform"]
                and "HEAVY_CLIPPING" not in reasons
            ),
            "c1_tone_future_candidate": False,
        },
        "review_reasons": reasons,
        "claim_boundary": [
            "display-referred sRGB comparison only",
            "not physical illuminant, camera EV or camera response recovery",
            "does not define product truth",
            "does not render, approve, reject or replace A0",
        ],
    }
