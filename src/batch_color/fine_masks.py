"""Fine-grained semantic mask contract for optional, review-gated colour work.

The stable A0 pipeline deliberately does not import this module.  A parser may
produce an ATR18 label map, but a colour operation is authorized only from the
validated masks exported here.  Automatic evidence needs per-pixel confidence;
reviewed overrides need an explicit reviewer identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from batch_color.encoding import inspect_encoding


ATR18_LABELS: tuple[str, ...] = (
    "background",
    "hat",
    "hair",
    "sunglasses",
    "upper_clothes",
    "skirt",
    "pants",
    "dress",
    "belt",
    "left_shoe",
    "right_shoe",
    "face",
    "left_leg",
    "right_leg",
    "left_arm",
    "right_arm",
    "bag",
    "scarf",
)

ATR18_GROUPS: dict[str, tuple[str, ...]] = {
    "garment": ("upper_clothes", "skirt", "pants", "dress"),
    "shoes": ("left_shoe", "right_shoe"),
    "skin": ("face", "left_leg", "right_leg", "left_arm", "right_arm"),
    "accessories": ("hat", "sunglasses", "belt", "bag", "scarf"),
    "person": tuple(name for name in ATR18_LABELS if name != "background"),
}

FINE_MASK_SCHEMA = "atr18-v1"


@dataclass(frozen=True)
class FineMaskBundle:
    schema: str
    label_status: str
    reviewed_by: str | None
    label_map: Image.Image
    confidence_map: Image.Image
    masks: dict[str, Image.Image]
    regions: dict[str, dict[str, object]]
    diagnostics: dict[str, object]


def _load_index_map(
    path: str | Path, size: tuple[int, int] | None = None
) -> np.ndarray:
    with Image.open(path) as opened:
        inspect_encoding(path, opened)
        if getattr(opened, "n_frames", 1) != 1:
            raise ValueError("Fine label maps must contain exactly one frame")
        if opened.mode not in {"L", "P"}:
            raise ValueError("Fine label maps must be 8-bit L or palette PNG/TIFF images")
        if "A" in opened.getbands() or "transparency" in opened.info:
            raise ValueError("Fine label maps cannot contain transparency")
        canonical = ImageOps.exif_transpose(opened)
        labels = np.asarray(canonical, dtype=np.uint8).copy()
    if size is not None and (labels.shape[1], labels.shape[0]) != size:
        raise ValueError(
            "Fine label map must exactly match the canonical image dimensions; resizing is unsafe"
        )
    if labels.size == 0 or int(labels.max()) >= len(ATR18_LABELS):
        raise ValueError("Fine label map contains a class outside the ATR18 0..17 schema")
    return labels


def load_atr18_label_map(
    path: str | Path, size: tuple[int, int] | None = None
) -> np.ndarray:
    """Load an opaque, unscaled ATR18 index map for validation or masking."""
    return _load_index_map(path, size)


def _load_confidence_map(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as opened:
        inspect_encoding(path, opened)
        if getattr(opened, "n_frames", 1) != 1:
            raise ValueError("Confidence maps must contain exactly one frame")
        if opened.mode != "L" or "transparency" in opened.info:
            raise ValueError("Confidence maps must be opaque 8-bit grayscale images")
        canonical = ImageOps.exif_transpose(opened)
        confidence = np.asarray(canonical, dtype=np.float32) / 255.0
    if (confidence.shape[1], confidence.shape[0]) != size:
        raise ValueError("Confidence map must exactly match the canonical image dimensions")
    if not np.all(np.isfinite(confidence)):
        raise ValueError("Confidence map contains non-finite values")
    return np.clip(confidence, 0.0, 1.0)


def inward_feather(mask: Image.Image, radius: float) -> Image.Image:
    """Soften only *inside* an authorized region; never spill into another object."""
    source = mask.convert("L")
    if radius <= 0:
        return source.copy()
    original = np.asarray(source, dtype=np.uint8)
    blurred = np.asarray(source.filter(ImageFilter.GaussianBlur(radius=float(radius))), dtype=np.uint8)
    safe = np.minimum(original, blurred)
    return Image.fromarray(safe, mode="L")


def build_fine_mask_bundle(
    label_map_path: str | Path,
    size: tuple[int, int],
    *,
    confidence_map_path: str | Path | None = None,
    label_status: str = "automatic",
    reviewed_by: str | None = None,
    confidence_threshold: float = 0.82,
    confidence_thresholds: dict[str, float] | None = None,
    min_authorized_fraction: float = 0.90,
    min_pixels: int = 128,
    feather_radius: float = 2.0,
) -> FineMaskBundle:
    labels = _load_index_map(label_map_path, size)
    confidence = (
        _load_confidence_map(confidence_map_path, size)
        if confidence_map_path is not None
        else None
    )
    return build_fine_mask_bundle_from_arrays(
        labels,
        confidence,
        label_status=label_status,
        reviewed_by=reviewed_by,
        confidence_threshold=confidence_threshold,
        confidence_thresholds=confidence_thresholds,
        min_authorized_fraction=min_authorized_fraction,
        min_pixels=min_pixels,
        feather_radius=feather_radius,
    )


def build_fine_mask_bundle_from_arrays(
    labels: np.ndarray,
    confidence: np.ndarray | None,
    *,
    label_status: str = "automatic",
    reviewed_by: str | None = None,
    confidence_threshold: float = 0.82,
    confidence_thresholds: dict[str, float] | None = None,
    min_authorized_fraction: float = 0.90,
    min_pixels: int = 128,
    feather_radius: float = 2.0,
) -> FineMaskBundle:
    if label_status not in {"automatic", "reviewed"}:
        raise ValueError("label_status must be automatic or reviewed")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in 0..1")
    threshold_overrides = dict(confidence_thresholds or {})
    valid_regions = set((*ATR18_LABELS, *ATR18_GROUPS.keys()))
    unknown_regions = sorted(set(threshold_overrides) - valid_regions)
    if unknown_regions:
        raise ValueError(
            "confidence_thresholds contains unsupported regions: "
            + ", ".join(unknown_regions)
        )
    for name, value in threshold_overrides.items():
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"confidence threshold for {name} must be in 0..1")
        threshold_overrides[name] = float(value)
    if not 0.0 <= min_authorized_fraction <= 1.0:
        raise ValueError("min_authorized_fraction must be in 0..1")
    if min_pixels < 1:
        raise ValueError("min_pixels must be positive")
    if label_status == "automatic" and confidence is None:
        raise ValueError("Automatic fine masks require a per-pixel confidence map")
    if label_status == "reviewed" and not (reviewed_by or "").strip():
        raise ValueError("Reviewed fine masks require a non-empty reviewer identity")

    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.size == 0:
        raise ValueError("Fine labels must be a non-empty HxW array")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Fine labels must contain integer ATR18 class indices")
    if int(labels.min()) < 0 or int(labels.max()) >= len(ATR18_LABELS):
        raise ValueError("Fine labels contain a class outside the ATR18 0..17 schema")
    labels = labels.astype(np.uint8, copy=True)
    if confidence is None:
        confidence = np.ones(labels.shape, dtype=np.float32)
    else:
        confidence = np.asarray(confidence, dtype=np.float32)
        if confidence.shape != labels.shape or not np.all(np.isfinite(confidence)):
            raise ValueError("Fine confidence must be finite and match the label geometry")
        confidence = np.clip(confidence, 0.0, 1.0).copy()
    class_thresholds = np.asarray(
        [threshold_overrides.get(name, confidence_threshold) for name in ATR18_LABELS],
        dtype=np.float32,
    )
    per_pixel_threshold = class_thresholds[labels]
    authorized = (
        confidence >= per_pixel_threshold
        if label_status == "automatic"
        else np.ones_like(labels, bool)
    )

    hard: dict[str, np.ndarray] = {
        name: labels == index for index, name in enumerate(ATR18_LABELS)
    }
    for group, members in ATR18_GROUPS.items():
        hard[group] = np.logical_or.reduce([hard[name] for name in members])

    masks: dict[str, Image.Image] = {}
    metrics: dict[str, dict[str, object]] = {}
    for name, region in hard.items():
        region_threshold = threshold_overrides.get(name, confidence_threshold)
        raw_pixels = int(np.count_nonzero(region))
        allowed = (
            region & authorized & (confidence >= region_threshold)
            if label_status == "automatic"
            else region.copy()
        )
        allowed_pixels = int(np.count_nonzero(allowed))
        fraction = 0.0 if raw_pixels == 0 else allowed_pixels / raw_pixels
        mean_confidence = 0.0 if raw_pixels == 0 else float(np.mean(confidence[region]))
        usable = bool(
            allowed_pixels >= min_pixels
            and fraction >= min_authorized_fraction
            and (label_status == "reviewed" or mean_confidence >= region_threshold)
        )
        binary = Image.fromarray(np.where(allowed, 255, 0).astype(np.uint8), mode="L")
        masks[name] = inward_feather(binary, feather_radius)
        metrics[name] = {
            "raw_pixels": raw_pixels,
            "authorized_pixels": allowed_pixels,
            "authorized_fraction": round(float(fraction), 6),
            "mean_confidence": round(mean_confidence, 6),
            "confidence_threshold": round(float(region_threshold), 6),
            "usable_for_colour": usable,
            "failure_reasons": [
                reason
                for condition, reason in (
                    (allowed_pixels < min_pixels, "too_few_authorized_pixels"),
                    (fraction < min_authorized_fraction, "insufficient_confident_coverage"),
                    (
                        label_status == "automatic" and mean_confidence < region_threshold,
                        "mean_confidence_below_threshold",
                    ),
                )
                if condition
            ],
        }

    unknown = np.where(~authorized, 255, 0).astype(np.uint8)
    masks["unknown"] = Image.fromarray(unknown, mode="L")
    metrics["unknown"] = {
        "raw_pixels": int(np.count_nonzero(~authorized)),
        "authorized_pixels": 0,
        "authorized_fraction": 0.0,
        "mean_confidence": round(float(np.mean(confidence[~authorized])) if np.any(~authorized) else 1.0, 6),
        "usable_for_colour": False,
        "failure_reasons": ["unknown_pixels_are_never_authorized"],
    }
    return FineMaskBundle(
        schema=FINE_MASK_SCHEMA,
        label_status=label_status,
        reviewed_by=reviewed_by.strip() if reviewed_by else None,
        label_map=Image.fromarray(labels, mode="L"),
        confidence_map=Image.fromarray(
            np.round(confidence * 255.0).astype(np.uint8), mode="L"
        ),
        masks=masks,
        regions=metrics,
        diagnostics={
            "confidence_threshold": confidence_threshold,
            "confidence_thresholds": {
                name: threshold_overrides.get(name, confidence_threshold)
                for name in (*ATR18_LABELS, *ATR18_GROUPS.keys())
            },
            "min_authorized_fraction": min_authorized_fraction,
            "min_pixels": min_pixels,
            "feather_radius": feather_radius,
            "label_classes_present": [
                ATR18_LABELS[index] for index in sorted(int(value) for value in np.unique(labels))
            ],
            "unknown_pixels": int(np.count_nonzero(~authorized)),
            "total_pixels": int(labels.size),
        },
    )


def region_names() -> tuple[str, ...]:
    return (*ATR18_LABELS, *ATR18_GROUPS.keys(), "unknown")
