"""Ground-truth validation for ATR18 fine parsing.

This module measures supplied predictions against supplied reviewed truth.  It
does not approve image quality, infer truth, or silently resize label maps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from batch_color.fine_masks import ATR18_GROUPS, ATR18_LABELS, load_atr18_label_map
from batch_color.safety import file_hash


def _region_mask(labels: np.ndarray, name: str) -> np.ndarray:
    if name in ATR18_LABELS:
        return labels == ATR18_LABELS.index(name)
    if name in ATR18_GROUPS:
        indices = [ATR18_LABELS.index(member) for member in ATR18_GROUPS[name]]
        return np.isin(labels, indices)
    raise ValueError(f"Unsupported ATR18 validation region: {name}")


def _boundary(mask: np.ndarray) -> np.ndarray:
    hard = np.asarray(mask, dtype=bool)
    if not np.any(hard):
        return hard
    eroded = np.asarray(
        Image.fromarray(np.where(hard, 255, 0).astype(np.uint8), mode="L").filter(
            ImageFilter.MinFilter(3)
        ),
        dtype=np.uint8,
    ) > 0
    return hard & ~eroded


def _dilate(mask: np.ndarray, tolerance: int) -> np.ndarray:
    if tolerance <= 0:
        return np.asarray(mask, dtype=bool)
    size = 2 * tolerance + 1
    return np.asarray(
        Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").filter(
            ImageFilter.MaxFilter(size)
        ),
        dtype=np.uint8,
    ) > 0


def _metrics(predicted: np.ndarray, truth: np.ndarray, tolerance: int) -> dict[str, object]:
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    predicted_pixels = int(np.count_nonzero(predicted))
    truth_pixels = int(np.count_nonzero(truth))
    union = tp + fp + fn
    precision = 1.0 if predicted_pixels == 0 and truth_pixels == 0 else tp / max(tp + fp, 1)
    recall = 1.0 if truth_pixels == 0 and predicted_pixels == 0 else tp / max(tp + fn, 1)
    iou = 1.0 if union == 0 else tp / union

    predicted_boundary = _boundary(predicted)
    truth_boundary = _boundary(truth)
    predicted_boundary_pixels = int(np.count_nonzero(predicted_boundary))
    truth_boundary_pixels = int(np.count_nonzero(truth_boundary))
    if predicted_boundary_pixels == 0 and truth_boundary_pixels == 0:
        boundary_precision = boundary_recall = boundary_f1 = 1.0
    else:
        boundary_precision = float(
            np.count_nonzero(predicted_boundary & _dilate(truth_boundary, tolerance))
            / max(predicted_boundary_pixels, 1)
        )
        boundary_recall = float(
            np.count_nonzero(truth_boundary & _dilate(predicted_boundary, tolerance))
            / max(truth_boundary_pixels, 1)
        )
        denominator = boundary_precision + boundary_recall
        boundary_f1 = 0.0 if denominator == 0 else 2.0 * boundary_precision * boundary_recall / denominator
    return {
        "truth_pixels": truth_pixels,
        "predicted_pixels": predicted_pixels,
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "iou": round(float(iou), 6),
        "boundary_precision": round(boundary_precision, 6),
        "boundary_recall": round(boundary_recall, 6),
        "boundary_f1": round(boundary_f1, 6),
    }


def _role_map(labels: np.ndarray) -> np.ndarray:
    roles = np.full(labels.shape, 5, dtype=np.uint8)  # other person / unresolved
    roles[labels == 0] = 0
    garment_members = (*ATR18_GROUPS["garment"], *ATR18_GROUPS["shoes"])
    roles[np.isin(labels, [ATR18_LABELS.index(name) for name in garment_members])] = 1
    roles[np.isin(labels, [ATR18_LABELS.index(name) for name in ATR18_GROUPS["skin"]])] = 2
    roles[labels == ATR18_LABELS.index("hair")] = 3
    roles[np.isin(labels, [ATR18_LABELS.index(name) for name in ATR18_GROUPS["accessories"]])] = 4
    return roles


def validate_fine_labels(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    required_regions: tuple[str, ...] = (),
    min_iou: float = 0.80,
    min_boundary_f1: float = 0.70,
    max_cross_role_leakage: float = 0.01,
    boundary_tolerance: int = 2,
) -> dict[str, object]:
    predicted = np.asarray(predicted)
    truth = np.asarray(truth)
    if predicted.shape != truth.shape or predicted.ndim != 2 or predicted.size == 0:
        raise ValueError("Predicted and truth ATR18 maps must have the same non-empty HxW geometry")
    for name, labels in (("predicted", predicted), ("truth", truth)):
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"{name} ATR18 map must contain integer indices")
        if int(labels.min()) < 0 or int(labels.max()) >= len(ATR18_LABELS):
            raise ValueError(f"{name} ATR18 map contains an index outside 0..17")
    if not 0.0 <= min_iou <= 1.0 or not 0.0 <= min_boundary_f1 <= 1.0:
        raise ValueError("Validation IoU and boundary thresholds must be in 0..1")
    if not 0.0 <= max_cross_role_leakage <= 1.0:
        raise ValueError("max_cross_role_leakage must be in 0..1")
    if boundary_tolerance < 0 or boundary_tolerance > 20:
        raise ValueError("boundary_tolerance must be in 0..20 pixels")
    allowed_regions = set((*ATR18_LABELS, *ATR18_GROUPS.keys()))
    unsupported = sorted(set(required_regions) - allowed_regions)
    if unsupported:
        raise ValueError("Unsupported required regions: " + ", ".join(unsupported))

    regions = {
        name: _metrics(
            _region_mask(predicted, name),
            _region_mask(truth, name),
            boundary_tolerance,
        )
        for name in (*ATR18_LABELS, *ATR18_GROUPS.keys())
    }
    evaluated = tuple(required_regions) or tuple(
        name
        for name in ("garment", "shoes", "skin", "hair", "accessories")
        if int(regions[name]["truth_pixels"]) > 0
    )
    reasons: list[str] = []
    if not evaluated:
        reasons.append("no_foreground_validation_region_present")
    for name in evaluated:
        metrics = regions[name]
        if int(metrics["truth_pixels"]) == 0:
            reasons.append(f"required_truth_region_absent:{name}")
            continue
        if float(metrics["iou"]) < min_iou:
            reasons.append(f"iou_below_threshold:{name}")
        if float(metrics["boundary_f1"]) < min_boundary_f1:
            reasons.append(f"boundary_f1_below_threshold:{name}")

    predicted_roles = _role_map(predicted)
    truth_roles = _role_map(truth)
    truth_foreground = truth_roles != 0
    cross_role = truth_foreground & (predicted_roles != truth_roles)
    leakage = float(np.count_nonzero(cross_role) / max(np.count_nonzero(truth_foreground), 1))
    if leakage > max_cross_role_leakage:
        reasons.append("cross_role_leakage_above_threshold")

    present_class_metrics = [
        regions[name]
        for name in ATR18_LABELS[1:]
        if int(regions[name]["truth_pixels"]) > 0
    ]
    macro_iou = (
        float(np.mean([float(item["iou"]) for item in present_class_metrics]))
        if present_class_metrics
        else 0.0
    )
    macro_boundary = (
        float(np.mean([float(item["boundary_f1"]) for item in present_class_metrics]))
        if present_class_metrics
        else 0.0
    )
    return {
        "schema_version": 1,
        "validation_kind": "atr18_prediction_against_reviewed_truth",
        "status": "review",
        "validation_result": "pass" if not reasons else "thresholds_not_met",
        "accepted": False,
        "checks_passed": not reasons,
        "claim_boundary": "label_accuracy_against_supplied_truth_only",
        "geometry": [int(predicted.shape[1]), int(predicted.shape[0])],
        "evaluated_regions": list(evaluated),
        "thresholds": {
            "min_iou": min_iou,
            "min_boundary_f1": min_boundary_f1,
            "max_cross_role_leakage": max_cross_role_leakage,
            "boundary_tolerance_pixels": boundary_tolerance,
        },
        "summary": {
            "pixel_accuracy": round(float(np.mean(predicted == truth)), 6),
            "macro_present_class_iou": round(macro_iou, 6),
            "macro_present_class_boundary_f1": round(macro_boundary, 6),
            "cross_role_leakage": round(leakage, 6),
        },
        "failure_reasons": reasons,
        "regions": regions,
        "artifacts": {},
    }


def validate_fine_label_files(
    predicted_path: str | Path,
    truth_path: str | Path,
    **kwargs: object,
) -> dict[str, object]:
    predicted = load_atr18_label_map(predicted_path)
    truth = load_atr18_label_map(truth_path)
    report = validate_fine_labels(predicted, truth, **kwargs)
    report["evidence"] = {
        "predicted": {
            "path": str(Path(predicted_path).resolve()),
            "sha256": file_hash(predicted_path),
        },
        "reviewed_truth": {
            "path": str(Path(truth_path).resolve()),
            "sha256": file_hash(truth_path),
        },
    }
    return report
