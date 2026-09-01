"""Review-gated, bounded colour matching for one explicitly authorized region."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageFilter

from batch_color.sku_pipeline import RegionPlan, apply_region_plans, region_distance, region_stats


SKU_ROLES: tuple[str, ...] = (
    "background",
    "skin_identity",
    "hair_identity",
    "target_sku",
    "accessory",
    "other_garment",
    "protected_object",
)

REFERENCE_POLICIES: tuple[str, ...] = (
    "scene_reference",
    "sku_approved_anchor",
    "source_identity",
    "protected",
)

_ROLE_REFERENCE_POLICY = {
    "background": "scene_reference",
    "skin_identity": "scene_reference",
    "hair_identity": "scene_reference",
    "target_sku": "sku_approved_anchor",
    "accessory": "sku_approved_anchor",
    "other_garment": "source_identity",
    "protected_object": "protected",
}

_SKIN_REGIONS = {"skin", "face", "left_leg", "right_leg", "left_arm", "right_arm"}
_GARMENT_REGIONS = {"garment", "upper_clothes", "skirt", "pants", "dress", "shoes", "left_shoe", "right_shoe"}
_ACCESSORY_REGIONS = {"accessories", "hat", "sunglasses", "belt", "bag", "scarf"}


@dataclass(frozen=True)
class RegionTargetPolicy:
    """Bind an edit to an object role and an explicit source of colour truth."""

    object_id: str
    sku_role: str
    reference_policy: str
    reference_id: str


def validate_region_target_policy(policy: RegionTargetPolicy, region: str) -> None:
    if not policy.object_id.strip() or not policy.reference_id.strip():
        raise ValueError("object_id and reference_id are required")
    if policy.sku_role not in SKU_ROLES:
        raise ValueError(f"Unsupported sku_role: {policy.sku_role}")
    if policy.reference_policy not in REFERENCE_POLICIES:
        raise ValueError(f"Unsupported reference_policy: {policy.reference_policy}")
    expected = _ROLE_REFERENCE_POLICY[policy.sku_role]
    if policy.reference_policy != expected:
        raise ValueError(
            f"sku_role={policy.sku_role} requires reference_policy={expected}"
        )
    if policy.reference_policy in {"source_identity", "protected"}:
        raise ValueError(
            f"sku_role={policy.sku_role} is protected and cannot authorize colour transfer"
        )
    if policy.sku_role == "background" and region != "background":
        raise ValueError("background role can only authorize the background region")
    if policy.sku_role == "skin_identity" and region not in _SKIN_REGIONS:
        raise ValueError("skin_identity role requires a skin region")
    if policy.sku_role == "hair_identity" and region != "hair":
        raise ValueError("hair_identity role can only authorize the hair region")
    if policy.sku_role == "target_sku" and region not in _GARMENT_REGIONS:
        raise ValueError("target_sku role requires a garment or shoe region")
    if policy.sku_role == "accessory" and region not in _ACCESSORY_REGIONS:
        raise ValueError("accessory role requires an accessory region")


def _boundary_residual(
    source: Image.Image,
    corrected: Image.Image,
    mask: Image.Image,
) -> dict[str, object]:
    hard = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    if not np.any(hard):
        return {
            "boundary_pixels": 0,
            "interior_pixels": 0,
            "boundary_median_change": 0.0,
            "interior_median_change": 0.0,
            "boundary_to_interior_ratio": None,
            "abrupt_falloff": False,
        }
    eroded = np.asarray(
        Image.fromarray(np.where(hard, 255, 0).astype(np.uint8), mode="L").filter(
            ImageFilter.MinFilter(5)
        ),
        dtype=np.uint8,
    ) > 0
    boundary = hard & ~eroded
    source_pixels = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    output_pixels = np.asarray(corrected.convert("RGB"), dtype=np.float32) / 255.0
    change = np.linalg.norm(output_pixels - source_pixels, axis=-1)
    boundary_median = float(np.median(change[boundary])) if np.any(boundary) else 0.0
    interior_median = float(np.median(change[eroded])) if np.any(eroded) else 0.0
    ratio = None if interior_median <= 1e-6 else boundary_median / interior_median
    return {
        "boundary_pixels": int(np.count_nonzero(boundary)),
        "interior_pixels": int(np.count_nonzero(eroded)),
        "boundary_median_change": round(boundary_median, 8),
        "interior_median_change": round(interior_median, 8),
        "boundary_to_interior_ratio": None if ratio is None else round(float(ratio), 6),
        "abrupt_falloff": bool(
            ratio is not None
            and interior_median >= (1.0 / 255.0)
            and np.count_nonzero(boundary) >= 16
            and ratio < 0.12
        ),
    }


def precision_region_match(
    source: Image.Image,
    reference: Image.Image,
    source_mask: Image.Image,
    reference_mask: Image.Image,
    *,
    protected_mask: Image.Image | None = None,
    region: str,
    target_policy: RegionTargetPolicy,
    strength: float = 0.55,
    luminance_cap: float = 0.045,
    chroma_cap: float = 0.028,
) -> tuple[Image.Image, dict[str, object]]:
    validate_region_target_policy(target_policy, region)
    if source_mask.size != source.size or reference_mask.size != reference.size:
        raise ValueError("Every region mask must exactly match its canonical image")
    if protected_mask is not None and protected_mask.size != source.size:
        raise ValueError("Protected mask must exactly match the canonical source image")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in 0..1")
    if not 0.0 <= luminance_cap <= 0.12:
        raise ValueError("luminance_cap must be in 0..0.12")
    if not 0.0 <= chroma_cap <= 0.08:
        raise ValueError("chroma_cap must be in 0..0.08")
    supplied_authorization = np.asarray(source_mask.convert("L"), dtype=np.uint8)
    protected = (
        np.zeros_like(supplied_authorization)
        if protected_mask is None
        else np.asarray(protected_mask.convert("L"), dtype=np.uint8)
    )
    # Protection can only remove authority.  Multiplication preserves soft
    # inward feathering and guarantees that protected pixels are never added
    # back by a later blur or blend.
    effective_authorization = np.round(
        supplied_authorization.astype(np.float32)
        * (1.0 - protected.astype(np.float32) / 255.0)
    ).astype(np.uint8)
    source_mask = Image.fromarray(effective_authorization, mode="L")
    source_stats = region_stats(source, source_mask)
    target_stats = region_stats(reference, reference_mask)
    before = region_distance(source_stats, target_stats)
    corrected, render = apply_region_plans(
        source,
        [
            RegionPlan(
                name=region,
                mask=source_mask,
                source=source_stats,
                target=target_stats,
                strength=strength,
                luminance_cap=luminance_cap,
                chroma_cap=chroma_cap,
            )
        ],
    )
    after_stats = region_stats(corrected, source_mask)
    after = region_distance(after_stats, target_stats)
    source_pixels = np.asarray(source.convert("RGB"), dtype=np.uint8)
    output_pixels = np.asarray(corrected.convert("RGB"), dtype=np.uint8)
    authorized = np.asarray(source_mask.convert("L"), dtype=np.uint8) > 0
    outside_changed = int(
        np.count_nonzero(np.any(source_pixels[~authorized] != output_pixels[~authorized], axis=-1))
    )
    quality_reasons = []
    if outside_changed:
        quality_reasons.append("pixels_changed_outside_authorized_region")
    if render["newly_clipped_percent_of_editable"] > 0.10:
        quality_reasons.append("new_clipping_above_0.10_percent")
    if after > before + 1e-6:
        quality_reasons.append("target_distance_increased")
    boundary_residual = _boundary_residual(source, corrected, source_mask)
    if boundary_residual["abrupt_falloff"]:
        quality_reasons.append("boundary_residual_abrupt_falloff")
    return corrected, {
        "region": region,
        "target_policy": asdict(target_policy),
        "source_stats": asdict(source_stats),
        "target_stats": asdict(target_stats),
        "corrected_stats": asdict(after_stats),
        "distance_before": round(before, 6),
        "distance_after": round(after, 6),
        "distance_improvement": round(before - after, 6),
        "outside_authorized_changed_pixels": outside_changed,
        "protection": {
            "supplied": protected_mask is not None,
            "protected_pixels": int(np.count_nonzero(protected)),
            "authorization_pixels_before_protection": int(
                np.count_nonzero(supplied_authorization)
            ),
            "authorization_pixels_after_protection": int(
                np.count_nonzero(effective_authorization)
            ),
            "contract": "protected-mask-can-only-subtract-edit-authority-v1",
        },
        "boundary_residual": boundary_residual,
        "render": render,
        "automatic_checks_passed": not quality_reasons,
        "review_reasons": quality_reasons or ["human_visual_review_required"],
        "accepted": False,
        "status": "review",
        "capability": "optional_b1_precision_region_match",
        "a0_modified": False,
    }
