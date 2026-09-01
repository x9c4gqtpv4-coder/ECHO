"""Deterministic, shadow-only workflow planning.

The planner may explain which capability is eligible, but it has no pixel,
publication or approval authority.  A0 remains the only default renderer.
"""
from __future__ import annotations

from batch_color.safety import payload_hash


def compile_shadow_plan(
    sku_profile: dict[str, object],
    runtime_compatibility: dict[str, object],
    *,
    authenticated_fine_masks: bool = False,
    c1_evidence_available: bool = False,
    protected_roles: tuple[str, ...] = (),
) -> dict[str, object]:
    """Compile a fail-closed advisory plan without selecting a pixel path."""

    garment = sku_profile.get("garment_anchor")
    garment_confirmed = bool(
        isinstance(garment, dict)
        and garment.get("status") == "confirmed"
        and garment.get("confirmed") is True
        and garment.get("sha256")
    )
    a0_compatible = runtime_compatibility.get("compatible") is True
    b1_ready = garment_confirmed and authenticated_fine_masks
    plan: dict[str, object] = {
        "schema_version": 1,
        "planner": "sku-capability-shadow-planner-v1",
        "mode": "shadow_no_pixel_authority",
        "selected_workflow": "sku_standard_a0",
        "automatic_route_switching": False,
        "target_scope": "one_sku_one_designated_scene",
        "capabilities": {
            "a0": {
                "role": "default_renderer",
                "runtime_compatible": a0_compatible,
                "contract": "background_plus_whole_person_same-transform",
                "planner_can_change_pixels": False,
            },
            "b1": {
                "role": "optional_bounded_residual_candidate",
                "eligible": b1_ready,
                "enabled": False,
                "prerequisites": {
                    "confirmed_garment_anchor": garment_confirmed,
                    "authenticated_recomputed_fine_masks": authenticated_fine_masks,
                    "protected_object_subtraction": True,
                    "human_review_after_render": True,
                },
            },
            "c1": {
                "role": "read_only_observer",
                "evidence_available": c1_evidence_available,
                "enabled_for_render": False,
                "can_veto_or_modify_a0": False,
            },
        },
        "protected_roles": sorted(set(protected_roles)),
        "decision": {
            "render_path": "a0_only",
            "reason": (
                "a0_runtime_compatible_shadow_plan"
                if a0_compatible
                else "a0_runtime_mismatch_requires_review_but_planner_does_not_switch_paths"
            ),
            "future_residual_requires_separate_command": True,
        },
        "claim_boundary": [
            "advisory plan only",
            "does not inspect or change rendered pixels",
            "does not authorize B1 masks",
            "does not approve output",
        ],
    }
    plan["plan_sha256"] = payload_hash(plan)
    return plan


def validate_shadow_plan(plan: dict[str, object]) -> None:
    """Validate the immutable planner contract used by review evidence."""

    if plan.get("schema_version") != 1:
        raise ValueError("Unsupported execution plan schema")
    if plan.get("mode") != "shadow_no_pixel_authority":
        raise ValueError("Execution plan has pixel authority")
    expected = plan.get("plan_sha256")
    clean = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if not isinstance(expected, str) or payload_hash(clean) != expected:
        raise RuntimeError("Execution plan fingerprint mismatch")
