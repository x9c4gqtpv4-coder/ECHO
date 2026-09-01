"""Immutable contracts for user-approved colour baselines.

The contract describes colour math, not a claim of production approval.  A
caller may still request experimental overrides, but the resulting report must
not identify that output as the frozen A0 baseline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from batch_color.safety import payload_hash


@dataclass(frozen=True)
class BaselineContract:
    schema_version: int
    baseline_id: str
    pipeline: str
    transform_space: str
    render_policy: str
    background_strength: float
    person_strength: float
    background_luminance_cap: float
    background_chroma_cap: float
    person_luminance_cap: float
    person_chroma_cap: float
    person_scene_luminance_scale: float
    person_scene_chroma_scale: float
    set_color_tolerance: float
    no_op_policy: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return payload_hash(self.as_dict())


A0_BASELINE = BaselineContract(
    schema_version=1,
    baseline_id="a0-person-background-v1",
    pipeline="person-background-two-anchor-v2",
    transform_space="oklab",
    render_policy="single-pass-float32-from-original",
    background_strength=0.78,
    person_strength=0.58,
    background_luminance_cap=0.10,
    background_chroma_cap=0.035,
    person_luminance_cap=0.060,
    person_chroma_cap=0.020,
    person_scene_luminance_scale=0.60,
    person_scene_chroma_scale=0.55,
    set_color_tolerance=2.0,
    no_op_policy="exact-transform-identity",
)

# Frozen A0 execution-closure identity.  This value changes only after the
# output golden test proves that a safety/identity change preserved A0 pixels.
A0_EXPECTED_CODE_FINGERPRINT = (
    "79563f8e360b2ffd6cbfaced0870f82ecc14169e887b8cac3534d38cb6677e86"
)
A0_EXPECTED_DEPENDENCIES = {"numpy": "2.3.5", "pillow": "12.3.0"}
A0_EXPECTED_PERSON_HELPER_SHA256 = (
    "12ee561ddaa9f40717f79f6b06ebbcb14552ec3c2ebb7b4c8c1695f18d7b32eb"
)


def a0_compatible(
    *,
    background_strength: float,
    person_strength: float,
    set_color_tolerance: float,
    runtime_compatibility: dict[str, object] | None = None,
) -> bool:
    parameters = (
        background_strength == A0_BASELINE.background_strength
        and person_strength == A0_BASELINE.person_strength
        and set_color_tolerance == A0_BASELINE.set_color_tolerance
    )
    return bool(
        parameters
        and runtime_compatibility is not None
        and runtime_compatibility.get("compatible") is True
    )
