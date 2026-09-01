from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, __version__ as pillow_version

from batch_color.color import srgb_to_oklab
from batch_color.image_io import ImageInfo, image_to_float, make_proxy
from batch_color.safety import atomic_json, file_hash, payload_hash
from batch_color import __version__

PROFILE_VERSION = 5
MAX_PROFILE_BYTES = 1_048_576
L_QUANTILES = (0.03, 0.10, 0.25, 0.50, 0.75, 0.90, 0.97)


@dataclass(frozen=True)
class RegionStatistics:
    l_quantiles: list[float]
    a_median: float
    b_median: float
    a_mad: float
    b_mad: float
    sample_count: int
    retained_fraction: float


@dataclass(frozen=True)
class SurfaceStatistics:
    coefficients: list[list[float]]
    residual: float
    sample_count: int
    model: str = "legacy-quadratic"
    trusted: bool = False
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ColorProfile:
    version: int
    name: str
    reference_filename: str
    reference_profile: str
    background: RegionStatistics
    background_surface: SurfaceStatistics
    background_sampling: str = "legacy-border"
    reference_pixels_sha256: str | None = None
    reference_mask_sha256: str | None = None
    reference_mask_backend: str | None = None
    reference_info: dict[str, object] = field(default_factory=dict)
    generator: dict[str, object] = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        validate_profile(self)
        atomic_json(path, asdict(self))

    @classmethod
    def from_json(cls, path: str | Path) -> "ColorProfile":
        """Legacy/statistics-only import. JSON never grants runtime evidence trust."""
        try:
            with Path(path).open("rb") as stream:
                data = stream.read(MAX_PROFILE_BYTES + 1)
            profile = profile_from_payload(strict_json(data))
        except (KeyError, TypeError, ValueError, OverflowError, AttributeError, RecursionError) as error:
            raise ValueError(f"Invalid Profile schema/statistics: {error}") from error
        return profile


def strict_json(data: bytes, limit: int = MAX_PROFILE_BYTES):
    if len(data) > limit:
        raise ValueError("JSON exceeds the supported size limit")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=unique, parse_constant=reject_constant)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid or excessively nested JSON") from error


def profile_from_payload(payload) -> ColorProfile:
    try:
        payload = dict(payload) if isinstance(payload, dict) else None
        if payload is None:
            raise ValueError("Profile must be an object")
        payload["background"] = RegionStatistics(**payload["background"])
        payload["background_surface"] = SurfaceStatistics(**payload["background_surface"])
        profile = ColorProfile(**payload)
        validate_profile(profile)
        return profile
    except (KeyError, TypeError, OverflowError, AttributeError, RecursionError) as error:
        raise ValueError(f"Invalid Profile schema/statistics: {error}") from error


def generator_identity() -> dict[str, object]:
    """Bind the recipe to the actual installed implementation, not only a label."""
    directory = Path(__file__).resolve().parent
    return {"version": __version__, "recipe_id": "masked-oklab-background-v1",
            "implementation": {name: file_hash(directory / name) for name in
                               ("profile.py", "surface.py", "color.py", "image_io.py")},
            "numpy": np.__version__, "pillow": pillow_version,
            "parameters": {"statistics_proxy": 768, "surface_proxy": 640,
                           "sample_limit": 60000, "core_min_filter": 5, "core_threshold": 250,
                           "coordinates": "normalized_image_xy_minus1_to1"}}


def reference_evidence_verified(profile: ColorProfile) -> bool:
    # A private runtime binding is deliberately NOT a dataclass/JSON field.
    # dataclasses.replace, JSON import, or mutation of nested fields loses trust.
    # This is an input-evidence contract, not a sandbox against arbitrary Python.
    binding = getattr(profile, "_reference_evidence_digest", None)
    return bool(binding and profile.version == PROFILE_VERSION
                and profile.background_sampling == "masked-core"
                and profile.reference_mask_sha256 and profile.reference_pixels_sha256
                and profile.generator == generator_identity()
                and binding == payload_hash(asdict(profile)))


def evidence_status(profile: ColorProfile) -> dict[str, object]:
    validate_profile(profile)
    verified = reference_evidence_verified(profile)
    return {"numeric_valid": True, "reference_evidence_verified": verified,
            "verification": "recomputed_reference_pixels_and_mask" if verified else "unverified_statistics_only",
            "sampling_reviewed": False, "human_review_required": True,
            "spatial_correspondence_verified": False,
            "reference_surface_support_passed": profile.background_surface.trusted,
            "reference_allows_spatial_candidate": bool(verified and profile.background_surface.trusted),
            "reference_pixels_sha256": profile.reference_pixels_sha256,
            "reference_mask_sha256": profile.reference_mask_sha256,
            "generator": profile.generator}


def _number(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a number, not a string/bool")
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    if not np.isfinite(np.float32(value)):
        raise ValueError(f"{name} is not safe in float32")


def surface_values_valid(coefficients):
    axis = np.linspace(-1, 1, 33)
    y, x = np.meshgrid(axis, axis)
    c = np.asarray(coefficients, dtype=np.float64)
    predicted = _surface_features(x, y).astype(np.float64) @ c.T
    dx = c[:, 1] + 2 * x[..., None] * c[:, 3] + y[..., None] * c[:, 5]
    dy = c[:, 2] + 2 * y[..., None] * c[:, 4] + x[..., None] * c[:, 5]
    return bool(np.all(np.isfinite(predicted)) and np.all(np.isfinite(dx)) and np.all(np.isfinite(dy))
                and predicted[..., 0].min() >= -0.02 and predicted[..., 0].max() <= 1.02
                and np.abs(predicted[..., 1:]).max() <= 0.5
                and np.maximum(np.linalg.norm(dx, axis=-1), np.linalg.norm(dy, axis=-1)).max() <= 0.75)


def validate_profile(profile: ColorProfile) -> None:
    """Validate direct API objects as well as JSON; reject coercion and non-finite metadata."""
    if not isinstance(profile, ColorProfile) or type(profile.version) is not int or profile.version not in {2, 3, 4, 5}:
        raise ValueError("Unsupported Profile type/version")
    for value in (profile.name, profile.reference_filename, profile.reference_profile, profile.background_sampling):
        if not isinstance(value, str) or not value:
            raise ValueError("Profile names and sampling mode must be nonempty strings")
    if not isinstance(profile.reference_info, dict):
        raise ValueError("reference_info must be an object")
    if not isinstance(profile.generator, dict):
        raise ValueError("generator must be an object")
    if profile.version == 5 and (not isinstance(profile.generator.get("version"), str)
                                or not isinstance(profile.generator.get("recipe_id"), str)
                                or not isinstance(profile.generator.get("implementation"), dict)):
        raise ValueError("Profile v5 requires a generator identity")
    if profile.reference_mask_backend is not None and not isinstance(profile.reference_mask_backend, str):
        raise ValueError("reference_mask_backend must be a string or null")
    for digest in (profile.reference_pixels_sha256, profile.reference_mask_sha256):
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                   or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Invalid reference content hash")
    stats, surface = profile.background, profile.background_surface
    if not isinstance(stats, RegionStatistics) or not isinstance(surface, SurfaceStatistics):
        raise ValueError("Invalid region/surface schema")
    if not isinstance(stats.l_quantiles, (list, tuple)) or len(stats.l_quantiles) != 7:
        raise ValueError("Expected seven luminance quantiles")
    for v in stats.l_quantiles:
        _number(v, "l_quantile", 0, 1)
    if np.any(np.diff(stats.l_quantiles) < 0):
        raise ValueError("Luminance quantiles must be nondecreasing")
    for name in ("a_median", "b_median"):
        _number(getattr(stats, name), name, -0.5, 0.5)
    for name in ("a_mad", "b_mad"):
        _number(getattr(stats, name), name, 0, 0.5)
    _number(stats.retained_fraction, "retained_fraction", 0, 1)
    if stats.retained_fraction == 0:
        raise ValueError("retained_fraction must be positive")
    for count in (stats.sample_count, surface.sample_count):
        if type(count) is not int or not 0 < count <= 1_000_000_000:
            raise ValueError("sample_count must be a positive integer")
    _number(surface.residual, "surface residual", 0, 2)
    if not isinstance(surface.coefficients, (list, tuple)) or len(surface.coefficients) != 3:
        raise ValueError("Surface needs three coefficient channels")
    for row in surface.coefficients:
        if not isinstance(row, (list, tuple)) or len(row) != 6:
            raise ValueError("Surface needs six coefficients per channel")
        for v in row:
            _number(v, "surface coefficient", -4, 4)
    if not surface_values_valid(surface.coefficients):
        raise ValueError("Surface exceeds supported color/gradient bounds on the standard grid")
    if type(surface.trusted) is not bool or not isinstance(surface.diagnostics, dict):
        raise ValueError("Invalid surface trust diagnostics")
    if surface.model not in {"constant", "plane", "quadratic", "legacy-quadratic"}:
        raise ValueError("Invalid surface model")
    c = np.asarray(surface.coefficients)
    if ((surface.model == "constant" and np.any(c[:, 1:] != 0))
            or (surface.model == "plane" and np.any(c[:, 3:] != 0))):
        raise ValueError("Surface coefficients do not match the declared model")
    if surface.trusted:
        from batch_color.surface import support_is_valid
        if profile.version not in {4, 5} or surface.model == "legacy-quadratic" or not support_is_valid(surface.diagnostics):
            raise ValueError("Trusted surface is missing valid support diagnostics")
        if profile.version == 5 and (profile.background_sampling != "masked-core"
                                    or not profile.reference_mask_sha256 or not profile.reference_pixels_sha256):
            raise ValueError("Profile v5 surface support requires a bound reference mask and pixels")
        errors = surface.diagnostics.get("blocked_validation_rmse", {})
        if not isinstance(errors, dict) or surface.model not in errors:
            raise ValueError("Trusted surface requires blocked validation evidence")
        for error in errors.values():
            _number(error, "blocked RMSE", 0, 2)
        if (errors[surface.model] > 0.05 or surface.diagnostics.get("selected_model") != surface.model
                or surface.diagnostics.get("reason") != "support_passed"):
            raise ValueError("Invalid surface validation decision")
    try:
        encoded = json.dumps(asdict(profile), allow_nan=False).encode("utf-8")
        if len(encoded) > MAX_PROFILE_BYTES:
            raise ValueError("Profile exceeds the supported metadata size")
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("Profile metadata must be finite JSON values") from error


def _border_samples(image: Image.Image, max_samples: int = 60_000) -> np.ndarray:
    """Sample likely studio background from top and side borders."""
    proxy = make_proxy(image, max_edge=768)
    rgb = image_to_float(proxy)
    h, w = rgb.shape[:2]
    yy, xx = np.ogrid[:h, :w]

    top = yy < max(2, int(round(h * 0.15)))
    sides = (xx < max(2, int(round(w * 0.075)))) | (
        xx >= w - max(2, int(round(w * 0.075)))
    )
    not_floor = yy < int(round(h * 0.90))
    border_mask = top | (sides & not_floor)
    samples = srgb_to_oklab(rgb[border_mask])

    if samples.shape[0] > max_samples:
        indices = np.linspace(0, samples.shape[0] - 1, max_samples, dtype=np.int64)
        samples = samples[indices]
    return samples


def _core_mask(mask: Image.Image, size: tuple[int, int]) -> np.ndarray:
    proxy_mask = mask.convert("L").resize(size, Image.Resampling.NEAREST)
    core = np.asarray(proxy_mask.filter(ImageFilter.MinFilter(5))) >= 250
    if np.count_nonzero(core) < max(100, int(core.size * 0.005)):
        raise ValueError("Insufficient background core; provide a reviewed mask instead of guessing")
    return core


def analyse_background(image: Image.Image, mask: Image.Image | None = None) -> RegionStatistics:
    if mask is None:
        samples = _border_samples(image)
    else:
        if mask.size != image.size:
            raise ValueError("Statistics mask geometry mismatch")
        proxy = make_proxy(image, max_edge=768)
        samples = srgb_to_oklab(image_to_float(proxy)[_core_mask(mask, proxy.size)])
        if len(samples) > 60_000:
            samples = samples[np.linspace(0, len(samples) - 1, 60_000, dtype=int)]
    center = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - center), axis=0)
    safe_mad = np.maximum(mad, np.array([0.012, 0.006, 0.006], dtype=np.float32))

    robust_distance = np.sqrt(np.sum(np.square((samples - center) / safe_mad), axis=1))
    retained = samples[robust_distance <= 6.0]
    if retained.shape[0] < max(100, int(samples.shape[0] * 0.25)):
        retained = samples

    retained_median = np.median(retained, axis=0)
    retained_mad = np.median(np.abs(retained - retained_median), axis=0)
    l_values = np.quantile(retained[:, 0], L_QUANTILES)
    return RegionStatistics(
        l_quantiles=[round(float(value), 8) for value in l_values],
        a_median=round(float(retained_median[1]), 8),
        b_median=round(float(retained_median[2]), 8),
        a_mad=round(float(retained_mad[1]), 8),
        b_mad=round(float(retained_mad[2]), 8),
        sample_count=int(retained.shape[0]),
        retained_fraction=round(float(retained.shape[0] / samples.shape[0]), 6),
    )


def _surface_features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.ones_like(x), x, y, x * x, y * y, x * y],
        axis=-1,
    ).astype(np.float32)


def analyse_background_surface(image: Image.Image, mask: Image.Image | None = None) -> SurfaceStatistics:
    """Fit a robust low-frequency Oklab surface from all image borders."""
    proxy = make_proxy(image, max_edge=640)
    rgb = image_to_float(proxy)
    lab = srgb_to_oklab(rgb)
    height, width = lab.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    band_y = max(3, int(round(height * 0.075)))
    band_x = max(3, int(round(width * 0.075)))
    border = (
        (yy < band_y)
        | (yy >= height - band_y)
        | (xx < band_x)
        | (xx >= width - band_x)
    )
    if mask is not None:
        if mask.size != image.size:
            raise ValueError("Surface mask geometry mismatch")
        border = _core_mask(mask, proxy.size)

    x = (2.0 * xx[border] / max(width - 1, 1) - 1.0).astype(np.float32)
    y = (2.0 * yy[border] / max(height - 1, 1) - 1.0).astype(np.float32)
    features = _surface_features(x, y)
    values = lab[border]
    if len(values) > 60_000:
        indices = np.linspace(0, len(values) - 1, 60_000, dtype=int)
        features, values = features[indices], values[indices]
        x, y = x[indices], y[indices]
    from batch_color.surface import choose_surface
    channel_first, residual, model, trusted, diagnostics = choose_surface(
        features, values, x, y, surface_values_valid)
    return SurfaceStatistics(
        coefficients=[
            [round(float(value), 9) for value in channel]
            for channel in channel_first
        ],
        residual=round(residual, 8),
        sample_count=int(values.shape[0]),
        model=model, trusted=trusted, diagnostics=diagnostics,
    )


def evaluate_surface(
    surface: SurfaceStatistics,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    features = _surface_features(x, y)
    coefficients = np.asarray(surface.coefficients, dtype=np.float32)
    result = features @ coefficients.T
    if not np.all(np.isfinite(result)):
        raise ValueError("Surface evaluation produced non-finite values")
    return result


def create_profile(
    image: Image.Image,
    image_info: ImageInfo,
    *,
    name: str,
    background_mask: Image.Image | None = None,
    mask_backend: str | None = None,
    mask_metadata: dict[str, object] | None = None,
) -> ColorProfile:
    if image.mode != "RGB" or image.size != (image_info.width, image_info.height):
        raise ValueError("Profile requires canonical 8-bit RGB pixels and matching ImageInfo geometry")
    if background_mask is not None:
        if background_mask.mode != "L" or background_mask.size != image.size:
            raise ValueError("Reference mask must be canonical 8-bit L and match reference geometry")
    surface = analyse_background_surface(image, background_mask)
    if background_mask is None:
        surface = replace(surface, trusted=False,
                          diagnostics={**surface.diagnostics, "reason": "reference_mask_missing"})
    profile = ColorProfile(
        version=PROFILE_VERSION,
        name=name,
        reference_filename=Path(image_info.path).name,
        reference_profile=image_info.source_profile,
        background=analyse_background(image, background_mask),
        background_surface=surface,
        background_sampling="masked-core" if background_mask is not None else "legacy-border",
        reference_pixels_sha256=hashlib.sha256(image.convert("RGB").tobytes()).hexdigest(),
        reference_mask_sha256=hashlib.sha256(background_mask.tobytes()).hexdigest() if background_mask is not None else None,
        reference_mask_backend=mask_backend,
        reference_info={**asdict(image_info), "warnings": list(image_info.warnings),
                        "mask_generation": dict(mask_metadata or {})},
        generator=generator_identity(),
    )
    validate_profile(profile)
    if background_mask is not None:
        object.__setattr__(profile, "_reference_evidence_digest", payload_hash(asdict(profile)))
    return profile
