"""Framework-neutral, in-memory adapter around the frozen ECHO color engine."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from threading import RLock
from typing import Any

import numpy as np
from PIL import Image

from batch_color import __version__
from batch_color.image_io import ImageInfo
from batch_color.masking import backend_identity, get_background_mask
from batch_color.profile import ColorProfile, create_profile
from batch_color.transfer import select_profile_path


_CACHE_LIMIT = 8
_CACHE_LOCK = RLock()


@dataclass(frozen=True)
class _ReferenceBundle:
    profile: ColorProfile
    mask_backend: str


_PROFILE_CACHE: OrderedDict[str, _ReferenceBundle] = OrderedDict()


def clear_profile_cache() -> None:
    """Clear only the small in-memory reference cache; no files are written."""
    with _CACHE_LOCK:
        _PROFILE_CACHE.clear()


def profile_cache_size() -> int:
    with _CACHE_LOCK:
        return len(_PROFILE_CACHE)


def _canonical_image(value: np.ndarray, *, name: str) -> Image.Image:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [H,W,3]")
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must be canonical uint8 RGB")
    if min(array.shape[:2]) < 8:
        raise ValueError(f"{name} is too small; both dimensions must be at least 8 pixels")
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def _canonical_batch(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [B,H,W,3]")
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must be canonical uint8 RGB")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array


def _canonical_masks(value: np.ndarray | None, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [B,H,W]")
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must be canonical uint8 L")
    return np.ascontiguousarray(array)


def _batch_index(length: int, index: int, batch_size: int, *, name: str) -> int:
    if length == 1:
        return 0
    if length == batch_size:
        return index
    raise ValueError(f"{name} batch must contain 1 or {batch_size} items, got {length}")


def _mask_image(
    masks: np.ndarray | None,
    index: int,
    batch_size: int,
    size: tuple[int, int],
    *,
    name: str,
) -> Image.Image | None:
    if masks is None:
        return None
    selected = masks[_batch_index(len(masks), index, batch_size, name=name)]
    expected = (size[1], size[0])
    if selected.shape != expected:
        raise ValueError(
            f"{name} geometry {selected.shape[::-1]} does not match image geometry {size}; "
            "resize the mask explicitly before ECHO"
        )
    return Image.fromarray(selected, mode="L")


def _image_info(image: Image.Image, role: str, digest: str) -> ImageInfo:
    return ImageInfo(
        path=f"comfyui://{role}/{digest}.png",
        width=image.width,
        height=image.height,
        source_profile="ComfyUI IMAGE tensor; assumed sRGB",
        converted_to_srgb=False,
        original_mode="RGB",
        warnings=("comfyui_tensor_assumed_srgb", "canonicalized_to_8bit_rgb"),
        original_format="ComfyUI IMAGE tensor",
        original_bit_depth=8,
    )


def _materialized_background(
    image: Image.Image,
    supplied: Image.Image | None,
    *,
    backend: str,
    role: str,
) -> tuple[Image.Image, str, dict[str, Any]]:
    if supplied is not None:
        return supplied, "external-supplied", {"role": role, "source": "connected MASK input"}
    result = get_background_mask(
        f"comfyui://{role}",
        image,
        backend=backend,
        quality="accurate",
    )
    if result.background_mask is None:  # Defensive: get_background_mask materializes it.
        raise RuntimeError("ECHO failed to materialize a background mask")
    return result.background_mask, result.backend, {
        "role": role,
        "source": "generated",
        "message": result.message,
        "fallback_reason": result.fallback_reason,
        "backend_identity": backend_identity(backend, "accurate"),
    }


def _reference_key(reference: Image.Image, mask: Image.Image | None, backend: str) -> str:
    digest = hashlib.sha256()
    digest.update(reference.tobytes())
    digest.update(str(reference.size).encode("ascii"))
    digest.update(backend.encode("utf-8"))
    if mask is not None:
        digest.update(b"external-mask\0")
        digest.update(mask.tobytes())
    else:
        digest.update(b"generated-mask\0")
        digest.update(json.dumps(backend_identity(backend, "accurate"), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _reference_bundle(
    reference: Image.Image,
    supplied_mask: Image.Image | None,
    *,
    backend: str,
) -> tuple[_ReferenceBundle, bool]:
    key = _reference_key(reference, supplied_mask, backend)
    with _CACHE_LOCK:
        cached = _PROFILE_CACHE.get(key)
        if cached is not None:
            _PROFILE_CACHE.move_to_end(key)
            return cached, True

    background_mask, mask_backend, metadata = _materialized_background(
        reference, supplied_mask, backend=backend, role="reference"
    )
    profile = create_profile(
        reference,
        _image_info(reference, "reference", key),
        name=f"ECHO ComfyUI reference {key[:12]}",
        background_mask=background_mask,
        mask_backend=mask_backend,
        mask_metadata=metadata,
    )
    bundle = _ReferenceBundle(profile=profile, mask_backend=mask_backend)
    with _CACHE_LOCK:
        _PROFILE_CACHE[key] = bundle
        _PROFILE_CACHE.move_to_end(key)
        while len(_PROFILE_CACHE) > _CACHE_LIMIT:
            _PROFILE_CACHE.popitem(last=False)
    return bundle, False


def _normalize_mode(mode: str) -> str:
    aliases = {
        "background + person": "both",
        "both": "both",
        "background only": "background",
        "background": "background",
    }
    try:
        return aliases[mode]
    except KeyError as error:
        raise ValueError("adjustment_mode must be background + person or background only") from error


def match_numpy(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    strength: float = 0.85,
    adjustment_mode: str = "background + person",
    transform_path: str = "auto",
    mask_backend: str = "heuristic",
    source_background_mask: np.ndarray | None = None,
    reference_background_mask: np.ndarray | None = None,
    protect_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Match canonical uint8 RGB batches and return output, source mask and JSON report.

    The reference and all optional masks may contain one item (broadcast) or the
    same number of items as the source batch.  White protection pixels remain
    byte-identical to the source after the one-pass render.
    """
    if mask_backend not in {"heuristic", "auto"}:
        raise ValueError("mask_backend must be heuristic or auto")
    if transform_path not in {"auto", "global", "surface"}:
        raise ValueError("transform_path must be auto, global or surface")
    if not np.isfinite(strength) or not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be finite and between 0 and 1")

    sources = _canonical_batch(source, name="source")
    references = _canonical_batch(reference, name="reference")
    source_masks = _canonical_masks(source_background_mask, name="source_background_mask")
    reference_masks = _canonical_masks(reference_background_mask, name="reference_background_mask")
    protection_masks = _canonical_masks(protect_mask, name="protect_mask")
    batch_size = len(sources)
    if len(references) not in {1, batch_size}:
        raise ValueError(
            f"reference batch must contain 1 or {batch_size} items, got {len(references)}"
        )

    outputs: list[np.ndarray] = []
    output_masks: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    engine_mode = _normalize_mode(adjustment_mode)

    for index, source_array in enumerate(sources):
        reference_index = _batch_index(len(references), index, batch_size, name="reference")
        reference_image = _canonical_image(references[reference_index], name="reference")
        source_image = _canonical_image(source_array, name="source")
        reference_mask = _mask_image(
            reference_masks,
            reference_index if len(references) > 1 else 0,
            len(references),
            reference_image.size,
            name="reference_background_mask",
        )
        source_mask = _mask_image(
            source_masks, index, batch_size, source_image.size, name="source_background_mask"
        )
        protected = _mask_image(
            protection_masks, index, batch_size, source_image.size, name="protect_mask"
        )

        bundle, cache_hit = _reference_bundle(
            reference_image, reference_mask, backend=mask_backend
        )
        source_mask, source_backend, source_mask_metadata = _materialized_background(
            source_image, source_mask, backend=mask_backend, role="source"
        )
        corrected, engine_report, used_mask = select_profile_path(
            source_image,
            bundle.profile,
            strength=float(strength),
            path=transform_path,
            background_mask=source_mask,
            mask_backend=source_backend,
            mode=engine_mode,
            protected_mask=protected,
        )
        outputs.append(np.asarray(corrected, dtype=np.uint8))
        output_masks.append(np.asarray(used_mask.convert("L"), dtype=np.uint8))
        reports.append(
            {
                "schema": "echo-comfyui/1",
                "engine_version": __version__,
                "batch_index": index,
                "status": "review",
                "approved": False,
                "reference_cache_hit": cache_hit,
                "reference_mask_backend": bundle.mask_backend,
                "source_mask": source_mask_metadata,
                "protection_mask_connected": protected is not None,
                "engine": engine_report.as_dict(),
            }
        )

    return (
        np.stack(outputs, axis=0),
        np.stack(output_masks, axis=0),
        json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False),
    )
