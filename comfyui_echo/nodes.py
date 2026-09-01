"""ComfyUI-facing tensor adapter for ECHO."""

from __future__ import annotations

import numpy as np

from .core import match_numpy


def _image_tensor_to_u8(tensor, *, name: str) -> np.ndarray:
    try:
        array = tensor.detach().cpu().numpy()
    except AttributeError as error:
        raise TypeError(f"{name} must be a ComfyUI IMAGE tensor") from error
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have ComfyUI IMAGE shape [B,H,W,3]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite pixels")
    return np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _mask_tensor_to_u8(tensor, *, name: str) -> np.ndarray | None:
    if tensor is None:
        return None
    try:
        array = tensor.detach().cpu().numpy()
    except AttributeError as error:
        raise TypeError(f"{name} must be a ComfyUI MASK tensor") from error
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"{name} must have ComfyUI MASK shape [B,H,W]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite pixels")
    return np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


class ECHOReferenceMatch:
    """Reference-guided, local-first color consistency in one ComfyUI node."""

    CATEGORY = "ECHO / 回响"
    FUNCTION = "match"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("corrected", "background_mask", "review_report")
    DESCRIPTION = (
        "Match source background and person color/exposure to a reference. "
        "Optional white protect_mask pixels remain unchanged. Outputs are review candidates."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "reference": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "adjustment_mode": (
                    ["background + person", "background only"],
                    {"default": "background + person"},
                ),
                "transform_path": (
                    ["auto", "global", "surface"],
                    {"default": "auto"},
                ),
                "mask_backend": (
                    ["heuristic", "auto"],
                    {"default": "heuristic"},
                ),
            },
            "optional": {
                "source_background_mask": ("MASK",),
                "reference_background_mask": ("MASK",),
                "protect_mask": ("MASK",),
            },
        }

    def match(
        self,
        source,
        reference,
        strength: float,
        adjustment_mode: str,
        transform_path: str,
        mask_backend: str,
        source_background_mask=None,
        reference_background_mask=None,
        protect_mask=None,
    ):
        source_u8 = _image_tensor_to_u8(source, name="source")
        reference_u8 = _image_tensor_to_u8(reference, name="reference")
        corrected, background_mask, report = match_numpy(
            source_u8,
            reference_u8,
            strength=strength,
            adjustment_mode=adjustment_mode,
            transform_path=transform_path,
            mask_backend=mask_backend,
            source_background_mask=_mask_tensor_to_u8(
                source_background_mask, name="source_background_mask"
            ),
            reference_background_mask=_mask_tensor_to_u8(
                reference_background_mask, name="reference_background_mask"
            ),
            protect_mask=_mask_tensor_to_u8(protect_mask, name="protect_mask"),
        )
        try:
            import torch
        except ImportError as error:  # ComfyUI always ships torch; useful error for bad installs.
            raise RuntimeError("ECHO must run inside a ComfyUI Python environment with torch") from error
        return (
            torch.from_numpy(corrected.astype(np.float32) / 255.0),
            torch.from_numpy(background_mask.astype(np.float32) / 255.0),
            report,
        )
