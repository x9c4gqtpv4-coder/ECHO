from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
from PIL import Image


MEDIAPIPE_MODEL_SHA256 = "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"


@dataclass(frozen=True)
class MulticlassEvidence:
    background: np.ndarray
    hair: np.ndarray
    body_skin: np.ndarray
    face_skin: np.ndarray
    clothes: np.ndarray
    accessories: np.ndarray
    backend: str
    model_path: str


def find_multiclass_model() -> Path | None:
    configured = os.environ.get("BATCH_COLOR_MULTICLASS_MODEL")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate
    candidates = [
        Path.cwd() / "models/selfie_multiclass_256x256.tflite",
        Path(__file__).resolve().parents[2]
        / "models/selfie_multiclass_256x256.tflite",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=2)
def _segmenter(model_path: str):
    try:
        import mediapipe as mp
    except ImportError as error:
        raise FileNotFoundError(
            "MediaPipe is unavailable; install the semantic dependency"
        ) from error
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    return mp.tasks.vision.ImageSegmenter.create_from_options(options)


def mediapipe_multiclass(
    image: Image.Image,
    *,
    model_path: str | Path | None = None,
) -> MulticlassEvidence:
    resolved = Path(model_path).expanduser() if model_path else find_multiclass_model()
    if resolved is None or not resolved.is_file():
        raise FileNotFoundError(
            "MediaPipe multiclass model is unavailable; run scripts/download_semantic_model.sh"
        )
    try:
        import mediapipe as mp
    except ImportError as error:
        raise FileNotFoundError(
            "MediaPipe is unavailable; install with: python -m pip install '.[semantic]'"
        ) from error
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    result = _segmenter(str(resolved.resolve())).segment(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    )
    masks = result.confidence_masks or []
    if len(masks) != 6:
        raise RuntimeError(f"MediaPipe multiclass returned {len(masks)} masks, expected 6")
    arrays = []
    for mask in masks:
        values = np.asarray(mask.numpy_view(), dtype=np.float32)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if values.shape != rgb.shape[:2] or not np.all(np.isfinite(values)):
            raise RuntimeError("MediaPipe multiclass returned invalid confidence geometry")
        arrays.append(np.clip(values, 0.0, 1.0).copy())
    return MulticlassEvidence(
        background=arrays[0],
        hair=arrays[1],
        body_skin=arrays[2],
        face_skin=arrays[3],
        clothes=arrays[4],
        accessories=arrays[5],
        backend="mediapipe-selfie-multiclass-256-v1",
        model_path=str(resolved.resolve()),
    )
