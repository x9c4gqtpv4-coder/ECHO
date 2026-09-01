from __future__ import annotations

import shutil
import subprocess
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from batch_color.image_io import load_mask, load_srgb, save_srgb
from batch_color.safety import file_hash


@dataclass(frozen=True)
class MaskResult:
    background_mask: Image.Image | None
    backend: str
    message: str | None = None
    fallback_reason: str | None = None
    cacheable: bool = True


def find_vision_helper() -> Path | None:
    configured = os.environ.get("BATCH_COLOR_PERSON_MASK_BIN")
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file() and configured_path.stat().st_mode & 0o111:
            return configured_path

    installed = shutil.which("batch-color-person-mask")
    if installed:
        return Path(installed)

    candidates = [
        Path.cwd() / "tools/person_mask/bin/batch-color-person-mask",
        Path(__file__).resolve().parents[2]
        / "tools/person_mask/bin/batch-color-person-mask",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate
    return None


def find_face_helper() -> Path | None:
    configured = os.environ.get("BATCH_COLOR_FACE_MASK_BIN")
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file() and configured_path.stat().st_mode & 0o111:
            return configured_path
    installed = shutil.which("batch-color-face-mask")
    if installed:
        return Path(installed)
    candidates = [
        Path.cwd() / "tools/face_mask/bin/batch-color-face-mask",
        Path(__file__).resolve().parents[2] / "tools/face_mask/bin/batch-color-face-mask",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate
    return None


def vision_face_mask(
    input_path: str | Path,
    *,
    executable: Path | None = None,
    canonical_image: Image.Image | None = None,
) -> Image.Image:
    helper = executable or find_face_helper()
    if helper is None:
        raise FileNotFoundError("macOS face helper is not built; run scripts/build_face_mask.sh")
    with tempfile.TemporaryDirectory(prefix="batch-color-face-") as directory:
        canonical = canonical_image if canonical_image is not None else load_srgb(input_path)[0]
        canonical_path = Path(directory) / "canonical.png"
        save_srgb(canonical, canonical_path)
        output_path = Path(directory) / "face-mask.png"
        completed = subprocess.run(
            [str(helper), str(canonical_path), str(output_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(message or "macOS Vision face detection failed")
        with Image.open(output_path) as opened:
            mask = opened.convert("L").copy()
        if mask.size != canonical.size:
            raise RuntimeError("Native face mask geometry does not match canonical pixels")
        return mask


def vision_person_mask(
    input_path: str | Path,
    *,
    quality: str = "accurate",
    executable: Path | None = None,
    canonical_image: Image.Image | None = None,
) -> Image.Image:
    helper = executable or find_vision_helper()
    if helper is None:
        raise FileNotFoundError(
            "macOS Vision helper is not built; run scripts/build_person_mask.sh"
        )

    with tempfile.TemporaryDirectory(prefix="batch-color-mask-") as directory:
        canonical = canonical_image if canonical_image is not None else load_srgb(input_path)[0]
        canonical_path = Path(directory) / "canonical.png"
        save_srgb(canonical, canonical_path)
        output_path = Path(directory) / "person-mask.png"
        completed = subprocess.run(
            [str(helper), str(canonical_path), str(output_path), quality],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(message or "macOS Vision person segmentation failed")
        with Image.open(output_path) as opened:
            mask = opened.convert("L").copy()
        if mask.size != canonical.size:
            raise RuntimeError("Native mask geometry does not match canonical pixels")
        return mask


def resolve_background_mask(
    input_path: str | Path,
    image_size: tuple[int, int],
    *,
    backend: str = "auto",
    quality: str = "accurate",
    canonical_image: Image.Image | None = None,
) -> MaskResult:
    if backend == "heuristic":
        return MaskResult(None, "heuristic-color")
    if backend not in {"auto", "vision"}:
        raise ValueError("mask backend must be one of: auto, vision, heuristic")

    helper = find_vision_helper()
    if helper is None:
        if backend == "vision":
            raise FileNotFoundError(
                "macOS Vision helper is not built; run scripts/build_person_mask.sh"
            )
        return MaskResult(
            None,
            "heuristic-color",
            "Vision helper unavailable; used heuristic mask",
            "native_unavailable", True,
        )

    try:
        person_mask = vision_person_mask(
            input_path,
            quality=quality,
            executable=helper,
            canonical_image=canonical_image,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        if backend == "vision":
            raise
        return MaskResult(
            None,
            "heuristic-color",
            f"Vision failed; used heuristic mask: {error}",
            "vision_timeout" if isinstance(error, subprocess.TimeoutExpired) else "vision_runtime_failure", False,
        )

    if person_mask.size != image_size:
        raise ValueError("Mask/image coordinate mismatch; use the canonical oriented image size")
    return MaskResult(
        ImageOps.invert(person_mask),
        f"vision-{quality}",
    )


def materialize_mask(image: Image.Image, result: MaskResult) -> MaskResult:
    if result.background_mask is not None:
        if result.background_mask.size != image.size:
            raise ValueError("Mask must match canonical image dimensions")
        return result
    from batch_color.profile import analyse_background, analyse_background_surface
    from batch_color.segmentation import estimate_studio_background_mask

    mask = estimate_studio_background_mask(
        image, analyse_background(image), analyse_background_surface(image)
    )
    return MaskResult(mask, result.backend, result.message, result.fallback_reason, result.cacheable)


def backend_identity(backend: str, quality: str) -> dict[str, str | None]:
    helper = find_vision_helper() if backend != "heuristic" else None
    return {"requested": backend, "quality": quality,
            "native_binary_sha256": file_hash(helper) if helper else None}


def get_background_mask(
    input_path: str | Path, image: Image.Image, *, backend: str = "auto",
    quality: str = "accurate", mask_path: str | Path | None = None,
) -> MaskResult:
    if mask_path is not None:
        return MaskResult(load_mask(mask_path, image.size), "external-supplied")
    return materialize_mask(image, resolve_background_mask(
        input_path, image.size, backend=backend, quality=quality, canonical_image=image
    ))
