from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageCms, ImageOps

from batch_color.safety import atomic_output, file_hash
from batch_color.encoding import inspect_encoding


@dataclass(frozen=True)
class ImageInfo:
    path: str
    width: int
    height: int
    source_profile: str
    converted_to_srgb: bool
    original_orientation: int = 1
    original_mode: str = "RGB"
    icc_sha256: str | None = None
    warnings: tuple[str, ...] = ()
    original_format: str = "unknown"
    original_bit_depth: int = 8


def _profile_name(profile: ImageCms.ImageCmsProfile) -> str:
    try:
        return ImageCms.getProfileName(profile).strip()
    except Exception:
        return "embedded ICC"


def load_srgb(
    path: str | Path,
    *,
    alpha_policy: str = "reject",
) -> tuple[Image.Image, ImageInfo]:
    """Decode ONE canonical 8-bit RGB image; never silently drop alpha/high bits."""
    if alpha_policy not in {"reject", "drop_near_opaque"}:
        raise ValueError("alpha_policy must be reject or drop_near_opaque")
    image_path = Path(path)
    with Image.open(image_path) as opened:
        original_format, original_bit_depth = inspect_encoding(image_path, opened)
        opened.load()
        if getattr(opened, "n_frames", 1) != 1:
            raise ValueError("Multi-frame images are not supported; export a single frame first")
        original_mode = opened.mode
        alpha_warning = None
        if "A" in opened.getbands() or "transparency" in opened.info:
            alpha = opened.convert("RGBA").getchannel("A")
            alpha_extrema = alpha.getextrema()
            if alpha_extrema != (255, 255):
                alpha_values = np.asarray(alpha, dtype=np.uint8)
                transparent_ratio = float(np.mean(alpha_values < 255))
                if not (
                    alpha_policy == "drop_near_opaque"
                    and alpha_extrema[0] >= 253
                    and transparent_ratio <= 0.001
                ):
                    raise ValueError("Transparent input needs an explicit compositing policy")
                alpha_warning = "near_opaque_alpha_dropped_preserving_rgb"
        orientation = int(opened.getexif().get(274, 1))
        icc_bytes = opened.info.get("icc_profile")
        oriented = ImageOps.exif_transpose(opened)
        if alpha_warning:
            oriented = oriented.convert("RGB")
        converted = False
        profile_name = "missing; assumed sRGB"
        warning_items: list[str] = []
        if alpha_warning:
            warning_items.append(alpha_warning)

        if icc_bytes:
            try:
                source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_bytes))
                profile_name = _profile_name(source_profile)
                srgb_profile = ImageCms.createProfile("sRGB")
                image = ImageCms.profileToProfile(
                    oriented,
                    source_profile,
                    srgb_profile,
                    outputMode="RGB",
                    renderingIntent=ImageCms.Intent.PERCEPTUAL,
                )
                converted = True
            except (OSError, ValueError, ImageCms.PyCMSError) as error:
                raise ValueError("Unreadable or incompatible ICC profile; refusing guessed colors") from error
        else:
            if original_mode in {"CMYK", "LAB"}:
                raise ValueError("CMYK/LAB input requires a valid embedded ICC profile")
            image = oriented.convert("RGB")
            warning_items.append("missing_icc_assumed_srgb")

        image = image.copy()
        # Native segmentation must see these exact oriented, ICC-normalized pixels.
        image.info.clear()
        image.info["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    return image, ImageInfo(
        path=str(image_path),
        width=image.width,
        height=image.height,
        source_profile=profile_name,
        converted_to_srgb=converted,
        original_orientation=orientation,
        original_mode=original_mode,
        icc_sha256=hashlib.sha256(icc_bytes).hexdigest() if icc_bytes else None,
        warnings=tuple(warning_items),
        original_format=original_format,
        original_bit_depth=original_bit_depth,
    )


def image_to_float(image: Image.Image) -> NDArray[np.float32]:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_srgb(image: Image.Image, path: str | Path, *, quality: int = 95) -> dict[str, object]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    srgb_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    save_args: dict[str, object] = {"icc_profile": srgb_profile}
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        save_args.update(quality=quality, subsampling=0, optimize=True)
    rgb = image.convert("RGB")
    with atomic_output(output_path) as staged:
        rgb.save(staged, **save_args)
        with Image.open(staged) as check:
            check.load()
            if check.size != rgb.size or check.mode != "RGB" or not check.info.get("icc_profile"):
                raise ValueError("Encoded image failed size/mode/ICC verification")
            if output_path.suffix.lower() in {".png", ".tif", ".tiff"}:
                if not np.array_equal(np.asarray(check), np.asarray(rgb)):
                    raise ValueError("Lossless output did not preserve computed pixels")
    return {"sha256": file_hash(output_path), "width": rgb.width, "height": rgb.height,
            "mode": "RGB", "bit_depth": 8, "reopened": True,
            "lossless": output_path.suffix.lower() in {".png", ".tif", ".tiff"}}


def save_mask(mask: Image.Image, path: str | Path) -> None:
    with atomic_output(path) as staged:
        mask.convert("L").save(staged, format="PNG")
        with Image.open(staged) as check:
            check.load()
            if check.size != mask.size or not np.array_equal(np.asarray(check), np.asarray(mask.convert("L"))):
                raise ValueError("Mask failed lossless verification")


def load_mask(path: str | Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as opened:
        inspect_encoding(path, opened)
        if getattr(opened, "n_frames", 1) != 1:
            raise ValueError("Multi-frame masks are not supported")
        if "A" in opened.getbands() or "transparency" in opened.info:
            if opened.convert("RGBA").getchannel("A").getextrema() != (255, 255):
                raise ValueError("Transparent masks need an explicit grayscale representation")
        mask = ImageOps.exif_transpose(opened).convert("L").copy()
    if mask.size != size:
        raise ValueError("External masks must match canonical image dimensions; automatic resizing is unsafe")
    return mask


def make_proxy(image: Image.Image, max_edge: int = 768) -> Image.Image:
    proxy = image.copy()
    proxy.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return proxy
