"""Inspect original sample precision BEFORE Pillow materializes pixels."""
from pathlib import Path
import struct

from PIL import Image


def jpeg_precision(path: str | Path) -> int:
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
           0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    with Path(path).open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            raise ValueError("Invalid JPEG signature")
        while stream.tell() < 16 * 1024 * 1024:
            if stream.read(1) != b"\xff":
                raise ValueError("Invalid JPEG marker before SOF")
            marker = stream.read(1)
            while marker == b"\xff":
                marker = stream.read(1)
            if not marker or marker[0] in {0x00, 0xDA, 0xD9}:
                break
            if marker[0] in {0x01, *range(0xD0, 0xD8)}:
                continue
            size = stream.read(2)
            if len(size) != 2:
                break
            length = struct.unpack(">H", size)[0]
            if length < 2:
                break
            data = stream.read(length - 2)
            if len(data) != length - 2:
                break
            if marker[0] in sof:
                if len(data) < 6:
                    break
                return data[0]
    raise ValueError("JPEG sample precision could not be established")


def inspect_encoding(path: str | Path, opened: Image.Image) -> tuple[str, int]:
    kind = opened.format
    if kind == "PNG":
        with Path(path).open("rb") as stream:
            header = stream.read(33)
        if (len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n"
                or header[8:16] != b"\0\0\0\rIHDR"):
            raise ValueError("Invalid PNG IHDR")
        bits, color_type = header[24:26]
        supported = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8},
                     4: {8, 16}, 6: {8, 16}}
        if bits not in supported.get(color_type, set()):
            raise ValueError("Invalid PNG bit depth/color type")
    elif kind == "TIFF":
        tags = opened.tag_v2
        samples = tags.get(258, (1,))
        formats = tags.get(339, (1,))
        samples = (samples,) if isinstance(samples, int) else samples
        formats = (formats,) if isinstance(formats, int) else formats
        if not samples or any(type(b) is not int or b <= 0 for b in samples):
            raise ValueError("Invalid TIFF BitsPerSample")
        if not formats or any(f != 1 for f in formats):
            raise ValueError("Only unsigned-integer TIFF samples are supported")
        bits = max(samples)
    elif kind == "JPEG":
        bits = jpeg_precision(path)
    elif kind == "WEBP":
        # Pillow/libwebp decodes the supported VP8/VP8L formats as 8-bit RGB(A).
        if opened.mode not in {"RGB", "RGBA"}:
            raise ValueError("Unsupported WebP pixel mode")
        bits = 8
    else:
        raise ValueError(f"Unsupported image format: {kind}; use PNG/JPEG/TIFF/WebP")
    if bits > 8 or opened.mode in {"I", "F"} or "16" in opened.mode:
        raise ValueError(f"High-bit-depth {kind} input ({bits} bits) is not supported; refusing conversion")
    if bits <= 0:
        raise ValueError("Invalid original sample precision")
    return kind, bits
