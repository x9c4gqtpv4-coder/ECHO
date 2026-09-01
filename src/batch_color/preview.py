from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from batch_color.image_io import save_srgb


def _panel(image: Image.Image, label: str, *, height: int) -> Image.Image:
    ratio = height / image.height
    resized = image.resize((max(1, round(image.width * ratio)), height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (resized.width, height + 54), "white")
    panel.paste(resized, (0, 54))
    draw = ImageDraw.Draw(panel)
    draw.text((16, 16), label, fill=(25, 25, 25), font=ImageFont.load_default(size=22))
    return panel


def save_comparison(
    source: Image.Image,
    corrected: Image.Image,
    reference: Image.Image | None,
    path: str | Path,
    *,
    height: int = 760,
) -> None:
    panels = [_panel(source, "SOURCE", height=height), _panel(corrected, "CANDIDATE / REVIEW", height=height)]
    if reference is not None:
        panels.append(_panel(reference, "REFERENCE", height=height))
    gap = 12
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels) + gap * (len(panels) - 1), height + 54),
        (235, 235, 235),
    )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    save_srgb(canvas, path)
