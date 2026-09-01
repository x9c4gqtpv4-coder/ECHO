"""Compare manually confirmed regions without claiming automatic skin diagnosis.

Usage: PYTHONPATH=src python scripts/audit_skin_transition.py --source in.png
  --corrected out.png --output audit --roi forehead:350,80,690,335
The output is diagnostic evidence, never an automatic quality approval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from batch_color.image_io import load_srgb, save_srgb


def compare_region(source: Image.Image, corrected: Image.Image, box: tuple[int, int, int, int]):
    a = np.asarray(source.crop(box), dtype=np.float32)
    b = np.asarray(corrected.crop(box), dtype=np.float32)
    delta = b - a
    gradient_a = np.concatenate([np.diff(a, axis=0).ravel(), np.diff(a, axis=1).ravel()])
    gradient_b = np.concatenate([np.diff(b, axis=0).ravel(), np.diff(b, axis=1).ravel()])
    gradient_delta = gradient_b - gradient_a
    denominator = np.linalg.norm(gradient_a) * np.linalg.norm(gradient_b)
    similarity = float(np.dot(gradient_a, gradient_b) / denominator) if denominator else None
    return {
        "box_xyxy": list(box),
        "rgb_mean_absolute_change_8bit": float(np.mean(np.abs(delta))),
        "rgb_p99_absolute_change_8bit": float(np.quantile(np.abs(delta), .99)),
        "rgb_max_absolute_change_8bit": float(np.max(np.abs(delta))),
        "mean_rgb_signed_change_8bit": np.mean(delta, axis=(0, 1)).tolist(),
        "change_field_gradient_p99_8bit_per_pixel": float(np.quantile(np.abs(gradient_delta), .99)),
        "rgb_gradient_cosine_similarity": similarity,
        "new_zero_channel_pixel_fraction": float(np.mean(np.any((b == 0) & (a > 0), axis=-1))),
        "new_255_channel_pixel_fraction": float(np.mean(np.any((b == 255) & (a < 255), axis=-1))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi", action="append", required=True, help="name:x1,y1,x2,y2 in source pixel coordinates")
    args = parser.parse_args()
    source, source_info = load_srgb(args.source)
    corrected, corrected_info = load_srgb(args.corrected)
    if source.size != corrected.size:
        parser.error("Input sizes must match; this tool does not register or resize images")
    parsed = []
    for value in args.roi:
        name, coords = value.split(":", 1)
        box = tuple(map(int, coords.split(",")))
        if not name or not all(c.isalnum() or c in "_-" for c in name):
            parser.error("Region name must use letters, numbers, underscores or dashes")
        if len(box) != 4 or not (0 <= box[0] < box[2] <= source.width and 0 <= box[1] < box[3] <= source.height):
            parser.error(f"Invalid region: {value}")
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            parser.error("Region must be at least 2x2 pixels")
        parsed.append((name, box))
    args.output.mkdir(parents=True, exist_ok=False)
    payload = {
        "source_filename": args.source.name,
        "corrected_filename": args.corrected.name,
        "size": list(source.size),
        "source_profile": source_info.source_profile,
        "corrected_profile": corrected_info.source_profile,
        "regions": {},
        "status": "diagnostic_only_not_approved",
        "limitations": [
            "Regions are manually specified, not automatically segmented skin.",
            "A small pixel difference or high gradient similarity does not prove absence of banding.",
            "This checks full-resolution files, not the rendering or resampling in a screenshot.",
        ],
    }
    for name, box in parsed:
        a, b = source.crop(box), corrected.crop(box)
        payload["regions"][name] = compare_region(source, corrected, box)
        delta = np.asarray(b, dtype=np.int16) - np.asarray(a, dtype=np.int16)
        amplified = Image.fromarray(np.clip(128 + delta * 12, 0, 255).astype(np.uint8))
        panels = [a, b, amplified]
        labels = ["SOURCE 1:1", "CORRECTED 1:1", "DIFFERENCE x12 (not photo)"]
        canvas = Image.new("RGB", (a.width * 3, a.height + 44), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (panel, label) in enumerate(zip(panels, labels)):
            x = a.width * index
            canvas.paste(panel, (x, 44))
            draw.text((x + 6, 12), label, fill="black", font=ImageFont.load_default(size=16))
        save_srgb(canvas, args.output / f"{name}_comparison.png")
    (args.output / "skin_transition_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
