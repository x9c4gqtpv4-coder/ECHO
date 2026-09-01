import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from batch_color.image_io import ImageInfo
from batch_color.profile import ColorProfile, create_profile
from batch_color.transfer import apply_profile, select_profile_path


def _studio_image(background: tuple[int, int, int]) -> Image.Image:
    pixels = np.full((180, 140, 3), background, dtype=np.uint8)
    pixels[45:165, 48:95] = (148, 101, 76)
    return Image.fromarray(pixels, mode="RGB")


class ProfileTests(unittest.TestCase):
    def test_profile_json_round_trip(self) -> None:
        image = _studio_image((180, 176, 168))
        info = ImageInfo("reference.jpg", 140, 180, "sRGB", True)
        profile = create_profile(image, info, name="test")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile.to_json(path)
            restored = ColorProfile.from_json(path)
        self.assertEqual(restored, profile)

    def test_transfer_moves_background_toward_reference(self) -> None:
        source = _studio_image((165, 170, 178))
        reference = _studio_image((190, 178, 160))
        info = ImageInfo("reference.jpg", 140, 180, "sRGB", True)
        profile = create_profile(reference, info, name="warm")
        output, report, mask = apply_profile(source, profile, strength=0.85, tile_rows=64)
        self.assertEqual(output.size, source.size)
        self.assertEqual(mask.size, source.size)
        self.assertLess(report.background_distance_after, report.background_distance_before)
        self.assertTrue(report.baseline_checks_passed)
        self.assertFalse(report.accepted)
        self.assertEqual(report.status, "review")

    def test_auto_selector_returns_a_named_path(self) -> None:
        source = _studio_image((165, 170, 178))
        reference = _studio_image((190, 178, 160))
        info = ImageInfo("reference.jpg", 140, 180, "sRGB", True)
        profile = create_profile(reference, info, name="warm")
        _, report, _ = select_profile_path(source, profile, strength=0.85)
        self.assertIn(report.path, {"spatial-surface", "global-monotone"})


if __name__ == "__main__":
    unittest.main()
