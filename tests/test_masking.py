import unittest

from PIL import Image

from batch_color.image_io import ImageInfo
from batch_color.masking import resolve_background_mask
from batch_color.profile import create_profile
from batch_color.transfer import select_profile_path


class MaskingTests(unittest.TestCase):
    def test_heuristic_backend_defers_mask_creation(self) -> None:
        result = resolve_background_mask(
            "unused.png",
            (80, 100),
            backend="heuristic",
        )
        self.assertIsNone(result.background_mask)
        self.assertEqual(result.backend, "heuristic-color")

    def test_external_mask_backend_is_recorded(self) -> None:
        source = Image.new("RGB", (80, 100), (170, 175, 180))
        reference = Image.new("RGB", (80, 100), (190, 178, 160))
        info = ImageInfo("reference.png", 80, 100, "sRGB", True)
        profile = create_profile(reference, info, name="test")
        background_mask = Image.new("L", source.size, 255)
        _, report, _ = select_profile_path(
            source,
            profile,
            background_mask=background_mask,
            mask_backend="unit-test",
        )
        self.assertEqual(report.mask_backend, "unit-test")


if __name__ == "__main__":
    unittest.main()

