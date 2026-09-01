import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

from batch_color.image_io import load_mask, load_srgb, save_srgb
from batch_color.masking import find_vision_helper, vision_person_mask


class CanonicalImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def oriented_file(self, orientation):
        pixels = np.arange(40 * 20 * 3, dtype=np.uint16).reshape(20, 40, 3).astype(np.uint8)
        image = Image.fromarray(pixels)
        exif = Image.Exif()
        exif[274] = orientation
        path = self.root / f"exif-{orientation}.jpg"
        image.save(path, exif=exif)
        return path

    def test_all_exif_orientations_are_applied_once(self):
        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                path = self.oriented_file(orientation)
                decoded, info = load_srgb(path)
                with Image.open(path) as opened:
                    expected = ImageOps.exif_transpose(opened).convert("RGB")
                np.testing.assert_array_equal(decoded, expected)
                self.assertEqual(info.original_orientation, orientation)
                self.assertNotIn(274, decoded.getexif())
                output = self.root / f"output-{orientation}.png"
                verification = save_srgb(decoded, output)
                np.testing.assert_array_equal(load_srgb(output)[0], decoded)
                self.assertTrue(verification["reopened"])

    def test_vision_receives_canonical_pixels_not_the_original_file(self):
        path = self.oriented_file(6)
        canonical, _ = load_srgb(path)

        def native(arguments, **kwargs):
            self.assertNotEqual(Path(arguments[1]), path)
            with Image.open(arguments[1]) as received:
                self.assertEqual(received.size, (20, 40))
                np.testing.assert_array_equal(received, canonical)
                self.assertNotIn(274, received.getexif())
            Image.new("L", canonical.size, 255).save(arguments[2])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("batch_color.masking.subprocess.run", side_effect=native):
            mask = vision_person_mask(path, executable=Path("native-test"), canonical_image=canonical)
        self.assertEqual(mask.size, canonical.size)

    def test_native_size_mismatch_is_not_silently_resized(self):
        path = self.oriented_file(6)

        def native(arguments, **kwargs):
            Image.new("L", (40, 20), 255).save(arguments[2])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("batch_color.masking.subprocess.run", side_effect=native):
            with self.assertRaises(RuntimeError):
                vision_person_mask(path, executable=Path("native-test"))

    @unittest.skipUnless(find_vision_helper(), "macOS native helper not built")
    def test_real_native_vision_exif_geometry(self):
        path = self.oriented_file(6)
        canonical, _ = load_srgb(path)
        mask = vision_person_mask(path, canonical_image=canonical)
        self.assertEqual(mask.size, (20, 40))

    def test_transparency_is_not_silently_dropped(self):
        path = self.root / "alpha.png"
        Image.new("RGBA", (12, 12), (80, 100, 120, 128)).save(path)
        with self.assertRaises(ValueError):
            load_srgb(path)

    def test_near_opaque_alpha_requires_and_records_explicit_policy(self):
        pixels = np.full((40, 40, 4), (80, 100, 120, 255), dtype=np.uint8)
        pixels[0, 0, 3] = 253
        path = self.root / "near-opaque.png"
        Image.fromarray(pixels, mode="RGBA").save(path)
        with self.assertRaises(ValueError):
            load_srgb(path)
        image, info = load_srgb(path, alpha_policy="drop_near_opaque")
        self.assertEqual(image.mode, "RGB")
        np.testing.assert_array_equal(np.asarray(image)[0, 0], pixels[0, 0, :3])
        self.assertIn("near_opaque_alpha_dropped_preserving_rgb", info.warnings)

    def test_explicit_near_opaque_policy_rejects_material_transparency(self):
        path = self.root / "material-alpha.png"
        Image.new("RGBA", (20, 20), (80, 100, 120, 252)).save(path)
        with self.assertRaises(ValueError):
            load_srgb(path, alpha_policy="drop_near_opaque")

    def test_high_bit_depth_is_not_silently_truncated(self):
        path = self.root / "depth.png"
        Image.fromarray(np.full((12, 12), 1024, dtype=np.uint16)).save(path)
        with self.assertRaises(ValueError):
            load_srgb(path)

    def test_invalid_icc_is_not_treated_as_srgb(self):
        path = self.root / "icc.png"
        Image.new("RGB", (12, 12)).save(path, icc_profile=b"invalid icc data")
        with self.assertRaises(ValueError):
            load_srgb(path)

    def test_external_mask_geometry_must_match(self):
        path = self.root / "mask.png"
        Image.new("L", (12, 24), 255).save(path)
        with self.assertRaises(ValueError):
            load_mask(path, (24, 12))


if __name__ == "__main__":
    unittest.main()
