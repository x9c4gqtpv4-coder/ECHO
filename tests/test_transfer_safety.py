import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from batch_color.image_io import ImageInfo, save_srgb
from batch_color.profile import analyse_background, analyse_background_surface, create_profile
from batch_color.transfer import _bounded_luminance_curve, _subject_luminance, apply_profile, select_profile_path


def studio():
    pixels = np.full((180, 140, 3), (160, 170, 180), np.uint8)
    pixels[45:155, 50:80] = (40, 105, 205)
    mask = np.full((180, 140), 255, np.uint8)
    mask[45:155, 50:80] = 0
    return Image.fromarray(pixels), Image.fromarray(mask)


def target(rgb):
    image = Image.new("RGB", (140, 180), rgb)
    return create_profile(image, ImageInfo("ref.png", 140, 180, "sRGB", True), name="test",
                          background_mask=Image.new("L", image.size, 255), mask_backend="unit-test")


class TransferSafetyTests(unittest.TestCase):
    def test_subject_curve_preserves_black_white_and_has_bounded_positive_derivative(self):
        grid = np.linspace(0, 1, 4097, dtype=np.float32)
        for source, reference in ((0.8, 0.2), (0.2, 0.8), (0.65, 0.68), (0.5, 0.5)):
            mapped = _subject_luminance(grid, source, reference)
            slopes = np.diff(mapped) / np.diff(grid)
            self.assertEqual(mapped[0], 0)
            self.assertEqual(mapped[-1], 1)
            self.assertGreaterEqual(float(slopes.min()), 0.499)
            self.assertLessEqual(float(slopes.max()), 2.001)

    def test_both_mode_does_not_turn_black_into_a_gray_veil(self):
        source, mask = studio()
        pixels = np.array(source)
        pixels[45:100, 50:80] = 0
        pixels[100:155, 50:80] = 255
        output, _, _ = select_profile_path(Image.fromarray(pixels), target((210, 200, 180)),
                                           mode="both", path="global", background_mask=mask)
        self.assertEqual(output.getpixel((60, 70)), (0, 0, 0))
        self.assertEqual(output.getpixel((60, 120)), (255, 255, 255))

    def test_default_background_mode_preserves_nonbackground_exactly(self):
        source, mask = studio()
        for path in ("global", "surface", "auto"):
            with self.subTest(path=path):
                output, report, _ = select_profile_path(source, target((190, 178, 160)), path=path, background_mask=mask)
                core = np.asarray(mask) == 0
                np.testing.assert_array_equal(np.asarray(output)[core], np.asarray(source)[core])
                self.assertFalse(report.accepted)
                self.assertEqual(report.mode, "background")

    def test_explicit_protection_preserves_product_in_both_mode_and_lossless_file(self):
        source, mask = studio()
        protect = Image.fromarray(255 - np.asarray(mask))
        output, report, _ = select_profile_path(source, target((190, 178, 160)), path="global",
                                               mode="both", background_mask=mask, protected_mask=protect)
        core = np.asarray(protect) == 255
        np.testing.assert_array_equal(np.asarray(output)[core], np.asarray(source)[core])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.png"
            save_srgb(output, path)
            with Image.open(path) as saved:
                np.testing.assert_array_equal(np.asarray(saved)[core], np.asarray(source)[core])
        self.assertFalse(report.accepted)

    def test_both_without_protection_is_explicitly_not_semantic_matching(self):
        source, mask = studio()
        _, report, _ = select_profile_path(source, target((190, 178, 160)), mode="both", background_mask=mask)
        self.assertIn("person_adjustment_is_background_driven_not_semantic_matching", report.review_reasons)
        self.assertFalse(report.accepted)

    def test_audited_gray_ramp_no_longer_reverses_after_all_mixing(self):
        pixels = np.full((180, 360, 3), 200, np.uint8)
        ramp = np.round(np.linspace(10, 240, 280)).astype(np.uint8)
        pixels[45:155, 40:320] = ramp[None, :, None]
        for weight in (0, 128, 255):
            mask = np.full((180, 360), 255, np.uint8)
            mask[45:155, 40:320] = weight
            for path in ("global", "surface"):
                for strength in (0.1, 0.85, 1.0):
                    with self.subTest(weight=weight, path=path, strength=strength):
                        output, report, _ = select_profile_path(Image.fromarray(pixels), target((30, 30, 30)),
                            path=path, strength=strength, mode="both", background_mask=Image.fromarray(mask))
                        mapped = np.asarray(output)[90, 40:320].astype(int)
                        self.assertTrue(np.all(np.diff(mapped, axis=0) >= 0))
                        self.assertFalse(report.accepted)

    def test_curve_is_bounded_and_positive_slope(self):
        rng = np.random.default_rng(9)
        for _ in range(30):
            source, reference = np.sort(rng.uniform(0, 1, (2, 7)), axis=1)
            x, y = _bounded_luminance_curve(source.tolist(), reference.tolist())
            slopes = np.diff(y) / np.diff(x)
            self.assertGreater(float(slopes.min()), 0)
            self.assertLessEqual(float(slopes.max()), 2.50001)
            self.assertGreaterEqual(float(y.min()), -1e-9)
            self.assertLessEqual(float(y.max()), 1.0 + 1e-9)

    def test_source_at_target_is_a_valid_no_op_not_forced_to_change(self):
        image = Image.new("RGB", (140, 180), (160, 160, 160))
        for path in ("global", "surface", "auto"):
            output, report, _ = select_profile_path(image, target((160, 160, 160)), path=path,
                                                   background_mask=Image.new("L", image.size, 255))
            np.testing.assert_array_equal(output, image)
            self.assertTrue(report.no_op)
            self.assertTrue(report.baseline_checks_passed)
            self.assertFalse(report.accepted)  # no-op is not an automated production certificate

    def test_tiny_improvement_is_not_production_approval(self):
        image = Image.new("RGB", (140, 180), (100, 100, 100))
        _, report, _ = select_profile_path(image, target((190, 190, 190)), path="global", strength=0.01,
                                          background_mask=Image.new("L", image.size, 255))
        self.assertLess(report.background_improvement_percent, 5)
        self.assertFalse(report.accepted)
        self.assertEqual(report.status, "review")

    def test_zero_strength_is_exact_identity(self):
        source, mask = studio()
        output, report, _ = select_profile_path(source, target((190, 178, 160)), strength=0,
                                               background_mask=mask, mode="both")
        np.testing.assert_array_equal(source, output)
        self.assertTrue(report.no_op)

    def test_invalid_strength_and_geometry_are_rejected(self):
        source, mask = studio()
        for strength in (-1, 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                apply_profile(source, target((190, 178, 160)), strength=strength)
        with self.assertRaises(ValueError):
            apply_profile(source, target((190, 178, 160)), background_mask=Image.new("L", (10, 10), 255))

    def test_tile_boundaries_do_not_change_pixels(self):
        source, mask = studio()
        a = apply_profile(source, target((190, 178, 160)), tile_rows=11, background_mask=mask, mode="both")[0]
        b = apply_profile(source, target((190, 178, 160)), tile_rows=180, background_mask=mask, mode="both")[0]
        np.testing.assert_array_equal(a, b)

    def test_mask_excludes_a_subject_touching_image_border_from_statistics(self):
        pixels = np.full((180, 140, 3), (160, 170, 180), np.uint8)
        mask = np.full((180, 140), 255, np.uint8)
        mask[:80, :100] = 0
        a = Image.fromarray(pixels)
        pixels[:80, :100] = (230, 40, 20)
        b = Image.fromarray(pixels)
        mask = Image.fromarray(mask)
        self.assertEqual(analyse_background(a, mask), analyse_background(b, mask))
        self.assertEqual(analyse_background_surface(a, mask), analyse_background_surface(b, mask))


if __name__ == "__main__":
    unittest.main()
