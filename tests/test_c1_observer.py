import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from batch_color.baseline import A0_EXPECTED_CODE_FINGERPRINT
from batch_color.c1 import C1_CONFIG, analyse_relative_illumination
from batch_color.cli import main
from batch_color.color import linear_to_srgb
from batch_color.runtime import a0_code_fingerprint
from batch_color.safety import file_hash


def _gray_gradient(*, gain: float = 1.0, gamma: float = 1.0) -> Image.Image:
    axis = np.linspace(0.08, 0.42, 96, dtype=np.float32)
    luminance = np.tile(axis[None, :], (96, 1))
    luminance = np.clip(np.power(luminance, gamma) * gain, 0.0, 1.0)
    linear = np.repeat(luminance[..., None], 3, axis=-1)
    encoded = np.round(np.clip(linear_to_srgb(linear), 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(encoded, mode="RGB")


class C1ObserverTests(unittest.TestCase):
    def test_known_half_stop_is_recovered_without_pixel_authority(self):
        source = _gray_gradient()
        target = _gray_gradient(gain=2.0**0.5)
        neutral = Image.new("L", source.size, 255)
        report = analyse_relative_illumination(
            source,
            target,
            source_region_mask=neutral,
            reference_region_mask=neutral,
            source_neutral_mask=neutral,
            reference_neutral_mask=neutral,
            neutral_evidence="same_entity",
            comparison_evidence="same_surface",
        )
        self.assertEqual(report["status"], "review")
        self.assertFalse(report["accepted"])
        self.assertFalse(report["pixel_output_changed"])
        self.assertAlmostEqual(
            report["exposure"]["relative_exposure_like_stops"], 0.5, delta=0.035
        )
        self.assertEqual(report["exposure"]["status"], "valid")
        self.assertTrue(report["applicability"]["c1_exposure_future_candidate"])
        self.assertEqual(
            report["applicability"]["a0"],
            "not_evaluated_observer_does_not_change_or_veto_a0",
        )

    def test_explicit_neutral_detects_warmer_target(self):
        source_linear = np.full((96, 96, 3), 0.25, dtype=np.float32)
        target_linear = source_linear * np.asarray([1.12, 1.0, 0.82], dtype=np.float32)
        source = Image.fromarray(
            np.round(linear_to_srgb(source_linear) * 255.0).astype(np.uint8), mode="RGB"
        )
        target = Image.fromarray(
            np.round(np.clip(linear_to_srgb(target_linear), 0.0, 1.0) * 255.0).astype(np.uint8),
            mode="RGB",
        )
        neutral = Image.new("L", source.size, 255)
        report = analyse_relative_illumination(
            source,
            target,
            source_neutral_mask=neutral,
            reference_neutral_mask=neutral,
            neutral_evidence="human_confirmed",
        )
        self.assertEqual(report["whitepoint"]["warm_cool_direction"], "target_warmer")
        self.assertEqual(report["whitepoint"]["status"], "valid_explicit_neutral")
        self.assertIsNone(report["whitepoint"]["source"]["apparent_cct_kelvin"])
        self.assertEqual(
            report["whitepoint"]["source"]["cct_reason"],
            "disabled_until_validated_planckian_locus_and_duv",
        )
        self.assertTrue(report["whitepoint"]["eligible_for_future_transform"])

    def test_coloured_surface_cannot_masquerade_as_explicit_neutral(self):
        source = Image.new("RGB", (96, 96), (190, 110, 90))
        target = Image.new("RGB", (96, 96), (180, 105, 85))
        neutral = Image.new("L", source.size, 255)
        report = analyse_relative_illumination(
            source,
            target,
            source_neutral_mask=neutral,
            reference_neutral_mask=neutral,
            neutral_evidence="human_confirmed",
        )
        self.assertEqual(report["whitepoint"]["status"], "explicit_surface_not_neutral")
        self.assertFalse(report["whitepoint"]["eligible_for_future_transform"])
        self.assertIsNone(report["whitepoint"]["source"]["apparent_cct_kelvin"])
        self.assertIn("EXPLICIT_SURFACE_NOT_NEUTRAL", report["review_reasons"])

    def test_automatic_neutral_is_hypothesis_and_does_not_emit_cct(self):
        report = analyse_relative_illumination(_gray_gradient(), _gray_gradient(gain=1.1))
        self.assertEqual(report["whitepoint"]["status"], "hypothesis_only")
        self.assertIsNone(report["whitepoint"]["source"]["apparent_cct_kelvin"])
        self.assertFalse(report["whitepoint"]["eligible_for_future_transform"])
        self.assertIn("AUTOMATIC_NEUTRAL_HYPOTHESIS_ONLY", report["review_reasons"])

    def test_nonparallel_tone_is_not_misreported_as_simple_exposure(self):
        report = analyse_relative_illumination(
            _gray_gradient(), _gray_gradient(gamma=0.72)
        )
        self.assertEqual(report["exposure"]["status"], "compound_or_composition_unstable")
        self.assertFalse(report["exposure"]["parallel_gain_supported"])
        self.assertIn("COMPOSITION_OR_TONE_UNSTABLE", report["review_reasons"])

    def test_explicit_evidence_requires_both_masks(self):
        with self.assertRaises(ValueError):
            analyse_relative_illumination(
                _gray_gradient(),
                _gray_gradient(),
                source_neutral_mask=Image.new("L", (96, 96), 255),
                neutral_evidence="same_entity",
            )

    def test_exposure_candidate_requires_confirmed_comparable_surface(self):
        report = analyse_relative_illumination(
            _gray_gradient(), _gray_gradient(gain=1.1)
        )
        self.assertFalse(report["applicability"]["c1_exposure_future_candidate"])
        self.assertIn("COMPARABLE_SURFACE_NOT_CONFIRMED", report["review_reasons"])

    def test_all_c1_thresholds_are_validated(self):
        invalid = (
            replace(C1_CONFIG, min_spatial_cell_pixels=0),
            replace(C1_CONFIG, min_linear_luminance=0.0),
            replace(C1_CONFIG, exposure_fit_review_stops=0.0),
            replace(C1_CONFIG, heavy_clipping_fraction=1.1),
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                analyse_relative_illumination(
                    _gray_gradient(), _gray_gradient(), config=config
                )

    def test_a0_fingerprint_is_unchanged(self):
        self.assertEqual(a0_code_fingerprint(), A0_EXPECTED_CODE_FINGERPRINT)


class C1CommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.png"
        self.reference = self.root / "reference.png"
        self.neutral = self.root / "neutral.png"
        _gray_gradient().save(self.source)
        _gray_gradient(gain=2.0**0.25).save(self.reference)
        Image.new("L", (96, 96), 255).save(self.neutral)

    def call(self, *extra: object) -> int:
        argv = [
            "c1-analyse",
            "--input",
            str(self.source),
            "--reference",
            str(self.reference),
            "--report",
            str(self.root / "report.json"),
            *map(str, extra),
        ]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(argv)

    def test_cli_publishes_only_a_hash_bound_read_only_report(self):
        before = {path: file_hash(path) for path in (self.source, self.reference, self.neutral)}
        code = self.call(
            "--source-neutral-mask",
            self.neutral,
            "--reference-neutral-mask",
            self.neutral,
            "--neutral-evidence",
            "same_entity",
        )
        self.assertEqual(code, 0)
        self.assertEqual(before, {path: file_hash(path) for path in before})
        report = json.loads((self.root / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "review")
        self.assertFalse(report["accepted"])
        self.assertFalse(report["pixel_output_changed"])
        self.assertEqual(report["artifacts"], {})
        self.assertEqual(report["analyzer_identity"]["pixel_authority"], "none")
        self.assertFalse(any(self.root.glob("*candidate*")))

    def test_report_cannot_alias_an_input(self):
        self.assertEqual(self.call("--report", self.source, "--overwrite"), 2)


if __name__ == "__main__":
    unittest.main()
