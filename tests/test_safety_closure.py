import contextlib
from dataclasses import asdict, replace
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np
from PIL import Image

from batch_color.batch import run_batch
from batch_color.cli import main
from batch_color.encoding import jpeg_precision
from batch_color.image_io import ImageInfo, load_mask, load_srgb
from batch_color.profile import ColorProfile, analyse_background_surface, create_profile
from batch_color.safety import atomic_json, file_hash
from batch_color.transaction import ArtifactTransaction
from batch_color.transfer import apply_profile, select_profile_path
from batch_color.profile import _surface_features, surface_values_valid
from batch_color.surface import choose_surface


def png16(path, color_type=2, alpha=65535):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    color = (0, 65535, 32769) + ((alpha,) if color_type == 6 else ())
    row = np.tile(np.array(color, dtype=">u2"), (16, 1)).tobytes()
    header = struct.pack(">IIBBBBB", 16, 16, 16, color_type, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress((b"\0" + row) * 16)) + chunk(b"IEND", b""))


class SafetyClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = Image.new("RGB", (80, 100), (160, 170, 180))
        self.mask = Image.new("L", self.image.size, 255)
        reference = Image.new("RGB", self.image.size, (190, 178, 160))
        self.profile = create_profile(reference, ImageInfo("reference.png", 80, 100, "sRGB", True),
                                      name="test", background_mask=self.mask)
        self.source = self.root / "source.png"
        self.image.save(self.source)
        self.mask_path = self.root / "mask.png"
        self.mask.save(self.mask_path)
        self.profile_path = self.root / "profile.json"
        self.profile.to_json(self.profile_path)
        self.output = self.root / "candidate.png"

    def call(self, *extra):
        args = ["apply", "--input", str(self.source), "--profile", str(self.profile_path),
                "--output", str(self.output), "--background-mask", str(self.mask_path), *map(str, extra)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(args)

    def test_rgb16_rejected_before_pixel_decode(self):
        png16(self.source)
        before = file_hash(self.source)
        with patch("PIL.PngImagePlugin.PngImageFile.load", side_effect=AssertionError("must not decode")):
            with self.assertRaisesRegex(ValueError, "High-bit-depth"):
                load_srgb(self.source)
        self.assertEqual(file_hash(self.source), before)

    def test_rgba16_all_alpha_cases_rejected_by_bit_depth(self):
        # Near-opaque 65534 also rounds to 255 in Pillow, so alpha alone is insufficient.
        for alpha in (65535, 65534, 65280, 32768, 0):
            with self.subTest(alpha=alpha):
                png16(self.source, 6, alpha)
                with self.assertRaisesRegex(ValueError, "High-bit-depth"):
                    load_srgb(self.source)

    def test_high_bit_depth_external_mask_is_rejected(self):
        png16(self.mask_path)
        with self.assertRaisesRegex(ValueError, "High-bit-depth"):
            load_mask(self.mask_path, (16, 16))

    def test_tiff_float_and_uint16_are_rejected(self):
        for dtype in (np.uint16, np.float32):
            path = self.root / f"{dtype.__name__}.tiff"
            Image.fromarray(np.full((16, 16), 100, dtype=dtype)).save(path)
            with self.assertRaises(ValueError):
                load_srgb(path)

    def test_supported_8bit_formats_record_original_precision(self):
        for extension in ("png", "jpg", "tiff", "webp"):
            path = self.root / ("valid." + extension)
            self.image.save(path)
            _, info = load_srgb(path)
            self.assertEqual(info.original_bit_depth, 8)
            self.assertNotEqual(info.original_format, "unknown")

    def test_unknown_formats_fail_closed(self):
        path = self.root / "image.gif"
        self.image.save(path)
        with self.assertRaisesRegex(ValueError, "Unsupported image format"):
            load_srgb(path)

    def test_jpeg_sof_precision_and_malformed_header(self):
        for precision in (8, 12, 16):
            path = self.root / "header.jpg"
            path.write_bytes(b"\xff\xd8\xff\xe0\x00\x04xx\xff\xc0\x00\x08"
                             + bytes([precision, 0, 16, 0, 16, 3]))
            self.assertEqual(jpeg_precision(path), precision)
        path.write_bytes(b"\xff\xd8\xff\xda")
        with self.assertRaises(ValueError):
            jpeg_precision(path)

    def test_profile_schema_errors_are_value_errors(self):
        for payload in ({}, [], None, {"background": []}, {**asdict(self.profile), "unexpected": 1}):
            self.profile_path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                ColorProfile.from_json(self.profile_path)

    def test_profile_numeric_semantics_reject_corruption(self):
        fields = (("a_median", 1e300), ("b_median", float("nan")), ("a_mad", -1),
                  ("b_mad", "0.1"), ("sample_count", 0), ("sample_count", True),
                  ("retained_fraction", 0), ("retained_fraction", 1.1))
        for field, value in fields:
            with self.subTest(field=field, value=value):
                payload = asdict(self.profile)
                payload["background"][field] = value
                self.profile_path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    ColorProfile.from_json(self.profile_path)

    def test_direct_api_profile_cannot_bypass_validation(self):
        profile = replace(self.profile, background=replace(self.profile.background, a_median=1e300))
        with self.assertRaises(ValueError):
            apply_profile(self.image, profile, background_mask=self.mask)

    def test_bad_surface_and_nonfinite_metadata_rejected(self):
        for change in ("residual", "coefficient", "metadata"):
            payload = asdict(self.profile)
            if change == "residual":
                payload["background_surface"]["residual"] = -1
            elif change == "coefficient":
                payload["background_surface"]["coefficients"][0][3] = 2
            else:
                payload["reference_info"]["bad"] = float("inf")
            self.profile_path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                ColorProfile.from_json(self.profile_path)

    def test_nan_rgb_is_rejected_before_uint8(self):
        with patch("batch_color.transfer.oklab_to_srgb", side_effect=lambda lab: np.full_like(lab, np.nan)):
            with self.assertRaisesRegex(ValueError, "non-finite RGB"):
                apply_profile(self.image, self.profile, background_mask=self.mask)

    def test_overflow_profile_creates_error_not_black_candidate(self):
        payload = asdict(self.profile)
        payload["background"]["a_median"] = 1e300
        self.profile_path.write_text(json.dumps(payload))
        self.assertEqual(self.call(), 2)
        self.assertFalse(self.output.exists())
        self.assertFalse(Path(str(self.output) + ".report.json").exists())
        reports = list((self.root / ".batch-color-errors").glob("*.json"))
        self.assertEqual(len(reports), 1)
        self.assertEqual(json.loads(reports[0].read_text())["status"], "error")

    def test_empty_profile_cli_has_no_traceback(self):
        self.profile_path.write_text("{}")
        self.assertEqual(self.call(), 2)

    def test_explicit_vision_timeout_is_a_controlled_cli_error(self):
        with patch("batch_color.masking.find_vision_helper", return_value=self.source), \
             patch("batch_color.masking.vision_person_mask", side_effect=subprocess.TimeoutExpired("fake", 90)):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
                code = main(["apply", "--input", str(self.source), "--profile", str(self.profile_path),
                             "--output", str(self.output), "--mask-backend", "vision"])
        self.assertEqual(code, 2)
        self.assertNotIn("Traceback", errors.getvalue())

    def test_narrow_reference_falls_back_without_artificial_gradient(self):
        pixels = np.full((300, 300, 3), 160, np.uint8)
        pixels[:, 150:154] = 161
        mask = np.zeros((300, 300), np.uint8)
        mask[:, 146:154] = 255
        profile = create_profile(Image.fromarray(pixels), ImageInfo("ref.png", 300, 300, "sRGB", True),
                                 name="narrow", background_mask=Image.fromarray(mask))
        self.assertFalse(profile.background_surface.trusted)
        self.assertEqual(profile.background_surface.model, "constant")
        source = Image.new("RGB", (300, 300), (160, 160, 160))
        for path in ("surface", "auto"):
            output, report, _ = select_profile_path(source, profile, path=path,
                                                    background_mask=Image.new("L", source.size, 255))
            self.assertEqual(report.path, "global-monotone")
            self.assertFalse(report.surface_enabled)
            self.assertLessEqual(int(np.ptp(np.asarray(output).astype(int))), 1)
            self.assertLessEqual(abs(int(np.asarray(output)[0, 0, 0]) - 160), 2)

    def test_source_support_is_checked_too(self):
        mask = np.zeros((100, 80), np.uint8)
        mask[:, 36:44] = 255
        _, report, _ = select_profile_path(self.image, self.profile, path="surface", background_mask=Image.fromarray(mask))
        self.assertFalse(report.surface_enabled)

    def test_flat_reference_does_not_select_quadratic_model(self):
        self.assertEqual(self.profile.background_surface.model, "constant")
        self.assertTrue(self.profile.background_surface.trusted)

    def test_legacy_profiles_do_not_claim_spatial_trust(self):
        payload = asdict(self.profile)
        payload["version"] = 3
        for key in ("model", "trusted", "diagnostics"):
            payload["background_surface"].pop(key)
        self.profile_path.write_text(json.dumps(payload))
        profile = ColorProfile.from_json(self.profile_path)
        self.assertFalse(profile.background_surface.trusted)
        _, report, _ = apply_profile(self.image, profile, background_mask=self.mask)
        self.assertFalse(report.surface_enabled)

    def test_auto_retries_after_transient_native_failure(self):
        inputs = self.root / "input"
        inputs.mkdir()
        self.image.save(inputs / "one.png")
        native = [RuntimeError("temporary failure"), Image.new("L", self.image.size, 0)]
        with patch("batch_color.masking.find_vision_helper", return_value=self.source), \
             patch("batch_color.masking.vision_person_mask", side_effect=native) as mock:
            kwargs = dict(input_directory=inputs, output_directory=self.root / "batch", profile_path=self.profile_path,
                          mask_backend="auto", save_previews=False)
            first, second, third = run_batch(**kwargs), run_batch(**kwargs), run_batch(**kwargs)
        self.assertEqual(mock.call_count, 2)
        self.assertEqual(first.items[0].mask_backend, "heuristic-color")
        self.assertEqual(second.items[0].mask_backend, "vision-accurate")
        self.assertEqual(second.skipped, 0)
        self.assertEqual(third.skipped, 1)
        self.assertTrue(all(run.review == 1 for run in (first, second, third)))

    def test_missing_native_does_not_force_endless_recompute(self):
        inputs = self.root / "input"
        inputs.mkdir()
        self.image.save(inputs / "one.png")
        with patch("batch_color.masking.find_vision_helper", return_value=None):
            kwargs = dict(input_directory=inputs, output_directory=self.root / "batch", profile_path=self.profile_path,
                          mask_backend="auto", save_previews=False)
            run_batch(**kwargs)
            self.assertEqual(run_batch(**kwargs).skipped, 1)

    def test_report_serialization_failure_publishes_no_artifacts(self):
        original = atomic_json
        def fail_report(path, payload):
            if Path(path).name == "report.json":
                raise ValueError("injected final report failure")
            return original(path, payload)
        with patch("batch_color.cli.atomic_json", side_effect=fail_report):
            self.assertEqual(self.call("--mask-output", self.root / "mask-out.png"), 2)
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / "mask-out.png").exists())
        self.assertFalse(Path(str(self.output) + ".report.json").exists())

    def test_failed_replacement_preserves_previous_complete_output(self):
        self.assertEqual(self.call(), 0)
        report = Path(str(self.output) + ".report.json")
        before = {p: file_hash(p) for p in (self.output, report)}
        with patch("batch_color.cli.save_comparison", side_effect=OSError("preview failed")):
            self.assertEqual(self.call("--overwrite", "--preview", self.root / "preview.jpg"), 2)
        self.assertEqual({p: file_hash(p) for p in before}, before)

    def test_batch_mask_write_failure_leaves_no_new_candidate(self):
        inputs = self.root / "input"
        inputs.mkdir()
        self.image.save(inputs / "one.png")
        with patch("batch_color.batch.save_mask", side_effect=OSError("injected failure")):
            summary = run_batch(input_directory=inputs, output_directory=self.root / "batch",
                                profile_path=self.profile_path, mask_backend="heuristic", save_previews=False)
        self.assertEqual(summary.errors, 1)
        self.assertFalse((self.root / "batch/candidates/one.png.png").exists())
        self.assertEqual(len(list((self.root / "batch/errors").glob("*.json"))), 1)


class TransactionTests(unittest.TestCase):
    def test_publish_failure_rolls_back_new_and_replaced_artifacts(self):
        for existing in (False, True):
            for failed_role in ("candidate", "mask", "report"):
                with self.subTest(existing=existing, failed_role=failed_role), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    outputs = {role: root / (role + ".json") for role in ("candidate", "mask", "report")}
                    if existing:
                        for path in outputs.values():
                            path.write_text("old " + path.name)
                    before = {r: p.read_bytes() if p.exists() else None for r, p in outputs.items()}
                    actual_replace = os.replace
                    def inject(src, dst):
                        if Path(dst) == outputs[failed_role] and Path(src).name == failed_role + ".json":
                            raise OSError("injected publish failure")
                        return actual_replace(src, dst)
                    with self.assertRaises(OSError), ArtifactTransaction(outputs) as job:
                        job.staged["candidate"].write_text("candidate")
                        job.staged["mask"].write_text("mask")
                        atomic_json(job.staged["report"], {"status": "review", "accepted": False,
                                                          "artifacts": job.artifact_records()})
                        with patch("batch_color.transaction.os.replace", side_effect=inject):
                            job.commit()
                    self.assertEqual({r: p.read_bytes() if p.exists() else None for r, p in outputs.items()}, before)
                    self.assertEqual(list(root.glob(".batch-color-run-*")), [])
                    self.assertEqual(list(root.glob("*.lock")), [])

    def test_bad_commit_manifest_cannot_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outputs = {"candidate": root / "candidate.png", "report": root / "report.json"}
            with self.assertRaises(ValueError), ArtifactTransaction(outputs) as job:
                job.staged["candidate"].write_bytes(b"test")
                atomic_json(job.staged["report"], {"status": "review", "accepted": False, "artifacts": {}})
                job.commit()
            self.assertFalse(outputs["candidate"].exists())


class SurfaceSelectionTests(unittest.TestCase):
    def fit(self, quadratic=False, noisy=False):
        y, x = np.mgrid[-1:1:80j, -1:1:80j]
        x, y = x.ravel(), y.ravel()
        features = _surface_features(x, y)
        lab = np.zeros((len(x), 3))
        lab[:, 0] = 0.6 + 0.03 * x + 0.02 * y
        if quadratic:
            lab[:, 0] += 0.08 * x ** 2 + 0.06 * x * y
        if noisy:
            lab[:, 0] += np.random.default_rng(7).uniform(-0.2, 0.2, len(x))
        return choose_surface(features, lab, x, y, surface_values_valid)

    def test_plane_selected_when_quadratic_has_no_holdout_gain(self):
        _, _, model, trusted, diagnostics = self.fit()
        self.assertTrue(trusted)
        self.assertEqual(model, "plane")
        self.assertLess(diagnostics["blocked_validation_rmse"]["plane"], 1e-6)

    def test_quadratic_requires_actual_blocked_holdout_gain(self):
        _, _, model, trusted, diagnostics = self.fit(quadratic=True)
        self.assertTrue(trusted)
        self.assertEqual(model, "quadratic")
        self.assertLess(diagnostics["blocked_validation_rmse"]["quadratic"],
                        diagnostics["blocked_validation_rmse"]["plane"] * 0.9)

    def test_large_heldout_error_disables_surface(self):
        _, _, model, trusted, diagnostics = self.fit(noisy=True)
        self.assertFalse(trusted)
        self.assertEqual(model, "constant")
        self.assertEqual(diagnostics["reason"], "holdout_error_too_large")


if __name__ == "__main__":
    unittest.main()
