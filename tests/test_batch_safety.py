import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from batch_color.batch import run_batch
from batch_color.cli import main
from batch_color.image_io import ImageInfo
from batch_color.profile import create_profile
from batch_color.safety import file_hash


class BatchSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs, self.outputs = self.root / "input", self.root / "output"
        self.inputs.mkdir()
        pixels = np.full((100, 80, 3), (160, 170, 180), np.uint8)
        pixels[30:80, 30:50] = (40, 105, 205)
        self.source = Image.fromarray(pixels)
        self.source.save(self.inputs / "one.png")
        self.profile = self.root / "profile.json"
        self.write_profile((190, 178, 160))

    def write_profile(self, rgb):
        image = Image.new("RGB", (80, 100), rgb)
        create_profile(image, ImageInfo("ref.png", 80, 100, "sRGB", True), name=str(rgb),
                       background_mask=Image.new("L", image.size, 255)).to_json(self.profile)

    def run_batch(self, **kwargs):
        return run_batch(input_directory=self.inputs, profile_path=self.profile, output_directory=self.outputs,
                         mask_backend="heuristic", save_previews=False, **kwargs)

    def test_cached_review_separates_execution_and_quality_exit(self):
        self.run_batch(strength=0)
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["batch", "--input", str(self.inputs), "--profile", str(self.profile),
                         "--output", str(self.outputs), "--mask-backend", "heuristic", "--strength", "0", "--no-previews"])
        summary = json.loads((self.outputs / "summary.json").read_text())
        self.assertEqual(code, 0)
        with contextlib.redirect_stdout(io.StringIO()):
            strict_code = main([
                "batch", "--input", str(self.inputs), "--profile", str(self.profile),
                "--output", str(self.outputs), "--mask-backend", "heuristic",
                "--strength", "0", "--no-previews", "--strict-quality-exit",
            ])
        self.assertEqual(strict_code, 3)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["review"], 1)
        self.assertEqual(summary["items"][0]["status"], "review")
        self.assertFalse(summary["items"][0]["accepted"])

    def test_profile_content_change_invalidates_cache(self):
        self.run_batch()
        report = self.outputs / "reports/one.png.json"
        old = json.loads(report.read_text())["cache_key"]
        self.write_profile((130, 150, 160))
        second = self.run_batch()
        self.assertEqual(second.skipped, 0)
        self.assertNotEqual(json.loads(report.read_text())["cache_key"], old)

    def test_input_content_change_invalidates_cache_even_with_same_size(self):
        self.run_batch()
        changed = np.array(self.source)
        changed[30:80, 30:50] = (190, 200, 210)
        Image.fromarray(changed).save(self.inputs / "one.png")
        self.assertEqual(self.run_batch().skipped, 0)

    def test_parameters_and_engine_identity_invalidate_cache(self):
        self.run_batch()
        self.assertEqual(self.run_batch(strength=0.5).skipped, 0)
        self.assertEqual(self.run_batch(strength=0.5).skipped, 1)
        with patch("batch_color.batch._engine_identity", return_value={"version": "new"}):
            self.assertEqual(self.run_batch(strength=0.5).skipped, 0)

    def test_same_stem_images_have_independent_artifacts_and_masks(self):
        self.source.save(self.inputs / "same.jpg")
        changed = np.array(self.source)
        changed[30:80, 30:50] = (160, 170, 180)
        changed[30:80, 55:75] = (40, 105, 205)
        Image.fromarray(changed).save(self.inputs / "same.png")
        summary = self.run_batch()
        self.assertEqual(summary.total, 3)
        self.assertEqual(summary.errors, 0)
        self.assertEqual(len(list((self.outputs / "reports").glob("*.json"))), 3)
        a, b = self.outputs / "masks/same.jpg.png", self.outputs / "masks/same.png.png"
        self.assertNotEqual(file_hash(a), file_hash(b))
        self.assertTrue(all(item.computation == "processed" for item in summary.items))

    def test_corrupt_artifacts_and_report_do_not_hit_cache(self):
        self.run_batch()
        for role in ("candidates", "masks"):
            (self.outputs / role / "one.png.png").write_bytes(b"corrupt")
            self.assertEqual(self.run_batch().skipped, 0)
        (self.outputs / "reports/one.png.json").write_text("{}")
        self.assertEqual(self.run_batch().skipped, 0)

    def test_partial_write_is_error_not_an_approved_or_cached_result(self):
        with patch("batch_color.batch.save_mask", side_effect=OSError("disk failure")):
            result = self.run_batch()
        self.assertEqual(result.errors, 1)
        self.assertFalse((self.outputs / "reports/one.png.json").exists())
        self.assertEqual(self.run_batch().skipped, 0)

    def test_foreign_hardlink_output_cannot_overwrite_input(self):
        candidate = self.outputs / "candidates/one.png.png"
        candidate.parent.mkdir(parents=True)
        os.link(self.inputs / "one.png", candidate)
        before = file_hash(self.inputs / "one.png")
        with self.assertRaises(ValueError):
            self.run_batch(overwrite=True)
        self.assertEqual(file_hash(self.inputs / "one.png"), before)

    def test_recursive_batch_does_not_ingest_its_outputs(self):
        inside = self.inputs / "results"
        for _ in range(2):
            summary = run_batch(input_directory=self.inputs, profile_path=self.profile, output_directory=inside,
                                recursive=True, mask_backend="heuristic", save_previews=False)
            self.assertEqual(summary.total, 1)

    def test_active_output_lock_is_not_removed_by_second_writer(self):
        self.outputs.mkdir()
        lock = self.outputs / ".batch-color.lock"
        lock.write_text("other writer")
        with self.assertRaises(FileExistsError):
            self.run_batch()
        self.assertEqual(lock.read_text(), "other writer")


if __name__ == "__main__":
    unittest.main()
