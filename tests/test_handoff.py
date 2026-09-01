import unittest

import numpy as np
from PIL import Image

from scripts.audit_skin_transition import compare_region
from scripts.export_source import is_source_path


class HandoffTests(unittest.TestCase):
    def test_identity_region_has_no_change(self):
        image = Image.new("RGB", (12, 12), (120, 90, 70))
        report = compare_region(image, image, (0, 0, 12, 12))
        self.assertEqual(report["rgb_max_absolute_change_8bit"], 0)
        self.assertEqual(report["change_field_gradient_p99_8bit_per_pixel"], 0)
        self.assertIsNone(report["rgb_gradient_cosine_similarity"])

    def test_uniform_shift_does_not_create_change_field_edges(self):
        source = Image.new("RGB", (12, 12), (120, 90, 70))
        output = Image.new("RGB", (12, 12), (122, 92, 72))
        report = compare_region(source, output, (0, 0, 12, 12))
        self.assertEqual(report["rgb_mean_absolute_change_8bit"], 2)
        self.assertEqual(report["change_field_gradient_p99_8bit_per_pixel"], 0)

    def test_injected_step_is_visible(self):
        source = Image.new("RGB", (12, 12), (120, 90, 70))
        pixels = np.array(source)
        pixels[:, 6:] += 10
        report = compare_region(source, Image.fromarray(pixels), (0, 0, 12, 12))
        self.assertGreater(report["change_field_gradient_p99_8bit_per_pixel"], 0)
        self.assertNotIn("accepted", report)

    def test_export_excludes_private_and_binary_files(self):
        for name in ["data/input/source.png", ".env", ".venv/lib/x.py", "models/a.onnx", "../secret.py", "src/../secret.py", "tools/person_mask/bin/helper"]:
            self.assertFalse(is_source_path(name), name)

    def test_export_includes_real_source_and_report(self):
        for name in ["src/batch_color/transfer.py", "tools/person_mask/main.swift", "docs/TECHNICAL_REPORT_2026-08-27.md", "scripts/audit_skin_transition.py", "requirements-tested.txt"]:
            self.assertTrue(is_source_path(name), name)


if __name__ == "__main__":
    unittest.main()
