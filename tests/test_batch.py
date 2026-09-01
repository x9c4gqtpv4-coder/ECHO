import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from batch_color.batch import run_batch
from batch_color.image_io import ImageInfo
from batch_color.profile import create_profile


class BatchTests(unittest.TestCase):
    def test_batch_processes_and_then_skips_existing_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_directory = root / "input"
            output_directory = root / "output"
            input_directory.mkdir()

            source_pixels = np.full((120, 90, 3), (168, 174, 182), dtype=np.uint8)
            source_pixels[30:110, 32:62] = (140, 95, 72)
            source = Image.fromarray(source_pixels, mode="RGB")
            source.save(input_directory / "one.png")

            reference_pixels = np.full((120, 90, 3), (190, 178, 160), dtype=np.uint8)
            reference = Image.fromarray(reference_pixels, mode="RGB")
            profile = create_profile(
                reference,
                ImageInfo("reference.png", 90, 120, "sRGB", True),
                name="test",
            )
            profile_path = root / "profile.json"
            profile.to_json(profile_path)

            first = run_batch(
                input_directory=input_directory,
                profile_path=profile_path,
                output_directory=output_directory,
                mask_backend="heuristic",
                save_previews=False,
            )
            self.assertEqual(first.total, 1)
            self.assertEqual(first.errors, 0)
            self.assertTrue((output_directory / "candidates/one.png.png").is_file())
            self.assertTrue((output_directory / "masks/one.png.png").is_file())
            self.assertFalse((output_directory / "corrected").exists())
            self.assertEqual(first.review, 1)
            self.assertEqual(first.accepted, 0)
            self.assertTrue((output_directory / "summary.csv").is_file())

            second = run_batch(
                input_directory=input_directory,
                profile_path=profile_path,
                output_directory=output_directory,
                mask_backend="heuristic",
                save_previews=False,
            )
            self.assertEqual(second.skipped, 1)
            self.assertEqual(second.review, 1)
            self.assertEqual(second.items[0].status, "review")
            self.assertEqual(second.items[0].computation, "cached")
            self.assertIsNotNone(second.items[0].background_after)


if __name__ == "__main__":
    unittest.main()
