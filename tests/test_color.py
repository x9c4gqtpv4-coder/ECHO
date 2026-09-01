import unittest

import numpy as np

from batch_color.color import oklab_to_srgb, srgb_to_oklab


class ColorConversionTests(unittest.TestCase):
    def test_oklab_round_trip(self) -> None:
        rgb = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.72, 0.43, 0.21],
                [0.12, 0.44, 0.91],
            ],
            dtype=np.float32,
        )
        restored = oklab_to_srgb(srgb_to_oklab(rgb))
        np.testing.assert_allclose(restored, rgb, atol=2e-5)

