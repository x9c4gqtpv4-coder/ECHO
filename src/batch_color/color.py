from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def srgb_to_linear(rgb: FloatArray) -> FloatArray:
    """Convert sRGB values in [0, 1] to linear-light RGB."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb(rgb: FloatArray) -> FloatArray:
    """Convert linear-light RGB to sRGB without clipping the input first."""
    rgb = np.asarray(rgb, dtype=np.float32)
    positive = np.maximum(rgb, 0.0)
    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(positive, 1.0 / 2.4) - 0.055,
    )


def linear_rgb_to_oklab(rgb: FloatArray) -> FloatArray:
    """Convert linear sRGB to Oklab."""
    rgb = np.asarray(rgb, dtype=np.float32)
    l = 0.4122214708 * rgb[..., 0] + 0.5363325363 * rgb[..., 1] + 0.0514459929 * rgb[..., 2]
    m = 0.2119034982 * rgb[..., 0] + 0.6806995451 * rgb[..., 1] + 0.1073969566 * rgb[..., 2]
    s = 0.0883024619 * rgb[..., 0] + 0.2817188376 * rgb[..., 1] + 0.6299787005 * rgb[..., 2]

    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)

    out = np.empty_like(rgb, dtype=np.float32)
    out[..., 0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    out[..., 1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    out[..., 2] = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return out


def oklab_to_linear_rgb(lab: FloatArray) -> FloatArray:
    """Convert Oklab to linear sRGB."""
    lab = np.asarray(lab, dtype=np.float32)
    l_ = lab[..., 0] + 0.3963377774 * lab[..., 1] + 0.2158037573 * lab[..., 2]
    m_ = lab[..., 0] - 0.1055613458 * lab[..., 1] - 0.0638541728 * lab[..., 2]
    s_ = lab[..., 0] - 0.0894841775 * lab[..., 1] - 1.2914855480 * lab[..., 2]

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    out = np.empty_like(lab, dtype=np.float32)
    out[..., 0] = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    out[..., 1] = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    out[..., 2] = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return out


def srgb_to_oklab(rgb: FloatArray) -> FloatArray:
    return linear_rgb_to_oklab(srgb_to_linear(rgb))


def oklab_to_srgb(lab: FloatArray) -> FloatArray:
    return linear_to_srgb(oklab_to_linear_rgb(lab))

