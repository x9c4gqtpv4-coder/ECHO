from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image, ImageFilter

from batch_color.color import srgb_to_oklab
from batch_color.image_io import image_to_float, make_proxy
from batch_color.profile import RegionStatistics, SurfaceStatistics, evaluate_surface


def _connected_to_border(eligible: np.ndarray) -> np.ndarray:
    """Return eligible pixels connected to any image border."""
    height, width = eligible.shape
    visited = np.zeros_like(eligible, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def seed(y: int, x: int) -> None:
        if eligible[y, x] and not visited[y, x]:
            visited[y, x] = True
            queue.append((y, x))

    for x in range(width):
        seed(0, x)
        seed(height - 1, x)
    for y in range(1, height - 1):
        seed(y, 0)
        seed(y, width - 1)

    while queue:
        y, x = queue.popleft()
        if y > 0 and eligible[y - 1, x] and not visited[y - 1, x]:
            visited[y - 1, x] = True
            queue.append((y - 1, x))
        if y + 1 < height and eligible[y + 1, x] and not visited[y + 1, x]:
            visited[y + 1, x] = True
            queue.append((y + 1, x))
        if x > 0 and eligible[y, x - 1] and not visited[y, x - 1]:
            visited[y, x - 1] = True
            queue.append((y, x - 1))
        if x + 1 < width and eligible[y, x + 1] and not visited[y, x + 1]:
            visited[y, x + 1] = True
            queue.append((y, x + 1))
    return visited


def estimate_studio_background_mask(
    image: Image.Image,
    statistics: RegionStatistics,
    surface: SurfaceStatistics,
    *,
    max_edge: int = 512,
) -> Image.Image:
    """Estimate a soft, border-connected studio background mask.

    This deliberately lightweight baseline works for uncluttered studio images.
    A learned segmentation adapter will replace it for complex scenes.
    """
    proxy = make_proxy(image, max_edge=max_edge)
    lab = srgb_to_oklab(image_to_float(proxy))
    height, width = lab.shape[:2]
    y_values = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x_values = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(y_values, x_values, indexing="ij")
    predicted_background = evaluate_surface(surface, grid_x, grid_y)

    l_scale = max(float(surface.residual) * 2.2, 0.045)
    a_scale = max(float(statistics.a_mad) * 5.0, 0.012)
    b_scale = max(float(statistics.b_mad) * 5.0, 0.012)

    distance = np.sqrt(
        np.square((lab[..., 0] - predicted_background[..., 0]) / l_scale)
        + np.square((lab[..., 1] - predicted_background[..., 1]) / a_scale)
        + np.square((lab[..., 2] - predicted_background[..., 2]) / b_scale)
    )
    eligible = distance <= 3.0
    connected = _connected_to_border(eligible)

    hard_mask = Image.fromarray((connected.astype(np.uint8) * 255), mode="L")
    blur_radius = max(2.0, min(proxy.size) / 120.0)
    soft_mask = hard_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return soft_mask.resize(image.size, Image.Resampling.BILINEAR)
