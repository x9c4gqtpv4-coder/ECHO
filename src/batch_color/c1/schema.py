from __future__ import annotations

from dataclasses import asdict, dataclass


C1_ANALYZER_ID = "c1-relative-illumination-v1"


@dataclass(frozen=True)
class C1AnalyzerConfig:
    """Frozen thresholds for the first read-only C1 observer.

    Values describe display-referred sRGB evidence.  They are not camera or
    physical-light calibration constants.
    """

    schema_version: int = 1
    max_edge: int = 1024
    grid_size: int = 3
    min_region_pixels: int = 512
    min_neutral_pixels: int = 256
    min_spatial_cell_pixels: int = 64
    clip_low_srgb: float = 2.0 / 255.0
    clip_high_srgb: float = 253.0 / 255.0
    min_linear_luminance: float = 1.0e-4
    neutral_oklab_chroma_max: float = 0.035
    explicit_neutral_oklab_chroma_max: float = 0.030
    neutral_lightness_low: float = 0.12
    neutral_lightness_high: float = 0.94
    exposure_parallel_mad_stops: float = 0.12
    exposure_fit_review_stops: float = 0.35
    spatial_exposure_review_stops: float = 0.35
    neutral_spatial_review_oklab: float = 0.012
    heavy_clipping_fraction: float = 0.10
    direction_deadband_oklab: float = 0.0015

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported C1 analyzer schema")
        if not 64 <= self.max_edge <= 4096:
            raise ValueError("C1 max_edge must be in 64..4096")
        if not 2 <= self.grid_size <= 8:
            raise ValueError("C1 grid_size must be in 2..8")
        if self.min_region_pixels < 128 or self.min_neutral_pixels < 32:
            raise ValueError("C1 evidence pixel thresholds are too small")
        if self.min_spatial_cell_pixels < 1:
            raise ValueError("C1 spatial cell pixel threshold must be positive")
        if not 0.0 <= self.clip_low_srgb < self.clip_high_srgb <= 1.0:
            raise ValueError("C1 clipping thresholds are invalid")
        if not 0.0 < self.neutral_oklab_chroma_max < 0.25:
            raise ValueError("C1 neutral chroma threshold is invalid")
        if not 0.0 < self.explicit_neutral_oklab_chroma_max < 0.25:
            raise ValueError("C1 explicit neutral chroma threshold is invalid")
        if not 0.0 <= self.neutral_lightness_low < self.neutral_lightness_high <= 1.0:
            raise ValueError("C1 neutral lightness range is invalid")
        if self.min_linear_luminance <= 0.0:
            raise ValueError("C1 minimum linear luminance must be positive")
        if min(
            self.exposure_parallel_mad_stops,
            self.exposure_fit_review_stops,
            self.spatial_exposure_review_stops,
            self.neutral_spatial_review_oklab,
        ) <= 0.0:
            raise ValueError("C1 fit and spatial thresholds must be positive")
        if not 0.0 <= self.heavy_clipping_fraction <= 1.0:
            raise ValueError("C1 heavy clipping fraction must be in 0..1")
        if self.direction_deadband_oklab < 0.0:
            raise ValueError("C1 direction deadband must not be negative")


C1_CONFIG = C1AnalyzerConfig()
