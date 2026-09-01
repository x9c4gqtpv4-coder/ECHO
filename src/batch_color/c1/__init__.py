"""Read-only C1 relative illumination and tone analysis.

C1 is deliberately separate from the frozen A0 renderer.  It produces
evidence and hypotheses only; importing or running it never changes pixels.
"""

from batch_color.c1.analysis import analyse_relative_illumination
from batch_color.c1.identity import c1_identity
from batch_color.c1.schema import C1_ANALYZER_ID, C1_CONFIG, C1AnalyzerConfig

__all__ = [
    "C1_ANALYZER_ID",
    "C1_CONFIG",
    "C1AnalyzerConfig",
    "analyse_relative_illumination",
    "c1_identity",
]
