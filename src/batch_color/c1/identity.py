from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import __version__ as pillow_version

from batch_color.c1.schema import C1_ANALYZER_ID, C1AnalyzerConfig
from batch_color.safety import file_hash, payload_hash


def c1_identity(config: C1AnalyzerConfig) -> dict[str, object]:
    """Bind a C1 report to its observer source, dependencies and thresholds."""

    package = Path(__file__).resolve().parent
    source = {
        path.name: file_hash(path)
        for path in sorted(package.glob("*.py"))
        if path.is_file()
    }
    payload: dict[str, object] = {
        "analyzer_id": C1_ANALYZER_ID,
        "schema_version": config.schema_version,
        "mode": "shadow_read_only",
        "pixel_authority": "none",
        "source": source,
        "source_fingerprint": payload_hash(source),
        "configuration": config.as_dict(),
        "configuration_fingerprint": payload_hash(config.as_dict()),
        "dependencies": {
            "numpy": np.__version__,
            "pillow": pillow_version,
        },
    }
    payload["identity_sha256"] = payload_hash(payload)
    return payload
