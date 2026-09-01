"""ComfyUI entry point for ECHO.

ComfyUI imports every directory in ``custom_nodes`` as a Python package.  The
color engine itself follows the normal ``src/`` layout, so this tiny adapter
adds that directory to the import path before exposing the node mappings.
"""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "src"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from .comfyui_echo import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
