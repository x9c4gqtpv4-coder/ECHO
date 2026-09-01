"""Public ComfyUI node mappings for ECHO."""

from .nodes import ECHOReferenceMatch


NODE_CLASS_MAPPINGS = {
    "ECHOReferenceMatch": ECHOReferenceMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ECHOReferenceMatch": "ECHO Reference Match / 回响·参考追色",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
