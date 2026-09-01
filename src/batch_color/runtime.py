"""Runtime identity and strict A0 compatibility evidence.

The runtime report binds a result to the implementation, dependencies, native
helper and git state that actually produced it.  It is evidence, not a quality
approval.
"""
from __future__ import annotations

import inspect
import json
import platform
import subprocess
import sys
from types import CodeType
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import __version__ as pillow_version

from batch_color import __version__
from batch_color.baseline import (
    A0_BASELINE,
    A0_EXPECTED_CODE_FINGERPRINT,
    A0_EXPECTED_DEPENDENCIES,
    A0_EXPECTED_PERSON_HELPER_SHA256,
)
from batch_color.masking import find_vision_helper
from batch_color.safety import file_hash, payload_hash


def _is_project_root(path: Path) -> bool:
    return (
        (path / ".git").exists()
        and (path / "pyproject.toml").is_file()
        and (path / "src/batch_color").is_dir()
    )


def _project_root(package: Path) -> Path | None:
    source_root = package.parents[1]
    if _is_project_root(source_root):
        return source_root
    try:
        direct = metadata.distribution("batch-color-standardizer").read_text(
            "direct_url.json"
        )
        payload = json.loads(direct) if direct else {}
        url = payload.get("url")
        if isinstance(url, str) and urlparse(url).scheme == "file":
            installed_from = Path(unquote(urlparse(url).path)).resolve()
            if _is_project_root(installed_from):
                return installed_from
    except (metadata.PackageNotFoundError, json.JSONDecodeError, OSError, TypeError):
        pass
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if _is_project_root(candidate):
            return candidate
    return None


def _git_identity(root: Path | None) -> dict[str, object]:
    if root is None:
        return {"commit": None, "working_tree_has_changes": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                stderr=subprocess.DEVNULL,
            )
        )
        return {"commit": commit, "working_tree_has_changes": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "working_tree_has_changes": None}


def _code_value(value: object) -> object:
    """Normalize bytecode evidence without installation-path metadata."""

    if isinstance(value, CodeType):
        return _code_identity(value)
    if isinstance(value, tuple):
        return [_code_value(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": type(value).__qualname__, "repr": repr(value)}


def _code_identity(code: CodeType) -> dict[str, object]:
    """Describe executable semantics while excluding co_filename/line tables."""

    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
        "code_hex": code.co_code.hex(),
        "constants": _code_value(code.co_consts),
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _callable_identity(function: object) -> dict[str, object]:
    """Return source and bytecode evidence for an actually bound A0 callable.

    Source hashes catch normal edits.  Bytecode identity also catches runtime
    replacement/monkey-patching of an imported helper, which was the gap in the
    original A0 fingerprint.
    """

    code = getattr(function, "__code__", None)
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        source = None
    return {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", None),
        "source": source,
        "bytecode_sha256": (
            payload_hash(_code_identity(code)) if isinstance(code, CodeType) else None
        ),
    }


def a0_execution_manifest() -> dict[str, object]:
    """Describe the complete conservative A0 execution closure.

    The file set intentionally errs on the side of invalidating A0 when shared
    colour, image or mask code changes.  The callable set binds the real runtime
    objects used by ``sku_pipeline`` so replacing an imported helper cannot keep
    the old fingerprint.
    """

    from batch_color import (
        color,
        encoding,
        image_io,
        masking,
        safety,
        sku,
        sku_pipeline,
        transfer,
    )
    from batch_color import workflow

    modules = (color, encoding, image_io, masking, safety, sku, transfer)
    module_hashes = {
        module.__name__: file_hash(Path(module.__file__).resolve()) for module in modules
    }
    functions = {
        # SKU target construction and rendering.
        "sku_pipeline.region_stats": sku_pipeline.region_stats,
        "sku_pipeline.region_distance": sku_pipeline.region_distance,
        "sku_pipeline.region_style_distance": sku_pipeline.region_style_distance,
        "sku_pipeline.normalized_garment_signature": sku_pipeline.normalized_garment_signature,
        "sku_pipeline.scene_style_target": sku_pipeline.scene_style_target,
        "sku_pipeline.background_style_target": sku_pipeline.background_style_target,
        "sku_pipeline.choose_anchor": sku_pipeline.choose_anchor,
        "sku_pipeline.apply_region_plans": sku_pipeline.apply_region_plans,
        "sku_pipeline._two_region_masks": sku_pipeline._two_region_masks,
        "sku_pipeline.run_sku_simple_pilot": sku_pipeline.run_sku_simple_pilot,
        # Imported runtime bindings that can change pixels without changing the
        # caller's source text.
        "sku_pipeline._bounded_luminance_curve": sku_pipeline._bounded_luminance_curve,
        "sku_pipeline.srgb_to_oklab": sku_pipeline.srgb_to_oklab,
        "sku_pipeline.oklab_to_srgb": sku_pipeline.oklab_to_srgb,
        "sku_pipeline.make_proxy": sku_pipeline.make_proxy,
        "sku_pipeline.get_background_mask": sku_pipeline.get_background_mask,
        # Anchor/profile evidence can change the common person target.
        "workflow.load_sku_profile": workflow.load_sku_profile,
        "workflow.profile_confirmed_garment": workflow.profile_confirmed_garment,
        "workflow.ensure_sku_profile": workflow.ensure_sku_profile,
    }
    return {
        "contract": A0_BASELINE.as_dict(),
        "module_sha256": module_hashes,
        "callables": {
            name: _callable_identity(function) for name, function in functions.items()
        },
    }


def a0_code_fingerprint() -> str:
    """Hash the frozen A0 contract and its conservative execution closure."""

    return payload_hash(a0_execution_manifest())


def runtime_identity() -> dict[str, object]:
    package = Path(__file__).resolve().parent
    root = _project_root(package)
    source = {path.name: file_hash(path) for path in sorted(package.glob("*.py"))}
    helper = find_vision_helper()
    payload = {
        "release_version": __version__,
        "git": _git_identity(root),
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy": np.__version__,
        "pillow": pillow_version,
        "operating_system": platform.system(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "source": source,
        "engine_source_sha256": payload_hash(source),
        "a0_code_fingerprint": a0_code_fingerprint(),
        "a0_execution_manifest": a0_execution_manifest(),
        "native_helpers": {
            "person_mask_path": str(helper) if helper else None,
            "person_mask_sha256": file_hash(helper) if helper else None,
        },
    }
    payload["identity_sha256"] = payload_hash(payload)
    return payload


def a0_runtime_compatibility(
    *,
    background_strength: float,
    person_strength: float,
    set_color_tolerance: float,
    identity: dict[str, object] | None = None,
) -> dict[str, object]:
    identity = identity or runtime_identity()
    parameter_match = (
        background_strength == A0_BASELINE.background_strength
        and person_strength == A0_BASELINE.person_strength
        and set_color_tolerance == A0_BASELINE.set_color_tolerance
    )
    dependency_match = (
        identity.get("numpy") == A0_EXPECTED_DEPENDENCIES["numpy"]
        and identity.get("pillow") == A0_EXPECTED_DEPENDENCIES["pillow"]
    )
    algorithm_match = identity.get("a0_code_fingerprint") == A0_EXPECTED_CODE_FINGERPRINT
    helper_identity = identity.get("native_helpers")
    actual_helper = (
        helper_identity.get("person_mask_sha256")
        if isinstance(helper_identity, dict)
        else None
    )
    helper_match = actual_helper == A0_EXPECTED_PERSON_HELPER_SHA256
    reasons = []
    if not parameter_match:
        reasons.append("a0_parameters_changed")
    if not algorithm_match:
        reasons.append("a0_algorithm_identity_changed")
    if not dependency_match:
        reasons.append("a0_dependency_identity_changed")
    if not helper_match:
        reasons.append("a0_native_helper_identity_changed_or_unavailable")
    return {
        "compatible": not reasons,
        "parameter_compatible": parameter_match,
        "algorithm_identity_match": algorithm_match,
        "dependency_identity_match": dependency_match,
        "native_helper_identity_match": helper_match,
        "expected": {
            "code_fingerprint": A0_EXPECTED_CODE_FINGERPRINT,
            "dependencies": A0_EXPECTED_DEPENDENCIES,
            "person_helper_sha256": A0_EXPECTED_PERSON_HELPER_SHA256,
        },
        "actual": {
            "code_fingerprint": identity.get("a0_code_fingerprint"),
            "dependencies": {
                "numpy": identity.get("numpy"),
                "pillow": identity.get("pillow"),
            },
            "person_helper_sha256": actual_helper,
        },
        "reasons": reasons,
    }
