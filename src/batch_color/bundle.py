"""Bound, recomputable background standards; no archive extraction or executable data.

A .bcp file is one atomically published ZIP containing canonical reference pixels,
the exact reference mask, claimed statistics, and a content manifest. Hashes prove
consistency, NOT authenticity or semantic correctness. Loading recomputes the
statistics with this implementation and returns that recomputed runtime object.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import stat
import struct
import zipfile
import zlib

from PIL import Image

from batch_color.image_io import ImageInfo
from batch_color.profile import (
    ColorProfile, MAX_PROFILE_BYTES, PROFILE_VERSION, create_profile,
    generator_identity, profile_from_payload, reference_evidence_verified, strict_json,
)
from batch_color.safety import atomic_output, validate_artifact_paths

BUNDLE_SCHEMA = 1
MAX_PIXELS = 24_000_000
MAX_EDGE = 12_000
MAX_BUNDLE_BYTES = 128 * 1024 * 1024
MEMBER_LIMITS = {"manifest.json": 64 * 1024, "profile.json": MAX_PROFILE_BYTES,
                 "provenance.json": 64 * 1024, "reference.png": 96 * 1024 * 1024,
                 "reference_mask.png": 32 * 1024 * 1024}


def validate_bundle_path(path: str | Path) -> None:
    if Path(path).suffix.lower() != ".bcp":
        raise ValueError("New standards require a .bcp evidence bundle; legacy JSON can only be read as global-only statistics")


def _geometry(image: Image.Image, mode: str) -> None:
    if (image.mode != mode or min(image.size) < 1 or max(image.size) > MAX_EDGE
            or image.width * image.height > MAX_PIXELS):
        raise ValueError(f"Bundle asset requires {mode}, at most {MAX_PIXELS} pixels and edge {MAX_EDGE}")


def _json_bytes(payload) -> bytes:
    return (json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def _png(image: Image.Image) -> bytes:
    # No EXIF/ICC reconversion on reload: these bytes are already canonical sRGB.
    clean = Image.frombytes(image.mode, image.size, image.tobytes())
    buffer = io.BytesIO()
    clean.save(buffer, format="PNG")
    return buffer.getvalue()


def _decode_png(data: bytes, mode: str) -> Image.Image:
    color_type = 2 if mode == "RGB" else 0
    if (len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[8:16] != b"\x00\x00\x00\rIHDR"
            or data[24] != 8 or data[25] != color_type):
        raise ValueError("Evidence assets must be canonical 8-bit RGB/L PNG, without alpha")
    width, height = struct.unpack(">II", data[16:24])
    if not 0 < width <= MAX_EDGE or not 0 < height <= MAX_EDGE or width * height > MAX_PIXELS:
        raise ValueError("Evidence image exceeds geometry/resource bounds")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            _geometry(opened, mode)
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError("Evidence images cannot have multiple frames")
            if any(key in opened.info for key in ("exif", "icc_profile", "transparency")):
                raise ValueError("Evidence PNG cannot carry a second orientation/color/alpha interpretation")
            opened.load()
            return opened.copy()
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Invalid evidence PNG") from error


def _same_tree(left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same_tree(a, b) for a, b in zip(left, right))
    if type(left) is float and type(right) is float:
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-7)
    return type(left) is type(right) and left == right


def _recompute(claimed: ColorProfile, reference: Image.Image, mask: Image.Image) -> ColorProfile:
    _geometry(reference, "RGB")
    _geometry(mask, "L")
    if mask.size != reference.size:
        raise ValueError("Reference/mask evidence geometry mismatch")
    if (claimed.version != PROFILE_VERSION or claimed.background_sampling != "masked-core"
            or claimed.generator != generator_identity()):
        raise ValueError("Unsupported bundle recipe/implementation; rebuild the standard with the current version")
    if (hashlib.sha256(reference.tobytes()).hexdigest() != claimed.reference_pixels_sha256
            or hashlib.sha256(mask.tobytes()).hexdigest() != claimed.reference_mask_sha256):
        raise ValueError("Reference/mask pixels do not match the Profile evidence binding")
    info = dict(claimed.reference_info)
    mask_metadata = info.pop("mask_generation", {})
    try:
        info = ImageInfo(**info)
        if info.original_bit_depth != 8:
            raise ValueError("Evidence origin must use the supported 8-bit input contract")
        actual = create_profile(reference, info, name=claimed.name, background_mask=mask,
                                mask_backend=claimed.reference_mask_backend, mask_metadata=mask_metadata)
    except (TypeError, KeyError) as error:
        raise ValueError("Invalid bundle ImageInfo/mask provenance") from error
    if not _same_tree(asdict(claimed), asdict(actual)):
        raise ValueError("Profile statistics/diagnostics disagree with recomputed reference evidence")
    # Use computed values, never the imported coefficients even within tolerance.
    return actual


def _provenance(profile: ColorProfile, reference: Image.Image) -> dict:
    return {"generator": profile.generator, "reference_pixels_sha256": profile.reference_pixels_sha256,
            "reference_mask_sha256": profile.reference_mask_sha256,
            "canonical": {"width": reference.width, "height": reference.height,
                          "color_space": "sRGB", "bit_depth": 8, "orientation": "already_applied",
                          "mask": "L8_255_background_0_protected"},
            "sampling_reviewed": False, "authenticity_verified": False,
            "purpose": "background_candidate_only_human_review_required"}


def save_bundle(path: str | Path, profile: ColorProfile, reference: Image.Image, reference_mask: Image.Image,
                *, inputs=(), overwrite: bool = False) -> None:
    validate_bundle_path(path)
    origin = profile.reference_info.get("path")
    sources = [*inputs, *([origin] if isinstance(origin, str) and origin else [])]
    validate_artifact_paths(sources, [path], overwrite=overwrite)
    if not reference_evidence_verified(profile):
        raise ValueError("Only profiles computed from bound reference pixels and a mask can be exported as evidence")
    profile = _recompute(profile, reference, reference_mask)
    members = {"profile.json": _json_bytes(asdict(profile)), "reference.png": _png(reference),
               "reference_mask.png": _png(reference_mask), "provenance.json": _json_bytes(_provenance(profile, reference))}
    manifest = {"schema": BUNDLE_SCHEMA, "kind": "batch-color-reference-standard",
                "members": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                            for name, data in members.items()}}
    members["manifest.json"] = _json_bytes(manifest)
    if any(len(data) > MEMBER_LIMITS[name] for name, data in members.items()):
        raise ValueError("Evidence bundle exceeds resource limits")
    with atomic_output(path) as staged:
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in members.items():
                entry = zipfile.ZipInfo(name)
                entry.external_attr = (stat.S_IFREG | 0o600) << 16
                # PNG is already compressed; compress only small JSON records.
                entry.compress_type = zipfile.ZIP_STORED if name.endswith(".png") else zipfile.ZIP_DEFLATED
                archive.writestr(entry, data)
        # Reopen and check the exact encoded assets before single-file publication.
        load_bundle(staged)
        validate_artifact_paths(sources, [path], overwrite=overwrite)


def load_bundle(path: str | Path) -> tuple[ColorProfile, Image.Image, Image.Image]:
    """No extraction; fixed member names, no paths/code/pickle, bounded decoding."""
    if Path(path).stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError("Evidence archive exceeds resource limits")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(MEMBER_LIMITS) or set(names) != set(MEMBER_LIMITS):
                raise ValueError("Evidence archive has missing, duplicate or unexpected members")
            members = {}
            for entry in entries:
                kind = stat.S_IFMT(entry.external_attr >> 16)
                limit = MEMBER_LIMITS[entry.filename]
                if (kind not in {0, stat.S_IFREG} or entry.flag_bits & 1
                        or entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                        or entry.file_size < 0 or entry.file_size > limit):
                    raise ValueError("Unsupported evidence member type or size")
                with archive.open(entry) as stream:
                    data = stream.read(limit + 1)
                if len(data) != entry.file_size or len(data) > limit:
                    raise ValueError("Evidence member size mismatch")
                members[entry.filename] = data
        manifest = strict_json(members["manifest.json"], MEMBER_LIMITS["manifest.json"])
        if (not isinstance(manifest, dict) or type(manifest.get("schema")) is not int
                or manifest["schema"] != BUNDLE_SCHEMA or manifest.get("kind") != "batch-color-reference-standard"
                or set(manifest) != {"schema", "kind", "members"} or not isinstance(manifest["members"], dict)
                or set(manifest["members"]) != set(MEMBER_LIMITS) - {"manifest.json"}):
            raise ValueError("Unsupported evidence manifest")
        for name, record in manifest["members"].items():
            if (not isinstance(record, dict) or set(record) != {"sha256", "bytes"}
                    or type(record["bytes"]) is not int or record["bytes"] != len(members[name])
                    or record["sha256"] != hashlib.sha256(members[name]).hexdigest()):
                raise ValueError("Evidence manifest hash/size mismatch")
        claimed = profile_from_payload(strict_json(members["profile.json"]))
        reference = _decode_png(members["reference.png"], "RGB")
        mask = _decode_png(members["reference_mask.png"], "L")
        provenance = strict_json(members["provenance.json"], MEMBER_LIMITS["provenance.json"])
        if provenance != _provenance(claimed, reference):
            raise ValueError("Evidence provenance does not match its assets/profile")
        return _recompute(claimed, reference, mask), reference, mask
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError, KeyError, TypeError, RuntimeError, RecursionError) as error:
        raise ValueError(f"Invalid evidence bundle: {error}") from error


def load_profile(path: str | Path) -> ColorProfile:
    suffix = Path(path).suffix.lower()
    if suffix == ".bcp":
        return load_bundle(path)[0]
    if suffix == ".json":
        return ColorProfile.from_json(path)
    raise ValueError("Profile must be a .bcp evidence bundle or legacy .json statistics")
