from __future__ import annotations

import csv
import io
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from batch_color.image_io import load_srgb, save_mask, save_srgb
from batch_color.masking import backend_identity, get_background_mask
from batch_color.preview import save_comparison
from batch_color.bundle import load_profile
from batch_color.profile import evidence_status
from batch_color.safety import atomic_json, atomic_text, file_hash, payload_hash, validate_artifact_paths
from batch_color.runtime import runtime_identity
from batch_color.transfer import select_profile_path
from batch_color.transaction import ArtifactTransaction

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
CACHE_SCHEMA = 5


@dataclass(frozen=True)
class BatchItemResult:
    input: str
    output: str | None
    status: str
    computation: str = "processed"
    selected_path: str | None = None
    mask_backend: str | None = None
    accepted: bool = False
    baseline_checks_passed: bool | None = None
    no_op: bool | None = None
    background_before: float | None = None
    background_after: float | None = None
    spatial_before: float | None = None
    spatial_after: float | None = None
    gamut_clipped_percent: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class BatchSummary:
    profile: str
    input_directory: str
    output_directory: str
    total: int
    accepted: int
    review: int
    skipped: int
    errors: int
    items: list[BatchItemResult]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _iter_images(input_directory: Path, output_directory: Path, recursive: bool) -> list[Path]:
    if input_directory == output_directory or input_directory.is_relative_to(output_directory):
        raise ValueError("Output directory must not equal or contain the input directory")
    iterator = input_directory.rglob("*") if recursive else input_directory.glob("*")
    return sorted((p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
                   and not p.resolve().is_relative_to(output_directory)), key=lambda p: str(p).casefold())


def _engine_identity() -> dict[str, object]:
    # Keep this wrapper for cache/test compatibility while using the same
    # complete identity evidence as the SKU workflow.
    return {**runtime_identity(), "schema": CACHE_SCHEMA}


def _read_cache(report_path: Path, key: str, artifacts: dict[str, Path]) -> dict | None:
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or data.get("cache_schema") != CACHE_SCHEMA
                or data.get("cache_key") != key or data.get("status") != "review"
                or data.get("accepted") is not False
                or data.get("mask_cacheable") is not True
                or not isinstance(data.get("review_reasons"), list)
                or not isinstance(data.get("baseline_checks_passed"), bool)):
            return None
        metrics = ("background_distance_before", "background_distance_after",
                   "spatial_distance_before", "spatial_distance_after", "gamut_clipped_percent")
        if not all(isinstance(data.get(name), (int, float)) and np.isfinite(data[name]) for name in metrics):
            return None
        for role, path in artifacts.items():
            recorded = data["artifacts"][role]
            if recorded["path"] != str(path) or recorded["sha256"] != file_hash(path):
                return None
        return data
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _item(source: Path, output: Path, payload: dict, computation: str) -> BatchItemResult:
    return BatchItemResult(
        input=str(source), output=str(output), status="review", computation=computation,
        selected_path=payload.get("path"), mask_backend=payload.get("mask_backend"), accepted=False,
        baseline_checks_passed=payload.get("baseline_checks_passed"), no_op=payload.get("no_op"),
        background_before=payload.get("background_distance_before"),
        background_after=payload.get("background_distance_after"),
        spatial_before=payload.get("spatial_distance_before"), spatial_after=payload.get("spatial_distance_after"),
        gamut_clipped_percent=payload.get("gamut_clipped_percent"),
    )


def run_batch(
    *, input_directory: str | Path, profile_path: str | Path, output_directory: str | Path,
    strength: float = 0.85, path: str = "auto", mask_backend: str = "auto",
    mask_quality: str = "accurate", recursive: bool = False, overwrite: bool = False,
    save_previews: bool = True, mode: str = "background",
    progress: Callable[[int, int, Path, str], None] | None = None,
) -> BatchSummary:
    input_root, output_root = Path(input_directory).resolve(), Path(output_directory).resolve()
    profile_file = Path(profile_path).resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_root}")
    if mode not in {"background", "both"} or path not in {"auto", "surface", "global"}:
        raise ValueError("Invalid mode or algorithm path")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be finite and between 0 and 1")
    profile_digest = file_hash(profile_file)
    profile = load_profile(profile_file)
    images = _iter_images(input_root, output_root, recursive)
    plans: list[tuple[Path, Path, dict[str, Path]]] = []
    error_paths = {}
    outputs = [output_root / "summary.json", output_root / "summary.csv", output_root / ".batch-color.lock"]
    for source in images:
        relative = source.relative_to(input_root)
        # Retain the entire original filename to distinguish same.jpg from same.png.
        artifacts = {"candidate": output_root / "candidates" / (str(relative) + ".png"),
                     "mask": output_root / "masks" / (str(relative) + ".png")}
        if save_previews:
            artifacts["preview"] = output_root / "previews" / (str(relative) + ".jpg")
        report_path = output_root / "reports" / (str(relative) + ".json")
        plans.append((source, report_path, artifacts))
        outputs.extend([report_path, *artifacts.values()])
        error_paths[source] = output_root / "errors" / (uuid.uuid4().hex + ".json")
        outputs.append(error_paths[source])
    validate_artifact_paths([profile_file, *images], outputs)
    identity = {"engine": _engine_identity(), "mask": backend_identity(mask_backend, mask_quality),
                "reference_evidence": evidence_status(profile),
                "profile_sha256": profile_digest, "profile_path": str(profile_file),
                "strength": float(strength), "path": path, "mode": mode, "previews": save_previews,
                "export": "8bit-srgb-lossless-png"}
    output_root.mkdir(parents=True, exist_ok=True)
    lock = output_root / ".batch-color.lock"
    # One serial writer per output directory. Crashed runs require explicit lock cleanup.
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("Batch writer active. Do not start another writer here.\n")
    results: list[BatchItemResult] = []
    try:
        for index, (source_path, report_path, artifacts) in enumerate(plans, start=1):
            try:
                source_digest = file_hash(source_path)
                key = payload_hash({**identity, "input_sha256": source_digest, "input": str(source_path),
                                    "artifacts": {k: str(v) for k, v in artifacts.items()}})
                payload = None if overwrite else _read_cache(report_path, key, artifacts)
                if payload is not None:
                    if file_hash(source_path) != source_digest or file_hash(profile_file) != profile_digest:
                        raise RuntimeError("Input/profile changed while validating cache")
                    result = _item(source_path, artifacts["candidate"], payload, "cached")
                else:
                    source, source_info = load_srgb(source_path)
                    mask_result = get_background_mask(source_path, source, backend=mask_backend, quality=mask_quality)
                    corrected, report, background_mask = select_profile_path(
                        source, profile, strength=strength, path=path, mode=mode,
                        background_mask=mask_result.background_mask, mask_backend=mask_result.backend,
                    )
                    with ArtifactTransaction({**artifacts, "report": report_path},
                                             inputs=[profile_file, *images]) as job:
                        verification = save_srgb(corrected, job.staged["candidate"])
                        save_mask(background_mask, job.staged["mask"])
                        if save_previews:
                            save_comparison(source, corrected, None, job.staged["preview"], height=640)
                        if file_hash(source_path) != source_digest or file_hash(profile_file) != profile_digest:
                            raise RuntimeError("Input/profile changed during processing; result cannot be cached")
                        payload = report.as_dict()
                        payload.update(input=str(source_path), output=str(artifacts["candidate"]),
                                   profile=str(profile_file), profile_name=profile.name,
                                   source_info=asdict(source_info), mask_note=mask_result.message,
                                   reference_info=profile.reference_info, schema_version=5,
                                   mask_configuration=identity["mask"], export_verification=verification,
                                   cache_schema=CACHE_SCHEMA, cache_key=key, run_identity=identity,
                                   mask_fallback_reason=mask_result.fallback_reason,
                                   mask_cacheable=mask_result.cacheable,
                                   publication="staged-validated-report-last", artifacts=job.artifact_records())
                        atomic_json(job.staged["report"], payload)
                        job.commit()
                    result = _item(source_path, artifacts["candidate"], payload, "processed")
            except Exception as error:
                result = BatchItemResult(input=str(source_path), output=None, status="error",
                                         computation="error", error=f"{type(error).__name__}: {error}")
                try:
                    atomic_json(error_paths[source_path], asdict(result))
                except OSError:
                    pass
            results.append(result)
            if progress:
                progress(index, len(images), source_path, f"{result.status}/{result.computation}")
        summary = BatchSummary(
            profile=str(profile_file), input_directory=str(input_root), output_directory=str(output_root),
            total=len(results), accepted=0, review=sum(r.status == "review" for r in results),
            skipped=sum(r.computation == "cached" for r in results), errors=sum(r.status == "error" for r in results),
            items=results,
        )
        atomic_json(output_root / "summary.json", summary.as_dict())
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(BatchItemResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(item) for item in results)
        atomic_text(output_root / "summary.csv", "\ufeff" + buffer.getvalue())
        return summary
    finally:
        lock.unlink(missing_ok=True)
