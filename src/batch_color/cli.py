from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
import subprocess
import uuid
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from batch_color.baseline import A0_BASELINE
from batch_color.batch import _engine_identity, run_batch
from batch_color import __version__
from batch_color.bundle import load_profile, save_bundle, validate_bundle_path
from batch_color.c1 import C1_CONFIG, analyse_relative_illumination, c1_identity
from batch_color.image_io import load_mask, load_srgb, save_mask, save_srgb
from batch_color.fine_masks import (
    build_fine_mask_bundle,
    build_fine_mask_bundle_from_arrays,
    region_names,
)
from batch_color.fine_parsing import segformer_atr18
from batch_color.fine_validation import validate_fine_label_files
from batch_color.masking import backend_identity, find_vision_helper, get_background_mask
from batch_color.precision import (
    REFERENCE_POLICIES,
    SKU_ROLES,
    RegionTargetPolicy,
    precision_region_match,
)
from batch_color.preview import save_comparison
from batch_color.profile import create_profile, evidence_status
from batch_color.safety import atomic_json, file_hash, validate_artifact_paths, validate_master_path
from batch_color.sku import scan_sku
from batch_color.sku_pipeline import run_sku_pilot, run_sku_simple_pilot
from batch_color.transfer import select_profile_path
from batch_color.transaction import ArtifactTransaction
from batch_color.workflow import (
    review_sku_output,
    save_new_sku_profile,
    verify_region_target_evidence,
)

EXPECTED_ERRORS = (OSError, ValueError, RuntimeError, subprocess.SubprocessError)


def _doctor() -> int:
    print(f"batch-color: {__version__}")
    print(f"Python: {platform.python_version()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Platform: {platform.system()} {platform.release()}")
    supported_python = (3, 11) <= sys.version_info[:2] < (3, 13)
    print(f"Supported Python: {'yes' if supported_python else 'no'}")
    print(f"Vision helper: {find_vision_helper() or 'not built'}")
    fine_dependencies = all(
        importlib.util.find_spec(module) is not None for module in ("torch", "transformers")
    )
    print("Fine ATR18 adapter: available; local audited safetensors model required")
    print(f"Fine parser dependencies installed: {'yes' if fine_dependencies else 'no (optional)'}")
    print("Automatic quality approval: unavailable; use sku-review for explicit human approval")
    return 0 if supported_python else 1


def _reference_profile(args, name):
    reference, info = load_srgb(args.reference)
    result = get_background_mask(args.reference, reference, backend=args.mask_backend,
                                 quality=args.mask_quality, mask_path=args.reference_mask)
    profile = create_profile(reference, info, name=name, background_mask=result.background_mask,
                             mask_backend=result.backend,
                             mask_metadata={"configuration": backend_identity(args.mask_backend, args.mask_quality),
                                            "message": result.message, "fallback_reason": result.fallback_reason,
                                            "cacheable": result.cacheable})
    return reference, profile, result.background_mask


def _create_profile(args: argparse.Namespace) -> int:
    inputs = [p for p in (args.reference, args.reference_mask) if p]
    outputs = {"profile": Path(args.output), "report": Path(str(args.output) + ".report.json")}
    if args.reference_mask_output:
        outputs["reference_mask"] = Path(args.reference_mask_output)
    args.error_report = str(outputs["report"].parent / ".batch-color-errors" / (uuid.uuid4().hex + ".json"))
    validate_artifact_paths(inputs, [*outputs.values(), args.error_report], overwrite=args.overwrite)
    validate_bundle_path(args.output)
    hashes = {str(Path(p)): file_hash(p) for p in inputs}
    try:
        reference, profile, mask = _reference_profile(args, args.name)
        with ArtifactTransaction(outputs, inputs=inputs, overwrite=args.overwrite) as job:
            save_bundle(job.staged["profile"], profile, reference, mask, inputs=inputs)
            if args.reference_mask_output:
                save_mask(mask, job.staged["reference_mask"])
            if any(file_hash(path) != digest for path, digest in hashes.items()):
                raise RuntimeError("Reference input changed while building the standard")
            atomic_json(job.staged["report"], {"status": "review", "accepted": False,
                        "schema_version": 5, "reference_evidence": evidence_status(profile),
                        "input_hashes": hashes, "reference_info": profile.reference_info,
                        "artifacts": job.artifact_records(), "publication": "staged-validated-report-last"})
            job.commit()
    except EXPECTED_ERRORS as error:
        _record_error(args, error)
        raise
    print(f"Recomputable background standard (human review required): {args.output}")
    return 0


def _verify_profile(args):
    profile = load_profile(args.profile)
    status = evidence_status(profile)
    print(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if status["reference_evidence_verified"] else 3


def _prepare_write_set(args, *, matching):
    if not args.report:
        args.report = str(args.output) + ".report.json"
    if not args.mask_output:
        args.mask_output = str(args.output) + ".mask.png"
    if matching:
        if not args.profile_output:
            args.profile_output = str(args.output) + ".reference.bcp"
        if not args.reference_mask_output:
            args.reference_mask_output = str(args.output) + ".reference-mask.png"
    inputs = [args.input, args.reference if matching else args.profile]
    inputs.extend(p for p in (args.background_mask, args.protected_mask) if p)
    if matching and args.reference_mask:
        inputs.append(args.reference_mask)
    outputs = [args.output, args.preview, args.mask_output, args.report]
    if matching:
        outputs.extend([args.profile_output, args.reference_mask_output])
    args.error_report = str(Path(args.report).parent / ".batch-color-errors" / (uuid.uuid4().hex + ".json"))
    outputs.append(args.error_report)
    validate_artifact_paths(inputs, outputs, overwrite=args.overwrite)
    validate_master_path(args.output)
    if matching:
        validate_bundle_path(args.profile_output)
    return {str(Path(p)): file_hash(p) for p in inputs}


def _finish(args, source, source_info, profile, input_hashes, reference=None, reference_mask=None):
    mask_result = get_background_mask(args.input, source, backend=args.mask_backend,
                                     quality=args.mask_quality, mask_path=args.background_mask)
    protection = load_mask(args.protected_mask, source.size) if args.protected_mask else None
    corrected, report, mask = select_profile_path(
        source, profile, strength=args.strength, path=args.path, mode=args.mode,
        background_mask=mask_result.background_mask, mask_backend=mask_result.backend,
        protected_mask=protection,
    )
    outputs = {"candidate": Path(args.output), "report": Path(args.report)}
    for role, value in (("mask", args.mask_output), ("profile", getattr(args, "profile_output", None)),
                        ("reference_mask", getattr(args, "reference_mask_output", None)),
                        ("preview", args.preview)):
        if value:
            outputs[role] = Path(value)
    with ArtifactTransaction(outputs, inputs=input_hashes, overwrite=args.overwrite) as job:
        verification = save_srgb(corrected, job.staged["candidate"])
        if args.mask_output:
            save_mask(mask, job.staged["mask"])
        if getattr(args, "profile_output", None):
            save_bundle(job.staged["profile"], profile, reference, reference_mask, inputs=input_hashes)
        if getattr(args, "reference_mask_output", None):
            save_mask(reference_mask, job.staged["reference_mask"])
        if args.preview:
            save_comparison(source, corrected, reference, job.staged["preview"])
        if any(file_hash(p) != expected for p, expected in input_hashes.items()):
            raise RuntimeError("An input changed during processing; candidate cannot be finalized")
        payload = report.as_dict()
        payload.update(schema_version=5, input=str(args.input), output=str(args.output),
                   engine_identity=_engine_identity(),
                   input_hashes=input_hashes, source_info=asdict(source_info),
                   reference_info=profile.reference_info, profile_name=profile.name,
                   mask_note=mask_result.message, export_verification=verification,
                   protected_mask=str(args.protected_mask) if args.protected_mask else None,
                   artifacts=job.artifact_records(), publication="staged-validated-report-last",
                   mask_fallback_reason=mask_result.fallback_reason, mask_cacheable=mask_result.cacheable)
        if hasattr(args, "reference"):
            payload["reference"] = str(args.reference)
        if hasattr(args, "profile"):
            payload["profile"] = str(args.profile)
        atomic_json(job.staged["report"], payload)
        job.commit()
    print(f"Candidate (REVIEW, not approved): {args.output}")
    print(f"Path: {report.path}; mode: {report.mode}; no-op: {report.no_op}")
    print(f"Mask backend: {report.mask_backend}")
    if mask_result.message:
        print(f"Mask note: {mask_result.message}")
    print(f"Background baseline: {report.background_distance_before:.3f} -> {report.background_distance_after:.3f}")
    print(f"Baseline checks passed: {report.baseline_checks_passed}; production accepted: False")
    print(f"Review reasons: {', '.join(report.review_reasons)}")
    print(f"Report: {args.report}")
    return 3 if args.strict_quality_exit else 0


def _match(args: argparse.Namespace) -> int:
    hashes = _prepare_write_set(args, matching=True)
    try:
        source, info = load_srgb(args.input)
        reference, profile, reference_mask = _reference_profile(args, args.profile_name or Path(args.reference).stem)
        return _finish(args, source, info, profile, hashes, reference, reference_mask)
    except EXPECTED_ERRORS as error:
        _record_error(args, error)
        raise


def _apply(args: argparse.Namespace) -> int:
    hashes = _prepare_write_set(args, matching=False)
    try:
        profile = load_profile(args.profile)
        source, info = load_srgb(args.input)
        return _finish(args, source, info, profile, hashes)
    except EXPECTED_ERRORS as error:
        _record_error(args, error)
        raise


def _record_error(args, error):
    try:
        atomic_json(args.error_report, {"status": "error", "accepted": False,
                    "error": f"{type(error).__name__}: {error}", "output": str(args.output),
                    "note": "No new complete job published; an older result may remain unchanged."})
        print(f"Error report: {args.error_report}", file=sys.stderr)
    except OSError:
        pass  # A disk failure must not conceal the original error.


def _batch(args: argparse.Namespace) -> int:
    def progress(index, total, source, status):
        print(f"[{index}/{total}] {source.name}: {status}")

    summary = run_batch(
        input_directory=args.input, profile_path=args.profile, output_directory=args.output,
        strength=args.strength, path=args.path, mode=args.mode, mask_backend=args.mask_backend,
        mask_quality=args.mask_quality, recursive=args.recursive, overwrite=args.overwrite,
        save_previews=not args.no_previews, progress=progress,
    )
    print(f"Batch: total={summary.total}, approved={summary.accepted}, review={summary.review}, "
          f"cached={summary.skipped}, errors={summary.errors}")
    print(f"Reports: {Path(summary.output_directory) / 'summary.json'}")
    if summary.errors:
        return 2
    return 3 if args.strict_quality_exit and summary.review else 0


def _sku_pilot(args: argparse.Namespace) -> int:
    output_root = args.output_root or str(Path(args.dataset_root) / "校色输出")
    if args.pipeline_mode == "person-background":
        final, summary = run_sku_simple_pilot(
            dataset_root=args.dataset_root,
            sku=args.sku,
            output_root=output_root,
            run_name=args.run_name,
            background_strength=(
                A0_BASELINE.background_strength
                if args.background_strength is None
                else args.background_strength
            ),
            person_strength=(
                A0_BASELINE.person_strength
                if args.person_strength is None
                else args.person_strength
            ),
            mask_backend=args.mask_backend,
            set_color_tolerance=args.set_color_tolerance,
            replace_output=args.replace_output,
        )
    else:
        final, summary = run_sku_pilot(
            dataset_root=args.dataset_root,
            sku=args.sku,
            output_root=output_root,
            run_name=args.run_name,
            garment_kind=args.garment_kind,
            garment_hint=args.garment_hint,
            background_strength=(
                0.68 if args.background_strength is None else args.background_strength
            ),
            garment_strength=args.garment_strength,
            skin_strength=args.skin_strength,
            hair_strength=args.hair_strength,
            mask_backend=args.mask_backend,
            parser_backend=args.parser_backend,
            pose_backend=args.pose_backend,
            garment_anchor=args.garment_anchor,
            set_color_tolerance=args.set_color_tolerance,
        )
    print(f"SKU pilot candidate (REVIEW, not approved): {final}")
    if "anchor" in summary:
        print(f"Anchor: {summary['anchor']}")
    print(
        "Baseline risk diagnostics clear: "
        f"{summary.get('baseline_diagnostics_passed', summary['automatic_checks_passed'])}"
    )
    print("Quality approval: pending human review")
    print(f"Contact sheet: {final / '整套对照.jpg'}")
    return 3 if args.strict_quality_exit else 0


def _sku_init(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root or Path(args.dataset_root) / "校色输出").resolve()
    manifest = scan_sku(args.dataset_root, args.sku)
    destination = output_root / args.sku / "sku-profile.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile = save_new_sku_profile(
        destination,
        manifest,
        product_anchor=args.product_anchor,
        garment_anchor=args.garment_anchor,
        confirm_product=args.confirm_product,
        confirm_garment=args.confirm_garment,
        overwrite=args.overwrite,
    )
    print(f"SKU profile: {destination}")
    print(f"Product truth: {profile['product_truth']['status']}")
    print(f"Garment anchor: {profile['garment_anchor']['status']}")
    print("A0 use of product truth: evidence only; no product-colour residual is enabled")
    return 0


def _sku_review(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    record_path, record = review_sku_output(
        output_root / args.sku,
        decision=args.decision,
        reviewer=args.reviewer,
        reason=args.reason,
        replace_approved=args.replace_approved,
    )
    print(f"Review decision: {record['decision']}")
    print(f"Review record: {record_path}")
    if record["accepted"]:
        print(f"Approved output: {output_root / args.sku / '已批准成品'}")
    return 0


def _confidence_threshold_policy(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("thresholds"), dict):
        payload = payload["thresholds"]
    if not isinstance(payload, dict):
        raise ValueError("Confidence policy must be a JSON object or contain a thresholds object")
    return {str(name): float(value) for name, value in payload.items()}


def _fine_masks(args: argparse.Namespace) -> int:
    source, source_info = load_srgb(args.input)
    confidence_thresholds = _confidence_threshold_policy(args.confidence_policy)
    parser_identity: dict[str, object]
    parser_inputs: list[str] = []
    if args.model_dir:
        if args.label_status != "automatic" or args.reviewed_by:
            raise ValueError(
                "Direct model output is automatic; export it first, correct it, then import as reviewed"
            )
        parsed = segformer_atr18(
            source,
            args.model_dir,
            device=args.fine_device,
            max_edge=args.fine_max_edge,
            threads=args.fine_threads,
        )
        bundle = build_fine_mask_bundle_from_arrays(
            parsed.labels,
            parsed.confidence,
            label_status="automatic",
            confidence_threshold=args.confidence_threshold,
            confidence_thresholds=confidence_thresholds,
            min_authorized_fraction=args.min_authorized_fraction,
            min_pixels=args.min_pixels,
            feather_radius=args.feather_radius,
        )
        parser_identity = parsed.identity
        parser_inputs.extend(parsed.input_files)
    else:
        bundle = build_fine_mask_bundle(
            args.label_map,
            source.size,
            confidence_map_path=args.confidence_map,
            label_status=args.label_status,
            reviewed_by=args.reviewed_by,
            confidence_threshold=args.confidence_threshold,
            confidence_thresholds=confidence_thresholds,
            min_authorized_fraction=args.min_authorized_fraction,
            min_pixels=args.min_pixels,
            feather_radius=args.feather_radius,
        )
        parser_identity = {
            "backend": "imported-atr18-label-map-v1",
            "automatic": args.label_status == "automatic",
        }
    # The exported 8-bit label/confidence artifacts are the durable evidence.
    # Rebuild the authoritative masks from those exact quantized bytes so a
    # later verifier can reproduce every authorization decision exactly.
    bundle = build_fine_mask_bundle_from_arrays(
        np.asarray(bundle.label_map, dtype=np.uint8),
        np.asarray(bundle.confidence_map, dtype=np.float32) / 255.0,
        label_status=bundle.label_status,
        reviewed_by=bundle.reviewed_by,
        confidence_threshold=args.confidence_threshold,
        confidence_thresholds=confidence_thresholds,
        min_authorized_fraction=args.min_authorized_fraction,
        min_pixels=args.min_pixels,
        feather_radius=args.feather_radius,
    )
    output_root = Path(args.output_dir).resolve()
    outputs = {
        f"mask_{name}": output_root / f"{name}.png" for name in bundle.masks
    }
    outputs["labels"] = output_root / "labels-atr18.png"
    outputs["confidence"] = output_root / "confidence.png"
    outputs["report"] = output_root / "fine-mask-report.json"
    inputs = [args.input, *parser_inputs]
    if args.label_map:
        inputs.append(args.label_map)
    if args.confidence_map:
        inputs.append(args.confidence_map)
    if args.confidence_policy:
        inputs.append(args.confidence_policy)
    input_hashes = {str(Path(path)): file_hash(path) for path in inputs}
    with ArtifactTransaction(outputs, inputs=inputs, overwrite=args.overwrite) as job:
        save_mask(bundle.label_map, job.staged["labels"])
        save_mask(bundle.confidence_map, job.staged["confidence"])
        for name, mask in bundle.masks.items():
            save_mask(mask, job.staged[f"mask_{name}"])
        if any(file_hash(path) != digest for path, digest in input_hashes.items()):
            raise RuntimeError("A fine-mask input changed during processing")
        payload = {
            "schema_version": 3,
            "fine_mask_schema": bundle.schema,
            "authorization_contract": "recompute-from-label-confidence-policy-v1",
            "status": "review",
            "accepted": False,
            "label_status": bundle.label_status,
            "reviewed_by": bundle.reviewed_by,
            "source": {
                "path": str(Path(args.input).resolve()),
                "sha256": file_hash(args.input),
                "canonical_size": list(source.size),
                "image_info": asdict(source_info),
            },
            "parser_identity": parser_identity,
            "label_map_input": (
                {
                    "path": str(Path(args.label_map).resolve()),
                    "sha256": file_hash(args.label_map),
                }
                if args.label_map
                else None
            ),
            "confidence_map": (
                {
                    "path": str(Path(args.confidence_map).resolve()),
                    "sha256": file_hash(args.confidence_map),
                }
                if args.confidence_map
                else None
            ),
            "regions": bundle.regions,
            "diagnostics": bundle.diagnostics,
            "input_hashes": input_hashes,
            "artifacts": job.artifact_records(),
            "publication": "staged-validated-report-last",
            "note": "Mask usability is not image quality approval; colour edits remain review candidates.",
        }
        atomic_json(job.staged["report"], payload)
        job.commit()
    usable = [name for name, data in bundle.regions.items() if data["usable_for_colour"]]
    print(f"Fine masks (REVIEW): {output_root}")
    print(f"Usable regions: {', '.join(usable) if usable else 'none'}")
    print(f"Report: {outputs['report']}")
    return 0


def _validate_fine(args: argparse.Namespace) -> int:
    inputs = [args.predicted_label_map, args.truth_label_map]
    output = Path(args.report)
    validate_artifact_paths(inputs, [output], overwrite=args.overwrite)
    report = validate_fine_label_files(
        args.predicted_label_map,
        args.truth_label_map,
        required_regions=tuple(args.required_region or ()),
        min_iou=args.min_iou,
        min_boundary_f1=args.min_boundary_f1,
        max_cross_role_leakage=args.max_cross_role_leakage,
        boundary_tolerance=args.boundary_tolerance,
    )
    with ArtifactTransaction({"report": output}, inputs=inputs, overwrite=args.overwrite) as job:
        atomic_json(job.staged["report"], report)
        if any(
            file_hash(path) != report["evidence"][key]["sha256"]
            for path, key in (
                (args.predicted_label_map, "predicted"),
                (args.truth_label_map, "reviewed_truth"),
            )
        ):
            raise RuntimeError("A fine validation input changed during processing")
        job.commit()
    print(f"ATR18 validation: {'PASS' if report['checks_passed'] else 'REVIEW'}")
    print(f"Report: {output}")
    print("Claim boundary: supplied prediction versus supplied reviewed truth only")
    return 3 if args.strict_quality_exit and not report["checks_passed"] else 0


def _verify_fine_mask_authorization(
    report_path: str | Path,
    mask_path: str | Path,
    image_path: str | Path,
    region: str,
    image_size: tuple[int, int],
) -> dict[str, object]:
    report_file = Path(report_path).resolve()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if (
        report.get("fine_mask_schema") != "atr18-v1"
        or report.get("schema_version") != 3
        or report.get("authorization_contract")
        != "recompute-from-label-confidence-policy-v1"
    ):
        raise ValueError("Fine-mask report schema is unsupported")
    source = report.get("source", {})
    if source.get("sha256") != file_hash(image_path):
        raise ValueError("Fine-mask report is not bound to the current canonical source image")
    if source.get("canonical_size") != list(image_size):
        raise ValueError("Fine-mask report canonical geometry does not match the image")

    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Fine-mask report has no artifact evidence")

    def evidence_path(role: str) -> Path:
        record = artifacts.get(role)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"Fine-mask report is missing artifact evidence: {role}")
        candidate = Path(record["path"])
        if not candidate.is_absolute():
            candidate = report_file.parent / candidate
        if candidate.is_symlink():
            raise ValueError("Fine-mask evidence must not be a symbolic link")
        candidate = candidate.resolve()
        if candidate.parent != report_file.parent or not candidate.is_file():
            raise ValueError("Fine-mask evidence must be regular files beside its report")
        if record.get("sha256") != file_hash(candidate):
            raise ValueError(f"Fine-mask artifact hash mismatch: {role}")
        return candidate

    labels_path = evidence_path("labels")
    confidence_path = evidence_path("confidence")
    recorded_mask_path = evidence_path(f"mask_{region}")
    supplied_mask_path = Path(mask_path).resolve()
    if recorded_mask_path != supplied_mask_path or file_hash(recorded_mask_path) != file_hash(
        supplied_mask_path
    ):
        raise ValueError("Fine mask bytes do not match the authorization evidence")

    label_status = report.get("label_status")
    if label_status == "reviewed" and not report.get("reviewed_by"):
        raise ValueError("Reviewed mask report is missing reviewer identity")
    if label_status not in {"automatic", "reviewed"}:
        raise ValueError("Fine mask report has an invalid label status")

    diagnostics = report.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("Fine-mask report is missing its reconstruction policy")
    thresholds = diagnostics.get("confidence_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("Fine-mask confidence policy is missing")
    try:
        reconstruction_policy = {
            "confidence_threshold": float(diagnostics["confidence_threshold"]),
            "confidence_thresholds": {
                str(name): float(value) for name, value in thresholds.items()
            },
            "min_authorized_fraction": float(
                diagnostics["min_authorized_fraction"]
            ),
            "min_pixels": int(diagnostics["min_pixels"]),
            "feather_radius": float(diagnostics["feather_radius"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Fine-mask reconstruction policy is invalid") from error
    rebuilt = build_fine_mask_bundle(
        labels_path,
        image_size,
        confidence_map_path=confidence_path,
        label_status=str(label_status),
        reviewed_by=report.get("reviewed_by"),
        **reconstruction_policy,
    )
    metrics = rebuilt.regions.get(region)
    if not isinstance(metrics, dict) or not metrics.get("usable_for_colour"):
        raise ValueError(f"Fine region is not authorized for colour work: {region}")
    supplied = load_mask(supplied_mask_path, image_size)
    if not np.array_equal(np.asarray(supplied), np.asarray(rebuilt.masks[region])):
        raise ValueError("Fine mask cannot be reproduced from label/confidence evidence")
    if report.get("regions", {}).get(region) != metrics:
        raise ValueError("Fine-mask region metrics cannot be reproduced")
    return {
        "report": str(report_file),
        "report_sha256": file_hash(report_path),
        "mask_sha256": file_hash(mask_path),
        "label_status": label_status,
        "reviewed_by": report.get("reviewed_by"),
        "metrics": metrics,
        "recomputed": True,
        "evidence_files": [
            str(labels_path),
            str(confidence_path),
            str(recorded_mask_path),
        ],
    }


def _precision_match(args: argparse.Namespace) -> int:
    inputs = [
        args.input,
        args.reference,
        args.source_mask,
        args.reference_mask,
        args.source_mask_report,
        args.reference_mask_report,
        args.sku_profile,
    ]
    if args.protected_mask:
        inputs.append(args.protected_mask)
    report_path = Path(args.report or (str(args.output) + ".report.json"))
    outputs = {"candidate": Path(args.output), "report": report_path}
    if args.preview:
        outputs["preview"] = Path(args.preview)
    validate_master_path(args.output)
    validate_artifact_paths(inputs, outputs.values(), overwrite=args.overwrite)
    reference_sha256 = file_hash(args.reference)
    if args.reference_id.casefold() != reference_sha256.casefold():
        raise ValueError("reference_id must equal the current reference image SHA-256")
    target_evidence = verify_region_target_evidence(
        args.sku_profile,
        object_id=args.object_id,
        sku_role=args.sku_role,
        reference_policy=args.reference_policy,
        reference_sha256=reference_sha256,
    )
    source, source_info = load_srgb(args.input)
    reference, reference_info = load_srgb(args.reference)
    source_mask = load_mask(args.source_mask, source.size)
    reference_mask = load_mask(args.reference_mask, reference.size)
    protected_mask = (
        load_mask(args.protected_mask, source.size) if args.protected_mask else None
    )
    source_authorization = _verify_fine_mask_authorization(
        args.source_mask_report, args.source_mask, args.input, args.region, source.size
    )
    reference_authorization = _verify_fine_mask_authorization(
        args.reference_mask_report,
        args.reference_mask,
        args.reference,
        args.region,
        reference.size,
    )
    for path in (
        *source_authorization["evidence_files"],
        *reference_authorization["evidence_files"],
    ):
        if path not in inputs:
            inputs.append(path)
    validate_artifact_paths(inputs, outputs.values(), overwrite=args.overwrite)
    input_hashes = {str(Path(path)): file_hash(path) for path in inputs}
    corrected, quality = precision_region_match(
        source,
        reference,
        source_mask,
        reference_mask,
        protected_mask=protected_mask,
        region=args.region,
        target_policy=RegionTargetPolicy(
            object_id=args.object_id,
            sku_role=args.sku_role,
            reference_policy=args.reference_policy,
            reference_id=args.reference_id,
        ),
        strength=args.strength,
        luminance_cap=args.luminance_cap,
        chroma_cap=args.chroma_cap,
    )
    with ArtifactTransaction(outputs, inputs=inputs, overwrite=args.overwrite) as job:
        verification = save_srgb(corrected, job.staged["candidate"])
        if args.preview:
            save_comparison(source, corrected, reference, job.staged["preview"])
        if any(file_hash(path) != digest for path, digest in input_hashes.items()):
            raise RuntimeError("A precision-match input changed during processing")
        payload = {
            **quality,
            "schema_version": 1,
            "input": str(Path(args.input).resolve()),
            "reference": str(Path(args.reference).resolve()),
            "output": str(Path(args.output).resolve()),
            "source_info": asdict(source_info),
            "reference_info": asdict(reference_info),
            "source_mask_authorization": source_authorization,
            "reference_mask_authorization": reference_authorization,
            "region_target_evidence": target_evidence,
            "protected_mask": (
                {
                    "path": str(Path(args.protected_mask).resolve()),
                    "sha256": file_hash(args.protected_mask),
                }
                if args.protected_mask
                else None
            ),
            "input_hashes": input_hashes,
            "export_verification": verification,
            "engine_identity": _engine_identity(),
            "artifacts": job.artifact_records(),
            "publication": "staged-validated-report-last",
        }
        atomic_json(job.staged["report"], payload)
        job.commit()
    print(f"Precision region candidate (REVIEW, not approved): {args.output}")
    print(f"Region: {args.region}")
    print(f"Target distance: {quality['distance_before']:.3f} -> {quality['distance_after']:.3f}")
    print(f"Outside authorized changed pixels: {quality['outside_authorized_changed_pixels']}")
    print(f"Report: {report_path}")
    return 3 if args.strict_quality_exit else 0


def _c1_analyse(args: argparse.Namespace) -> int:
    inputs = [args.input, args.reference]
    inputs.extend(
        value
        for value in (
            args.source_region_mask,
            args.reference_region_mask,
            args.source_neutral_mask,
            args.reference_neutral_mask,
        )
        if value
    )
    report_path = Path(args.report)
    validate_artifact_paths(inputs, [report_path], overwrite=args.overwrite)
    input_hashes = {str(Path(path)): file_hash(path) for path in inputs}
    source, source_info = load_srgb(args.input)
    reference, reference_info = load_srgb(args.reference)
    source_region = (
        load_mask(args.source_region_mask, source.size) if args.source_region_mask else None
    )
    reference_region = (
        load_mask(args.reference_region_mask, reference.size)
        if args.reference_region_mask
        else None
    )
    source_neutral = (
        load_mask(args.source_neutral_mask, source.size) if args.source_neutral_mask else None
    )
    reference_neutral = (
        load_mask(args.reference_neutral_mask, reference.size)
        if args.reference_neutral_mask
        else None
    )
    config = replace(C1_CONFIG, max_edge=args.max_edge)
    payload = analyse_relative_illumination(
        source,
        reference,
        source_region_mask=source_region,
        reference_region_mask=reference_region,
        source_neutral_mask=source_neutral,
        reference_neutral_mask=reference_neutral,
        neutral_evidence=args.neutral_evidence,
        comparison_evidence=args.comparison_evidence,
        region_name=args.region_name,
        config=config,
    )
    payload.update(
        source=str(Path(args.input).resolve()),
        reference=str(Path(args.reference).resolve()),
        source_info=asdict(source_info),
        reference_info=asdict(reference_info),
        input_hashes=input_hashes,
        analyzer_identity=c1_identity(config),
        publication="staged-validated-report-last",
    )
    with ArtifactTransaction(
        {"report": report_path}, inputs=inputs, overwrite=args.overwrite
    ) as job:
        if any(file_hash(path) != digest for path, digest in input_hashes.items()):
            raise RuntimeError("A C1 observer input changed during analysis")
        payload["artifacts"] = job.artifact_records()
        atomic_json(job.staged["report"], payload)
        job.commit()
    print("C1 relative illumination analysis: REVIEW (read-only)")
    print(f"Analysis status: {payload['analysis_status']}")
    print(
        "Exposure-like stops: "
        f"{payload['exposure'].get('relative_exposure_like_stops')}"
    )
    print(f"Warm/cool: {payload['whitepoint']['warm_cool_direction']}")
    print("Pixel output changed: False; A0 changed or vetoed: False")
    print(f"Report: {report_path}")
    return 3 if args.strict_quality_exit and payload["analysis_status"] != "valid" else 0


def _mask_options(parser):
    parser.add_argument("--mask-backend", choices=("auto", "vision", "heuristic"), default="auto")
    parser.add_argument("--mask-quality", choices=("accurate", "balanced", "fast"), default="accurate")


def _transfer_options(parser):
    _mask_options(parser)
    parser.add_argument("--strength", type=float, default=0.85, help="有限强度 0..1")
    parser.add_argument("--path", choices=("auto", "global", "surface"), default="auto")
    parser.add_argument("--mode", choices=("background", "both"), default="background",
                        help="默认仅背景；both 为背景驱动人物调整实验，不是独立肤色匹配")
    parser.add_argument("--overwrite", action="store_true", help="允许重算/替换输出，绝不允许覆盖输入")
    parser.add_argument(
        "--strict-quality-exit",
        action="store_true",
        help="把“已生成但待复核”作为退出码 3；默认生成成功退出 0",
    )


def _single_outputs(parser):
    parser.add_argument("--output", required=True, help="待复核 PNG/TIFF 无损母版")
    parser.add_argument("--report", help="默认 <output>.report.json")
    parser.add_argument("--preview", help="对比预览，可用 JPEG")
    parser.add_argument("--mask-output", help="输出实际源图背景蒙版 PNG；默认 <output>.mask.png")
    parser.add_argument("--background-mask", help="输入背景蒙版，白色=背景；必须与转正后图像同尺寸")
    parser.add_argument("--protected-mask", help="输入保护蒙版，白色=完全不改，黑色=允许调整")
    _transfer_options(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="batch-color", description="参考追色实验工具；所有候选必须复核")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="检查本机环境")
    profile = commands.add_parser("profile", help="创建可复算背景标准 .bcp（包含参考图和蒙版，不可公开上传）")
    profile.add_argument("--reference", required=True)
    profile.add_argument("--reference-mask", help="参考图背景蒙版")
    profile.add_argument("--reference-mask-output", help="另存实际参考蒙版 PNG；标准包内始终保存")
    profile.add_argument("--name", required=True)
    profile.add_argument("--output", required=True)
    profile.add_argument("--overwrite", action="store_true")
    _mask_options(profile)
    verify = commands.add_parser("verify-profile", help="只读检查标准包并复算证据；不批准画质")
    verify.add_argument("--profile", required=True)

    match = commands.add_parser("match", help="向参考图追色并输出待复核候选")
    match.add_argument("--input", required=True)
    match.add_argument("--reference", required=True)
    match.add_argument("--reference-mask", help="参考图背景蒙版")
    match.add_argument("--reference-mask-output", help="实际参考蒙版；默认 <output>.reference-mask.png")
    match.add_argument("--profile-name")
    match.add_argument("--profile-output", help="可复算标准 .bcp；默认 <output>.reference.bcp")
    _single_outputs(match)

    apply = commands.add_parser("apply", help="应用背景 Profile")
    apply.add_argument("--input", required=True)
    apply.add_argument("--profile", required=True)
    _single_outputs(apply)

    batch = commands.add_parser("batch", help="串行处理；缓存不等于质量通过")
    batch.add_argument("--input", required=True)
    batch.add_argument("--profile", required=True)
    batch.add_argument("--output", required=True)
    batch.add_argument("--recursive", action="store_true")
    batch.add_argument("--no-previews", action="store_true")
    _transfer_options(batch)

    sku_pilot = commands.add_parser("sku-pilot", help="按 SKU 双锚点生成整套待复核候选")
    sku_pilot.add_argument("--dataset-root", required=True)
    sku_pilot.add_argument("--sku", required=True)
    sku_pilot.add_argument("--output-root", help="默认 <dataset-root>/校色输出")
    sku_pilot.add_argument("--run-name", required=True)
    sku_pilot.add_argument(
        "--pipeline-mode", choices=("person-background", "semantic"), default="person-background"
    )
    sku_pilot.add_argument("--garment-kind", choices=("top", "dress", "bottom", "set"), default="set")
    sku_pilot.add_argument("--garment-hint", choices=("light", "dark", "midtone", "any"), default="any")
    sku_pilot.add_argument("--background-strength", type=float)
    sku_pilot.add_argument("--garment-strength", type=float, default=0.58)
    sku_pilot.add_argument("--person-strength", type=float)
    sku_pilot.add_argument("--skin-strength", type=float, default=0.22)
    sku_pilot.add_argument("--hair-strength", type=float, default=0.12)
    sku_pilot.add_argument(
        "--parser-backend", choices=("mediapipe", "auto", "none"), default="mediapipe"
    )
    sku_pilot.add_argument(
        "--pose-backend", choices=("vision", "auto", "none"), default="vision"
    )
    sku_pilot.add_argument(
        "--garment-anchor", help="可选：固定使用本 SKU 某张成品动作作为服装色标准"
    )
    sku_pilot.add_argument(
        "--set-color-tolerance", type=float, default=A0_BASELINE.set_color_tolerance
    )
    sku_pilot.add_argument(
        "--replace-output",
        action="store_true",
        help="替换该 SKU 的当前校色产物，保留不相关旧测试目录",
    )
    sku_pilot.add_argument(
        "--strict-quality-exit",
        action="store_true",
        help="把待人工复核状态作为退出码 3；默认候选包生成成功退出 0",
    )
    _mask_options(sku_pilot)
    sku_pilot.set_defaults(mask_backend="vision", mask_quality="accurate")

    sku_init = commands.add_parser("sku-init", help="创建持久 SKU Profile，记录场景、产品图和候选服装锚点")
    sku_init.add_argument("--dataset-root", required=True)
    sku_init.add_argument("--sku", required=True)
    sku_init.add_argument("--output-root", help="默认 <dataset-root>/校色输出")
    sku_init.add_argument("--product-anchor", help="产品图文件名；A0 中只作证据")
    sku_init.add_argument("--garment-anchor", help="已选定的成品动作文件名")
    sku_init.add_argument("--confirm-product", action="store_true")
    sku_init.add_argument("--confirm-garment", action="store_true")
    sku_init.add_argument("--overwrite", action="store_true")

    sku_review = commands.add_parser("sku-review", help="对 SKU 候选包做可追溯的批准或驳回")
    sku_review.add_argument("--output-root", required=True)
    sku_review.add_argument("--sku", required=True)
    sku_review.add_argument("--decision", choices=("approve", "reject"), required=True)
    sku_review.add_argument("--reviewer", required=True)
    sku_review.add_argument("--reason", required=True)
    sku_review.add_argument("--replace-approved", action="store_true")

    fine_masks = commands.add_parser(
        "fine-masks",
        help="验证并导出 ATR18 精细部位蒙版；自动结果必须带置信度，默认不进入 A0",
    )
    fine_masks.add_argument("--input", required=True)
    fine_source = fine_masks.add_mutually_exclusive_group(required=True)
    fine_source.add_argument("--label-map", help="与转正图像同尺寸的 ATR18 0..17 标签图")
    fine_source.add_argument(
        "--model-dir", help="本地、已审计许可证的 ATR18 SegFormer safetensors 模型目录"
    )
    fine_masks.add_argument("--confidence-map", help="自动标签必需：同尺寸 8 位灰度置信度图")
    fine_masks.add_argument("--label-status", choices=("automatic", "reviewed"), default="automatic")
    fine_masks.add_argument("--reviewed-by", help="label-status=reviewed 时必需")
    fine_masks.add_argument("--confidence-threshold", type=float, default=0.82)
    fine_masks.add_argument(
        "--confidence-policy",
        help="可选 JSON：按 ATR18 类别/分组覆盖置信度阈值，文件会纳入证据哈希",
    )
    fine_masks.add_argument("--min-authorized-fraction", type=float, default=0.90)
    fine_masks.add_argument("--min-pixels", type=int, default=128)
    fine_masks.add_argument("--feather-radius", type=float, default=2.0)
    fine_masks.add_argument("--fine-device", choices=("cpu", "mps"), default="cpu")
    fine_masks.add_argument("--fine-max-edge", type=int, default=768)
    fine_masks.add_argument("--fine-threads", type=int, default=2)
    fine_masks.add_argument("--output-dir", required=True)
    fine_masks.add_argument("--overwrite", action="store_true")

    validate_fine = commands.add_parser(
        "validate-fine",
        help="将 ATR18 预测标签与人工复核真值逐像素对比；不批准成图画质",
    )
    validate_fine.add_argument("--predicted-label-map", required=True)
    validate_fine.add_argument("--truth-label-map", required=True)
    validate_fine.add_argument(
        "--required-region",
        choices=region_names()[:-1],
        action="append",
        help="必须达到阈值的类别/分组，可重复；默认验证真值中存在的主要分组",
    )
    validate_fine.add_argument("--min-iou", type=float, default=0.80)
    validate_fine.add_argument("--min-boundary-f1", type=float, default=0.70)
    validate_fine.add_argument("--max-cross-role-leakage", type=float, default=0.01)
    validate_fine.add_argument("--boundary-tolerance", type=int, default=2)
    validate_fine.add_argument("--report", required=True)
    validate_fine.add_argument("--overwrite", action="store_true")
    validate_fine.add_argument("--strict-quality-exit", action="store_true")

    precision = commands.add_parser(
        "precision-match",
        help="只在已授权精细部位内进行有界追色；生成结果始终待人工复核",
    )
    precision.add_argument("--input", required=True)
    precision.add_argument("--reference", required=True)
    precision.add_argument("--source-mask", required=True)
    precision.add_argument("--reference-mask", required=True)
    precision.add_argument("--source-mask-report", required=True)
    precision.add_argument("--reference-mask-report", required=True)
    precision.add_argument(
        "--protected-mask",
        help="可选受保护对象蒙版，白色区域从修改权限中扣除（如书、商标、配饰）",
    )
    precision.add_argument(
        "--sku-profile",
        required=True,
        help="带指纹且包含已确认场景/服装锚点的 sku-profile.json",
    )
    precision.add_argument("--region", choices=region_names(), required=True)
    precision.add_argument("--object-id", required=True, help="被调整对象的稳定标识，如 SKU+部位")
    precision.add_argument("--sku-role", choices=SKU_ROLES, required=True)
    precision.add_argument("--reference-policy", choices=REFERENCE_POLICIES, required=True)
    precision.add_argument(
        "--reference-id",
        required=True,
        help="参考图文件的 SHA-256；防止角色策略与实际参考图脱钩",
    )
    precision.add_argument("--strength", type=float, default=0.55)
    precision.add_argument("--luminance-cap", type=float, default=0.045)
    precision.add_argument("--chroma-cap", type=float, default=0.028)
    precision.add_argument("--output", required=True)
    precision.add_argument("--report")
    precision.add_argument("--preview")
    precision.add_argument("--overwrite", action="store_true")
    precision.add_argument("--strict-quality-exit", action="store_true")

    c1 = commands.add_parser(
        "c1-analyse",
        help="只读分解表观冷暖、曝光型增益和显示色调；不修改像素或A0",
    )
    c1.add_argument("--input", required=True)
    c1.add_argument("--reference", required=True)
    c1.add_argument("--source-region-mask", help="源图可比区域蒙版，白色=分析")
    c1.add_argument("--reference-region-mask", help="参考图可比区域蒙版，白色=分析")
    c1.add_argument("--source-neutral-mask", help="源图已确认中性证据蒙版")
    c1.add_argument("--reference-neutral-mask", help="参考图已确认中性证据蒙版")
    c1.add_argument(
        "--neutral-evidence",
        choices=("automatic", "human_confirmed", "same_entity"),
        default="automatic",
        help="显式证据必须同时提供源/参考中性蒙版",
    )
    c1.add_argument(
        "--comparison-evidence",
        choices=("automatic", "human_confirmed", "same_surface"),
        default="automatic",
        help="曝光候选只在显式确认可比表面且同时提供两张区域蒙版时放行",
    )
    c1.add_argument("--region-name", default="scene")
    c1.add_argument("--max-edge", type=int, default=C1_CONFIG.max_edge)
    c1.add_argument("--report", required=True)
    c1.add_argument("--overwrite", action="store_true")
    c1.add_argument("--strict-quality-exit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"doctor": lambda _: _doctor(), "profile": _create_profile,
                "match": _match, "apply": _apply, "batch": _batch,
                "verify-profile": _verify_profile, "sku-pilot": _sku_pilot,
                "sku-init": _sku_init, "sku-review": _sku_review,
                "fine-masks": _fine_masks, "validate-fine": _validate_fine,
                "precision-match": _precision_match, "c1-analyse": _c1_analyse}
    try:
        return handlers[args.command](args)
    except EXPECTED_ERRORS as error:
        print(f"batch-color: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
