"""Persistent SKU truth profiles and explicit human review records."""
from __future__ import annotations

import json
import os
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from batch_color.baseline import A0_BASELINE
from batch_color.planning import validate_shadow_plan
from batch_color.safety import atomic_json, file_hash, payload_hash
from batch_color.sku import SKUInput, validate_inputs_unchanged


PROFILE_SCHEMA = 1
REVIEW_SCHEMA = 1
CANDIDATE_DIRECTORY = "待复核候选"
APPROVED_DIRECTORY = "已批准成品"
REJECTED_DIRECTORY = "已拒绝"
REVIEWS_DIRECTORY = "审核记录"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: str | Path, root: str | Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _select_member(paths: tuple[str, ...], value: str | None, label: str) -> str | None:
    if value is None:
        return None
    candidate = Path(value)
    matches = [
        path
        for path in paths
        if Path(path).name.casefold() == candidate.name.casefold()
        or Path(path).resolve() == candidate.resolve()
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} must identify exactly one scanned SKU image: {value}")
    return matches[0]


def _profile_fingerprint(payload: dict[str, object]) -> str:
    clean = {key: value for key, value in payload.items() if key != "profile_fingerprint"}
    return payload_hash(clean)


def create_sku_profile(
    manifest: SKUInput,
    *,
    product_anchor: str | None = None,
    garment_anchor: str | None = None,
    confirm_product: bool = False,
    confirm_garment: bool = False,
    auto_garment_candidate: str | None = None,
) -> dict[str, object]:
    product = _select_member(manifest.product_images, product_anchor, "product anchor")
    if product is None and manifest.product_images:
        product = manifest.product_images[0]
    garment = _select_member(manifest.targets, garment_anchor, "garment anchor")
    if garment is None and auto_garment_candidate is not None:
        garment = _select_member(manifest.targets, auto_garment_candidate, "auto garment candidate")
    if confirm_product and product is None:
        raise ValueError("Cannot confirm product truth: this SKU has no product image")
    if confirm_garment and garment is None:
        raise ValueError("Cannot confirm garment anchor without a target image")
    root = manifest.directory
    payload: dict[str, object] = {
        "schema_version": PROFILE_SCHEMA,
        "profile_kind": "sku-workflow-and-truth-profile",
        "sku_id": manifest.sku,
        "mode": "look-consistency",
        "capability_claim": "look_consistency_only_not_physical_product_colour",
        "created_at": _now(),
        "scene_reference": {
            "file": _relative(manifest.scene_reference, root),
            "sha256": manifest.input_hashes[manifest.scene_reference],
            "confirmed": True,
        },
        "product_truth": {
            "status": "confirmed" if confirm_product else ("candidate" if product else "unavailable"),
            "anchor_file": _relative(product, root) if product else None,
            "anchor_sha256": manifest.input_hashes[product] if product else None,
            "confirmed": bool(confirm_product),
            "images": [
                {
                    "file": _relative(path, root),
                    "sha256": manifest.input_hashes[path],
                }
                for path in manifest.product_images
            ],
            "a0_effect": "evidence_only_not_applied_to_colour_target",
        },
        "garment_anchor": {
            "status": "confirmed" if confirm_garment else ("candidate" if garment else "unavailable"),
            "file": _relative(garment, root) if garment else None,
            "sha256": manifest.input_hashes[garment] if garment else None,
            "confirmed": bool(confirm_garment),
        },
        "mask_overrides": {},
        "baseline": {
            "id": A0_BASELINE.baseline_id,
            "fingerprint": A0_BASELINE.fingerprint,
        },
        "residual_policy": "disabled_a0_only",
        "qa_policy": "a0-look-consistency-review-v1",
    }
    payload["profile_fingerprint"] = _profile_fingerprint(payload)
    return payload


def load_sku_profile(path: str | Path, manifest: SKUInput) -> dict[str, object]:
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or profile.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError("Unsupported SKU profile")
    if profile.get("sku_id") != manifest.sku:
        raise ValueError("SKU profile belongs to a different SKU")
    if profile.get("profile_fingerprint") != _profile_fingerprint(profile):
        raise ValueError("SKU profile fingerprint mismatch")
    root = Path(manifest.directory)
    evidence = [profile.get("scene_reference")]
    product = profile.get("product_truth")
    if isinstance(product, dict):
        evidence.extend(product.get("images", []))
        recorded_products = {
            str(item.get("file"))
            for item in product.get("images", [])
            if isinstance(item, dict) and item.get("file")
        }
        current_products = {
            _relative(path, root) for path in manifest.product_images
        }
        if recorded_products != current_products:
            raise RuntimeError("SKU product evidence set changed; rebuild or review the profile")
    garment = profile.get("garment_anchor")
    if isinstance(garment, dict) and garment.get("file"):
        evidence.append(garment)
    for record in evidence:
        if not isinstance(record, dict) or not record.get("file"):
            continue
        source = (root / str(record["file"])).resolve()
        if not source.is_relative_to(root.resolve()) or not source.is_file():
            raise ValueError(f"SKU profile evidence is missing or outside the SKU: {source}")
        if file_hash(source) != record.get("sha256"):
            raise RuntimeError(f"SKU profile evidence changed: {source}")
    return profile


def profile_confirmed_garment(profile: dict[str, object] | None) -> str | None:
    if not profile:
        return None
    garment = profile.get("garment_anchor")
    if isinstance(garment, dict) and garment.get("confirmed") is True:
        value = garment.get("file")
        return str(value) if value else None
    return None


def verify_region_target_evidence(
    profile_path: str | Path,
    *,
    object_id: str,
    sku_role: str,
    reference_policy: str,
    reference_sha256: str,
) -> dict[str, object]:
    """Verify that a declared region target is backed by a fingerprinted SKU profile.

    This intentionally verifies only profile identity and the selected target
    record.  Full source-set verification still belongs to ``load_sku_profile``
    when a current SKU manifest is available.
    """
    path = Path(profile_path).resolve()
    profile = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or profile.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError("Unsupported SKU profile evidence")
    if profile.get("profile_fingerprint") != _profile_fingerprint(profile):
        raise ValueError("SKU profile evidence fingerprint mismatch")
    sku_id = str(profile.get("sku_id", ""))
    if not sku_id or not object_id.startswith(f"{sku_id}:"):
        raise ValueError("object_id must begin with the SKU profile id followed by ':'")
    if reference_policy == "scene_reference":
        if sku_role not in {"background", "skin_identity", "hair_identity"}:
            raise ValueError("Scene reference cannot authorize this SKU role")
        target = profile.get("scene_reference")
        target_name = "scene_reference"
    elif reference_policy == "sku_approved_anchor":
        if sku_role not in {"target_sku", "accessory"}:
            raise ValueError("SKU anchor cannot authorize this SKU role")
        target = profile.get("garment_anchor")
        target_name = "garment_anchor"
    else:
        raise ValueError("Protected or source-identity policies do not authorize transfer")
    if not isinstance(target, dict) or target.get("confirmed") is not True:
        raise ValueError(f"SKU profile {target_name} is not confirmed")
    if str(target.get("sha256", "")).casefold() != reference_sha256.casefold():
        raise ValueError(f"Reference image does not match confirmed {target_name}")
    return {
        "profile": str(path),
        "profile_sha256": file_hash(path),
        "profile_fingerprint": profile["profile_fingerprint"],
        "sku_id": sku_id,
        "target_record": target_name,
        "target_file": target.get("file"),
        "target_sha256": target.get("sha256"),
        "confirmed": True,
    }


def ensure_sku_profile(
    manifest: SKUInput,
    *,
    existing_path: str | Path,
    staged_path: str | Path,
    auto_garment_candidate: str,
) -> dict[str, object]:
    existing = Path(existing_path)
    if existing.is_file():
        profile = load_sku_profile(existing, manifest)
    else:
        profile = create_sku_profile(
            manifest, auto_garment_candidate=auto_garment_candidate
        )
    atomic_json(staged_path, profile)
    return profile


def save_new_sku_profile(
    path: str | Path,
    manifest: SKUInput,
    *,
    product_anchor: str | None,
    garment_anchor: str | None,
    confirm_product: bool,
    confirm_garment: bool,
    overwrite: bool,
) -> dict[str, object]:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"SKU profile exists; use --overwrite: {destination}")
    profile = create_sku_profile(
        manifest,
        product_anchor=product_anchor,
        garment_anchor=garment_anchor,
        confirm_product=confirm_product,
        confirm_garment=confirm_garment,
    )
    atomic_json(destination, profile)
    return profile


def _verified_candidates(root: Path, summary: dict[str, object]) -> dict[str, str]:
    records: dict[str, str] = {}
    items = summary.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("SKU summary has no candidates")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Invalid SKU summary item")
        relative = Path(str(item.get("output", "")))
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Candidate is missing or outside the SKU output: {relative}")
        digest = file_hash(path)
        if digest != item.get("output_sha256"):
            raise RuntimeError(f"Candidate hash mismatch: {relative}")
        records[relative.as_posix()] = digest
    return records


@contextmanager
def _exclusive_review_lock(root: Path):
    """Serialize review/approval for one SKU and retain no silent stale bypass."""

    lock = root / ".sku-review.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"pid": os.getpid(), "created_at": _now()},
                    ensure_ascii=False,
                )
                + "\n"
            )
    except FileExistsError as error:
        raise FileExistsError(
            f"SKU review is already locked; inspect/remove a stale lock only after confirming no review is running: {lock}"
        ) from error
    try:
        yield lock
    finally:
        lock.unlink(missing_ok=True)


def _copied_candidate_hashes(
    copied_root: Path, candidates: dict[str, str]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative, expected in candidates.items():
        parts = Path(relative).parts
        if not parts or parts[0] != CANDIDATE_DIRECTORY:
            raise ValueError(f"Candidate output is outside the owned candidate directory: {relative}")
        copied = copied_root.joinpath(*parts[1:]).resolve()
        if not copied.is_relative_to(copied_root.resolve()) or not copied.is_file():
            raise RuntimeError(f"Approved staging copy is incomplete: {relative}")
        actual = file_hash(copied)
        if actual != expected:
            raise RuntimeError(f"Candidate changed while it was copied for approval: {relative}")
        result[relative] = actual
    return result


def review_sku_output(
    sku_output: str | Path,
    *,
    decision: str,
    reviewer: str,
    reason: str,
    replace_approved: bool = False,
) -> tuple[Path, dict[str, object]]:
    root = Path(sku_output).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SKU candidate package does not exist: {root}")
    with _exclusive_review_lock(root):
        return _review_sku_output_locked(
            root,
            decision=decision,
            reviewer=reviewer,
            reason=reason,
            replace_approved=replace_approved,
        )


def _review_sku_output_locked(
    root: Path,
    *,
    decision: str,
    reviewer: str,
    reason: str,
    replace_approved: bool,
) -> tuple[Path, dict[str, object]]:
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    if not reviewer.strip() or not reason.strip():
        raise ValueError("reviewer and reason are required")
    summary_path = root / "summary.json"
    manifest_path = root / "input-manifest.json"
    profile_path = root / "sku-profile.json"
    identity_path = root / "run-identity.json"
    plan_path = root / "execution-plan.json"
    if not all(
        path.is_file()
        for path in (summary_path, manifest_path, profile_path, identity_path, plan_path)
    ):
        raise FileNotFoundError("Incomplete SKU candidate package")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        manifest = SKUInput(
            sku=str(manifest_payload["sku"]),
            directory=str(manifest_payload["directory"]),
            scene_reference=str(manifest_payload["scene_reference"]),
            product_images=tuple(manifest_payload["product_images"]),
            targets=tuple(manifest_payload["targets"]),
            input_hashes=dict(manifest_payload["input_hashes"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid SKU input manifest") from error
    validate_inputs_unchanged(manifest)
    load_sku_profile(profile_path, manifest)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("sku") != manifest.sku:
        raise ValueError("Summary SKU does not match the input manifest")
    if summary.get("status") not in {"candidate", "review"} or summary.get("accepted") is not False:
        raise ValueError("Only an unapproved candidate package can be reviewed")
    for item in summary.get("items", []):
        if not isinstance(item, dict):
            raise ValueError("Invalid SKU summary item")
        source = str(item.get("input", ""))
        expected = manifest.input_hashes.get(source)
        if expected is None or item.get("input_sha256") != expected:
            raise RuntimeError("Candidate source evidence does not match the input manifest")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity_sha256 = identity.get("identity_sha256") if isinstance(identity, dict) else None
    if not isinstance(identity_sha256, str):
        raise ValueError("Run identity is missing its fingerprint")
    identity_payload = {key: value for key, value in identity.items() if key != "identity_sha256"}
    if payload_hash(identity_payload) != identity_sha256:
        raise RuntimeError("Run identity fingerprint mismatch")
    if summary.get("run_identity_sha256") != identity_sha256:
        raise RuntimeError("Summary is not bound to the run identity")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("Execution plan must be a JSON object")
    validate_shadow_plan(plan)
    if summary.get("execution_plan_sha256") != file_hash(plan_path):
        raise RuntimeError("Summary is not bound to the execution plan")
    if summary.get("execution_plan_fingerprint") != plan.get("plan_sha256"):
        raise RuntimeError("Summary execution plan fingerprint mismatch")
    candidates = _verified_candidates(root, summary)
    mask_hashes = {
        path.relative_to(root).as_posix(): file_hash(path)
        for path in sorted((root / "蒙版").rglob("*"))
        if path.is_file()
    }
    evidence_hashes = {
        "input_manifest_sha256": file_hash(manifest_path),
        "summary_sha256": file_hash(summary_path),
        "sku_profile_sha256": file_hash(profile_path),
        "run_identity_file_sha256": file_hash(identity_path),
        "execution_plan_file_sha256": file_hash(plan_path),
    }
    review_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    record: dict[str, object] = {
        "schema_version": REVIEW_SCHEMA,
        "review_id": review_id,
        "sku": summary.get("sku"),
        "decision": "approved" if decision == "approve" else "rejected",
        "accepted": decision == "approve",
        "reviewer": reviewer.strip(),
        "reason": reason.strip(),
        "reviewed_at": _now(),
        **evidence_hashes,
        "candidate_outputs": candidates,
        "mask_bundle_sha256": payload_hash(mask_hashes),
        "baseline": summary.get("baseline"),
        "run_identity_sha256": identity_sha256,
    }
    review_directory = root / REVIEWS_DIRECTORY
    record_path = review_directory / f"{review_id}.json"
    shareable_path = review_directory / f"{review_id}.shareable.json"
    status_path = root / "review-status.json"
    transaction = root / f".review-{review_id}.staging"
    transaction.mkdir(parents=True, exist_ok=False)
    destination: Path | None = None
    backup: Path | None = None
    old_status = transaction / "old-review-status.json"
    published_record = published_shareable = published_status = published_decision = False
    try:
        decision_staged = transaction / "decision"
        if decision == "approve":
            source = root / CANDIDATE_DIRECTORY
            destination = root / APPROVED_DIRECTORY
            if destination.exists() and not replace_approved:
                raise FileExistsError(
                    f"Approved output exists; use --replace-approved: {destination}"
                )
            shutil.copytree(source, decision_staged)
            record["approved_outputs"] = _copied_candidate_hashes(
                decision_staged, candidates
            )
        else:
            destination = root / REJECTED_DIRECTORY / review_id
            decision_staged.mkdir()
            if (root / "整套对照.jpg").is_file():
                shutil.copy2(
                    root / "整套对照.jpg",
                    decision_staged / "整套对照.jpg",
                )
            atomic_json(decision_staged / "rejection.json", record)

        # Recheck every file that was validated before approval. Cooperative
        # locking prevents our tools from racing; this second check also catches
        # external edits during the verification/copy interval.
        current_evidence = {
            "input_manifest_sha256": file_hash(manifest_path),
            "summary_sha256": file_hash(summary_path),
            "sku_profile_sha256": file_hash(profile_path),
            "run_identity_file_sha256": file_hash(identity_path),
            "execution_plan_file_sha256": file_hash(plan_path),
        }
        if current_evidence != evidence_hashes:
            raise RuntimeError("SKU review evidence changed during approval")
        current_masks = {
            path.relative_to(root).as_posix(): file_hash(path)
            for path in sorted((root / "蒙版").rglob("*"))
            if path.is_file()
        }
        if current_masks != mask_hashes:
            raise RuntimeError("SKU mask evidence changed during approval")
        if _verified_candidates(root, summary) != candidates:
            raise RuntimeError("SKU candidates changed during approval")

        shareable = {key: value for key, value in record.items() if key != "reviewer"}
        status = {
            "status": record["decision"],
            "accepted": record["accepted"],
            "latest_review": record_path.relative_to(root).as_posix(),
            "reviewed_at": record["reviewed_at"],
        }
        atomic_json(transaction / "record.json", record)
        atomic_json(transaction / "shareable.json", shareable)
        atomic_json(transaction / "status.json", status)

        review_directory.mkdir(parents=True, exist_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if status_path.is_file():
            shutil.copy2(status_path, old_status)
        if destination.exists():
            backup = transaction / "old-decision"
            destination.rename(backup)
        decision_staged.rename(destination)
        published_decision = True
        if decision == "approve" and _copied_candidate_hashes(
            destination, candidates
        ) != record["approved_outputs"]:
            raise RuntimeError("Published approved files do not match the review record")
        os.replace(transaction / "record.json", record_path)
        published_record = True
        os.replace(transaction / "shareable.json", shareable_path)
        published_shareable = True
        os.replace(transaction / "status.json", status_path)
        published_status = True
    except BaseException:
        if published_status:
            if old_status.is_file():
                os.replace(old_status, status_path)
            else:
                status_path.unlink(missing_ok=True)
        if published_shareable:
            shareable_path.unlink(missing_ok=True)
        if published_record:
            record_path.unlink(missing_ok=True)
        if published_decision and destination is not None and destination.exists():
            shutil.rmtree(destination)
        if backup and backup.exists() and destination is not None:
            backup.rename(destination)
        raise
    finally:
        shutil.rmtree(transaction, ignore_errors=True)
    return record_path, record
