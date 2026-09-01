"""Repair legacy SKU report paths after flattening outputs; never edit images."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

from batch_color.safety import atomic_json, file_hash


MANAGED_MARKERS = ("校色成品", "蒙版", "报告", "预览")


def managed_relative(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    parts = path.parts
    for marker in MANAGED_MARKERS:
        if marker in parts:
            index = parts.index(marker)
            return Path(*parts[index:]).as_posix()
    raise ValueError(f"Absolute artifact path is outside managed output folders: {value}")


def _repair_item(item: dict[str, object], sku_root: Path) -> None:
    item["output"] = managed_relative(str(item["output"]))
    mask_paths = item.get("mask_paths", {})
    if not isinstance(mask_paths, dict):
        raise ValueError("mask_paths must be an object")
    item["mask_paths"] = {
        str(name): managed_relative(str(path)) for name, path in mask_paths.items()
    }
    output = sku_root / str(item["output"])
    if not output.is_file():
        raise FileNotFoundError(f"Relocated output is missing: {output}")
    expected = item.get("output_sha256")
    if expected and file_hash(output) != expected:
        raise RuntimeError(f"Relocated output hash mismatch: {output}")
    for relative in item["mask_paths"].values():
        if not (sku_root / str(relative)).is_file():
            raise FileNotFoundError(f"Relocated mask is missing: {sku_root / str(relative)}")
    item["path_policy"] = "relative_to_sku_output"


def repair_sku_payload(payload: dict[str, object], sku_root: Path) -> dict[str, object]:
    repaired = deepcopy(payload)
    if "items" in repaired:
        items = repaired["items"]
        if not isinstance(items, list):
            raise ValueError("summary items must be a list")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("summary item must be an object")
            _repair_item(item, sku_root)
    elif "output" in repaired:
        _repair_item(repaired, sku_root)
    else:
        raise ValueError("Unrecognized SKU report payload")

    reference_masks = repaired.get("reference_mask_paths")
    if isinstance(reference_masks, dict):
        repaired["reference_mask_paths"] = {
            str(name): managed_relative(str(path)) for name, path in reference_masks.items()
        }
        for relative in repaired["reference_mask_paths"].values():
            if not (sku_root / str(relative)).is_file():
                raise FileNotFoundError(
                    f"Relocated reference mask is missing: {sku_root / str(relative)}"
                )
    configuration = repaired.get("configuration")
    if isinstance(configuration, dict):
        configuration["path_policy"] = "relative_to_sku_output"
        configuration["output_layout"] = "flat-sku-v1"
    repaired["report_relocation"] = {
        "schema": 1,
        "images_modified": False,
        "path_policy": "relative_to_sku_output",
    }
    return repaired


def migrate(output_root: Path, *, apply: bool) -> dict[str, object]:
    root = output_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    changes: list[tuple[Path, dict[str, object]]] = []
    for sku_root in sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("sz")):
        candidates = [sku_root / "summary.json", *sorted((sku_root / "报告").glob("*.json"))]
        for report in candidates:
            if not report.is_file():
                continue
            payload = json.loads(report.read_text(encoding="utf-8"))
            repaired = repair_sku_payload(payload, sku_root)
            if repaired != payload:
                changes.append((report, repaired))

    aggregate_changes = 0
    for report in sorted(root.glob("*-batch-summary.json")):
        payload = json.loads(report.read_text(encoding="utf-8"))
        repaired = deepcopy(payload)
        for item in repaired.get("items", []):
            if isinstance(item, dict) and isinstance(item.get("sku"), str):
                sku_root = root / item["sku"]
                if not sku_root.is_dir():
                    raise FileNotFoundError(f"Aggregate SKU output is missing: {sku_root}")
                item["output"] = item["sku"]
        repaired["output_path_policy"] = "relative_to_output_root"
        repaired["report_relocation"] = {"schema": 1, "images_modified": False}
        if repaired != payload:
            changes.append((report, repaired))
            aggregate_changes += 1

    backup = None
    if apply and changes:
        backup_dir = root / ".report-backups"
        backup_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = backup_dir / f"before-v0.4.1-path-migration-{stamp}.zip"
        with zipfile.ZipFile(backup, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, _payload in changes:
                archive.write(path, path.relative_to(root).as_posix())
        with zipfile.ZipFile(backup) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Report backup verification failed")
        for path, payload in changes:
            atomic_json(path, payload)

    return {
        "output_root": str(root),
        "mode": "applied" if apply else "dry-run",
        "reports_changed": len(changes),
        "aggregate_reports_changed": aggregate_changes,
        "backup": None if backup is None else str(backup),
        "images_modified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(migrate(args.output_root, apply=args.apply), ensure_ascii=False, indent=2))
