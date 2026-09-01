"""Export a text-only, auditable source handoff; never include user photographs."""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import tomllib
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT_FILES = {".gitignore", "LICENSE", "README.md", "pyproject.toml", "requirements-tested.txt"}
EXTENSIONS = {
    "src": {".py"}, "tests": {".py"}, "scripts": {".py", ".sh"},
    "configs": {".json", ".md"}, "docs": {".md", ".txt"},
    ".github": {".yml", ".yaml"}, "ci": {".example"},
}


def is_source_path(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "__pycache__" in path.parts:
        return False
    if name in ROOT_FILES or name in {
        "tools/person_mask/main.swift",
        "tools/face_mask/main.swift",
        "tools/pose_evidence/main.swift",
        "models/README.md",
    }:
        return True
    if path.parts and path.parts[0] == "data":
        return name in {f"data/{part}/.gitkeep" for part in ("input", "references", "output", "cache")}
    return bool(path.parts and path.suffix in EXTENSIONS.get(path.parts[0], set()))


def export_source(root: Path, destination: Path) -> dict:
    root = root.resolve()
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    )
    names = sorted({n.decode("utf-8") for n in raw.split(b"\0") if n and is_source_path(n.decode("utf-8"))})
    files = []
    contents = {}
    for name in names:
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Refusing symlink or out-of-root source: {name}")
        data = path.read_bytes()
        data.decode("utf-8")  # Every shipped file must be readable text.
        mode = 0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644
        files.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "mode": oct(mode)})
        contents[name] = data
    required = {
        "LICENSE",
        "src/batch_color/transfer.py",
        "src/batch_color/sku_pipeline.py",
        "tools/person_mask/main.swift",
        "tools/face_mask/main.swift",
        "tools/pose_evidence/main.swift",
        "docs/UPGRADE_REPORT_0.4.2_2026-08-31.md",
        "docs/UPGRADE_REPORT_0.5.1_VALIDATION_READINESS.md",
        "docs/UPGRADE_REPORT_0.5.2_C1_OBSERVER.md",
        "docs/UPGRADE_REPORT_0.5.3_SAFETY_PLANNER.md",
    }
    if not required.issubset(contents):
        raise ValueError("Required source or report is missing")
    version = tomllib.loads(contents["pyproject.toml"].decode("utf-8"))["project"]["version"]
    primary_report = "docs/UPGRADE_REPORT_0.5.3_SAFETY_PLANNER.md"
    if primary_report not in contents:
        primary_report = "docs/UPGRADE_REPORT_0.5.2_C1_OBSERVER.md"
    if primary_report not in contents:
        primary_report = "docs/UPGRADE_REPORT_0.5.1_VALIDATION_READINESS.md"
    if primary_report not in contents:
        primary_report = "docs/UPGRADE_REPORT_0.5.0_FINE_PRECISION.md"
    if primary_report not in contents:
        primary_report = "docs/UPGRADE_REPORT_0.4.2_2026-08-31.md"
    if primary_report not in contents:
        primary_report = "docs/UPGRADE_REPORT_0.4.1_2026-08-30.md"
    if primary_report not in contents:
        primary_report = "docs/CURRENT_SOURCE_STATUS_0.4.0a2_2026-08-30.md"
    if primary_report not in contents:
        primary_report = "docs/CURRENT_SOURCE_STATUS_0.4.0a1_2026-08-30.md"
    if primary_report not in contents:
        primary_report = "docs/UPGRADE_REPORT_0.3.2_2026-08-27.md"
    if primary_report not in contents:
        primary_report = "docs/REPAIR_REPORT_0.3.1_2026-08-27.md"
    if primary_report not in contents:
        primary_report = "docs/REPAIR_REPORT_2026-08-27.md"
    if primary_report not in contents:
        primary_report = "docs/TECHNICAL_REPORT_2026-08-27.md"
    manifest = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "local_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "working_tree_has_changes": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        "runtime_status": f"experimental_{version}_review_only",
        "version": version,
        "primary_report": primary_report,
        "source_files": files,
        "excludes": ["photos", "screenshots", "credentials", "models", ".venv", "compiled binaries", ".git history"],
    }
    destination.mkdir(parents=True, exist_ok=False)
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    (destination / "MANIFEST.json").write_bytes(manifest_bytes)
    report = contents[primary_report]
    (destination / "完整技术报告.txt").write_bytes(report)
    listing = [f"完整源码合集：{version} 实验版及诊断工具。所有候选必须人工复核，未提供自动批准。\n"]
    for item in files:
        name = item["path"]
        if Path(name).suffix in {".py", ".swift", ".sh", ".toml", ".json", ".txt"} or name == ".gitignore":
            listing.append(f"\n{'=' * 72}\nFILE: {name}\nSHA256: {item['sha256']}\n{'=' * 72}\n")
            listing.append(contents[name].decode("utf-8"))
            listing.append("\n")
    (destination / "完整源代码.txt").write_text("".join(listing), encoding="utf-8")
    archive = destination / f"batch-color-standardizer-source-v{version}.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for item in files:
            entry = zipfile.ZipInfo("batch-color-standardizer/" + item["path"])
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = (stat.S_IFREG | int(item["mode"], 8)) << 16
            zipped.writestr(entry, contents[item["path"]])
        zipped.writestr("batch-color-standardizer/MANIFEST.json", manifest_bytes)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        for item in files:
            actual = zipped.read("batch-color-standardizer/" + item["path"])
            assert hashlib.sha256(actual).hexdigest() == item["sha256"]
    checksums = {
        name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
        for name in (archive.name, "完整技术报告.txt", "完整源代码.txt", "MANIFEST.json")
    }
    (destination / "SHA256SUMS.json").write_text(json.dumps(checksums, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"destination": str(destination.resolve()), "source_file_count": len(files), "archive_bytes": archive.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_source(Path(__file__).resolve().parents[1], args.output), ensure_ascii=False))
