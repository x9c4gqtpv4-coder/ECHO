from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image

from batch_color.image_io import load_srgb, save_srgb


@dataclass(frozen=True)
class PosePoint:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class PoseEvidence:
    width: int
    height: int
    faces: tuple[dict[str, object], ...]
    bodies: tuple[dict[str, PosePoint], ...]
    hands: tuple[dict[str, PosePoint], ...]
    backend: str


def find_pose_helper() -> Path | None:
    configured = os.environ.get("BATCH_COLOR_POSE_EVIDENCE_BIN")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate
    installed = shutil.which("batch-color-pose-evidence")
    if installed:
        return Path(installed)
    candidates = [
        Path.cwd() / "tools/pose_evidence/bin/batch-color-pose-evidence",
        Path(__file__).resolve().parents[2]
        / "tools/pose_evidence/bin/batch-color-pose-evidence",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate
    return None


def _point(value: object) -> PosePoint:
    if not isinstance(value, dict):
        raise ValueError("Pose point must be an object")
    return PosePoint(
        float(value["x"]),
        float(value["y"]),
        float(value.get("confidence", 1.0)),
    )


def _points(value: object) -> dict[str, PosePoint]:
    if not isinstance(value, dict):
        raise ValueError("Pose points must be an object")
    return {str(name): _point(point) for name, point in value.items()}


def vision_pose_evidence(
    input_path: str | Path,
    *,
    executable: Path | None = None,
    canonical_image: Image.Image | None = None,
) -> PoseEvidence:
    helper = executable or find_pose_helper()
    if helper is None:
        raise FileNotFoundError(
            "macOS pose helper is not built; run scripts/build_pose_evidence.sh"
        )
    with tempfile.TemporaryDirectory(prefix="batch-color-pose-") as directory:
        canonical = canonical_image if canonical_image is not None else load_srgb(input_path)[0]
        canonical_path = Path(directory) / "canonical.png"
        save_srgb(canonical, canonical_path)
        completed = subprocess.run(
            [str(helper), str(canonical_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(message or "macOS Vision pose evidence failed")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("macOS Vision pose evidence returned invalid JSON") from error
    width, height = int(payload["width"]), int(payload["height"])
    if (width, height) != canonical.size:
        raise RuntimeError("Native pose evidence geometry does not match canonical pixels")
    faces = payload.get("faces", [])
    bodies = payload.get("bodies", [])
    hands = payload.get("hands", [])
    if not isinstance(faces, list) or not isinstance(bodies, list) or not isinstance(hands, list):
        raise RuntimeError("Native pose evidence collections are invalid")
    return PoseEvidence(
        width=width,
        height=height,
        faces=tuple(item for item in faces if isinstance(item, dict)),
        bodies=tuple(_points(item.get("points", {})) for item in bodies if isinstance(item, dict)),
        hands=tuple(_points(item.get("points", {})) for item in hands if isinstance(item, dict)),
        backend=str(payload.get("backend", "apple-vision-pose-evidence")),
    )
