"""Artifact identity, input protection, and atomic single-file publication."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _path_key(path: Path) -> str:
    # Conservative even on case-sensitive volumes; also handles macOS Unicode names.
    return unicodedata.normalize("NFC", str(path.resolve())).casefold()


def validate_artifact_paths(
    inputs: Iterable[str | Path], outputs: Iterable[str | Path | None], *, overwrite: bool = True
) -> None:
    """Validate the entire write set BEFORE opening or writing any artifact."""
    sources = [Path(value) for value in inputs]
    destinations = [Path(value) for value in outputs if value is not None]
    source_keys = {_path_key(path) for path in sources}
    source_ids = {(p.stat().st_dev, p.stat().st_ino) for p in sources if p.exists()}
    output_keys: set[str] = set()
    output_ids: set[tuple[int, int]] = set()
    for path in destinations:
        key = _path_key(path)
        if path.is_symlink():
            raise ValueError(f"Refusing symlink output: {path}")
        if key in source_keys or key in output_keys:
            raise ValueError(f"Output aliases an input or another artifact: {path}")
        if path.exists():
            if not path.is_file():
                raise ValueError(f"Output is not a regular file: {path}")
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in source_ids or identity in output_ids:
                raise ValueError(f"Output shares a file identity with input/artifact: {path}")
            if not overwrite:
                raise FileExistsError(f"Output exists; choose a new path or --overwrite: {path}")
            output_ids.add(identity)
        for parent in path.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"Output parent is not a directory: {parent}")
        output_keys.add(key)
    all_keys = source_keys | output_keys
    for key in all_keys:
        for parent in Path(key).parents:
            if str(parent) in output_keys:
                raise ValueError(f"An output is also another path's parent: {parent}")


@contextmanager
def atomic_output(path: str | Path) -> Iterator[Path]:
    """Stage next to destination; old artifact survives writer/verification failure."""
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError(f"Refusing symlink output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=destination.suffix, dir=destination.parent
    )
    os.close(descriptor)
    staged = Path(temporary)
    try:
        yield staged
        with staged.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


def atomic_text(path: str | Path, text: str) -> None:
    with atomic_output(path) as staged:
        staged.write_text(text, encoding="utf-8")


def atomic_json(path: str | Path, payload: object) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def validate_master_path(path: str | Path) -> None:
    if Path(path).suffix.lower() not in {".png", ".tif", ".tiff"}:
        raise ValueError("Candidate masters must be lossless PNG/TIFF; JPEG is for previews only")
