"""Stage a whole job, validate it, then publish with rollback and a report commit marker.

Separate filesystem paths cannot be atomically renamed as a group. Cooperative
locks, a durable journal and report-last publication are used. Ordinary exceptions
roll back; hard termination retains staging/backups/locks for explicit recovery.
Readers must verify the committed report's hashes, not just a candidate filename.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

from batch_color.safety import atomic_json, file_hash, validate_artifact_paths


class ArtifactTransaction:
    def __init__(self, outputs: dict[str, Path], *, inputs=(), overwrite=True):
        self.outputs = {role: Path(path).absolute() for role, path in outputs.items()}
        if "report" not in outputs:
            raise ValueError("A transaction needs a final report commit marker")
        self.inputs, self.overwrite = list(inputs), overwrite
        self.directory = None
        self.staged = {}
        self.locks = []
        self.keep_recovery = False
        self.committed = False

    def __enter__(self):
        validate_artifact_paths(self.inputs, self.outputs.values(), overwrite=self.overwrite)
        try:
            for destination in sorted(self.outputs.values()):
                destination.parent.mkdir(parents=True, exist_ok=True)
                lock = destination.with_name(f".{destination.name}.batch-color.lock")
                with lock.open("x", encoding="utf-8") as handle:
                    handle.write("Output reserved by batch-color transaction.\n")
                self.locks.append(lock)
            self.directory = Path(tempfile.mkdtemp(prefix=".batch-color-run-", dir=self.outputs["report"].parent))
            device = self.directory.stat().st_dev
            if any(p.parent.stat().st_dev != device for p in self.outputs.values()):
                raise ValueError("A job's outputs must reside on the same filesystem")
            self.before = {role: file_hash(p) if p.exists() else None for role, p in self.outputs.items()}
            for role, destination in self.outputs.items():
                self.staged[role] = self.directory / (role + destination.suffix)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def artifact_records(self):
        return {role: {"path": str(self.outputs[role]), "sha256": file_hash(path)}
                for role, path in self.staged.items() if role != "report"}

    def commit(self):
        validate_artifact_paths(self.inputs, self.outputs.values(), overwrite=self.overwrite)
        if self.committed:
            raise RuntimeError("Transaction already committed")
        for role, destination in self.outputs.items():
            if (file_hash(destination) if destination.exists() else None) != self.before[role]:
                raise RuntimeError("Output changed while the job was staged")
            if not self.staged[role].is_file():
                raise ValueError(f"Missing staged artifact: {role}")
        # Serialization, image reopen validation and hashes must all precede commit.
        import json
        payload = json.loads(self.staged["report"].read_text())
        if payload.get("status") != "review" or payload.get("accepted") is not False:
            raise ValueError("Only complete review reports can be published")
        if payload.get("artifacts") != self.artifact_records():
            raise ValueError("Final report does not match staged artifact hashes")
        json.dumps(payload, allow_nan=False)
        backups = {}
        for role, destination in self.outputs.items():
            if self.before[role] is not None:
                backup = self.directory / ("backup-" + role + destination.suffix)
                shutil.copy2(destination, backup)
                if file_hash(backup) != self.before[role]:
                    raise RuntimeError("Backup verification failed")
                backups[role] = backup
        journal = {"state": "prepared", "outputs": {k: str(p) for k, p in self.outputs.items()},
                   "backups": {k: str(p) for k, p in backups.items()}, "before": self.before,
                   "after": {k: file_hash(p) for k, p in self.staged.items()}}
        for p in [*self.staged.values(), *backups.values()]:
            with p.open("rb") as stream:
                os.fsync(stream.fileno())
        atomic_json(self.directory / "journal.json", journal)
        fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        published = []
        try:
            # Report is last. During publication an old report will fail hash checks.
            for role in [r for r in self.outputs if r != "report"] + ["report"]:
                os.replace(self.staged[role], self.outputs[role])
                published.append(role)
            for parent in {p.parent for p in self.outputs.values()}:
                fd = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            self.committed = True
        except BaseException as original:
            failures = []
            for role in reversed(published):
                try:
                    if role in backups:
                        os.replace(backups[role], self.outputs[role])
                    else:
                        self.outputs[role].unlink()
                except OSError as error:
                    failures.append(str(error))
            if failures:
                self.keep_recovery = True
                raise RuntimeError(f"Rollback incomplete; retain locks/backups at {self.directory}: {failures}") from original
            raise

    def __exit__(self, *args):
        if not self.keep_recovery:
            if self.directory is not None:
                shutil.rmtree(self.directory)
            for lock in self.locks:
                lock.unlink(missing_ok=True)
