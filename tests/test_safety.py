import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from batch_color.cli import main
from batch_color.safety import atomic_output, file_hash, validate_artifact_paths


class PathSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.reference = self.root / "source.png", self.root / "reference.png"
        Image.new("RGB", (140, 180), (140, 150, 160)).save(self.source)
        Image.new("RGB", (140, 180), (190, 180, 160)).save(self.reference)

    def call(self, *extra):
        args = ["match", "--input", str(self.source), "--reference", str(self.reference),
                "--output", str(self.root / "candidate.png"), "--mask-backend", "heuristic", *map(str, extra)]
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            return main(args)

    def test_every_output_role_refuses_to_overwrite_either_input(self):
        before = {p: file_hash(p) for p in (self.source, self.reference)}
        for flag in ("--output", "--preview", "--mask-output", "--reference-mask-output", "--profile-output", "--report"):
            for target in (self.source, self.reference):
                with self.subTest(flag=flag, target=target):
                    self.assertEqual(self.call(flag, target, "--overwrite"), 2)
                    self.assertFalse((self.root / "candidate.png").exists())
                    self.assertEqual({p: file_hash(p) for p in before}, before)

    def test_all_outputs_are_checked_against_each_other(self):
        same = self.root / "conflict.png"
        self.assertEqual(self.call("--preview", same, "--report", same), 2)
        self.assertFalse((self.root / "candidate.png").exists())

    def test_hardlink_is_rejected_even_with_overwrite(self):
        alias = self.root / "linked.png"
        os.link(self.source, alias)
        before = file_hash(self.source)
        self.assertEqual(self.call("--output", alias, "--overwrite"), 2)
        self.assertEqual(file_hash(self.source), before)

    def test_symlink_is_rejected(self):
        alias = self.root / "linked.png"
        alias.symlink_to(self.reference)
        self.assertEqual(self.call("--preview", alias, "--overwrite"), 2)

    def test_case_aliases_are_rejected_before_files_exist(self):
        with self.assertRaises(ValueError):
            validate_artifact_paths([], [self.root / "Foo.png", self.root / "foo.png"])

    def test_parent_child_artifacts_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_artifact_paths([], [self.root / "reports", self.root / "reports/one.json"])

    def test_profile_command_cannot_overwrite_reference(self):
        before = file_hash(self.reference)
        with contextlib.redirect_stderr(io.StringIO()):
            code = main(["profile", "--reference", str(self.reference), "--output", str(self.reference),
                         "--name", "test", "--overwrite"])
        self.assertEqual(code, 2)
        self.assertEqual(file_hash(self.reference), before)

    def test_existing_output_needs_explicit_overwrite(self):
        output = self.root / "candidate.png"
        output.write_bytes(b"existing user file")
        self.assertEqual(self.call(), 2)
        self.assertEqual(output.read_bytes(), b"existing user file")

    def test_lossy_master_is_rejected(self):
        self.assertEqual(self.call("--output", self.root / "candidate.jpg"), 2)

    def test_atomic_failure_preserves_existing_file(self):
        path = self.root / "artifact.json"
        path.write_text("original")
        with self.assertRaises(RuntimeError):
            with atomic_output(path) as staged:
                staged.write_text("partial")
                raise RuntimeError("injected failure")
        self.assertEqual(path.read_text(), "original")
        self.assertEqual(list(self.root.glob(".artifact.json.*")), [])

    def test_success_is_a_review_candidate_with_automatic_sidecar(self):
        self.assertEqual(self.call(), 0)
        self.assertTrue((self.root / "candidate.png.report.json").is_file())
        with Image.open(self.root / "candidate.png") as saved:
            self.assertEqual(saved.size, (140, 180))

    def test_strict_quality_exit_preserves_review_code_three(self):
        self.assertEqual(self.call("--strict-quality-exit"), 3)
        self.assertTrue((self.root / "candidate.png.report.json").is_file())


if __name__ == "__main__":
    unittest.main()
