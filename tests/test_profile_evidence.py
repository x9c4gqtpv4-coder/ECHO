"""Evidence is recomputed, not asserted by an imported trusted flag or manifest."""
import contextlib
from dataclasses import asdict, replace
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np
from PIL import Image

from batch_color.batch import run_batch
from batch_color.bundle import load_bundle, load_profile, save_bundle, _decode_png
from batch_color.cli import main
from batch_color.image_io import ImageInfo
from batch_color.profile import ColorProfile, create_profile, evidence_status, reference_evidence_verified
from batch_color.safety import file_hash
from batch_color.transfer import select_profile_path


class ProfileEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = Image.new("RGB", (80, 100), (140, 140, 140))
        self.source = Image.new("RGB", self.reference.size, (160, 160, 160))
        self.mask = Image.new("L", self.reference.size, 255)
        self.reference_path = self.root / "reference.png"
        self.source_path = self.root / "source.png"
        self.mask_path = self.root / "mask.png"
        self.reference.save(self.reference_path)
        self.source.save(self.source_path)
        self.mask.save(self.mask_path)
        self.info = ImageInfo(str(self.reference_path), 80, 100, "sRGB", True)
        self.profile = create_profile(self.reference, self.info, name="test", background_mask=self.mask,
                                      mask_backend="external-supplied", mask_metadata={"cacheable": True})
        self.bundle = self.root / "standard.bcp"
        save_bundle(self.bundle, self.profile, self.reference, self.mask)

    def call(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(list(map(str, args)))

    def rewrite(self, mutate, *, rehash=True):
        with zipfile.ZipFile(self.bundle) as package:
            members = {name: package.read(name) for name in package.namelist()}
        mutate(members)
        if rehash:
            manifest = json.loads(members["manifest.json"])
            for name, record in manifest["members"].items():
                if name in members:
                    record.update(bytes=len(members[name]), sha256=hashlib.sha256(members[name]).hexdigest())
            members["manifest.json"] = json.dumps(manifest).encode()
        output = self.root / "modified.bcp"
        with zipfile.ZipFile(output, "w") as package:
            for name, data in members.items():
                package.writestr(name, data)
        return output

    def forged_payload(self):
        payload = asdict(self.profile)
        surface = payload["background_surface"]
        surface["model"] = "plane"
        surface["coefficients"][0][1] = 0.025
        surface["diagnostics"].update(selected_model="plane", reason="support_passed",
                                       blocked_validation_rmse={"constant": 0.04, "plane": 0.005})
        return payload

    def test_bundle_recomputes_and_preserves_exact_reference_assets(self):
        restored, reference, mask = load_bundle(self.bundle)
        self.assertTrue(reference_evidence_verified(restored))
        self.assertEqual(asdict(restored), asdict(self.profile))
        self.assertEqual(reference.tobytes(), self.reference.tobytes())
        self.assertEqual(mask.tobytes(), self.mask.tobytes())
        status = evidence_status(restored)
        self.assertFalse(status["sampling_reviewed"])
        self.assertFalse(status["spatial_correspondence_verified"])

    def test_plain_json_roundtrip_never_grants_evidence_trust(self):
        path = self.root / "profile.json"
        self.profile.to_json(path)
        restored = load_profile(path)
        self.assertEqual(asdict(restored), asdict(self.profile))
        self.assertFalse(reference_evidence_verified(restored))
        self.assertNotIn("_reference_evidence_digest", path.read_text())
        for selected in ("auto", "surface"):
            _, report, _ = select_profile_path(self.source, restored, path=selected, background_mask=self.mask)
            self.assertFalse(report.surface_enabled)
            self.assertIn("reference_evidence_unverified_global_only_rebuild_bundle", report.review_reasons)

    def test_audited_forged_gradient_is_global_only_for_json_and_direct_api(self):
        path = self.root / "forged.json"
        path.write_text(json.dumps(self.forged_payload()))
        imported = ColorProfile.from_json(path)
        direct = replace(self.profile, background_surface=imported.background_surface)
        for profile in (imported, direct):
            for selected in ("auto", "surface"):
                output, report, _ = select_profile_path(self.source, profile, path=selected, background_mask=self.mask)
                pixels = np.asarray(output)
                self.assertEqual(int(np.ptp(pixels.astype(int))), 0)
                self.assertFalse(report.surface_enabled)
                self.assertFalse(report.accepted)

    def test_mutating_nested_data_invalidates_runtime_binding(self):
        self.profile.background_surface.diagnostics["condition_number"] += 0.01
        self.assertFalse(reference_evidence_verified(self.profile))
        _, report, _ = select_profile_path(self.source, self.profile, path="surface", background_mask=self.mask)
        self.assertFalse(report.surface_enabled)

    def test_json_cannot_supply_private_runtime_proof(self):
        payload = asdict(self.profile)
        payload["_reference_evidence_digest"] = getattr(self.profile, "_reference_evidence_digest")
        path = self.root / "proof.json"
        path.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            load_profile(path)

    def test_no_reference_mask_never_claims_spatial_support_or_evidence(self):
        profile = create_profile(self.reference, self.info, name="border")
        self.assertEqual(profile.background_sampling, "legacy-border")
        self.assertFalse(profile.background_surface.trusted)
        self.assertFalse(reference_evidence_verified(profile))

    def test_verified_bundle_does_not_override_insufficient_spatial_support(self):
        pixels = np.zeros((100, 80), np.uint8)
        pixels[:, 36:44] = 255
        mask = Image.fromarray(pixels)
        profile = create_profile(self.reference, self.info, name="narrow", background_mask=mask)
        bundle = self.root / "narrow.bcp"
        save_bundle(bundle, profile, self.reference, mask)
        restored = load_profile(bundle)
        self.assertTrue(reference_evidence_verified(restored))
        self.assertFalse(restored.background_surface.trusted)
        _, report, _ = select_profile_path(self.source, restored, path="surface", background_mask=self.mask)
        self.assertFalse(report.surface_enabled)

    def test_legacy_v4_trusted_flag_cannot_enable_spatial_path(self):
        payload = self.forged_payload()
        payload.update(version=4, background_sampling="legacy-border", reference_mask_sha256=None)
        payload.pop("generator")
        path = self.root / "legacy.json"
        path.write_text(json.dumps(payload))
        imported = load_profile(path)
        _, report, _ = select_profile_path(self.source, imported, path="surface", background_mask=self.mask)
        self.assertFalse(report.surface_enabled)

    def test_forged_coefficients_rejected_even_with_rehashed_manifest(self):
        modified = self.rewrite(lambda members: members.update({"profile.json": json.dumps(self.forged_payload()).encode()}))
        with self.assertRaisesRegex(ValueError, "recomputed"):
            load_bundle(modified)

    def test_forged_diagnostics_rejected_even_with_rehashed_manifest(self):
        payload = asdict(self.profile)
        payload["background_surface"]["diagnostics"]["condition_number"] += 1
        modified = self.rewrite(lambda members: members.update({"profile.json": json.dumps(payload).encode()}))
        with self.assertRaisesRegex(ValueError, "recomputed"):
            load_bundle(modified)

    def test_pixels_or_mask_cannot_be_swapped(self):
        for name, image in (("reference.png", self.source), ("reference_mask.png", Image.new("L", self.mask.size, 254))):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            modified = self.rewrite(lambda members: members.update({name: buffer.getvalue()}))
            with self.assertRaisesRegex(ValueError, "binding"):
                load_bundle(modified)

    def test_manifest_corruption_rejected_before_profile_use(self):
        modified = self.rewrite(lambda members: members.update({"profile.json": b"{}"}), rehash=False)
        with self.assertRaisesRegex(ValueError, "hash/size"):
            load_bundle(modified)

    def test_broken_deflate_stream_is_a_controlled_error(self):
        with zipfile.ZipFile(self.bundle) as archive:
            offset = archive.getinfo("manifest.json").header_offset
        data = bytearray(self.bundle.read_bytes())
        name_length, extra_length = struct.unpack_from("<HH", data, offset + 26)
        data[offset + 30 + name_length + extra_length] = 0x07  # forbidden DEFLATE block type
        damaged = self.root / "deflate-error.bcp"
        damaged.write_bytes(data)
        with self.assertRaises(ValueError):
            load_bundle(damaged)
        self.assertEqual(self.call("verify-profile", "--profile", damaged), 2)

    def test_missing_or_unexpected_assets_are_rejected_without_extraction(self):
        for mutation in (
            lambda m: m.pop("reference_mask.png"),
            lambda m: m.update({"../escape.png": b"not an image"}),
            lambda m: m.update({"evidence.npz": b"not supported"}),
        ):
            modified = self.rewrite(mutation)
            with self.assertRaisesRegex(ValueError, "members"):
                load_bundle(modified)
        self.assertFalse((self.root.parent / "escape.png").exists())

    def test_duplicate_archive_member_is_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.bundle, "a") as package:
                package.writestr("profile.json", b"{}")
        with self.assertRaisesRegex(ValueError, "members"):
            load_bundle(self.bundle)

    def test_symlink_archive_member_is_rejected(self):
        with zipfile.ZipFile(self.bundle) as package:
            members = {name: package.read(name) for name in package.namelist()}
        modified = self.root / "symlink.bcp"
        with zipfile.ZipFile(modified, "w") as package:
            for name, data in members.items():
                entry = zipfile.ZipInfo(name)
                if name == "reference.png":
                    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                package.writestr(entry, data)
        with self.assertRaisesRegex(ValueError, "type or size"):
            load_bundle(modified)

    def test_resource_limits_checked_before_archive_decode(self):
        with patch("batch_color.bundle.MAX_BUNDLE_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "resource limits"):
                load_bundle(self.bundle)

    def test_png_high_bits_alpha_and_oversize_rejected_before_decode(self):
        header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        with patch("batch_color.bundle.Image.open", side_effect=AssertionError("must not decode")):
            for width, height, depth, kind in ((80, 100, 16, 2), (80, 100, 8, 6), (12001, 100, 8, 2)):
                data = header + struct.pack(">IIBBBBB", width, height, depth, kind, 0, 0, 0) + b"0000"
                with self.assertRaises(ValueError):
                    _decode_png(data, "RGB")

    def test_recipe_version_change_requires_rebuild(self):
        with patch("batch_color.bundle.generator_identity", return_value={"recipe_id": "changed"}):
            with self.assertRaisesRegex(ValueError, "rebuild"):
                load_bundle(self.bundle)

    def test_duplicate_json_keys_and_oversized_json_are_rejected(self):
        path = self.root / "malformed.json"
        for content in ('{"version": 4, "version": 5}', ' ' * (1_048_576 + 1)):
            path.write_text(content)
            with self.assertRaises(ValueError):
                load_profile(path)

    def test_save_refuses_unverified_profile_or_mismatched_assets(self):
        with self.assertRaises(ValueError):
            save_bundle(self.root / "unverified.bcp", replace(self.profile), self.reference, self.mask)
        with self.assertRaises(ValueError):
            save_bundle(self.root / "wrong-pixels.bcp", self.profile, self.source, self.mask)
        self.assertFalse((self.root / "unverified.bcp").exists())
        self.assertFalse((self.root / "wrong-pixels.bcp").exists())

    def test_failed_bundle_reopen_preserves_existing_standard(self):
        before = file_hash(self.bundle)
        with patch("batch_color.bundle.load_bundle", side_effect=ValueError("injected reopen failure")):
            with self.assertRaises(ValueError):
                save_bundle(self.bundle, self.profile, self.reference, self.mask, overwrite=True)
        self.assertEqual(file_hash(self.bundle), before)

    def test_bundle_output_cannot_alias_reference_by_hardlink(self):
        alias = self.root / "reference-alias.bcp"
        os.link(self.reference_path, alias)
        before = file_hash(self.reference_path)
        with self.assertRaises(ValueError):
            save_bundle(alias, self.profile, self.reference, self.mask, overwrite=True)
        self.assertEqual(file_hash(self.reference_path), before)

    def test_profile_cli_saves_bundle_mask_and_committed_report(self):
        output, mask = self.root / "cli.bcp", self.root / "reference-export.png"
        code = self.call("profile", "--reference", self.reference_path, "--reference-mask", self.mask_path,
                         "--name", "test", "--output", output, "--reference-mask-output", mask)
        self.assertEqual(code, 0)
        self.assertTrue(reference_evidence_verified(load_profile(output)))
        with Image.open(mask) as opened:
            self.assertEqual(opened.tobytes(), self.mask.tobytes())
        report = json.loads(Path(str(output) + ".report.json").read_text())
        self.assertFalse(report["accepted"])
        self.assertEqual(report["artifacts"]["reference_mask"]["sha256"], file_hash(mask))

    def test_match_persists_both_masks_and_reference_bundle_by_default(self):
        output = self.root / "candidate.png"
        code = self.call("match", "--input", self.source_path, "--reference", self.reference_path,
                         "--background-mask", self.mask_path, "--reference-mask", self.mask_path, "--output", output)
        self.assertEqual(code, 0)
        report = json.loads(Path(str(output) + ".report.json").read_text())
        self.assertEqual(set(report["artifacts"]), {"candidate", "mask", "reference_mask", "profile"})
        for record in report["artifacts"].values():
            self.assertEqual(record["sha256"], file_hash(record["path"]))
        self.assertTrue(report["reference_evidence"]["reference_evidence_verified"])
        self.assertFalse(report["accepted"])

    def test_reference_mask_stage_failure_leaves_no_candidate_or_bundle(self):
        output = self.root / "candidate.png"
        from batch_color.cli import save_mask as real_save

        def fail_reference_mask(mask, path):
            if Path(path).name == "reference_mask.png":
                raise OSError("injected reference mask failure")
            return real_save(mask, path)

        with patch("batch_color.cli.save_mask", side_effect=fail_reference_mask):
            code = self.call("match", "--input", self.source_path, "--reference", self.reference_path,
                             "--background-mask", self.mask_path, "--reference-mask", self.mask_path, "--output", output)
        self.assertEqual(code, 2)
        self.assertFalse(output.exists())
        self.assertFalse(Path(str(output) + ".reference.bcp").exists())
        self.assertFalse(Path(str(output) + ".report.json").exists())

    def test_new_reference_artifact_publish_failures_restore_previous_job(self):
        output = self.root / "candidate.png"
        args = ("match", "--input", self.source_path, "--reference", self.reference_path,
                "--background-mask", self.mask_path, "--reference-mask", self.mask_path, "--output", output)
        self.assertEqual(self.call(*args), 0)
        report_path = Path(str(output) + ".report.json")
        report = json.loads(report_path.read_text())
        expected = {Path(record["path"]): record["sha256"] for record in report["artifacts"].values()}
        expected[report_path] = file_hash(report_path)
        Image.new("RGB", self.source.size, (200, 200, 200)).save(self.source_path)
        for role in ("profile", "reference_mask"):
            target = Path(report["artifacts"][role]["path"])
            real_replace = os.replace
            triggered = [False]

            def fail_one_publish(source, destination):
                if Path(destination) == target and not triggered[0]:
                    triggered[0] = True
                    raise OSError("injected reference artifact publish failure")
                return real_replace(source, destination)

            with patch("batch_color.transaction.os.replace", side_effect=fail_one_publish):
                self.assertEqual(self.call(*args, "--overwrite"), 2)
            self.assertTrue(triggered[0])
            self.assertEqual({path: file_hash(path) for path in expected}, expected)

    def test_verify_profile_is_read_only_and_not_quality_approval(self):
        before = {p.name: file_hash(p) for p in self.root.iterdir() if p.is_file()}
        self.assertEqual(self.call("verify-profile", "--profile", self.bundle), 0)
        self.assertEqual({p.name: file_hash(p) for p in self.root.iterdir() if p.is_file()}, before)
        path = self.root / "legacy.json"
        self.profile.to_json(path)
        self.assertEqual(self.call("verify-profile", "--profile", path), 3)

    def test_corrupt_bundle_cli_creates_error_not_candidate(self):
        modified = self.rewrite(lambda members: members.update({"profile.json": json.dumps(self.forged_payload()).encode()}))
        output = self.root / "candidate.png"
        self.assertEqual(self.call("apply", "--input", self.source_path, "--profile", modified,
                                   "--output", output, "--background-mask", self.mask_path), 2)
        self.assertFalse(output.exists())
        self.assertFalse(Path(str(output) + ".report.json").exists())

    def test_bundle_batch_recomputes_once_then_caches_without_approval(self):
        inputs, outputs = self.root / "input", self.root / "batch"
        inputs.mkdir()
        self.source.save(inputs / "one.png")
        self.source.save(inputs / "two.png")
        from batch_color.bundle import _recompute as real_recompute
        with patch("batch_color.bundle._recompute", wraps=real_recompute) as recompute:
            options = dict(input_directory=inputs, profile_path=self.bundle, output_directory=outputs,
                           mask_backend="heuristic", save_previews=False)
            first, second = run_batch(**options), run_batch(**options)
        self.assertEqual(recompute.call_count, 2)  # once per batch, not per photograph
        self.assertEqual((first.review, first.errors, first.skipped), (2, 0, 0))
        self.assertEqual((second.review, second.errors, second.skipped), (2, 0, 2))
        self.assertEqual(second.accepted, 0)


if __name__ == "__main__":
    unittest.main()
