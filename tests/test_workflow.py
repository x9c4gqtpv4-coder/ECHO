import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from batch_color.baseline import (
    A0_BASELINE,
    A0_EXPECTED_CODE_FINGERPRINT,
    A0_EXPECTED_DEPENDENCIES,
    A0_EXPECTED_PERSON_HELPER_SHA256,
    a0_compatible,
)
from batch_color.planning import compile_shadow_plan
from batch_color.runtime import _code_identity, a0_code_fingerprint, a0_runtime_compatibility
from batch_color.safety import atomic_json, file_hash, payload_hash
from batch_color.sku import scan_sku
from batch_color.workflow import (
    APPROVED_DIRECTORY,
    CANDIDATE_DIRECTORY,
    create_sku_profile,
    load_sku_profile,
    review_sku_output,
    verify_region_target_evidence,
)


class SKUProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sku = self.root / "SKU-001"
        self.sku.mkdir()
        Image.new("RGB", (32, 32), (160, 150, 140)).save(self.sku / "指定场景.png")
        Image.new("RGB", (32, 32), (30, 60, 90)).save(self.sku / "产品图.png")
        Image.new("RGB", (32, 32), (130, 120, 110)).save(self.sku / "成品动作1.png")
        self.manifest = scan_sku(self.root, self.sku.name)

    def test_profile_consumes_and_verifies_product_evidence_without_claiming_a0_use(self):
        profile = create_sku_profile(
            self.manifest,
            product_anchor="产品图.png",
            garment_anchor="成品动作1.png",
            confirm_product=True,
            confirm_garment=True,
        )
        self.assertEqual(profile["product_truth"]["status"], "confirmed")
        self.assertEqual(
            profile["product_truth"]["a0_effect"],
            "evidence_only_not_applied_to_colour_target",
        )
        path = self.root / "sku-profile.json"
        atomic_json(path, profile)
        self.assertEqual(load_sku_profile(path, self.manifest)["profile_fingerprint"], profile["profile_fingerprint"])
        evidence = verify_region_target_evidence(
            path,
            object_id="SKU-001:upper-clothes",
            sku_role="target_sku",
            reference_policy="sku_approved_anchor",
            reference_sha256=profile["garment_anchor"]["sha256"],
        )
        self.assertTrue(evidence["confirmed"])
        with self.assertRaises(ValueError):
            verify_region_target_evidence(
                path,
                object_id="SKU-001:upper-clothes",
                sku_role="target_sku",
                reference_policy="sku_approved_anchor",
                reference_sha256="0" * 64,
            )
        Image.new("RGB", (32, 32), (200, 0, 0)).save(self.sku / "产品图.png")
        with self.assertRaises(RuntimeError):
            load_sku_profile(path, self.manifest)


class RuntimeCompatibilityTests(unittest.TestCase):
    def _identity(self):
        return {
            "a0_code_fingerprint": A0_EXPECTED_CODE_FINGERPRINT,
            "numpy": A0_EXPECTED_DEPENDENCIES["numpy"],
            "pillow": A0_EXPECTED_DEPENDENCIES["pillow"],
            "native_helpers": {
                "person_mask_sha256": A0_EXPECTED_PERSON_HELPER_SHA256,
            },
        }

    def test_a0_compatibility_binds_algorithm_dependencies_and_native_helper(self):
        result = a0_runtime_compatibility(
            background_strength=A0_BASELINE.background_strength,
            person_strength=A0_BASELINE.person_strength,
            set_color_tolerance=A0_BASELINE.set_color_tolerance,
            identity=self._identity(),
        )
        self.assertTrue(result["compatible"])
        self.assertTrue(
            a0_compatible(
                background_strength=A0_BASELINE.background_strength,
                person_strength=A0_BASELINE.person_strength,
                set_color_tolerance=A0_BASELINE.set_color_tolerance,
                runtime_compatibility=result,
            )
        )
        changed = self._identity()
        changed["a0_code_fingerprint"] = "changed"
        mismatch = a0_runtime_compatibility(
            background_strength=A0_BASELINE.background_strength,
            person_strength=A0_BASELINE.person_strength,
            set_color_tolerance=A0_BASELINE.set_color_tolerance,
            identity=changed,
        )
        self.assertFalse(mismatch["compatible"])
        self.assertIn("a0_algorithm_identity_changed", mismatch["reasons"])

    def test_a0_fingerprint_detects_replaced_imported_pixel_helper(self):
        original = a0_code_fingerprint()

        def replacement(*args, **kwargs):
            return args[0]

        with patch(
            "batch_color.sku_pipeline._bounded_luminance_curve", replacement
        ):
            self.assertNotEqual(a0_code_fingerprint(), original)

    def test_bytecode_identity_is_independent_of_installation_path(self):
        def sample(value):
            return value + 1

        relocated = sample.__code__.replace(
            co_filename="/another/install/location/runtime.py",
            co_firstlineno=sample.__code__.co_firstlineno + 100,
        )
        self.assertEqual(_code_identity(sample.__code__), _code_identity(relocated))


class ReviewWorkflowTests(unittest.TestCase):
    @staticmethod
    def _package(base: Path, colour: str = "red") -> tuple[Path, Path]:
        dataset = base / "dataset"
        sku = dataset / "SKU-001"
        sku.mkdir(parents=True)
        scene = sku / "指定场景.png"
        target = sku / "成品动作1.png"
        Image.new("RGB", (24, 24), "gray").save(scene)
        Image.new("RGB", (24, 24), "white").save(target)
        manifest = scan_sku(dataset, sku.name)
        target_path = manifest.targets[0]
        profile = create_sku_profile(
            manifest, auto_garment_candidate=target.name
        )
        root = base / "output"
        candidate = root / CANDIDATE_DIRECTORY
        masks = root / "蒙版"
        candidate.mkdir(parents=True)
        masks.mkdir()
        output = candidate / "one.png"
        Image.new("RGB", (24, 24), colour).save(output)
        Image.new("L", (24, 24), 255).save(masks / "person.png")
        identity = {"release_version": "test"}
        identity["identity_sha256"] = payload_hash(identity)
        atomic_json(root / "input-manifest.json", manifest.as_dict())
        atomic_json(root / "sku-profile.json", profile)
        atomic_json(root / "run-identity.json", identity)
        plan = compile_shadow_plan(profile, {"compatible": True})
        atomic_json(root / "execution-plan.json", plan)
        atomic_json(
            root / "summary.json",
            {
                "sku": "SKU-001",
                "status": "candidate",
                "accepted": False,
                "baseline": {"id": A0_BASELINE.baseline_id},
                "run_identity_sha256": identity["identity_sha256"],
                "execution_plan_sha256": file_hash(root / "execution-plan.json"),
                "execution_plan_fingerprint": plan["plan_sha256"],
                "items": [
                    {
                        "input": target_path,
                        "input_sha256": manifest.input_hashes[target_path],
                        "output": f"{CANDIDATE_DIRECTORY}/one.png",
                        "output_sha256": file_hash(output),
                    }
                ],
            },
        )
        atomic_json(root / "review-status.json", {"status": "candidate", "accepted": False})
        return root, output

    def test_approval_is_hash_bound_and_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = self._package(Path(directory), "red")
            record_path, record = review_sku_output(
                root,
                decision="approve",
                reviewer="internal-reviewer",
                reason="contact sheet and protected regions checked",
            )
            self.assertTrue(record["accepted"])
            approved = root / APPROVED_DIRECTORY / "one.png"
            self.assertEqual(file_hash(approved), file_hash(output))
            status = json.loads((root / "review-status.json").read_text())
            self.assertEqual(status["status"], "approved")
            shareable = json.loads(record_path.with_suffix(".shareable.json").read_text())
            self.assertNotIn("reviewer", shareable)

    def test_changed_candidate_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = self._package(Path(directory), "red")
            Image.new("RGB", (24, 24), "blue").save(output)
            with self.assertRaises(RuntimeError):
                review_sku_output(
                    root,
                    decision="approve",
                    reviewer="reviewer",
                    reason="must fail hash verification",
                )

    def test_existing_review_lock_blocks_a_second_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self._package(Path(directory), "red")
            (root / ".sku-review.lock").write_text("active\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                review_sku_output(
                    root,
                    decision="approve",
                    reviewer="reviewer",
                    reason="must serialize reviews",
                )

    def test_candidate_mutation_during_copy_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = self._package(Path(directory), "red")
            import batch_color.workflow as workflow

            original_copytree = workflow.shutil.copytree

            def racing_copytree(source, destination, *args, **kwargs):
                Image.new("RGB", (24, 24), "blue").save(output)
                return original_copytree(source, destination, *args, **kwargs)

            with patch("batch_color.workflow.shutil.copytree", racing_copytree):
                with self.assertRaises(RuntimeError):
                    review_sku_output(
                        root,
                        decision="approve",
                        reviewer="reviewer",
                        reason="candidate changed during approval",
                    )
            self.assertFalse((root / APPROVED_DIRECTORY).exists())
            self.assertFalse(any((root / "审核记录").glob("*.json")) if (root / "审核记录").exists() else False)


if __name__ == "__main__":
    unittest.main()
