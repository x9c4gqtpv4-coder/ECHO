import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from batch_color.cli import main
from batch_color.fine_masks import (
    build_fine_mask_bundle_from_arrays,
    inward_feather,
)
from batch_color.fine_validation import validate_fine_labels
from batch_color.precision import RegionTargetPolicy, precision_region_match
from batch_color.fine_parsing import _model_files, _normalized_label
from batch_color.safety import atomic_json, file_hash, payload_hash


class FineMaskTests(unittest.TestCase):
    def labels(self):
        labels = np.zeros((80, 100), dtype=np.uint8)
        labels[10:30, 35:65] = 11  # face
        labels[30:55, 25:75] = 4  # upper clothes
        labels[55:78, 25:75] = 6  # pants
        return labels

    def test_reviewed_bundle_exposes_fine_parts_and_groups(self):
        bundle = build_fine_mask_bundle_from_arrays(
            self.labels(),
            None,
            label_status="reviewed",
            reviewed_by="qa-user",
            feather_radius=2,
        )
        self.assertTrue(bundle.regions["upper_clothes"]["usable_for_colour"])
        self.assertTrue(bundle.regions["pants"]["usable_for_colour"])
        self.assertTrue(bundle.regions["garment"]["usable_for_colour"])
        self.assertEqual(bundle.reviewed_by, "qa-user")
        self.assertEqual(bundle.masks["bag"].getbbox(), None)
        self.assertEqual(bundle.masks["unknown"].getbbox(), None)

    def test_automatic_bundle_requires_confidence_and_fails_closed(self):
        with self.assertRaises(ValueError):
            build_fine_mask_bundle_from_arrays(self.labels(), None, label_status="automatic")
        confidence = np.ones((80, 100), dtype=np.float32)
        confidence[self.labels() == 4] = 0.4
        bundle = build_fine_mask_bundle_from_arrays(
            self.labels(),
            confidence,
            label_status="automatic",
            confidence_threshold=0.82,
        )
        self.assertFalse(bundle.regions["upper_clothes"]["usable_for_colour"])
        self.assertIn(
            "insufficient_confident_coverage",
            bundle.regions["upper_clothes"]["failure_reasons"],
        )
        self.assertIsNone(bundle.masks["upper_clothes"].getbbox())

    def test_automatic_bundle_applies_region_specific_confidence_thresholds(self):
        confidence = np.full((80, 100), 0.85, dtype=np.float32)
        bundle = build_fine_mask_bundle_from_arrays(
            self.labels(),
            confidence,
            label_status="automatic",
            confidence_threshold=0.82,
            confidence_thresholds={"upper_clothes": 0.90},
        )
        self.assertFalse(bundle.regions["upper_clothes"]["usable_for_colour"])
        self.assertTrue(bundle.regions["pants"]["usable_for_colour"])
        self.assertEqual(bundle.regions["upper_clothes"]["confidence_threshold"], 0.9)
        self.assertEqual(bundle.regions["pants"]["confidence_threshold"], 0.82)
        self.assertEqual(
            bundle.regions["garment"]["authorized_pixels"],
            int(np.count_nonzero(self.labels() == 6)),
        )

    def test_inward_feather_never_authorizes_outside_pixels(self):
        binary = np.zeros((40, 40), dtype=np.uint8)
        binary[10:30, 8:32] = 255
        feathered = np.asarray(inward_feather(Image.fromarray(binary), 4), dtype=np.uint8)
        self.assertEqual(int(np.count_nonzero(feathered[binary == 0])), 0)
        self.assertGreater(int(np.count_nonzero(feathered[binary > 0])), 0)

    def test_invalid_class_is_rejected(self):
        labels = self.labels().astype(np.int16)
        labels[0, 0] = 18
        with self.assertRaises(ValueError):
            build_fine_mask_bundle_from_arrays(
                labels, None, label_status="reviewed", reviewed_by="qa"
            )

    def test_segformer_contract_normalizes_known_atr_names_and_requires_safetensors(self):
        self.assertEqual(_normalized_label("Upper-clothes"), "upper_clothes")
        self.assertEqual(_normalized_label("Left-shoe"), "left_shoe")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            with self.assertRaises(ValueError):
                _model_files(root)
            (root / "model.safetensors").write_bytes(b"test-only")
            self.assertEqual(len(_model_files(root)), 2)


class PrecisionRegionTests(unittest.TestCase):
    @staticmethod
    def pair():
        source = np.full((80, 100, 3), (25, 30, 35), dtype=np.uint8)
        reference = source.copy()
        source[20:65, 25:75] = (95, 75, 55)
        reference[20:65, 25:75] = (145, 105, 75)
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[20:65, 25:75] = 255
        return Image.fromarray(source), Image.fromarray(reference), Image.fromarray(mask)

    def test_precision_match_changes_only_authorized_region_and_improves_distance(self):
        source, reference, mask = self.pair()
        corrected, report = precision_region_match(
            source,
            reference,
            mask,
            mask,
            region="upper_clothes",
            target_policy=RegionTargetPolicy(
                object_id="sku-001:upper",
                sku_role="target_sku",
                reference_policy="sku_approved_anchor",
                reference_id="sku-001:approved-garment-anchor",
            ),
            strength=0.7,
        )
        source_array = np.asarray(source)
        corrected_array = np.asarray(corrected)
        authorized = np.asarray(mask) > 0
        np.testing.assert_array_equal(source_array[~authorized], corrected_array[~authorized])
        self.assertEqual(report["outside_authorized_changed_pixels"], 0)
        self.assertLess(report["distance_after"], report["distance_before"])
        self.assertIn("boundary_to_interior_ratio", report["boundary_residual"])
        self.assertEqual(report["target_policy"]["sku_role"], "target_sku")

    def test_precision_match_rejects_role_reference_mismatch_and_protected_roles(self):
        source, reference, mask = self.pair()
        with self.assertRaises(ValueError):
            precision_region_match(
                source,
                reference,
                mask,
                mask,
                region="upper_clothes",
                target_policy=RegionTargetPolicy(
                    object_id="sku-001:upper",
                    sku_role="target_sku",
                    reference_policy="scene_reference",
                    reference_id="scene-01",
                ),
            )

    def test_protected_mask_subtracts_authority_and_preserves_pixels(self):
        source, reference, mask = self.pair()
        protected = np.zeros((80, 100), dtype=np.uint8)
        protected[20:65, 25:50] = 255
        corrected, report = precision_region_match(
            source,
            reference,
            mask,
            mask,
            protected_mask=Image.fromarray(protected, mode="L"),
            region="upper_clothes",
            target_policy=RegionTargetPolicy(
                object_id="sku-001:upper",
                sku_role="target_sku",
                reference_policy="sku_approved_anchor",
                reference_id="sku-001:approved-garment-anchor",
            ),
            strength=0.7,
        )
        np.testing.assert_array_equal(
            np.asarray(source)[protected > 0], np.asarray(corrected)[protected > 0]
        )
        self.assertTrue(report["protection"]["supplied"])
        self.assertLess(
            report["protection"]["authorization_pixels_after_protection"],
            report["protection"]["authorization_pixels_before_protection"],
        )
        with self.assertRaises(ValueError):
            precision_region_match(
                source,
                reference,
                mask,
                mask,
                region="bag",
                target_policy=RegionTargetPolicy(
                    object_id="bag-01",
                    sku_role="protected_object",
                    reference_policy="protected",
                    reference_id="source-identity",
                ),
            )

    def test_cli_requires_hash_bound_mask_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, reference, _ = self.pair()
            source_path, reference_path = root / "source.png", root / "reference.png"
            source.save(source_path)
            reference.save(reference_path)
            sku_profile = root / "sku-profile.json"
            profile = {
                "schema_version": 1,
                "sku_id": "SKU-001",
                "scene_reference": {
                    "file": reference_path.name,
                    "sha256": file_hash(reference_path),
                    "confirmed": True,
                },
                "garment_anchor": {
                    "status": "confirmed",
                    "file": reference_path.name,
                    "sha256": file_hash(reference_path),
                    "confirmed": True,
                },
            }
            profile["profile_fingerprint"] = payload_hash(profile)
            atomic_json(sku_profile, profile)
            labels = np.zeros((80, 100), dtype=np.uint8)
            labels[20:65, 25:75] = 4
            source_labels, reference_labels = root / "source-labels.png", root / "ref-labels.png"
            Image.fromarray(labels, mode="L").save(source_labels)
            Image.fromarray(labels, mode="L").save(reference_labels)
            source_masks, reference_masks = root / "source-masks", root / "ref-masks"
            for image, label, output in (
                (source_path, source_labels, source_masks),
                (reference_path, reference_labels, reference_masks),
            ):
                self.assertEqual(
                    main(
                        [
                            "fine-masks",
                            "--input",
                            str(image),
                            "--label-map",
                            str(label),
                            "--label-status",
                            "reviewed",
                            "--reviewed-by",
                            "qa-user",
                            "--output-dir",
                            str(output),
                        ]
                    ),
                    0,
                )
            candidate = root / "candidate.png"
            self.assertEqual(
                main(
                    [
                        "precision-match",
                        "--input",
                        str(source_path),
                        "--reference",
                        str(reference_path),
                        "--source-mask",
                        str(source_masks / "upper_clothes.png"),
                        "--reference-mask",
                        str(reference_masks / "upper_clothes.png"),
                        "--source-mask-report",
                        str(source_masks / "fine-mask-report.json"),
                        "--reference-mask-report",
                        str(reference_masks / "fine-mask-report.json"),
                        "--sku-profile",
                        str(sku_profile),
                        "--region",
                        "upper_clothes",
                        "--object-id",
                        "SKU-001:upper",
                        "--sku-role",
                        "target_sku",
                        "--reference-policy",
                        "sku_approved_anchor",
                        "--reference-id",
                        file_hash(reference_path),
                        "--output",
                        str(candidate),
                    ]
                ),
                0,
            )
            report = json.loads(Path(str(candidate) + ".report.json").read_text())
            self.assertEqual(report["outside_authorized_changed_pixels"], 0)
            self.assertFalse(report["accepted"])
            self.assertEqual(report["capability"], "optional_b1_precision_region_match")

            # Changing the mask after review invalidates the authorization binding.
            changed = np.asarray(Image.open(source_masks / "upper_clothes.png")).copy()
            changed[0, 0] = 255
            Image.fromarray(changed).save(source_masks / "upper_clothes.png")
            self.assertEqual(
                main(
                    [
                        "precision-match",
                        "--input",
                        str(source_path),
                        "--reference",
                        str(reference_path),
                        "--source-mask",
                        str(source_masks / "upper_clothes.png"),
                        "--reference-mask",
                        str(reference_masks / "upper_clothes.png"),
                        "--source-mask-report",
                        str(source_masks / "fine-mask-report.json"),
                        "--reference-mask-report",
                        str(reference_masks / "fine-mask-report.json"),
                        "--sku-profile",
                        str(sku_profile),
                        "--region",
                        "upper_clothes",
                        "--object-id",
                        "SKU-001:upper",
                        "--sku-role",
                        "target_sku",
                        "--reference-policy",
                        "sku_approved_anchor",
                        "--reference-id",
                        file_hash(reference_path),
                        "--output",
                        str(root / "should-not-exist.png"),
                    ]
                ),
                2,
            )

            # Even if an attacker rewrites the report's artifact hash to match
            # the tampered mask, authorization is rebuilt from labels and
            # confidence and the forged mask is still rejected.
            fine_report_path = source_masks / "fine-mask-report.json"
            fine_report = json.loads(fine_report_path.read_text(encoding="utf-8"))
            fine_report["artifacts"]["mask_upper_clothes"]["sha256"] = file_hash(
                source_masks / "upper_clothes.png"
            )
            atomic_json(fine_report_path, fine_report)
            self.assertEqual(
                main(
                    [
                        "precision-match",
                        "--input", str(source_path),
                        "--reference", str(reference_path),
                        "--source-mask", str(source_masks / "upper_clothes.png"),
                        "--reference-mask", str(reference_masks / "upper_clothes.png"),
                        "--source-mask-report", str(fine_report_path),
                        "--reference-mask-report", str(reference_masks / "fine-mask-report.json"),
                        "--sku-profile", str(sku_profile),
                        "--region", "upper_clothes",
                        "--object-id", "SKU-001:upper",
                        "--sku-role", "target_sku",
                        "--reference-policy", "sku_approved_anchor",
                        "--reference-id", file_hash(reference_path),
                        "--output", str(root / "forged-should-not-exist.png"),
                    ]
                ),
                2,
            )


class FineValidationTests(unittest.TestCase):
    def test_perfect_truth_match_passes_all_metrics(self):
        labels = np.zeros((48, 64), dtype=np.uint8)
        labels[8:22, 20:44] = 11
        labels[22:44, 14:50] = 4
        report = validate_fine_labels(
            labels,
            labels.copy(),
            required_regions=("skin", "garment"),
        )
        self.assertTrue(report["checks_passed"])
        self.assertEqual(report["summary"]["pixel_accuracy"], 1.0)
        self.assertEqual(report["regions"]["garment"]["iou"], 1.0)
        self.assertEqual(report["regions"]["skin"]["boundary_f1"], 1.0)
        self.assertFalse(report["accepted"])

    def test_cross_role_confusion_fails_validation(self):
        truth = np.zeros((48, 64), dtype=np.uint8)
        truth[12:40, 16:48] = 4
        predicted = truth.copy()
        predicted[12:40, 16:48] = 11
        report = validate_fine_labels(
            predicted,
            truth,
            required_regions=("garment",),
            min_iou=0.8,
            max_cross_role_leakage=0.01,
        )
        self.assertFalse(report["checks_passed"])
        self.assertIn("iou_below_threshold:garment", report["failure_reasons"])
        self.assertIn("cross_role_leakage_above_threshold", report["failure_reasons"])

    def test_validation_cli_writes_hash_bound_truth_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.zeros((32, 40), dtype=np.uint8)
            labels[8:26, 10:30] = 4
            predicted = root / "predicted.png"
            truth = root / "truth.png"
            Image.fromarray(labels, mode="L").save(predicted)
            Image.fromarray(labels, mode="L").save(truth)
            report_path = root / "validation.json"
            self.assertEqual(
                main(
                    [
                        "validate-fine",
                        "--predicted-label-map",
                        str(predicted),
                        "--truth-label-map",
                        str(truth),
                        "--required-region",
                        "garment",
                        "--report",
                        str(report_path),
                    ]
                ),
                0,
            )
            report = json.loads(report_path.read_text())
            self.assertTrue(report["checks_passed"])
            self.assertEqual(report["evidence"]["reviewed_truth"]["sha256"], file_hash(truth))


if __name__ == "__main__":
    unittest.main()
