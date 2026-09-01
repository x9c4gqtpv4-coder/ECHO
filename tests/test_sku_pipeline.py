import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from batch_color.baseline import (
    A0_BASELINE,
    A0_EXPECTED_CODE_FINGERPRINT,
    A0_EXPECTED_DEPENDENCIES,
    A0_EXPECTED_PERSON_HELPER_SHA256,
)
from batch_color.masking import MaskResult
from batch_color.semantic import build_semantic_masks
from batch_color.runtime import runtime_identity
from batch_color.safety import payload_hash
from batch_color.semantic import (
    _clean_authorized_edit_region,
    _continuous_edit_weight,
    _garment_seed_prior,
    _robust_skin_probability,
    _seed_connected_support,
)
from batch_color.sku import scan_sku
from batch_color.sku_pipeline import (
    RegionPlan,
    RegionStats,
    apply_region_plans,
    background_style_target,
    choose_anchor,
    normalized_garment_signature,
    region_distance,
    region_style_distance,
    region_stats,
    run_sku_simple_pilot,
    scene_style_target,
)


class SKUScanTests(unittest.TestCase):
    def test_scan_is_case_insensitive_and_excludes_pose_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sku = root / "sz123"
            sku.mkdir()
            Image.new("RGB", (16, 16), "gray").save(sku / "指定场景.JPEG")
            Image.new("RGB", (16, 16), "red").save(sku / "成品动作1.PNG")
            Image.new("RGB", (16, 16), "blue").save(sku / "成品动作2.png")
            Image.new("RGB", (16, 16), "green").save(sku / "确定动作1.png")
            result = scan_sku(root, "sz123")
            self.assertEqual(len(result.targets), 2)
            self.assertTrue(result.targets[0].endswith("成品动作1.PNG"))
            self.assertFalse(any("确定动作" in path for path in result.targets))

    def test_scan_accepts_safe_non_sz_sku_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sku = root / "SKU-ABC-001"
            sku.mkdir()
            Image.new("RGB", (16, 16), "gray").save(sku / "指定场景.png")
            Image.new("RGB", (16, 16), "red").save(sku / "成品动作1.png")
            self.assertEqual(scan_sku(root, sku.name).sku, sku.name)


class SemanticMaskTests(unittest.TestCase):
    def test_semantic_edit_weight_is_continuous_at_core_threshold(self):
        score = np.array([[0.1599, 0.16, 0.30, 0.5499, 0.55, 0.5501]], dtype=np.float32)
        result = _continuous_edit_weight(
            score,
            np.ones_like(score),
            lower=0.16,
            core=0.55,
            transition_floor=0.38,
        )
        self.assertEqual(float(result[0, 0]), 0.0)
        self.assertGreater(float(result[0, 1]), 0.0)
        self.assertLess(abs(float(result[0, 4] - result[0, 3])), 0.001)
        self.assertLess(abs(float(result[0, 5] - result[0, 4])), 0.001)

    def test_clean_face_authorization_uses_uniform_interior_not_confidence_opacity(self):
        candidate = np.zeros((48, 64), dtype=np.float32)
        candidate[8:40, 10:54] = 0.28
        candidate[8:40, 10:32] = 0.92
        candidate[20:28, 27:37] = 0.0  # protected glasses/lips-sized opening
        seed = np.zeros_like(candidate)
        seed[12:18, 20:28] = 1.0
        result = _clean_authorized_edit_region(
            candidate,
            seed,
            threshold=0.16,
            closing_size=3,
            feather_radius=1.5,
        )
        self.assertGreater(float(result[16, 20]), 0.95)
        self.assertGreater(float(result[16, 44]), 0.95)
        self.assertEqual(float(result[24, 32]), 0.0)
        self.assertEqual(float(result[2, 2]), 0.0)

    def test_clean_garment_authorization_fills_narrow_textile_confidence_gaps(self):
        candidate = np.zeros((64, 64), dtype=np.float32)
        candidate[10:54, 10:54] = 0.78
        candidate[10:54, 20:22] = 0.0
        candidate[10:54, 34:36] = 0.0
        seed = np.zeros_like(candidate)
        seed[24:40, 14:50] = 1.0
        result = _clean_authorized_edit_region(
            candidate,
            seed,
            threshold=0.12,
            closing_size=7,
            feather_radius=1.5,
        )
        self.assertGreater(float(result[32, 20]), 0.95)
        self.assertGreater(float(result[32, 35]), 0.95)
        self.assertGreater(float(result[32, 48]), 0.95)
        self.assertEqual(float(result[4, 4]), 0.0)

    def test_missing_semantic_seed_fails_closed(self):
        candidate = np.ones((30, 30), dtype=np.float32)
        seed = np.zeros((30, 30), dtype=np.float32)
        self.assertEqual(int(np.count_nonzero(_seed_connected_support(candidate, seed))), 0)

    def test_missing_colour_seeds_do_not_remove_skin_or_garment_priors(self):
        lab = np.zeros((30, 30, 3), dtype=np.float32)
        missing = np.zeros((30, 30), dtype=np.float32)
        skin, skin_pixels = _robust_skin_probability(lab, missing)
        garment, garment_pixels = _garment_seed_prior(lab, missing)
        self.assertEqual(skin_pixels, 0)
        self.assertEqual(garment_pixels, 0)
        self.assertEqual(int(np.count_nonzero(skin)), 0)
        self.assertEqual(int(np.count_nonzero(garment)), 0)

    def test_conservative_light_top_mask_stays_inside_person(self):
        image = np.full((240, 180, 3), (210, 190, 170), np.uint8)
        image[20:220, 35:145] = (28, 27, 30)  # person/hair base
        image[45:90, 70:110] = (205, 150, 125)  # face
        image[90:155, 55:125] = (235, 232, 225)  # light garment
        image[100:140, 40:55] = (235, 232, 225)  # left sleeve
        image[100:140, 125:140] = (235, 232, 225)  # right sleeve
        person = np.zeros((240, 180), np.uint8)
        person[20:220, 35:145] = 255
        background = Image.fromarray(255 - person, mode="L")
        with patch(
            "batch_color.semantic.get_background_mask",
            return_value=MaskResult(background, "unit-test"),
        ):
            masks = build_semantic_masks(
                "dummy.png",
                Image.fromarray(image),
                garment_kind="top",
                garment_hint="light",
                mask_backend="vision",
                parser_backend="none",
                pose_backend="none",
                proxy_edge=512,
            )
        garment = np.asarray(masks.garment)
        self.assertGreater(int(np.count_nonzero(garment >= 128)), 500)
        self.assertEqual(int(garment[0, 0]), 0)
        self.assertGreater(int(garment[120, 90]), 128)


class SKUTransferTests(unittest.TestCase):
    def test_a0_contract_and_rendering_golden_are_stable(self):
        self.assertEqual(
            A0_BASELINE.fingerprint,
            "5b6681f3a27269bc019c378430d601f443d44c83031bc3dccb0c4e80ee5c27bd",
        )
        height, width = 24, 32
        y, x = np.mgrid[0:height, 0:width]
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        pixels[..., 0] = 80 + x * 3
        pixels[..., 1] = 70 + y * 4
        pixels[..., 2] = 60 + (x + y) * 2
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[4:20, 6:26] = 255
        source = RegionStats(
            320, (0.25, 0.30, 0.38, 0.46, 0.55, 0.63, 0.68), 0.018, 0.025, 0.0
        )
        target = RegionStats(
            320, (0.28, 0.34, 0.42, 0.50, 0.59, 0.67, 0.72), 0.012, 0.032, 0.0
        )
        output, report = apply_region_plans(
            Image.fromarray(pixels),
            [
                RegionPlan(
                    "person",
                    Image.fromarray(mask),
                    source,
                    target,
                    A0_BASELINE.person_strength,
                    A0_BASELINE.person_luminance_cap,
                    A0_BASELINE.person_chroma_cap,
                )
            ],
            tile_rows=7,
        )
        digest = hashlib.sha256(np.asarray(output).tobytes()).hexdigest()
        self.assertEqual(
            digest, "0862bb6d939fc3c497ea1a419bdb70b068e3cacff693986887c93cc6aadf827b"
        )
        self.assertFalse(report["no_op"])

    def test_exact_identity_plan_is_no_op(self):
        image = Image.new("RGB", (40, 30), (120, 110, 100))
        mask = Image.new("L", image.size, 255)
        stats = RegionStats(1200, (0.4,) * 7, 0.01, 0.02, 0.0)
        output, report = apply_region_plans(
            image, [RegionPlan("person", mask, stats, stats, 0.58, 0.06, 0.02)]
        )
        np.testing.assert_array_equal(np.asarray(output), np.asarray(image))
        self.assertTrue(report["no_op"])
        self.assertEqual(report["no_op_reason"], "exact_transform_identity")

    def test_region_plan_improves_target_and_preserves_unauthorized_pixels(self):
        pixels = np.full((120, 100, 3), (110, 105, 100), np.uint8)
        pixels[30:95, 25:75] = (180, 170, 160)
        image = Image.fromarray(pixels)
        mask_array = np.zeros((120, 100), np.uint8)
        mask_array[30:95, 25:75] = 255
        mask = Image.fromarray(mask_array)
        target_image = image.copy()
        target_pixels = np.asarray(target_image).copy()
        target_pixels[30:95, 25:75] = (205, 190, 175)
        target_image = Image.fromarray(target_pixels)
        source_stats = region_stats(image, mask)
        target_stats = region_stats(target_image, mask)
        output, report = apply_region_plans(
            image,
            [RegionPlan("garment", mask, source_stats, target_stats, 0.8, 0.12, 0.04)],
            tile_rows=17,
        )
        after_stats = region_stats(output, mask)
        self.assertLess(region_distance(after_stats, target_stats), region_distance(source_stats, target_stats))
        outside = mask_array == 0
        np.testing.assert_array_equal(np.asarray(output)[outside], pixels[outside])
        self.assertEqual(report["unauthorized_changed_pixels"], 0)

    def test_anchor_is_color_medoid_not_first_item(self):
        def stats(l_value):
            return RegionStats(1000, (l_value,) * 7, 0.01, 0.02, 0.0)

        records = [
            {"path": "far.png", "garment_stats": stats(0.25)},
            {"path": "middle.png", "garment_stats": stats(0.52)},
            {"path": "near.png", "garment_stats": stats(0.55)},
        ]
        index, ranking = choose_anchor(records)
        self.assertEqual(index, 1)
        self.assertEqual(ranking[0]["file"], "middle.png")

    def test_background_style_target_preserves_shadow_shape(self):
        source = RegionStats(1000, (0.20, 0.35, 0.50, 0.60, 0.70, 0.78, 0.82), 0.03, 0.01, 0.0)
        reference = RegionStats(1000, (0.60, 0.64, 0.68, 0.72, 0.77, 0.82, 0.86), 0.01, 0.04, 0.0)
        target = background_style_target(source, reference)
        self.assertAlmostEqual(target.l_quantiles[3], reference.l_quantiles[3], places=6)
        self.assertLess(target.l_quantiles[0], 0.40)
        self.assertEqual(target.a_median, reference.a_median)
        self.assertLess(region_style_distance(target, reference), region_style_distance(source, reference))

    def test_scene_style_target_keeps_subject_relative_identity(self):
        subject = RegionStats(1000, (0.30,) * 7, 0.05, 0.02, 0.0)
        source_background = RegionStats(1000, (0.50,) * 7, 0.02, 0.01, 0.0)
        reference_background = RegionStats(1000, (0.60,) * 7, 0.03, 0.04, 0.0)
        target = scene_style_target(
            subject,
            source_background,
            reference_background,
            luminance_scale=0.5,
            chroma_scale=0.5,
        )
        self.assertAlmostEqual(target.l_quantiles[3], 0.35, places=6)
        self.assertAlmostEqual(target.a_median, 0.055, places=6)
        self.assertAlmostEqual(target.b_median, 0.035, places=6)

    def test_normalized_signature_removes_shared_scene_offset(self):
        garment = RegionStats(1000, (0.7,) * 7, 0.04, 0.06, 0.0)
        background = RegionStats(1000, (0.5,) * 7, 0.01, 0.02, 0.0)
        signature = normalized_garment_signature(garment, background)
        self.assertEqual(signature.l_quantiles, (0.2,) * 7)
        self.assertAlmostEqual(signature.a_median, 0.03)
        self.assertAlmostEqual(signature.b_median, 0.04)


class SKUOutputContractTests(unittest.TestCase):
    @staticmethod
    def _masks(_path, image, *, mask_backend):
        person = np.zeros((image.height, image.width), dtype=np.uint8)
        person[10:-10, image.width // 3 : 2 * image.width // 3] = 255
        background = 255 - person
        masks = {
            "background": Image.fromarray(background),
            "background_core": Image.fromarray(background),
            "person": Image.fromarray(person),
            "person_core": Image.fromarray(person),
            "transition": Image.new("L", image.size, 0),
        }
        diagnostics = {
            "backend": f"unit-{mask_backend}",
            "person_core_pixels": int(np.count_nonzero(person)),
            "background_core_pixels": int(np.count_nonzero(background)),
            "transition_pixels": 0,
            "warnings": [],
            "human_review_required": True,
        }
        return masks, diagnostics

    def test_flat_output_relative_paths_and_safe_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sku = "sz123"
            source = root / sku
            source.mkdir()
            scene = np.full((80, 64, 3), (165, 155, 145), dtype=np.uint8)
            scene[10:-10, 22:42] = (185, 165, 150)
            first = np.full((80, 64, 3), (145, 140, 135), dtype=np.uint8)
            first[10:-10, 22:42] = (180, 150, 135)
            second = np.full((80, 64, 3), (135, 130, 125), dtype=np.uint8)
            second[10:-10, 22:42] = (170, 145, 130)
            Image.fromarray(scene).save(source / "指定场景.png")
            Image.fromarray(first).save(source / "成品动作1.png")
            Image.fromarray(second).save(source / "成品动作2.png")

            output_root = root / "校色输出"
            legacy = output_root / sku / "legacy-test"
            legacy.mkdir(parents=True)
            identity = runtime_identity()
            identity["a0_code_fingerprint"] = A0_EXPECTED_CODE_FINGERPRINT
            identity["numpy"] = A0_EXPECTED_DEPENDENCIES["numpy"]
            identity["pillow"] = A0_EXPECTED_DEPENDENCIES["pillow"]
            identity["native_helpers"] = {
                "person_mask_sha256": A0_EXPECTED_PERSON_HELPER_SHA256
            }
            identity.pop("identity_sha256", None)
            identity["identity_sha256"] = payload_hash(identity)
            with patch("batch_color.sku_pipeline._two_region_masks", side_effect=self._masks), patch(
                "batch_color.sku_pipeline.runtime_identity", return_value=identity
            ):
                final, summary = run_sku_simple_pilot(
                    dataset_root=root,
                    sku=sku,
                    output_root=output_root,
                    run_name="golden-a0",
                    mask_backend="vision",
                )

            self.assertEqual(final, (output_root / sku).resolve())
            self.assertTrue(legacy.is_dir())
            self.assertFalse((final / "golden-a0").exists())
            self.assertEqual(summary["schema_version"], 4)
            self.assertEqual(summary["status"], "candidate")
            self.assertTrue(summary["baseline"]["compatible"])
            self.assertEqual(summary["baseline"]["fingerprint"], A0_BASELINE.fingerprint)
            for item in summary["items"]:
                self.assertFalse(Path(item["output"]).is_absolute())
                self.assertTrue((final / item["output"]).is_file())
                for mask_path in item["mask_paths"].values():
                    self.assertFalse(Path(mask_path).is_absolute())
                    self.assertTrue((final / mask_path).is_file())
            for mask_path in summary["reference_mask_paths"].values():
                self.assertFalse(Path(mask_path).is_absolute())
                self.assertTrue((final / mask_path).is_file())
            stored = json.loads((final / "summary.json").read_text())
            self.assertEqual(stored["items"][0]["output"], summary["items"][0]["output"])

            with patch("batch_color.sku_pipeline._two_region_masks", side_effect=self._masks), patch(
                "batch_color.sku_pipeline.runtime_identity", return_value=identity
            ):
                with self.assertRaises(FileExistsError):
                    run_sku_simple_pilot(
                        dataset_root=root,
                        sku=sku,
                        output_root=output_root,
                        run_name="second",
                        mask_backend="vision",
                    )
                _final, custom_summary = run_sku_simple_pilot(
                    dataset_root=root,
                    sku=sku,
                    output_root=output_root,
                    run_name="second",
                    mask_backend="vision",
                    person_strength=0.57,
                    replace_output=True,
                )
            self.assertTrue(legacy.is_dir())
            self.assertFalse(custom_summary["baseline"]["compatible"])
            self.assertEqual(
                len(list((final / "待复核候选").glob("*.png"))), 2
            )
            self.assertTrue(any((final / "历史").iterdir()))


if __name__ == "__main__":
    unittest.main()
