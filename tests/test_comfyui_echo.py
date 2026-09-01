from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from comfyui_echo import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from comfyui_echo.core import clear_profile_cache, match_numpy, profile_cache_size
from comfyui_echo.nodes import _image_tensor_to_u8, _mask_tensor_to_u8


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def _pair():
    height, width = 96, 128
    source = np.full((height, width, 3), (205, 215, 225), dtype=np.uint8)
    reference = np.full((height, width, 3), (184, 174, 160), dtype=np.uint8)
    # A differently coloured central subject prevents the masks from being a
    # vacuous all-background test.
    source[18:88, 42:88] = (76, 91, 108)
    reference[18:88, 42:88] = (95, 78, 65)
    background = np.full((height, width), 255, dtype=np.uint8)
    background[16:90, 40:90] = 0
    return source, reference, background


class ComfyUIEchoCoreTests(unittest.TestCase):
    def setUp(self):
        clear_profile_cache()

    def test_node_discovery_is_conventional_and_torch_is_lazy(self):
        self.assertIn("ECHOReferenceMatch", NODE_CLASS_MAPPINGS)
        self.assertIn("ECHOReferenceMatch", NODE_DISPLAY_NAME_MAPPINGS)
        node = NODE_CLASS_MAPPINGS["ECHOReferenceMatch"]
        self.assertEqual(node.CATEGORY, "ECHO / 回响")
        self.assertEqual(node.RETURN_TYPES, ("IMAGE", "MASK", "STRING"))

    def test_repository_root_loads_like_a_comfyui_custom_node(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "echo_comfyui_test_package",
            root / "__init__.py",
            submodule_search_locations=[str(root)],
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
        self.assertIn("ECHOReferenceMatch", module.NODE_CLASS_MAPPINGS)

    def test_reference_match_protects_white_mask_and_reports_review(self):
        source, reference, background = _pair()
        protected = np.zeros(background.shape, dtype=np.uint8)
        protected[30:58, 50:78] = 255

        corrected, used_mask, report_json = match_numpy(
            source,
            reference,
            strength=0.8,
            adjustment_mode="background + person",
            transform_path="global",
            source_background_mask=background,
            reference_background_mask=background,
            protect_mask=protected,
        )

        self.assertEqual(corrected.shape, (1, *source.shape))
        self.assertEqual(used_mask.shape, (1, *background.shape))
        np.testing.assert_array_equal(corrected[0, 30:58, 50:78], source[30:58, 50:78])
        self.assertFalse(np.array_equal(corrected[0], source))
        report = json.loads(report_json)[0]
        self.assertEqual(report["schema"], "echo-comfyui/1")
        self.assertEqual(report["status"], "review")
        self.assertIs(report["approved"], False)
        self.assertIs(report["protection_mask_connected"], True)
        self.assertIs(report["engine"]["accepted"], False)

    def test_single_reference_broadcast_and_profile_cache(self):
        source, reference, background = _pair()
        sources = np.stack([source, np.clip(source.astype(int) + 7, 0, 255).astype(np.uint8)])
        masks = np.stack([background, background])
        output, _, first_report = match_numpy(
            sources,
            reference,
            transform_path="global",
            source_background_mask=masks,
            reference_background_mask=background,
        )
        self.assertEqual(output.shape[0], 2)
        first = json.loads(first_report)
        self.assertFalse(first[0]["reference_cache_hit"])
        self.assertTrue(first[1]["reference_cache_hit"])
        self.assertEqual(profile_cache_size(), 1)

        _, _, second_report = match_numpy(
            source,
            reference,
            transform_path="global",
            source_background_mask=background,
            reference_background_mask=background,
        )
        self.assertTrue(json.loads(second_report)[0]["reference_cache_hit"])

    def test_mask_geometry_is_never_silently_resized(self):
        source, reference, background = _pair()
        with self.assertRaisesRegex(ValueError, "resize the mask explicitly"):
            match_numpy(
                source,
                reference,
                transform_path="global",
                source_background_mask=background[:40, :40],
                reference_background_mask=background,
            )

    def test_tensor_adapters_clip_and_validate_without_importing_torch(self):
        image = _FakeTensor(np.array([[[[-1.0, 0.5, 2.0]]]], dtype=np.float32))
        converted = _image_tensor_to_u8(image, name="image")
        np.testing.assert_array_equal(converted, [[[[0, 128, 255]]]])
        mask = _FakeTensor(np.array([[0.0, 1.0]], dtype=np.float32))
        np.testing.assert_array_equal(_mask_tensor_to_u8(mask, name="mask"), [[[0, 255]]])

        invalid = _FakeTensor(np.full((1, 2, 2, 3), np.nan, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _image_tensor_to_u8(invalid, name="invalid")

    def test_public_node_runs_end_to_end_with_comfy_tensor_contract(self):
        source, reference, background = _pair()
        source_tensor = _FakeTensor(source[None].astype(np.float32) / 255.0)
        reference_tensor = _FakeTensor(reference[None].astype(np.float32) / 255.0)
        mask_tensor = _FakeTensor(background[None].astype(np.float32) / 255.0)
        fake_torch = SimpleNamespace(from_numpy=lambda value: value)
        node = NODE_CLASS_MAPPINGS["ECHOReferenceMatch"]()
        with patch.dict(sys.modules, {"torch": fake_torch}):
            corrected, mask, report = node.match(
                source_tensor,
                reference_tensor,
                0.85,
                "background + person",
                "global",
                "heuristic",
                source_background_mask=mask_tensor,
                reference_background_mask=mask_tensor,
            )
        self.assertEqual(corrected.shape, (1, *source.shape))
        self.assertEqual(mask.shape, (1, *background.shape))
        self.assertEqual(json.loads(report)[0]["status"], "review")


if __name__ == "__main__":
    unittest.main()
