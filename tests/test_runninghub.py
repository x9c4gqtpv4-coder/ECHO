from __future__ import annotations

import unittest

from batch_color.runninghub import (
    advanced_task_payload,
    echo_node_overrides,
    native_comfy_proxy,
    output_urls_from_response,
    task_id_from_response,
)


class RunningHubContractTests(unittest.TestCase):
    def test_echo_overrides_keep_stable_node_contract(self):
        overrides = echo_node_overrides(
            source_file="api/source.png",
            reference_file="api/reference.png",
        )
        fields = [(item.nodeId, item.fieldName, item.fieldValue) for item in overrides]
        self.assertEqual(fields[:2], [
            ("1", "image", "api/source.png"),
            ("2", "image", "api/reference.png"),
        ])
        self.assertIn(("3", "strength", 0.85), fields)

    def test_advanced_payload_supports_saved_or_inline_workflows(self):
        overrides = echo_node_overrides(source_file="s.png", reference_file="r.png")
        saved = advanced_task_payload(
            api_key="secret-placeholder",
            workflow_id="12345",
            overrides=overrides,
        )
        self.assertEqual(saved["workflowId"], "12345")
        self.assertFalse(saved["addMetadata"])
        self.assertEqual(saved["nodeInfoList"][0]["nodeId"], "1")

        inline = advanced_task_payload(
            api_key="secret-placeholder",
            workflow_json='{"1": {}}',
            webhook_url="https://example.com/runninghub/callback",
        )
        self.assertEqual(inline["workflow"], '{"1": {}}')
        self.assertEqual(inline["webhookUrl"], "https://example.com/runninghub/callback")

    def test_proxy_and_response_validation(self):
        self.assertEqual(
            native_comfy_proxy("abc", plus=True),
            "https://www.runninghub.ai/proxy-plus/abc",
        )
        task = {"code": 0, "data": {"taskId": "987", "taskStatus": "QUEUED"}}
        self.assertEqual(task_id_from_response(task), "987")
        outputs = {"code": 0, "data": [{"fileUrl": "https://example.com/output.png"}]}
        self.assertEqual(output_urls_from_response(outputs), ["https://example.com/output.png"])
        with self.assertRaises(ValueError):
            output_urls_from_response({"code": 0, "data": [{"fileUrl": "http://bad.test/a.png"}]})

    def test_invalid_webhook_and_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            advanced_task_payload(
                api_key="key",
                workflow_id="1",
                webhook_url="http://localhost/callback",
            )
        with self.assertRaises(ValueError):
            echo_node_overrides(source_file="s", reference_file="r", strength=float("nan"))


if __name__ == "__main__":
    unittest.main()
