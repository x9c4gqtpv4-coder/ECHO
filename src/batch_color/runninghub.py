"""Pure payload helpers for RunningHub integration.

This module deliberately performs no network requests and never reads an API
key from disk.  It gives UI, CLI and server adapters one validated contract for
the same ECHO workflow node IDs and fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np


RUNNINGHUB_DOMAINS = {"global": "www.runninghub.ai", "china": "www.runninghub.cn"}


@dataclass(frozen=True)
class NodeOverride:
    nodeId: str
    fieldName: str
    fieldValue: object
    description: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.description is None:
            payload.pop("description")
        return payload


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a string/integer identifier")
    normalized = str(value).strip()
    if not normalized or len(normalized) > 128:
        raise ValueError(f"{name} must be a nonempty identifier")
    return normalized


def runninghub_base(region: str = "global") -> str:
    try:
        return f"https://{RUNNINGHUB_DOMAINS[region]}"
    except KeyError as error:
        raise ValueError("region must be global or china") from error


def native_comfy_proxy(api_key: str, *, plus: bool = False, region: str = "global") -> str:
    key = _identifier(api_key, "api_key")
    route = "proxy-plus" if plus else "proxy"
    return f"{runninghub_base(region)}/{route}/{key}"


def echo_node_overrides(
    *,
    source_file: str,
    reference_file: str,
    strength: float = 0.85,
    adjustment_mode: str = "background + person",
    transform_path: str = "auto",
    mask_backend: str = "heuristic",
    source_node_id: str = "1",
    reference_node_id: str = "2",
    echo_node_id: str = "3",
) -> list[NodeOverride]:
    if not isinstance(source_file, str) or not source_file.strip():
        raise ValueError("source_file must be the uploaded RunningHub fileName")
    if not isinstance(reference_file, str) or not reference_file.strip():
        raise ValueError("reference_file must be the uploaded RunningHub fileName")
    if not np.isfinite(strength) or not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be finite and between 0 and 1")
    if adjustment_mode not in {"background + person", "background only"}:
        raise ValueError("invalid adjustment_mode")
    if transform_path not in {"auto", "global", "surface"}:
        raise ValueError("invalid transform_path")
    if mask_backend not in {"heuristic", "auto"}:
        raise ValueError("invalid mask_backend")
    source_id = _identifier(source_node_id, "source_node_id")
    reference_id = _identifier(reference_node_id, "reference_node_id")
    echo_id = _identifier(echo_node_id, "echo_node_id")
    return [
        NodeOverride(source_id, "image", source_file, "source image"),
        NodeOverride(reference_id, "image", reference_file, "reference image"),
        NodeOverride(echo_id, "strength", round(float(strength), 4)),
        NodeOverride(echo_id, "adjustment_mode", adjustment_mode),
        NodeOverride(echo_id, "transform_path", transform_path),
        NodeOverride(echo_id, "mask_backend", mask_backend),
    ]


def advanced_task_payload(
    *,
    api_key: str,
    workflow_id: str | int | None = None,
    overrides: Iterable[NodeOverride] = (),
    workflow_json: str | None = None,
    webhook_url: str | None = None,
    add_metadata: bool = False,
) -> dict[str, Any]:
    """Build the official `/task/openapi/create` JSON body.

    Supply a saved `workflow_id`, or a full API-format JSON string, or both.
    RunningHub documents `workflow` as overriding `workflowId` when present.
    """
    key = _identifier(api_key, "api_key")
    if workflow_id is None and workflow_json is None:
        raise ValueError("workflow_id or workflow_json is required")
    payload: dict[str, Any] = {
        "apiKey": key,
        "nodeInfoList": [item.as_dict() for item in overrides],
        "addMetadata": bool(add_metadata),
    }
    if workflow_id is not None:
        payload["workflowId"] = _identifier(workflow_id, "workflow_id")
    if workflow_json is not None:
        if not isinstance(workflow_json, str) or not workflow_json.strip():
            raise ValueError("workflow_json must be a nonempty API-format JSON string")
        payload["workflow"] = workflow_json
    if webhook_url is not None:
        parsed = urlparse(webhook_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("webhook_url must be a public HTTPS URL without user info")
        payload["webhookUrl"] = webhook_url
    return payload


def task_id_from_response(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict) or payload.get("code") != 0:
        message = payload.get("msg") if isinstance(payload, dict) else None
        raise ValueError(f"RunningHub task creation failed: {message or 'invalid response'}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("RunningHub task response has no data object")
    return _identifier(data.get("taskId"), "taskId")


def output_urls_from_response(payload: dict[str, Any]) -> list[str]:
    if not isinstance(payload, dict) or payload.get("code") != 0:
        raise ValueError("RunningHub output query failed")
    data = payload.get("data")
    if data in (None, ""):
        return []
    if not isinstance(data, list):
        raise ValueError("RunningHub output data must be a list")
    urls: list[str] = []
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("fileUrl"), str):
            raise ValueError("RunningHub output item has no fileUrl")
        parsed = urlparse(item["fileUrl"])
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("RunningHub returned a non-HTTPS output URL")
        urls.append(item["fileUrl"])
    return urls
