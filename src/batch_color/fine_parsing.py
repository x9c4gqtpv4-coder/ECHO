"""Optional local-only SegFormer adapter for the ATR18 fine-mask contract.

No weights are downloaded or bundled.  Callers must provide a local model
directory whose license and provenance have been reviewed separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from batch_color.fine_masks import ATR18_LABELS
from batch_color.image_io import make_proxy
from batch_color.safety import file_hash, payload_hash


@dataclass(frozen=True)
class FineParserOutput:
    labels: np.ndarray
    confidence: np.ndarray
    backend: str
    identity: dict[str, object]
    input_files: tuple[str, ...]


def _normalized_label(value: object) -> str:
    text = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "upperclothes": "upper_clothes",
        "upper_cloth": "upper_clothes",
        "leftshoe": "left_shoe",
        "rightshoe": "right_shoe",
        "leftleg": "left_leg",
        "rightleg": "right_leg",
        "leftarm": "left_arm",
        "rightarm": "right_arm",
    }
    return aliases.get(text, text)


def _model_files(model_directory: Path) -> tuple[Path, ...]:
    if model_directory.is_symlink() or not model_directory.is_dir():
        raise ValueError("Fine parser model must be a real local directory, not a symlink")
    files = tuple(sorted(path for path in model_directory.rglob("*") if path.is_file()))
    if not files or not any(path.name.endswith(".safetensors") for path in files):
        raise ValueError("Fine parser requires local safetensors weights; pickle weights are refused")
    if any(path.is_symlink() for path in files):
        raise ValueError("Fine parser model files cannot be symlinks")
    total = sum(path.stat().st_size for path in files)
    if total > 2 * 1024 * 1024 * 1024:
        raise ValueError("Fine parser model directory exceeds the 2 GiB safety limit")
    return files


def segformer_atr18(
    image: Image.Image,
    model_directory: str | Path,
    *,
    device: str = "cpu",
    max_edge: int = 768,
    threads: int = 2,
) -> FineParserOutput:
    if device not in {"cpu", "mps"}:
        raise ValueError("Fine parser device must be cpu or mps")
    if not 256 <= max_edge <= 1536:
        raise ValueError("Fine parser max_edge must be in 256..1536")
    if not 1 <= threads <= 8:
        raise ValueError("Fine parser threads must be in 1..8")
    model_root = Path(model_directory).expanduser().resolve()
    files = _model_files(model_root)
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    except ImportError as error:
        raise FileNotFoundError(
            "Fine parser dependencies are absent; install the optional fine-semantic extra"
        ) from error

    torch.set_num_threads(threads)
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable on this Mac")
    processor = AutoImageProcessor.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForSemanticSegmentation.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    id2label = getattr(model.config, "id2label", {})
    ordered = tuple(
        _normalized_label(id2label.get(index, id2label.get(str(index), "")))
        for index in range(len(ATR18_LABELS))
    )
    if ordered != ATR18_LABELS:
        raise ValueError(
            "Local SegFormer labels do not exactly match ATR18; refusing an unsafe class remap"
        )
    if int(getattr(model.config, "num_labels", 0)) != len(ATR18_LABELS):
        raise ValueError("Local SegFormer must expose exactly 18 ATR classes")

    proxy = make_proxy(image.convert("RGB"), max_edge=max_edge)
    target = torch.device(device)
    model.to(target)
    model.eval()
    inputs = processor(images=proxy, return_tensors="pt")
    inputs = {name: tensor.to(target) for name, tensor in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
        logits = torch.nn.functional.interpolate(
            logits,
            size=(proxy.height, proxy.width),
            mode="bilinear",
            align_corners=False,
        )
        probabilities = torch.softmax(logits.float(), dim=1)
        confidence, labels = torch.max(probabilities, dim=1)
    labels_proxy = labels[0].detach().cpu().numpy().astype(np.uint8)
    confidence_proxy = confidence[0].detach().cpu().numpy().astype(np.float32)
    del inputs, logits, probabilities, confidence, labels, model
    if device == "mps":
        torch.mps.empty_cache()

    full_labels = np.asarray(
        Image.fromarray(labels_proxy, mode="L").resize(image.size, Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    full_confidence = np.asarray(
        Image.fromarray(confidence_proxy, mode="F").resize(image.size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    records = [
        {"relative_path": str(path.relative_to(model_root)), "sha256": file_hash(path)}
        for path in files
    ]
    identity = {
        "backend": "segformer-atr18-local-safetensors-v1",
        "model_directory": str(model_root),
        "model_manifest_sha256": payload_hash(records),
        "files": records,
        "device": device,
        "proxy_size": list(proxy.size),
        "max_edge": max_edge,
        "threads": threads,
        "network_access": False,
        "weights_bundled": False,
        "license_must_be_audited_separately": True,
    }
    return FineParserOutput(
        labels=full_labels,
        confidence=np.clip(full_confidence, 0.0, 1.0),
        backend="segformer-atr18-local-safetensors-v1",
        identity=identity,
        input_files=tuple(str(path) for path in files),
    )
