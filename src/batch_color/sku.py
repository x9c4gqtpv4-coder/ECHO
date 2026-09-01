from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from batch_color.safety import file_hash


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SKUInput:
    sku: str
    directory: str
    scene_reference: str
    product_images: tuple[str, ...]
    targets: tuple[str, ...]
    input_hashes: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _role_files(directory: Path, prefix: str) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
            and path.stem.startswith(prefix)
        ),
        key=lambda path: path.name.casefold(),
    )


def scan_sku(dataset_root: str | Path, sku: str) -> SKUInput:
    root = Path(dataset_root).resolve()
    directory = (root / sku).resolve()
    if not directory.is_dir() or directory.parent != root:
        raise NotADirectoryError(f"SKU directory does not exist directly under dataset root: {sku}")
    if sku in {"", ".", ".."} or Path(sku).name != sku:
        raise ValueError("SKU must be one direct child directory name")

    scenes = _role_files(directory, "指定场景")
    targets = _role_files(directory, "成品动作")
    products = _role_files(directory, "产品图")
    if len(scenes) != 1:
        raise ValueError(f"Expected exactly one 指定场景 image for {sku}; found {len(scenes)}")
    if not targets:
        raise ValueError(f"No 成品动作 images found for {sku}")
    if len({path.name.casefold() for path in targets}) != len(targets):
        raise ValueError(f"Case-insensitive duplicate target names found for {sku}")

    inputs = [scenes[0], *products, *targets]
    return SKUInput(
        sku=sku,
        directory=str(directory),
        scene_reference=str(scenes[0]),
        product_images=tuple(str(path) for path in products),
        targets=tuple(str(path) for path in targets),
        input_hashes={str(path): file_hash(path) for path in inputs},
    )


def validate_inputs_unchanged(manifest: SKUInput) -> None:
    for path, expected in manifest.input_hashes.items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Input changed during SKU processing: {path}")
