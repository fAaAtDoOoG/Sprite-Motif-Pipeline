from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import requests

from .config import DEFAULTS

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class ModelAsset:
    label: str
    filename: str
    subdir: str
    url: str

    def path_under(self, models_root: Path) -> Path:
        return models_root / self.subdir / self.filename


MODEL_ASSETS: tuple[ModelAsset, ...] = (
    ModelAsset(
        label="Qwen-Image-2512 diffusion model",
        filename=DEFAULTS.diffusion_model,
        subdir="diffusion_models",
        url="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors",
    ),
    ModelAsset(
        label="Qwen2.5-VL text encoder",
        filename=DEFAULTS.text_encoder,
        subdir="text_encoders",
        url="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    ),
    ModelAsset(
        label="Qwen Image VAE",
        filename=DEFAULTS.vae,
        subdir="vae",
        url="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
    ),
    ModelAsset(
        label="Qwen-Image-2512 Pixel Art LoRA",
        filename=DEFAULTS.pixel_lora,
        subdir="loras",
        url="https://huggingface.co/prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA/resolve/main/Qwen-Image-2512-Master-Pixel-Art-LoRA.safetensors",
    ),
)


def default_models_root() -> Path:
    env = os.environ.get("SPRITEPIPE_COMFY_MODELS_DIR")
    if env:
        return Path(env)

    candidates = [
        Path("D:/AI/ComfyUI/models"),
        Path("C:/AI/ComfyUI/models"),
        Path.home() / "ComfyUI" / "models",
        Path.cwd() / "ComfyUI" / "models",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def missing_local_assets(models_root: Path, assets: Iterable[ModelAsset] = MODEL_ASSETS) -> list[ModelAsset]:
    missing: list[ModelAsset] = []
    for asset in assets:
        path = asset.path_under(models_root)
        if not path.exists() or path.stat().st_size == 0:
            missing.append(asset)
    return missing


def assets_for_filenames(filenames: Iterable[str]) -> list[ModelAsset]:
    requested = set(filenames)
    return [asset for asset in MODEL_ASSETS if asset.filename in requested]


def download_assets(models_root: Path, assets: Iterable[ModelAsset], progress: ProgressCallback | None = None) -> list[Path]:
    downloaded: list[Path] = []
    for asset in assets:
        downloaded.append(download_asset(models_root, asset, progress=progress))
    return downloaded


def download_asset(models_root: Path, asset: ModelAsset, progress: ProgressCallback | None = None) -> Path:
    target = asset.path_under(models_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        _emit(progress, f"exists {target}")
        return target

    temp = target.with_suffix(target.suffix + ".part")
    headers: dict[str, str] = {}
    existing_size = temp.stat().st_size if temp.exists() else 0
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"

    _emit(progress, f"download {asset.label} -> {target}")
    with requests.get(asset.url, stream=True, timeout=60, headers=headers) as response:
        if response.status_code == 416:
            temp.replace(target)
            return target
        if response.status_code not in {200, 206}:
            response.raise_for_status()

        mode = "ab" if response.status_code == 206 and existing_size else "wb"
        if mode == "wb":
            existing_size = 0
        total = _content_total(response, existing_size)
        written = existing_size
        last_reported = -1
        with temp.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if total:
                    percent = int(written * 100 / total)
                    if percent >= last_reported + 5:
                        last_reported = percent
                        _emit(progress, f"{asset.filename}: {percent}% ({human_bytes(written)} / {human_bytes(total)})")
                else:
                    _emit(progress, f"{asset.filename}: {human_bytes(written)}")

    temp.replace(target)
    _emit(progress, f"done {target}")
    return target


def human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def _content_total(response: requests.Response, existing_size: int) -> int | None:
    if response.status_code == 206:
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            tail = content_range.rsplit("/", 1)[-1]
            if tail.isdigit():
                return int(tail)
    length = response.headers.get("Content-Length")
    if length and length.isdigit():
        return int(length) + existing_size
    return None


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
