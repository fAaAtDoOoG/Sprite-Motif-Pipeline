from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from .config import DEFAULTS, ModelDefaults


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout_s: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.client_id = str(uuid.uuid4())

    def object_info(self, node_type: str | None = None) -> dict[str, Any]:
        suffix = "/object_info" if node_type is None else f"/object_info/{node_type}"
        response = requests.get(f"{self.base_url}{suffix}", timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def queue_prompt(self, prompt: dict[str, Any]) -> str:
        response = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": prompt, "client_id": self.client_id},
            timeout=self.timeout_s,
        )
        if response.status_code >= 400:
            raise ComfyError(f"ComfyUI rejected prompt: {response.status_code} {response.text}")
        data = response.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyError(f"ComfyUI response did not include prompt_id: {data}")
        return str(prompt_id)

    def wait_for_history(self, prompt_id: str, timeout_s: int = 900, poll_s: float = 2.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=self.timeout_s)
            response.raise_for_status()
            history = response.json()
            if prompt_id in history:
                record = history[prompt_id]
                if record.get("status", {}).get("completed") is False:
                    raise ComfyError(f"ComfyUI prompt failed: {record.get('status')}")
                return record
            time.sleep(poll_s)
        raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}")

    def download_images(self, history_record: dict[str, Any], output_dir: Path, stem: str) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        images: list[Path] = []
        image_index = 0
        for node_output in history_record.get("outputs", {}).values():
            for image in node_output.get("images", []):
                query = urlencode(
                    {
                        "filename": image["filename"],
                        "subfolder": image.get("subfolder", ""),
                        "type": image.get("type", "output"),
                    }
                )
                response = requests.get(f"{self.base_url}/view?{query}", timeout=self.timeout_s)
                response.raise_for_status()
                suffix = Path(image["filename"]).suffix or ".png"
                path = output_dir / f"{stem}_{image_index}{suffix}"
                path.write_bytes(response.content)
                images.append(path)
                image_index += 1
        if not images:
            raise ComfyError(f"No images found in ComfyUI history output: {history_record.keys()}")
        return images


def validate_required_nodes(client: ComfyClient, required: set[str]) -> list[str]:
    info = client.object_info()
    missing = sorted(node for node in required if node not in info)
    return missing


def validate_model_assets(client: ComfyClient, defaults: ModelDefaults = DEFAULTS) -> dict[str, str]:
    checks = {
        "UNETLoader.unet_name": ("UNETLoader", "unet_name", defaults.diffusion_model),
        "CLIPLoader.clip_name": ("CLIPLoader", "clip_name", defaults.text_encoder),
        "VAELoader.vae_name": ("VAELoader", "vae_name", defaults.vae),
        "LoraLoaderModelOnly.lora_name": ("LoraLoaderModelOnly", "lora_name", defaults.pixel_lora),
    }
    missing: dict[str, str] = {}
    for label, (node_type, input_name, expected) in checks.items():
        values = _combo_values(client.object_info(node_type), node_type, input_name)
        if expected not in values:
            missing[label] = expected
    return missing


def _combo_values(info: dict[str, Any], node_type: str, input_name: str) -> list[str]:
    required = info.get(node_type, {}).get("input", {}).get("required", {})
    spec = required.get(input_name)
    if isinstance(spec, list) and spec and isinstance(spec[0], list):
        return [str(value) for value in spec[0]]
    return []
