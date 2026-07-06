from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode, urlparse

import requests

from .config import DEFAULTS, ModelDefaults

ProgressCallback = Callable[[str], None]
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"


class ComfyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComfyLaunchPlan:
    command: tuple[str, ...]
    cwd: Path
    root: Path
    label: str


class ComfyClient:
    def __init__(self, base_url: str = DEFAULT_COMFY_URL, timeout_s: int = 30):
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


def default_comfy_dir(models_root: Path | None = None) -> Path:
    env = os.environ.get("SPRITEPIPE_COMFY_DIR", "").strip()
    if env:
        return Path(env)

    candidates: list[Path] = []
    if models_root:
        root = models_root.expanduser()
        if root.name.lower() == "models":
            candidates.append(root.parent)
    candidates.extend(
        [
            Path("D:/AI/ComfyUI"),
            Path("C:/AI/ComfyUI"),
            Path("D:/AI/ComfyUI_windows_portable"),
            Path("C:/AI/ComfyUI_windows_portable"),
            Path.home() / "ComfyUI",
            Path.cwd() / "ComfyUI",
        ]
    )
    for candidate in candidates:
        if _looks_like_comfyui_root(candidate):
            return candidate
    return candidates[0] if candidates else Path("D:/AI/ComfyUI")


def start_comfyui_server(
    base_url: str = DEFAULT_COMFY_URL,
    *,
    comfy_dir: str | Path = "",
    models_root: str | Path = "",
    progress: ProgressCallback | None = None,
    timeout_s: int = 180,
) -> dict[str, str]:
    base_url = (base_url or DEFAULT_COMFY_URL).rstrip("/")
    if comfyui_is_ready(base_url):
        return {"status": "ready", "message": "ComfyUI is already running.", "comfy_dir": str(comfy_dir or "")}

    if not _is_local_endpoint(base_url):
        raise RuntimeError(f"Cannot auto-start non-local ComfyUI endpoint: {base_url}")

    root = resolve_comfyui_dir(comfy_dir, models_root)
    plan = build_comfy_launch_plan(root, base_url)
    log_path = Path.cwd() / "logs" / "comfyui.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _emit(progress, f"starting ComfyUI with {plan.label}")
    _emit(progress, f"log {log_path}")

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    log_handle = log_path.open("ab")
    try:
        process = subprocess.Popen(
            plan.command,
            cwd=plan.cwd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    finally:
        log_handle.close()

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if comfyui_is_ready(base_url):
            _emit(progress, "ComfyUI is running")
            return {
                "status": "started",
                "message": "ComfyUI is running.",
                "comfy_dir": str(plan.root),
                "log_path": str(log_path),
            }
        exit_code = process.poll()
        if exit_code is not None:
            log_tail = _tail_text(log_path)
            detail = f" Last log lines: {log_tail}" if log_tail else ""
            raise RuntimeError(f"ComfyUI exited early with code {exit_code}.{detail} Check {log_path}.")
        time.sleep(2)
        _emit(progress, "waiting for ComfyUI")

    raise RuntimeError(f"ComfyUI was launched but is not reachable at {base_url}. Check {log_path}.")


def comfyui_is_ready(base_url: str = DEFAULT_COMFY_URL) -> bool:
    base_url = (base_url or DEFAULT_COMFY_URL).rstrip("/")
    for path in ("/system_stats", "/object_info"):
        try:
            response = requests.get(f"{base_url}{path}", timeout=3)
            if response.ok:
                return True
        except requests.RequestException:
            continue
    return False


def resolve_comfyui_dir(comfy_dir: str | Path = "", models_root: str | Path = "") -> Path:
    candidates: list[Path] = []
    if comfy_dir:
        candidates.append(Path(comfy_dir).expanduser())
    if models_root:
        models_path = Path(models_root).expanduser()
        if models_path.name.lower() == "models":
            candidates.append(models_path.parent)
    candidates.append(default_comfy_dir(Path(models_root).expanduser() if models_root else None))

    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate.absolute()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if _looks_like_comfyui_root(resolved):
            return resolved
    raise FileNotFoundError(f"Could not find a ComfyUI launchable folder. Tried: {', '.join(str(path) for path in candidates)}")


def build_comfy_launch_plan(root: Path, base_url: str = DEFAULT_COMFY_URL) -> ComfyLaunchPlan:
    root = root.expanduser().resolve()
    host, port = _host_port(base_url)
    batch_names = (
        "run_nvidia_gpu.bat",
        "run_nvidia_gpu_fast_fp16_accumulation.bat",
        "run_cpu.bat",
    )
    for name in batch_names:
        script = root / name
        if script.exists():
            return ComfyLaunchPlan(
                command=("cmd.exe", "/c", str(script)),
                cwd=root,
                root=root,
                label=name,
            )

    portable_main = root / "ComfyUI" / "main.py"
    portable_python = root / "python_embeded" / "python.exe"
    if portable_main.exists() and portable_python.exists():
        return ComfyLaunchPlan(
            command=(str(portable_python), "-s", str(portable_main), "--listen", host, "--port", str(port)),
            cwd=root,
            root=root,
            label="portable python_embeded",
        )

    main_py = root / "main.py"
    if main_py.exists():
        interpreter = _comfy_python(root)
        return ComfyLaunchPlan(
            command=(interpreter, str(main_py), "--listen", host, "--port", str(port)),
            cwd=root,
            root=root,
            label="main.py",
        )

    raise FileNotFoundError(f"No supported ComfyUI launch script was found in {root}.")


def _combo_values(info: dict[str, Any], node_type: str, input_name: str) -> list[str]:
    required = info.get(node_type, {}).get("input", {}).get("required", {})
    spec = required.get(input_name)
    if isinstance(spec, list) and spec and isinstance(spec[0], list):
        return [str(value) for value in spec[0]]
    return []


def _looks_like_comfyui_root(path: Path) -> bool:
    return any(
        candidate.exists()
        for candidate in (
            path / "main.py",
            path / "run_nvidia_gpu.bat",
            path / "run_cpu.bat",
            path / "ComfyUI" / "main.py",
        )
    )


def _is_local_endpoint(endpoint: str) -> bool:
    host = urlparse(endpoint).hostname
    return host in {None, "localhost", "127.0.0.1", "::1"}


def _host_port(endpoint: str) -> tuple[str, int]:
    parsed = urlparse(endpoint or DEFAULT_COMFY_URL)
    host = parsed.hostname or "127.0.0.1"
    if host == "localhost":
        host = "127.0.0.1"
    return host, parsed.port or 8188


def _comfy_python(root: Path) -> str:
    candidates = (
        root / ".venv" / "Scripts" / "python.exe",
        root / "venv" / "Scripts" / "python.exe",
        root / "python_embeded" / "python.exe",
        root / "python_embedded" / "python.exe",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _tail_text(path: Path, max_chars: int = 2000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:].strip()


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
