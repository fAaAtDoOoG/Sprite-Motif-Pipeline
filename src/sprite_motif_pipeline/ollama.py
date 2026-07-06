from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import requests

from .model_assets import human_bytes

ProgressCallback = Callable[[str], None]

DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class OllamaValidation:
    endpoint: str
    model: str
    server_available: bool
    cli_available: bool
    model_present: bool
    version: str = ""
    models: tuple[str, ...] = ()
    message: str = ""


def normalize_ollama_endpoint(endpoint: str | None) -> str:
    return (endpoint or DEFAULT_OLLAMA_ENDPOINT).strip().rstrip("/") or DEFAULT_OLLAMA_ENDPOINT


def find_ollama_executable() -> Path | None:
    env_path = os.environ.get("OLLAMA_EXE", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    resolved = shutil.which("ollama")
    if resolved:
        candidates.append(Path(resolved))

    local_app_data = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", "")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe")
    if program_files:
        candidates.append(Path(program_files) / "Ollama" / "ollama.exe")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def validate_ollama_model(
    endpoint: str,
    model: str,
    *,
    auto_start: bool = True,
    progress: ProgressCallback | None = None,
) -> OllamaValidation:
    endpoint = normalize_ollama_endpoint(endpoint)
    model = model.strip()
    cli = find_ollama_executable()
    version = ollama_version(endpoint)

    if not version and auto_start:
        start_ollama_server(endpoint, progress=progress)
        version = ollama_version(endpoint)

    if not version:
        return OllamaValidation(
            endpoint=endpoint,
            model=model,
            server_available=False,
            cli_available=cli is not None,
            model_present=False,
            message="Ollama server is not reachable.",
        )

    models = list_ollama_models(endpoint)
    present = ollama_model_present(model, models)
    return OllamaValidation(
        endpoint=endpoint,
        model=model,
        server_available=True,
        cli_available=cli is not None,
        model_present=present,
        version=version,
        models=models,
        message="ready" if present else "model is missing",
    )


def start_ollama_server(endpoint: str, *, progress: ProgressCallback | None = None, timeout_s: int = 30) -> bool:
    endpoint = normalize_ollama_endpoint(endpoint)
    if ollama_version(endpoint):
        return True
    if not _is_local_endpoint(endpoint):
        _emit(progress, f"cannot auto-start non-local Ollama endpoint: {endpoint}")
        return False

    executable = find_ollama_executable()
    if executable is None:
        _emit(progress, "ollama executable not found")
        return False

    _emit(progress, f"starting Ollama server: {executable}")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        [str(executable), "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if ollama_version(endpoint):
            _emit(progress, "Ollama server is running")
            return True
        time.sleep(0.5)
    _emit(progress, "Ollama server did not become ready in time")
    return False


def ollama_version(endpoint: str) -> str:
    endpoint = normalize_ollama_endpoint(endpoint)
    try:
        response = requests.get(f"{endpoint}/api/version", timeout=3)
        response.raise_for_status()
        version = response.json().get("version", "")
    except (requests.RequestException, ValueError):
        return ""
    return str(version or "unknown")


def list_ollama_models(endpoint: str) -> tuple[str, ...]:
    endpoint = normalize_ollama_endpoint(endpoint)
    response = requests.get(f"{endpoint}/api/tags", timeout=10)
    response.raise_for_status()
    data = response.json()
    models = data.get("models", [])
    names = [str(item.get("name", "")).strip() for item in models if isinstance(item, dict)]
    return tuple(name for name in names if name)


def ollama_model_present(model: str, models: tuple[str, ...] | list[str]) -> bool:
    requested = model.strip()
    if not requested:
        return False
    if requested in models:
        return True
    if ":" not in requested:
        prefix = f"{requested}:"
        return any(name.startswith(prefix) for name in models)
    return False


def pull_ollama_model(endpoint: str, model: str, *, progress: ProgressCallback | None = None) -> str:
    endpoint = normalize_ollama_endpoint(endpoint)
    model = model.strip()
    if not model:
        raise ValueError("Ollama model name is empty.")
    if not ollama_version(endpoint) and not start_ollama_server(endpoint, progress=progress):
        raise RuntimeError("Ollama is not running and could not be started automatically.")

    _emit(progress, f"pull Ollama model {model}")
    with requests.post(
        f"{endpoint}/api/pull",
        json={"name": model, "stream": True},
        stream=True,
        timeout=(10, 600),
    ) as response:
        response.raise_for_status()
        last_message = ""
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            event = json.loads(raw_line)
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            message = format_pull_progress(model, event)
            if message and message != last_message:
                last_message = message
                _emit(progress, message)

    models = list_ollama_models(endpoint)
    if not ollama_model_present(model, models):
        raise RuntimeError(f"Ollama pull finished, but model '{model}' is still not listed.")
    _emit(progress, f"done Ollama model {model}")
    return model


def unload_ollama_model(endpoint: str, model: str, *, progress: ProgressCallback | None = None) -> str:
    endpoint = normalize_ollama_endpoint(endpoint)
    model = model.strip()
    if not model:
        raise ValueError("Ollama model name is empty.")
    if not ollama_version(endpoint):
        raise RuntimeError("Ollama is not running.")

    _emit(progress, f"unload Ollama model {model}")
    response = requests.post(
        f"{endpoint}/api/chat",
        json={"model": model, "messages": [], "keep_alive": 0},
        timeout=30,
    )
    response.raise_for_status()
    _emit(progress, f"unloaded Ollama model {model}")
    return model


def format_pull_progress(model: str, event: dict[str, object]) -> str:
    status = str(event.get("status", "")).strip()
    completed = _int_or_none(event.get("completed"))
    total = _int_or_none(event.get("total"))
    if completed is not None and total:
        percent = max(0, min(100, int(completed * 100 / total)))
        detail = f"{human_bytes(completed)} / {human_bytes(total)}"
        return f"{model}: {percent}% ({detail}) {status}".strip()
    return f"{model}: {status}".strip() if status else ""


def _is_local_endpoint(endpoint: str) -> bool:
    host = urlparse(endpoint).hostname
    return host in {None, "localhost", "127.0.0.1", "::1"}


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
