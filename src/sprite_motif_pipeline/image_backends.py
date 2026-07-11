from __future__ import annotations

import base64
import binascii
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes, urljoin, urlparse, urlunparse

import requests

IMAGE_BACKEND_QWEN = "qwen-comfy"
IMAGE_BACKEND_CUSTOM_COMFY = "custom-comfy"
IMAGE_BACKEND_OPENAI = "openai-images"
IMAGE_BACKENDS = (
    IMAGE_BACKEND_QWEN,
    IMAGE_BACKEND_CUSTOM_COMFY,
    IMAGE_BACKEND_OPENAI,
)

DEFAULT_OPENAI_IMAGES_ENDPOINT = "https://api.openai.com/v1/images/generations"
DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-1"

CUSTOM_WORKFLOW_PLACEHOLDERS = {
    "positive_prompt",
    "negative_prompt",
    "width",
    "height",
    "seed",
    "steps",
    "cfg",
    "filename_prefix",
    "model",
    "lora_name",
    "lora_strength",
}

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def normalize_image_backend(value: str | None) -> str:
    backend = (value or IMAGE_BACKEND_QWEN).strip().lower()
    aliases = {
        "qwen": IMAGE_BACKEND_QWEN,
        "comfy": IMAGE_BACKEND_QWEN,
        "comfyui": IMAGE_BACKEND_QWEN,
        "custom": IMAGE_BACKEND_CUSTOM_COMFY,
        "openai": IMAGE_BACKEND_OPENAI,
        "api": IMAGE_BACKEND_OPENAI,
    }
    backend = aliases.get(backend, backend)
    if backend not in IMAGE_BACKENDS:
        raise ValueError(f"unsupported image backend '{value}'")
    return backend


def load_custom_workflow(path: str | Path | None) -> dict[str, Any]:
    if path is None or not str(path).strip():
        raise ValueError("A custom ComfyUI API workflow JSON path is required.")
    workflow_path = Path(path).expanduser()
    if not workflow_path.is_file():
        raise FileNotFoundError(f"Custom ComfyUI workflow not found: {workflow_path}")
    try:
        data = json.loads(workflow_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Custom ComfyUI workflow is not valid JSON: {workflow_path}") from exc

    if isinstance(data, dict) and isinstance(data.get("prompt"), dict):
        data = data["prompt"]
    _validate_comfy_workflow(data)
    return data


def render_custom_workflow(
    template: dict[str, Any],
    *,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    filename_prefix: str,
    model: str = "",
    lora_name: str = "",
    lora_strength: float = 0.0,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "filename_prefix": filename_prefix,
        "model": model,
        "lora_name": lora_name,
        "lora_strength": lora_strength,
    }
    rendered = _replace_placeholders(template, values)
    _validate_comfy_workflow(rendered)
    return rendered


def custom_workflow_node_types(template: dict[str, Any]) -> set[str]:
    _validate_comfy_workflow(template)
    return {str(node["class_type"]) for node in template.values()}


def build_openai_image_request(
    *,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    model: str,
) -> dict[str, Any]:
    prompt = positive_prompt.strip()
    negative = negative_prompt.strip()
    if negative:
        prompt = f"{prompt}\n\nAvoid these visual traits: {negative}"
    request: dict[str, Any] = {
        "prompt": prompt,
        "n": 1,
        "size": f"{width}x{height}",
    }
    if model.strip():
        request["model"] = model.strip()
    return request


def generate_openai_image(
    request: dict[str, Any],
    *,
    endpoint: str,
    api_key: str,
    timeout_s: int,
    output_stem: Path,
) -> Path:
    url = normalize_openai_images_endpoint(endpoint)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = requests.post(url, headers=headers, json=request, timeout=timeout_s)
    except requests.RequestException as exc:
        raise RuntimeError(f"Images API is not reachable at {public_endpoint(url)}.") from exc
    if response.status_code >= 400:
        raise RuntimeError(f"Images API rejected the request with HTTP {response.status_code}.")
    try:
        payload = response.json()
        item = payload["data"][0]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Images API response did not contain data[0].b64_json or data[0].url.") from exc

    image_bytes = _image_bytes_from_item(item, timeout_s=timeout_s, api_endpoint=url, api_key=api_key)
    suffix = _image_suffix(image_bytes)
    output_path = output_stem.with_suffix(suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    return output_path


def normalize_openai_images_endpoint(endpoint: str | None) -> str:
    value = (endpoint or DEFAULT_OPENAI_IMAGES_ENDPOINT).strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Images API endpoint must be an http:// or https:// URL.")
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1/images/generations"
    elif path.endswith("/v1"):
        path = f"{path}/images/generations"
    return urlunparse(parsed._replace(path=path, query=parsed.query, fragment=""))


def public_endpoint(endpoint: str | None) -> str:
    if not endpoint:
        return ""
    parsed = urlparse(endpoint)
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urlunparse(parsed._replace(netloc=netloc, params="", query="", fragment=""))


def _validate_comfy_workflow(workflow: Any) -> None:
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("Custom ComfyUI workflow must be a non-empty API-format JSON object.")
    invalid = [
        str(node_id)
        for node_id, node in workflow.items()
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str) or not node["class_type"].strip()
    ]
    if invalid:
        raise ValueError(
            "Custom ComfyUI workflow must use API format; invalid node(s): " + ", ".join(invalid[:10])
        )


def _replace_placeholders(value: Any, replacements: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _replace_placeholders(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_placeholders(item, replacements) for item in value]
    if not isinstance(value, str):
        return value

    full_match = _PLACEHOLDER.fullmatch(value)
    if full_match:
        name = full_match.group(1)
        if name not in CUSTOM_WORKFLOW_PLACEHOLDERS or name not in replacements:
            raise ValueError(f"Unknown custom workflow placeholder '{{{{{name}}}}}'.")
        return replacements[name]

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in CUSTOM_WORKFLOW_PLACEHOLDERS or name not in replacements:
            raise ValueError(f"Unknown custom workflow placeholder '{{{{{name}}}}}'.")
        return str(replacements[name])

    rendered = _PLACEHOLDER.sub(replace, value)
    if "{{" in rendered or "}}" in rendered:
        raise ValueError(f"Unsupported or malformed custom workflow placeholder in '{value}'.")
    return rendered


def _image_bytes_from_item(item: Any, *, timeout_s: int, api_endpoint: str, api_key: str) -> bytes:
    if not isinstance(item, dict):
        raise RuntimeError("Images API data[0] must be an object.")
    encoded = item.get("b64_json")
    if isinstance(encoded, str) and encoded:
        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError("Images API returned invalid base64 image data.") from exc

    image_url = item.get("url")
    if not isinstance(image_url, str) or not image_url:
        raise RuntimeError("Images API response did not contain b64_json or url image data.")
    if image_url.startswith("data:"):
        return _decode_data_url(image_url)
    image_url = urljoin(api_endpoint, image_url)
    headers: dict[str, str] = {}
    if api_key and _same_origin(api_endpoint, image_url):
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = requests.get(image_url, headers=headers, timeout=timeout_s)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError("Images API returned an image URL that could not be downloaded.") from exc
    return response.content


def _same_origin(left: str, right: str) -> bool:
    first = urlparse(left)
    second = urlparse(right)
    first_port = first.port or (443 if first.scheme == "https" else 80)
    second_port = second.port or (443 if second.scheme == "https" else 80)
    return (first.scheme.lower(), (first.hostname or "").lower(), first_port) == (
        second.scheme.lower(),
        (second.hostname or "").lower(),
        second_port,
    )


def _decode_data_url(value: str) -> bytes:
    try:
        header, data = value.split(",", 1)
    except ValueError as exc:
        raise RuntimeError("Images API returned an invalid data URL.") from exc
    if ";base64" in header.lower():
        try:
            return base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError("Images API returned an invalid base64 data URL.") from exc
    return unquote_to_bytes(data)


def _image_suffix(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return ".png"
