import base64
import json

import pytest

from sprite_motif_pipeline import image_backends
from sprite_motif_pipeline.image_backends import (
    build_openai_image_request,
    custom_workflow_node_types,
    generate_openai_image,
    load_custom_workflow,
    normalize_openai_images_endpoint,
    public_endpoint,
    render_custom_workflow,
)


def test_custom_workflow_replaces_typed_and_embedded_placeholders(tmp_path):
    path = tmp_path / "custom.json"
    path.write_text(
        json.dumps(
            {
                "1": {
                    "class_type": "CustomSampler",
                    "inputs": {
                        "text": "{{positive_prompt}}",
                        "negative": "{{negative_prompt}}",
                        "width": "{{width}}",
                        "seed": "{{seed}}",
                        "cfg": "{{cfg}}",
                        "model": "{{model}}",
                        "lora": "{{lora_name}}",
                        "strength": "{{lora_strength}}",
                    },
                },
                "2": {
                    "class_type": "SaveImage",
                    "inputs": {"filename_prefix": "run/{{filename_prefix}}"},
                },
            }
        ),
        encoding="utf-8",
    )

    template = load_custom_workflow(path)
    rendered = render_custom_workflow(
        template,
        positive_prompt="Pixel Art, subject",
        negative_prompt="blur",
        width=1024,
        height=1024,
        seed=42,
        steps=30,
        cfg=4.5,
        filename_prefix="candidate_00",
        model="local-model.safetensors",
        lora_name="pixel-style.safetensors",
        lora_strength=0.75,
    )

    assert rendered["1"]["inputs"]["width"] == 1024
    assert rendered["1"]["inputs"]["seed"] == 42
    assert rendered["1"]["inputs"]["cfg"] == 4.5
    assert rendered["1"]["inputs"]["model"] == "local-model.safetensors"
    assert rendered["1"]["inputs"]["lora"] == "pixel-style.safetensors"
    assert rendered["1"]["inputs"]["strength"] == 0.75
    assert rendered["2"]["inputs"]["filename_prefix"] == "run/candidate_00"
    assert custom_workflow_node_types(template) == {"CustomSampler", "SaveImage"}


def test_custom_workflow_rejects_ui_format(tmp_path):
    path = tmp_path / "ui.json"
    path.write_text('{"nodes": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="API format"):
        load_custom_workflow(path)


def test_openai_request_keeps_negative_prompt_separate_in_instruction():
    request = build_openai_image_request(
        positive_prompt="Pixel Art, shadow creature",
        negative_prompt="teeth, armor",
        width=1024,
        height=1024,
        model="image-model",
    )

    assert request["model"] == "image-model"
    assert request["size"] == "1024x1024"
    assert "Pixel Art, shadow creature" in request["prompt"]
    assert "Avoid these visual traits: teeth, armor" in request["prompt"]


def test_generate_openai_image_decodes_base64_without_exporting_key(monkeypatch, tmp_path):
    png = b"\x89PNG\r\n\x1a\nminimal"
    observed = {}

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"data": [{"b64_json": base64.b64encode(png).decode("ascii")}]}

    def fake_post(url, *, headers, json, timeout):
        observed.update(url=url, headers=headers, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(image_backends.requests, "post", fake_post)
    request = {"model": "local-image", "prompt": "Pixel Art, subject", "size": "1024x1024", "n": 1}

    output = generate_openai_image(
        request,
        endpoint="http://127.0.0.1:9000/v1",
        api_key="secret-token",
        timeout_s=90,
        output_stem=tmp_path / "candidate",
    )

    assert output.suffix == ".png"
    assert output.read_bytes() == png
    assert observed["url"] == "http://127.0.0.1:9000/v1/images/generations"
    assert observed["headers"]["Authorization"] == "Bearer secret-token"
    assert "secret-token" not in json.dumps(observed["json"])


def test_endpoint_helpers_remove_credentials_and_query_values():
    endpoint = "https://user:secret@example.com/v1/images/generations?token=private"

    assert public_endpoint(endpoint) == "https://example.com/v1/images/generations"
    assert normalize_openai_images_endpoint("http://localhost:8080") == "http://localhost:8080/v1/images/generations"
    assert image_backends._same_origin("https://api.example.com/v1/images", "https://api.example.com/output.png")
    assert not image_backends._same_origin("https://api.example.com/v1/images", "https://cdn.example.com/output.png")


def test_relative_image_url_uses_auth_only_on_the_api_origin(monkeypatch, tmp_path):
    png = b"\x89PNG\r\n\x1a\nrelative"
    observed = {}

    class ApiResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"data": [{"url": "/generated/candidate.png"}]}

    class ImageResponse:
        content = png

        @staticmethod
        def raise_for_status():
            return None

    monkeypatch.setattr(image_backends.requests, "post", lambda *_args, **_kwargs: ApiResponse())

    def fake_get(url, *, headers, timeout):
        observed.update(url=url, headers=headers, timeout=timeout)
        return ImageResponse()

    monkeypatch.setattr(image_backends.requests, "get", fake_get)

    output = generate_openai_image(
        {"prompt": "Pixel Art, subject"},
        endpoint="http://localhost:9000/v1/images/generations",
        api_key="local-secret",
        timeout_s=60,
        output_stem=tmp_path / "candidate",
    )

    assert output.read_bytes() == png
    assert observed["url"] == "http://localhost:9000/generated/candidate.png"
    assert observed["headers"] == {"Authorization": "Bearer local-secret"}


def test_connection_error_does_not_expose_endpoint_credentials(monkeypatch, tmp_path):
    raw_endpoint = "https://user:password@example.com/v1/images/generations?token=query-secret"

    def fail_post(*_args, **_kwargs):
        raise image_backends.requests.ConnectionError(f"could not connect to {raw_endpoint}")

    monkeypatch.setattr(image_backends.requests, "post", fail_post)

    with pytest.raises(RuntimeError) as captured:
        generate_openai_image(
            {"prompt": "Pixel Art, subject"},
            endpoint=raw_endpoint,
            api_key="header-secret",
            timeout_s=30,
            output_stem=tmp_path / "candidate",
        )

    message = str(captured.value)
    assert "https://example.com/v1/images/generations" in message
    assert "password" not in message
    assert "query-secret" not in message
    assert "header-secret" not in message
