from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from sprite_motif_pipeline import web_gui
from sprite_motif_pipeline.config import DEFAULT_PROMPT_MODEL, DEFAULT_PROMPT_MODEL_NUM_GPU
from sprite_motif_pipeline.image_backends import IMAGE_BACKEND_CUSTOM_COMFY, IMAGE_BACKEND_OPENAI, IMAGE_BACKEND_QWEN
from sprite_motif_pipeline.prompting import PromptSpec
from sprite_motif_pipeline.runner import GenerationOptions
from sprite_motif_pipeline.web_gui import APP_JS, INDEX_HTML, WebAppState, default_payload, format_prompt_preview, format_prompt_previews


def test_heartbeat_can_arm_and_disarm_auto_shutdown():
    state = WebAppState(auto_shutdown_after_s=12)

    armed = state.heartbeat(auto_shutdown=True)
    assert armed["auto_shutdown_enabled"] is True
    assert armed["auto_shutdown_after_s"] == 12

    disarmed = state.heartbeat(auto_shutdown=False)
    assert disarmed["auto_shutdown_enabled"] is False


def test_shutdown_rejects_active_job_without_force():
    state = WebAppState()
    with state.lock:
        state.job.active = True

    with pytest.raises(RuntimeError, match="job is still running"):
        state.request_shutdown("test")

    assert state.request_shutdown("test", force=True)["ok"] is True


def test_browser_ui_contains_comparison_viewer_controls():
    assert "highPreview" in INDEX_HTML
    assert "lowPreview" in INDEX_HTML
    assert "viewerStage" in INDEX_HTML
    assert "zoomIn" in INDEX_HTML
    assert "showCandidateComparison" in APP_JS
    assert "setZoomAround" in APP_JS


def test_browser_ui_contains_local_server_start_controls():
    assert "startComfy" in INDEX_HTML
    assert "comfyDir" in INDEX_HTML
    assert "startLlm" in INDEX_HTML
    assert "/api/start-comfy" in APP_JS
    assert "/api/start-llm" in APP_JS


def test_browser_ui_contains_user_input_history():
    assert "inputHistory" in INDEX_HTML
    assert "renderUserInputs" in APP_JS
    assert "app.js?v=13" in INDEX_HTML


def test_browser_ui_is_text_to_image_only():
    assert "generationBackend" not in INDEX_HTML
    assert "referenceFile" not in INDEX_HTML
    assert "upload-reference" not in APP_JS
    assert "loraName" in INDEX_HTML


def test_browser_ui_defaults_to_32b_ollama_gpu_prompt_model():
    payload = default_payload()

    assert payload["llm_provider"] == "ollama"
    assert payload["llm_model"] == DEFAULT_PROMPT_MODEL
    assert payload["llm_num_gpu"] == DEFAULT_PROMPT_MODEL_NUM_GPU
    assert "llmNumGpu" in INDEX_HTML
    assert "llm_num_gpu" in APP_JS


def test_browser_ui_exposes_image_and_prompt_provider_credentials():
    payload = default_payload()

    assert payload["image_backend"] == IMAGE_BACKEND_QWEN
    assert "imageBackend" in INDEX_HTML
    assert "customWorkflow" in INDEX_HTML
    assert "imageApiKey" in INDEX_HTML
    assert "llmApiKey" in INDEX_HTML
    assert 'type="password"' in INDEX_HTML
    assert "image_api_key" in APP_JS
    assert "llm_api_key" in APP_JS


def test_browser_generation_does_not_prevalidate_and_leave_ollama_running():
    assert "ensurePromptModelReady" not in APP_JS


def test_default_payload_has_no_preloaded_character_description():
    payload = default_payload()

    assert payload["description"] == ""


def test_prompt_phase_starts_rewrites_and_stops_ollama(monkeypatch):
    events = []

    @contextmanager
    def fake_temporary_ollama(*_args, **_kwargs):
        events.append("ollama_start")
        try:
            yield object()
        finally:
            events.append("ollama_stop")

    monkeypatch.setattr(web_gui, "temporary_ollama_server", fake_temporary_ollama)
    monkeypatch.setattr(
        web_gui,
        "validate_ollama_model",
        lambda *_args, **_kwargs: SimpleNamespace(model_present=True),
    )
    payload = default_payload()
    config = web_gui.llm_config_from_payload(payload)
    expected = PromptSpec("Pixel Art, subject", "blur", "test")

    result = web_gui.run_with_temporary_prompt_service(
        payload,
        config,
        lambda *_args: None,
        lambda: events.append("rewrite") or expected,
    )

    assert result == expected
    assert events == ["ollama_start", "rewrite", "ollama_stop"]


def test_direct_prompt_phase_does_not_start_ollama(monkeypatch):
    monkeypatch.setattr(
        web_gui,
        "temporary_ollama_server",
        lambda *_args, **_kwargs: pytest.fail("direct prompt should skip Ollama"),
    )
    payload = {**default_payload(), "mode": "prompt"}
    expected = PromptSpec("Pixel Art, subject", "blur", "direct")

    result = web_gui.run_with_temporary_prompt_service(
        payload,
        web_gui.llm_config_from_payload(payload),
        lambda *_args: None,
        lambda: expected,
    )

    assert result == expected


def test_comfy_phase_starts_validates_generates_and_stops(monkeypatch):
    events = []

    @contextmanager
    def fake_temporary_comfy(*_args, **_kwargs):
        events.append("comfy_start")
        try:
            yield object()
        finally:
            events.append("comfy_stop")

    class FakeComfyClient:
        def __init__(self, _base_url):
            events.append("client")

    monkeypatch.setattr(web_gui, "temporary_comfyui_server", fake_temporary_comfy)
    monkeypatch.setattr(web_gui, "ComfyClient", FakeComfyClient)
    monkeypatch.setattr(web_gui, "validate_required_nodes", lambda *_args: events.append("nodes") or [])
    monkeypatch.setattr(web_gui, "validate_model_assets", lambda *_args: events.append("assets") or {})
    expected = web_gui.Path("run_test")

    result = web_gui.generate_with_temporary_comfy_service(
        default_payload(),
        GenerationOptions(),
        lambda *_args: None,
        lambda: events.append("generate") or expected,
    )

    assert result == expected
    assert events == ["comfy_start", "client", "nodes", "assets", "generate", "comfy_stop"]


def test_images_api_phase_does_not_start_comfy(monkeypatch):
    monkeypatch.setattr(
        web_gui,
        "temporary_comfyui_server",
        lambda *_args, **_kwargs: pytest.fail("Images API must not start ComfyUI"),
    )
    expected = web_gui.Path("run_api")
    events = []

    result = web_gui.generate_with_temporary_comfy_service(
        default_payload(),
        GenerationOptions(image_backend=IMAGE_BACKEND_OPENAI),
        lambda message, *_args: events.append(message),
        lambda: expected,
    )

    assert result == expected
    assert any("Images API" in message for message in events)


def test_custom_comfy_phase_validates_only_custom_workflow_nodes(monkeypatch, tmp_path):
    events = []
    workflow = tmp_path / "custom.json"
    workflow.write_text('{"1":{"class_type":"LocalImageNode","inputs":{}}}', encoding="utf-8")

    @contextmanager
    def fake_temporary_comfy(*_args, **_kwargs):
        events.append("comfy_start")
        try:
            yield object()
        finally:
            events.append("comfy_stop")

    monkeypatch.setattr(web_gui, "temporary_comfyui_server", fake_temporary_comfy)
    monkeypatch.setattr(web_gui, "ComfyClient", lambda _url: object())

    def fake_validate(_client, required):
        events.append(required)
        return []

    monkeypatch.setattr(web_gui, "validate_required_nodes", fake_validate)
    monkeypatch.setattr(
        web_gui,
        "validate_model_assets",
        lambda *_args: pytest.fail("custom workflows must not use Qwen asset validation"),
    )

    result = web_gui.generate_with_temporary_comfy_service(
        default_payload(),
        GenerationOptions(image_backend=IMAGE_BACKEND_CUSTOM_COMFY, custom_workflow=workflow),
        lambda *_args: None,
        lambda: web_gui.Path("run_custom"),
    )

    assert result == web_gui.Path("run_custom")
    assert {"LocalImageNode"} in events
    assert events[0] == "comfy_start"
    assert events[-1] == "comfy_stop"


def test_images_api_validation_never_contacts_comfy(monkeypatch):
    monkeypatch.setattr(web_gui, "ComfyClient", lambda *_args: pytest.fail("Images API validation must skip ComfyUI"))

    result = web_gui.validate_comfy_response({**default_payload(), "image_backend": IMAGE_BACKEND_OPENAI})

    assert result["status"] == "not_required"


def test_web_payload_accepts_transient_api_keys(monkeypatch):
    monkeypatch.delenv("SPRITEPIPE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("SPRITEPIPE_IMAGE_API_KEY", raising=False)
    payload = {
        **default_payload(),
        "image_backend": IMAGE_BACKEND_OPENAI,
        "image_model": "local-image-model",
        "image_api_key": "image-secret",
        "llm_provider": "openai-compatible",
        "llm_api_key": "prompt-secret",
    }

    options = web_gui.generation_options_from_payload(payload)
    config = web_gui.llm_config_from_payload(payload)

    assert options.image_api_key == "image-secret"
    assert config.api_key == "prompt-secret"


def test_prompt_preview_includes_negative_prompt():
    spec = PromptSpec(
        positive_prompt="Pixel Art, character",
        negative_prompt="photorealistic rendering",
        source="test",
    )

    assert format_prompt_preview(spec) == "Pixel Art, character\n\nNegative:\nphotorealistic rendering"


def test_batch_prompt_preview_labels_each_candidate():
    specs = [
        PromptSpec("Pixel Art, direction one", "negative one", "test"),
        PromptSpec("Pixel Art, direction two", "negative two", "test"),
    ]

    preview = format_prompt_previews(specs)

    assert "Candidate 1" in preview
    assert "Pixel Art, direction one" in preview
    assert "Candidate 2" in preview
    assert "negative two" in preview
