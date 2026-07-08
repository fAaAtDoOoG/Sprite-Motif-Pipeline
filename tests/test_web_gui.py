import pytest

from sprite_motif_pipeline.prompting import PromptSpec
from sprite_motif_pipeline import web_gui
from sprite_motif_pipeline.web_gui import APP_JS, INDEX_HTML, WebAppState, auto_start_comfy_job, format_prompt_preview, schedule_auto_start_comfy


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
    assert "app.js?v=8" in INDEX_HTML


def test_auto_start_comfy_job_reuses_running_comfy(monkeypatch):
    calls = {}

    def fake_start_comfy(payload, progress):
        calls["payload"] = payload
        progress("ComfyUI is already running", 100)
        return {"kind": "comfy_start", "status": "ready", "message": "ComfyUI is already running."}

    monkeypatch.setattr(web_gui, "start_comfy_job", fake_start_comfy)
    messages = []

    result = auto_start_comfy_job(lambda message, percent=None: messages.append((message, percent)))

    assert calls["payload"]["comfy_url"]
    assert result["kind"] == "comfy_start"
    assert result["auto"] is True
    assert result["status"] == "ready"
    assert ("ComfyUI is already running", 100) in messages


def test_auto_start_comfy_job_skips_missing_folder(monkeypatch):
    def fake_start_comfy(_payload, _progress):
        raise FileNotFoundError("missing test ComfyUI")

    monkeypatch.setattr(web_gui, "start_comfy_job", fake_start_comfy)

    result = auto_start_comfy_job(lambda _message, _percent=None: None)

    assert result["kind"] == "comfy_start"
    assert result["auto"] is True
    assert result["status"] == "not_found"
    assert "missing test ComfyUI" in result["message"]


def test_schedule_auto_start_comfy_uses_job_state(monkeypatch):
    def fake_auto_start(progress):
        progress("done fake comfy startup", 100)
        return {"kind": "comfy_start", "status": "ready"}

    monkeypatch.setattr(web_gui, "auto_start_comfy_job", fake_auto_start)
    state = WebAppState()

    result = schedule_auto_start_comfy(state)

    assert result and result["job_id"] == 1


def test_prompt_preview_includes_negative_prompt():
    spec = PromptSpec(
        positive_prompt="Pixel Art, character",
        negative_prompt="photorealistic rendering",
        source="test",
    )

    assert format_prompt_preview(spec) == "Pixel Art, character\n\nNegative:\nphotorealistic rendering"
