import pytest

from sprite_motif_pipeline.prompting import PromptSpec
from sprite_motif_pipeline.web_gui import APP_JS, INDEX_HTML, WebAppState, format_prompt_preview


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


def test_prompt_preview_includes_negative_prompt():
    spec = PromptSpec(
        positive_prompt="Pixel Art, character",
        negative_prompt="photorealistic rendering",
        source="test",
    )

    assert format_prompt_preview(spec) == "Pixel Art, character\n\nNegative:\nphotorealistic rendering"
