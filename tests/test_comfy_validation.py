from sprite_motif_pipeline import comfy
from sprite_motif_pipeline.comfy import ComfyClient, ComfyServerLease, validate_model_assets


class FakeClient:
    def __init__(self, values):
        self.values = values

    def object_info(self, node_type=None):
        field = {
            "UNETLoader": "unet_name",
            "CLIPLoader": "clip_name",
            "VAELoader": "vae_name",
            "LoraLoaderModelOnly": "lora_name",
        }[node_type]
        return {
            node_type: {
                "input": {
                    "required": {
                        field: [self.values.get(node_type, []), {}],
                    }
                }
            }
        }


def test_validate_model_assets_reports_missing_defaults():
    missing = validate_model_assets(FakeClient({"VAELoader": ["qwen_image_vae.safetensors"]}))
    assert "UNETLoader.unet_name" in missing
    assert "CLIPLoader.clip_name" in missing
    assert "LoraLoaderModelOnly.lora_name" in missing
    assert "VAELoader.vae_name" not in missing


def test_free_memory_requests_model_unload(monkeypatch):
    calls = {}

    class Response:
        def raise_for_status(self):
            return None

    def fake_post(url, *, json, timeout):
        calls.update(url=url, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(comfy.requests, "post", fake_post)

    ComfyClient("http://127.0.0.1:8188", timeout_s=7).free_memory()

    assert calls["url"] == "http://127.0.0.1:8188/free"
    assert calls["json"] == {"unload_models": True, "free_memory": True}
    assert calls["timeout"] == 7


def test_stop_comfy_terminates_pipeline_owned_process(monkeypatch):
    process = object()
    calls = []
    monkeypatch.setattr(comfy, "terminate_process_tree", lambda value: calls.append(value))

    status = comfy.stop_comfyui_server(
        ComfyServerLease("http://127.0.0.1:8188", process, True),  # type: ignore[arg-type]
    )

    assert status == "stopped"
    assert calls == [process]


def test_stop_comfy_releases_but_keeps_preexisting_server(monkeypatch):
    calls = []

    class ReusedClient:
        def __init__(self, base_url, timeout_s):
            calls.append((base_url, timeout_s))

        def free_memory(self):
            calls.append("freed")

    monkeypatch.setattr(comfy, "comfyui_is_ready", lambda _base_url: True)
    monkeypatch.setattr(comfy, "ComfyClient", ReusedClient)

    status = comfy.stop_comfyui_server(ComfyServerLease("http://127.0.0.1:8188", None, False))

    assert status == "reused"
    assert calls[-1] == "freed"


def test_progress_encoding_error_cannot_abort_comfy_lifecycle():
    messages = []

    def cp1252_progress(message):
        message.encode("cp1252")
        messages.append(message)

    comfy._emit(cp1252_progress, "log C:/用户/comfyui.log")

    assert messages == [r"log C:/\u7528\u6237/comfyui.log"]
