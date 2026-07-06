from sprite_motif_pipeline.comfy import validate_model_assets


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
