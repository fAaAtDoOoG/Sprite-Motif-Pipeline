from sprite_motif_pipeline.cli import build_parser
from sprite_motif_pipeline.image_backends import IMAGE_BACKEND_CUSTOM_COMFY, IMAGE_BACKEND_OPENAI


def test_generate_cli_accepts_images_api_and_prompt_api_settings():
    args = build_parser().parse_args(
        [
            "generate",
            "--description",
            "small mage",
            "--llm-provider",
            "openai-compatible",
            "--llm-model",
            "local-chat",
            "--llm-endpoint",
            "http://localhost:8000/v1/chat/completions",
            "--llm-api-key",
            "prompt-secret",
            "--image-backend",
            IMAGE_BACKEND_OPENAI,
            "--image-model",
            "local-image",
            "--image-endpoint",
            "http://localhost:9000/v1/images/generations",
            "--image-api-key",
            "image-secret",
        ]
    )

    assert args.llm_model == "local-chat"
    assert args.llm_api_key == "prompt-secret"
    assert args.image_backend == IMAGE_BACKEND_OPENAI
    assert args.image_model == "local-image"
    assert args.image_api_key == "image-secret"


def test_iterate_cli_accepts_custom_comfy_workflow():
    args = build_parser().parse_args(
        [
            "iterate",
            "runs/run_test",
            "--index",
            "0",
            "--feedback",
            "make it taller",
            "--llm-provider",
            "ollama",
            "--llm-model",
            "qwen3:32b",
            "--image-backend",
            IMAGE_BACKEND_CUSTOM_COMFY,
            "--custom-workflow",
            "workflows/custom.json",
        ]
    )

    assert args.llm_model == "qwen3:32b"
    assert args.image_backend == IMAGE_BACKEND_CUSTOM_COMFY
    assert args.custom_workflow.name == "custom.json"
