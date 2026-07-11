import json
from pathlib import Path

from sprite_motif_pipeline.prompting import PromptSpec
from sprite_motif_pipeline.image_backends import IMAGE_BACKEND_CUSTOM_COMFY, IMAGE_BACKEND_OPENAI
from sprite_motif_pipeline.runner import GenerationOptions, generate_batch, seeds_for_batch
from sprite_motif_pipeline.session import load_manifest, user_input_history


def test_seeds_for_batch_are_sequential_with_base_seed():
    assert seeds_for_batch(10, 3) == [10, 11, 12]


def test_seeds_for_batch_random_count():
    assert len(seeds_for_batch(None, 4)) == 4


def test_generate_batch_records_original_user_prompt(tmp_path):
    spec = PromptSpec(
        positive_prompt="Pixel Art, one original full-body character",
        negative_prompt="photorealistic rendering",
        source="direct",
    )

    run_dir = generate_batch(
        spec,
        description="raw user prompt",
        options=GenerationOptions(batch_size=1, output_dir=tmp_path, dry_run=True),
        user_input_kind="direct_prompt",
        user_input_text="raw user prompt",
    )

    manifest = load_manifest(run_dir)
    assert manifest.user_inputs[0].kind == "direct_prompt"
    assert manifest.user_inputs[0].text == "raw user prompt"
    assert manifest.user_inputs[0].created_at


def test_generate_batch_uses_a_distinct_prompt_pair_for_each_candidate(tmp_path):
    specs = [
        PromptSpec("Pixel Art, candidate direction one", "negative direction one", "ollama"),
        PromptSpec("Pixel Art, candidate direction two", "negative direction two", "ollama"),
    ]

    run_dir = generate_batch(
        specs,
        description="raw description",
        options=GenerationOptions(batch_size=2, output_dir=tmp_path, dry_run=True),
    )

    manifest = load_manifest(run_dir)
    assert [candidate.positive_prompt for candidate in manifest.candidates] == [spec.positive_prompt for spec in specs]
    assert [candidate.negative_prompt for candidate in manifest.candidates] == [spec.negative_prompt for spec in specs]
    for candidate, spec in zip(manifest.candidates, specs, strict=True):
        api_prompt = json.loads(Path(candidate.api_prompt_path).read_text(encoding="utf-8"))
        serialized = json.dumps(api_prompt)
        assert spec.positive_prompt in serialized
        assert spec.negative_prompt in serialized


def test_generate_batch_can_export_a_custom_comfy_workflow(tmp_path):
    workflow = tmp_path / "workflow.json"
    workflow.write_text(
        json.dumps(
            {
                "1": {
                    "class_type": "CustomTextToImage",
                    "inputs": {
                        "prompt": "{{positive_prompt}}",
                        "negative": "{{negative_prompt}}",
                        "width": "{{width}}",
                        "height": "{{height}}",
                        "seed": "{{seed}}",
                        "model": "{{model}}",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    spec = PromptSpec("Pixel Art, custom subject", "blur", "direct")

    run_dir = generate_batch(
        spec,
        description="custom subject",
        options=GenerationOptions(
            batch_size=1,
            output_dir=tmp_path / "runs",
            dry_run=True,
            image_backend=IMAGE_BACKEND_CUSTOM_COMFY,
            image_model="my-local-model",
            custom_workflow=workflow,
        ),
    )

    manifest = load_manifest(run_dir)
    request = json.loads(Path(manifest.candidates[0].api_prompt_path).read_text(encoding="utf-8"))
    assert request["1"]["inputs"]["model"] == "my-local-model"
    assert request["1"]["inputs"]["width"] == 1024
    assert manifest.image_backend == IMAGE_BACKEND_CUSTOM_COMFY
    assert manifest.image_model == "my-local-model"


def test_builtin_qwen_backend_uses_the_selected_diffusion_filename(tmp_path):
    spec = PromptSpec("Pixel Art, selected model subject", "blur", "direct")

    run_dir = generate_batch(
        spec,
        description="selected model subject",
        options=GenerationOptions(
            batch_size=1,
            output_dir=tmp_path,
            dry_run=True,
            image_model="alternate_qwen_model.safetensors",
        ),
    )

    manifest = load_manifest(run_dir)
    request = json.loads(Path(manifest.candidates[0].api_prompt_path).read_text(encoding="utf-8"))
    assert request["1"]["inputs"]["unet_name"] == "alternate_qwen_model.safetensors"
    assert manifest.image_model == "alternate_qwen_model.safetensors"


def test_images_api_dry_run_never_writes_api_key_to_artifacts(tmp_path):
    spec = PromptSpec("Pixel Art, API subject", "blur", "direct")

    run_dir = generate_batch(
        spec,
        description="API subject",
        options=GenerationOptions(
            batch_size=1,
            output_dir=tmp_path,
            dry_run=True,
            image_backend=IMAGE_BACKEND_OPENAI,
            image_model="local-image-model",
            image_endpoint="https://user:password@example.com/v1/images/generations?token=secret",
            image_api_key="top-secret",
        ),
    )

    manifest_path = run_dir / "manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = load_manifest(run_dir)
    request_text = Path(manifest.candidates[0].api_prompt_path).read_text(encoding="utf-8")
    assert "top-secret" not in manifest_text
    assert "password" not in manifest_text
    assert "token=secret" not in manifest_text
    assert "top-secret" not in request_text
    assert manifest.image_endpoint == "https://example.com/v1/images/generations"


def test_images_api_direct_runner_use_gets_an_api_model_default(tmp_path):
    spec = PromptSpec("Pixel Art, API default subject", "blur", "direct")

    run_dir = generate_batch(
        spec,
        description="API default subject",
        options=GenerationOptions(
            batch_size=1,
            output_dir=tmp_path,
            dry_run=True,
            image_backend=IMAGE_BACKEND_OPENAI,
        ),
    )

    manifest = load_manifest(run_dir)
    request = json.loads(Path(manifest.candidates[0].api_prompt_path).read_text(encoding="utf-8"))
    assert request["model"] == "gpt-image-1"
    assert not request["model"].endswith(".safetensors")


def test_user_input_history_falls_back_to_description(tmp_path):
    run_dir = tmp_path / "run_old"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        '{"run_id":"run_old","description":"old user text","prompt_source":"fallback","prompt_notes":"","high_res":"1024x1024","low_res":"64x64","candidates":[]}',
        encoding="utf-8",
    )

    manifest = load_manifest(run_dir)
    history = user_input_history(manifest)

    assert history[0].kind == "description"
    assert history[0].text == "old user text"


def test_old_reference_metadata_is_ignored_when_loading_a_manifest(tmp_path):
    run_dir = tmp_path / "run_old_reference"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        '{"run_id":"run_old_reference","description":"old user text","prompt_source":"fallback","prompt_notes":"","high_res":"1024x1024","low_res":"64x64","generation_backend":"reference","reference_image":"reference.png","candidates":[]}',
        encoding="utf-8",
    )

    manifest = load_manifest(run_dir)

    assert manifest.run_id == "run_old_reference"
    assert not hasattr(manifest, "generation_backend")
