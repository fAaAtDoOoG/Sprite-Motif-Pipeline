from sprite_motif_pipeline.prompting import PromptSpec
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
