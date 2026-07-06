from sprite_motif_pipeline.prompting import compose_prompt


def test_fallback_prompt_enforces_pixel_sprite_constraints():
    spec = compose_prompt("红发女骑士")
    assert spec.positive_prompt.startswith("Pixel Art")
    assert "facing right" in spec.positive_prompt
    assert "64x64" in spec.positive_prompt
    assert spec.source == "fallback"


def test_direct_prompt_can_force_trigger():
    spec = compose_prompt(direct_prompt="a tiny knight", force_pixel_trigger=True)
    assert spec.positive_prompt.startswith("Pixel Art")
    assert spec.source == "direct"
