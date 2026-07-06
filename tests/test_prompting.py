import pytest

from sprite_motif_pipeline.prompting import LLMConfig, compose_prompt


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


def test_llm_failure_can_be_strict_instead_of_fallback():
    with pytest.raises(ValueError, match="unsupported LLM provider"):
        compose_prompt(
            "red knight",
            llm_config=LLMConfig(provider="unsupported-local-model"),
            allow_fallback=False,
        )
