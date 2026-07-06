import pytest

from sprite_motif_pipeline.prompting import LLMConfig, _coerce_keep_alive, compose_prompt


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


def test_ollama_keep_alive_defaults_to_unload(monkeypatch):
    monkeypatch.delenv("SPRITEPIPE_LLM_KEEP_ALIVE", raising=False)

    assert LLMConfig.from_env().keep_alive == "0"
    assert _coerce_keep_alive("0") == 0
    assert _coerce_keep_alive("5m") == "5m"
