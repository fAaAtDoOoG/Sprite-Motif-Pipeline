import pytest

import sprite_motif_pipeline.prompting as prompting
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


def test_llm_negative_prompt_is_preserved_and_enforced(monkeypatch):
    def fake_call_ollama(messages, config):
        return (
            '{"positive_prompt":"Pixel Art, one original full-body shadow knight, static pose, centered, facing right, '
            'plain neutral background, no readable text, designed to downscale cleanly to 64x64",'
            '"negative_prompt":"extra swords, sunny cheerful mood"}'
        )

    monkeypatch.setattr(prompting, "_call_ollama", fake_call_ollama)

    spec = compose_prompt(
        "shadow knight with one broken sword",
        llm_config=LLMConfig(provider="ollama", model="fake"),
        allow_fallback=False,
    )

    assert "extra swords" in spec.negative_prompt
    assert "photorealistic rendering" in spec.negative_prompt
    assert "busy background" in spec.negative_prompt
    assert "watermark" in spec.negative_prompt
    assert spec.source == "ollama"


def test_llm_receives_previous_negative_prompt(monkeypatch):
    captured = {}

    def fake_call_ollama(messages, config):
        captured["messages"] = messages
        return (
            '{"positive_prompt":"Pixel Art, one original full-body mage, static pose, centered, facing right, '
            'plain neutral background, no readable text, designed to downscale cleanly to 64x64",'
            '"negative_prompt":"no staff glow"}'
        )

    monkeypatch.setattr(prompting, "_call_ollama", fake_call_ollama)

    compose_prompt(
        "forest mage",
        previous_prompt="Pixel Art, forest mage with staff",
        previous_negative_prompt="glowing staff, busy forest background",
        feedback="remove glow",
        llm_config=LLMConfig(provider="ollama", model="fake"),
        allow_fallback=False,
    )

    payload = captured["messages"][1]["content"]
    assert "previous_negative_prompt" in payload
    assert "glowing staff" in payload


def test_ollama_keep_alive_defaults_to_unload(monkeypatch):
    monkeypatch.delenv("SPRITEPIPE_LLM_KEEP_ALIVE", raising=False)

    assert LLMConfig.from_env().keep_alive == "0"
    assert _coerce_keep_alive("0") == 0
    assert _coerce_keep_alive("5m") == "5m"
