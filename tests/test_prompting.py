import json

import pytest

import sprite_motif_pipeline.prompting as prompting
from sprite_motif_pipeline.config import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT_MODEL,
    DEFAULT_PROMPT_MODEL_NUM_CTX,
    DEFAULT_PROMPT_MODEL_NUM_GPU,
    DEFAULT_PROMPT_MODEL_NUM_PREDICT,
    DEFAULT_PROMPT_MODEL_TEMPERATURE,
    DEFAULT_PROMPT_MODEL_TIMEOUT,
)
from sprite_motif_pipeline.prompting import (
    LLMConfig,
    _coerce_keep_alive,
    _enforce_negative_constraints,
    compose_prompt,
    compose_prompt_batch,
)


def test_fallback_prompt_enforces_pixel_sprite_constraints():
    spec = compose_prompt("红发女骑士")
    assert spec.positive_prompt.startswith("Pixel Art")
    assert "facing right" in spec.positive_prompt
    assert "64x64" in spec.positive_prompt
    assert "one original full-body subject" in spec.positive_prompt
    assert "side-view" not in spec.positive_prompt
    assert "bold outer outline" not in spec.positive_prompt
    assert "limited color palette" not in spec.positive_prompt
    assert "crisp square-pixel shapes" not in spec.positive_prompt
    assert spec.source == "fallback"


def test_direct_prompt_can_force_trigger():
    spec = compose_prompt(direct_prompt="a tiny knight", force_pixel_trigger=True)
    assert spec.positive_prompt.startswith("Pixel Art")
    assert spec.source == "direct"


def test_global_prompt_defaults_are_content_neutral():
    forbidden = (
        "anime key visual",
        "soft airbrush",
        "heavy antialiasing",
        "muddy colors",
        "noisy edges",
        "duplicate limbs",
        "malformed hands",
        "deformed face",
        "weapon motion blur",
        "large background props",
        "tiny details that disappear",
    )
    combined = "\n".join((DEFAULT_NEGATIVE_PROMPT, prompting.SYSTEM_PROMPT)).lower()

    assert all(phrase not in combined for phrase in forbidden)
    assert "side-view" not in prompting.SYSTEM_PROMPT
    assert "three-quarter" not in prompting.SYSTEM_PROMPT
    assert "limited palette" not in prompting.SYSTEM_PROMPT
    assert "strong contrast" not in prompting.SYSTEM_PROMPT
    assert "black twisted shadow creature" not in prompting.SYSTEM_PROMPT


def test_llm_failure_can_be_strict_instead_of_fallback():
    with pytest.raises(ValueError, match="unsupported LLM provider"):
        compose_prompt(
            "red knight",
            llm_config=LLMConfig(provider="unsupported-local-model"),
            allow_fallback=False,
        )


def test_fallback_does_not_hardcode_negative_splitting():
    spec = compose_prompt(
        "\u9ed1\u5f71\u602a\u7269\uff0c\u8eab\u4f53\u7626\u5f31\uff0c\u65e0\u7259\u9f7f\uff0c\u53ef\u89c1\u808c\u8089\uff0c\u5916\u58f3\uff0c\u76d4\u7532"
    )

    assert "\u8eab\u4f53\u7626\u5f31" in spec.positive_prompt
    assert "\u65e0\u7259\u9f7f" in spec.positive_prompt
    assert "\u53ef\u89c1\u808c\u8089" in spec.positive_prompt
    assert "\u5916\u58f3" in spec.positive_prompt
    assert "\u76d4\u7532" in spec.positive_prompt
    assert "teeth" not in spec.negative_prompt
    assert "visible muscles" not in spec.negative_prompt
    assert "\u808c\u8089" not in spec.negative_prompt
    assert "\u5916\u58f3" not in spec.negative_prompt
    assert "\u76d4\u7532" not in spec.negative_prompt


def test_llm_receives_raw_user_text_for_semantic_classification(monkeypatch):
    captured = {"calls": []}

    def fake_call_ollama(messages, config):
        captured["calls"].append(
            {"system": messages[0]["content"], "payload": json.loads(messages[1]["content"]), "keep_alive": config.keep_alive}
        )
        return (
            '{"positive_prompt":"Pixel Art, one original full-body shadow creature, static pose, centered, facing right, '
            'visible muscles, shell armor, plain neutral background, no readable text, designed to downscale cleanly to 64x64",'
            '"negative_prompt":"teeth"}'
        )

    monkeypatch.setattr(prompting, "_call_ollama", fake_call_ollama)

    spec = compose_prompt(
        "\u9ed1\u5f71\u602a\u7269\uff0c\u65e0\u7259\u9f7f\uff0c\u53ef\u89c1\u808c\u8089\uff0c\u5916\u58f3\uff0c\u76d4\u7532",
        llm_config=LLMConfig(provider="ollama", model="fake"),
        allow_fallback=False,
    )

    assert len(captured["calls"]) == 1
    rewrite_call = captured["calls"][0]
    assert rewrite_call["keep_alive"] == "0"
    assert rewrite_call["payload"]["description"] == "\u9ed1\u5f71\u602a\u7269\uff0c\u65e0\u7259\u9f7f\uff0c\u53ef\u89c1\u808c\u8089\uff0c\u5916\u58f3\uff0c\u76d4\u7532"
    assert "negative_constraints" not in rewrite_call["payload"]
    assert "raw_user_description" not in rewrite_call["payload"]
    assert rewrite_call["payload"]["candidate_count"] == 1
    assert "bilingual game character concept artist" in rewrite_call["system"]
    assert "Every candidate preserves every stated" in rewrite_call["system"]
    assert "Guessing is required" in rewrite_call["system"]
    assert "different coherent solution" in rewrite_call["system"]
    assert "may not contradict an explicit fact" in rewrite_call["system"]
    assert "negative_prompt separately includes every explicit absence" in rewrite_call["system"]
    assert 'use "teeth", not "no teeth"' in rewrite_call["system"]
    assert "shadow/dark creature" not in rewrite_call["system"]
    assert "black twisted shadow creature" not in rewrite_call["system"]
    assert "teeth" in spec.negative_prompt
    assert "visible muscles" in spec.positive_prompt
    assert "shell armor" in spec.positive_prompt


def test_llm_creates_one_distinct_prompt_pair_per_batch_candidate(monkeypatch):
    captured = []
    candidate_data = [
        {
            "positive_prompt": (
                f"Pixel Art, one original full-body subject with a red cloak, static pose, centered, facing right, "
                f"plain neutral background, distinct visual direction {index}, designed to downscale cleanly to 64x64"
            ),
            "negative_prompt": f"blue cloak, duplicate direction {index}",
        }
        for index in range(1, 5)
    ]

    def fake_call_ollama(messages, config):
        captured.append(json.loads(messages[1]["content"]))
        return json.dumps({"candidates": candidate_data})

    monkeypatch.setattr(prompting, "_call_ollama", fake_call_ollama)

    specs = compose_prompt_batch(
        "a character with a red cloak and no blue cloak",
        candidate_count=4,
        llm_config=LLMConfig(provider="ollama", model="fake"),
        allow_fallback=False,
    )

    assert len(specs) == 4
    assert len({spec.positive_prompt for spec in specs}) == 4
    assert len({spec.negative_prompt for spec in specs}) == 4
    assert all("red cloak" in spec.positive_prompt for spec in specs)
    assert all("blue cloak" in spec.negative_prompt for spec in specs)
    assert captured[0]["candidate_count"] == 4
    assert len(captured) == 1


def test_core_constraints_do_not_duplicate_full_body_subject(monkeypatch):
    def fake_call_ollama(messages, config):
        return (
            '{"positive_prompt":"Pixel Art, one full-body black twisted shadow creature, static pose, centered, '
            'facing right, white eye on the head, plain neutral background, no readable text, '
            'designed to downscale clearly to 64x64",'
            '"negative_prompt":"teeth, muscles, exoskeleton, shell, armor"}'
        )

    monkeypatch.setattr(prompting, "_call_ollama", fake_call_ollama)

    spec = compose_prompt(
        "black twisted shadow creature",
        llm_config=LLMConfig(provider="ollama", model="fake"),
        allow_fallback=False,
    )

    assert spec.positive_prompt.count("full-body") == 1
    assert "one original full-body subject" not in spec.positive_prompt


def test_llm_semantic_judgment_is_not_keyword_sanitized(monkeypatch):
    def fake_call_ollama(messages, config):
        return (
            '{"positive_prompt":"Pixel Art, one original full-body shadow creature, static pose, centered, facing right, '
            'plain neutral background, no readable text, designed to downscale cleanly to 64x64",'
            '"negative_prompt":"\u808c\u8089, \u5916\u58f3, \u76d4\u7532, sunny cheerful mood"}'
        )

    monkeypatch.setattr(prompting, "_call_ollama", fake_call_ollama)

    spec = compose_prompt(
        "\u9ed1\u5f71\u602a\u7269\uff0c\u65e0\u7259\u9f7f\uff0c\u53ef\u89c1\u808c\u8089\uff0c\u5916\u58f3\uff0c\u76d4\u7532",
        llm_config=LLMConfig(provider="ollama", model="fake"),
        allow_fallback=False,
    )

    assert "\u808c\u8089" in spec.negative_prompt
    assert "\u5916\u58f3" in spec.negative_prompt
    assert "\u76d4\u7532" in spec.negative_prompt
    assert "sunny cheerful mood" in spec.negative_prompt


def test_llm_negative_prompt_is_preserved_and_enforced(monkeypatch):
    def fake_call_ollama(messages, config):
        return (
            '{"positive_prompt":"Pixel Art, one original full-body shadow knight, static pose, centered, facing right, '
            'plain neutral background, designed to downscale cleanly to 64x64",'
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
    assert "text" in spec.negative_prompt
    assert "watermark" in spec.negative_prompt
    assert "no readable text" not in spec.positive_prompt
    assert spec.source == "ollama"


def test_negated_positive_trait_is_mirrored_into_negative_prompt():
    negative = _enforce_negative_constraints("blur", "Pixel Art, an entity with no teeth, centered")

    assert "teeth" in negative


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


def test_ollama_prompt_model_defaults_for_gpu(monkeypatch):
    monkeypatch.setenv("SPRITEPIPE_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("SPRITEPIPE_LLM_MODEL", raising=False)
    monkeypatch.delenv("SPRITEPIPE_LLM_TIMEOUT", raising=False)
    monkeypatch.delenv("SPRITEPIPE_OLLAMA_NUM_GPU", raising=False)
    monkeypatch.delenv("SPRITEPIPE_OLLAMA_NUM_CTX", raising=False)
    monkeypatch.delenv("SPRITEPIPE_OLLAMA_NUM_PREDICT", raising=False)
    monkeypatch.delenv("SPRITEPIPE_LLM_TEMPERATURE", raising=False)

    config = LLMConfig.from_env()

    assert config.model == DEFAULT_PROMPT_MODEL
    assert config.timeout_s == DEFAULT_PROMPT_MODEL_TIMEOUT
    assert config.temperature == DEFAULT_PROMPT_MODEL_TEMPERATURE == 0.55
    assert config.ollama_num_gpu == DEFAULT_PROMPT_MODEL_NUM_GPU
    assert config.ollama_num_ctx == DEFAULT_PROMPT_MODEL_NUM_CTX
    assert config.ollama_num_predict == DEFAULT_PROMPT_MODEL_NUM_PREDICT
    assert config.think is False
