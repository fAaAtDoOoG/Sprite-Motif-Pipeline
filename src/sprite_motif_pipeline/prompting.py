from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any, Iterable

import requests

from .config import DEFAULT_NEGATIVE_PROMPT

PIXEL_TRIGGER = "Pixel Art"

SYSTEM_PROMPT = """You rewrite short user descriptions into prompts for a text-to-image model.
The image target is always a static 2D pixel-art game character motif, generated high resolution and later downscaled to a tiny sprite.
Return strict JSON with keys positive_prompt and negative_prompt.

Hard requirements:
- The positive prompt must start with "Pixel Art".
- Describe exactly one full-body character, static, centered, facing right in side view or three-quarter side view.
- Make the silhouette readable at 64x64: clean outline, limited palette, strong contrast, large simple shapes.
- Prefer a neutral plain background and no readable text.
- Do not mention copyrighted characters, brands, logos, or living artists unless the user explicitly supplied them.
- Preserve the user's core identity, costume, mood, and revision request.
- If previous_prompt is provided, keep its successful character identity and only adjust the requested parts.
"""


@dataclass(frozen=True)
class PromptSpec:
    positive_prompt: str
    negative_prompt: str
    source: str
    notes: str = ""


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "none"
    model: str = ""
    endpoint: str = ""
    api_key: str = ""
    temperature: float = 0.2
    timeout_s: int = 60

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.environ.get("SPRITEPIPE_LLM_PROVIDER", "none").strip().lower()
        return cls(
            provider=provider,
            model=os.environ.get("SPRITEPIPE_LLM_MODEL", "").strip(),
            endpoint=os.environ.get("SPRITEPIPE_LLM_ENDPOINT", "").strip(),
            api_key=os.environ.get("SPRITEPIPE_LLM_API_KEY", "").strip(),
            temperature=float(os.environ.get("SPRITEPIPE_LLM_TEMPERATURE", "0.2")),
            timeout_s=int(os.environ.get("SPRITEPIPE_LLM_TIMEOUT", "60")),
        )


def compose_prompt(
    description: str | None = None,
    *,
    direct_prompt: str | None = None,
    feedback: str | None = None,
    previous_prompt: str | None = None,
    llm_config: LLMConfig | None = None,
    force_pixel_trigger: bool = False,
    allow_fallback: bool = True,
) -> PromptSpec:
    if direct_prompt:
        positive = _clean_one_line(direct_prompt)
        if force_pixel_trigger:
            positive = ensure_pixel_trigger(positive)
        return PromptSpec(
            positive_prompt=positive,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            source="direct",
            notes="Direct prompt mode skips LLM rewriting.",
        )

    clean_description = _clean_one_line(description or "")
    if not clean_description:
        clean_description = "an original adventurer character"

    config = llm_config or LLMConfig.from_env()
    if config.provider and config.provider != "none":
        try:
            return _compose_with_llm(clean_description, feedback, previous_prompt, config)
        except Exception as exc:  # noqa: BLE001 - fallback is an intentional UX feature.
            if not allow_fallback:
                raise
            fallback = _compose_fallback(clean_description, feedback, previous_prompt)
            return PromptSpec(
                positive_prompt=fallback.positive_prompt,
                negative_prompt=fallback.negative_prompt,
                source="fallback",
                notes=f"LLM prompt rewrite failed, used deterministic fallback: {exc}",
            )

    return _compose_fallback(clean_description, feedback, previous_prompt)


def ensure_pixel_trigger(prompt: str) -> str:
    stripped = prompt.strip()
    if stripped.lower().startswith(PIXEL_TRIGGER.lower()):
        return stripped
    return f"{PIXEL_TRIGGER}, {stripped}"


def _compose_fallback(
    description: str,
    feedback: str | None,
    previous_prompt: str | None,
) -> PromptSpec:
    revision = _clean_one_line(feedback or "")
    previous = _clean_one_line(previous_prompt or "")
    subject = description
    if previous and revision:
        subject = f"{description}. Keep the established design from the previous prompt while applying this revision: {revision}"
    elif revision:
        subject = f"{description}. Revision request: {revision}"

    positive = (
        "Pixel Art, 2D pixel art game character sprite motif, one original full-body character, "
        "static pose, centered composition, facing right, side-view or subtle three-quarter side-view, "
        "clean readable silhouette, bold outer outline, limited color palette, crisp square-pixel shapes, "
        "clear costume blocks, expressive but simple face, hands and feet simplified for sprite readability, "
        "plain neutral background, no readable text. "
        f"Character concept: {subject}. "
        "Design for high-resolution generation that will be downscaled cleanly to 64x64 game art."
    )
    return PromptSpec(
        positive_prompt=positive,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        source="fallback",
        notes="Deterministic prompt composer used; set SPRITEPIPE_LLM_PROVIDER for LLM rewriting.",
    )


def _compose_with_llm(
    description: str,
    feedback: str | None,
    previous_prompt: str | None,
    config: LLMConfig,
) -> PromptSpec:
    user_payload = {
        "description": description,
        "previous_prompt": previous_prompt or "",
        "feedback": feedback or "",
        "examples": _select_examples(description, feedback),
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    if config.provider == "ollama":
        content = _call_ollama(messages, config)
    elif config.provider in {"openai", "openai-compatible", "compatible"}:
        content = _call_openai_compatible(messages, config)
    else:
        raise ValueError(f"unsupported LLM provider '{config.provider}'")

    parsed = _extract_json_object(content)
    positive = ensure_pixel_trigger(_clean_one_line(str(parsed.get("positive_prompt", ""))))
    negative = _clean_one_line(str(parsed.get("negative_prompt", "")))
    if not positive:
        raise ValueError("LLM returned an empty positive_prompt")
    if not negative:
        negative = DEFAULT_NEGATIVE_PROMPT

    positive = _enforce_core_constraints(positive)
    return PromptSpec(
        positive_prompt=positive,
        negative_prompt=negative,
        source=config.provider,
        notes=f"Prompt rewritten by {config.provider} model '{config.model}'.",
    )


def _call_openai_compatible(messages: list[dict[str, str]], config: LLMConfig) -> str:
    endpoint = config.endpoint or "https://api.openai.com/v1/chat/completions"
    if not config.model:
        raise ValueError("SPRITEPIPE_LLM_MODEL is required for OpenAI-compatible providers")
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    response = requests.post(
        endpoint,
        headers=headers,
        json={
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "response_format": {"type": "json_object"},
        },
        timeout=config.timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _call_ollama(messages: list[dict[str, str]], config: LLMConfig) -> str:
    base = (config.endpoint or "http://127.0.0.1:11434").rstrip("/")
    if not config.model:
        raise ValueError("SPRITEPIPE_LLM_MODEL is required for Ollama")
    try:
        response = requests.post(
            f"{base}/api/chat",
            json={
                "model": config.model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {"temperature": config.temperature},
            },
            timeout=config.timeout_s,
        )
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise RuntimeError(f"Ollama is not reachable at {base}. Use Prompt model Validate/Download, or start Ollama.") from exc
    except requests.Timeout as exc:
        raise RuntimeError(f"Ollama timed out at {base} while using model '{config.model}'.") from exc
    except requests.HTTPError as exc:
        detail = response.text.strip()[:500]
        raise RuntimeError(f"Ollama request failed for model '{config.model}': {response.status_code} {detail}") from exc
    return response.json()["message"]["content"]


def _extract_json_object(content: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM response was not a JSON object")
    return data


def _enforce_core_constraints(prompt: str) -> str:
    required_phrases = [
        "one original full-body character",
        "static pose",
        "facing right",
        "plain neutral background",
        "no readable text",
        "designed to downscale cleanly to 64x64",
    ]
    lower = prompt.lower()
    additions = [phrase for phrase in required_phrases if phrase not in lower]
    if additions:
        prompt = f"{prompt}, {', '.join(additions)}"
    return prompt


def _select_examples(description: str, feedback: str | None) -> list[dict[str, str]]:
    examples = list(load_prompt_examples())
    haystack = f"{description} {feedback or ''}".lower()
    scored: list[tuple[int, dict[str, str]]] = []
    for example in examples:
        score = sum(1 for word in re.findall(r"[a-zA-Z\u4e00-\u9fff]+", haystack) if word in example["description"].lower())
        scored.append((score, example))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:3]]


def load_prompt_examples() -> Iterable[dict[str, str]]:
    with resources.files(__package__).joinpath("prompt_training_examples.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _clean_one_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
