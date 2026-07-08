from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any, Iterable

import requests

from .config import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT_MODEL,
    DEFAULT_PROMPT_MODEL_NUM_CTX,
    DEFAULT_PROMPT_MODEL_NUM_GPU,
    DEFAULT_PROMPT_MODEL_NUM_PREDICT,
    DEFAULT_PROMPT_MODEL_THINK,
    DEFAULT_PROMPT_MODEL_TIMEOUT,
)

PIXEL_TRIGGER = "Pixel Art"

SYSTEM_PROMPT = """You convert user descriptions into image-generation prompt tags for a text-to-image model.
The target is a static 2D pixel-art game character motif, generated high resolution and later downscaled to a tiny sprite.
Return strict JSON with exactly two string keys: positive_prompt and negative_prompt.

Output format:
- positive_prompt and negative_prompt must be comma-separated prompt tags or short prompt phrases.
- Do not write prose sentences, explanations, bullet points, markdown fences, paragraphs, or quoted lists inside the JSON values.
- Use English prompt terms only. Translate non-English user concepts into concise English tags.
- Each tag should usually be 1 to 6 words.
- positive_prompt must start with "Pixel Art".

Semantic task:
- Read the full user wording and decide which concepts are desired visual traits and which concepts are exclusions or rejections.
- Use contextual language understanding, not a fixed keyword list. The same concept can be positive in one request and negative in another.
- Internally identify visual concepts the user wants to appear.
- Internally identify visual concepts the user says are absent, excluded, rejected, removed, or forbidden.
- Write positive_prompt as tag phrases for the intended design.
- Write negative_prompt starting with every user-specific excluded or absent visual concept as plain concept tags, followed by generic bad-quality tags.
- Put desired traits only in positive_prompt.
- Put excluded, absent, forbidden, or rejected visual concepts in negative_prompt.
- A positive tag may include absence or negation language when it describes the intended design condition.
- However, every excluded, absent, forbidden, or rejected visual concept from the user's wording must also appear as a plain underlying concept tag in negative_prompt.
- Do not rely only on a positive absence phrase. If a visual concept is negated, also include that visual concept in negative_prompt.
- User-specific exclusion tags in negative_prompt must be clean underlying nouns or noun phrases, without negation grammar and without extra adjectives.
- For every user language, use contextual language understanding to detect absence, nonexistence, exclusion, removal, or avoidance.
- Do not invent extra user-specific negative tags that the user did not exclude. Generic bad-quality tags are allowed after user-specific exclusions.
- Do not invent replacement traits just to explain an exclusion. Only add a positive replacement when the user actually describes it.
- Preserve the user's intended character identity, body shape, materials, surface details, mood, and revision request.

Abstract rule example, not content to copy:
- Input meaning: desired A, no B, without C.
- Correct positive_prompt pattern: Pixel Art, A, no B, without C.
- Correct negative_prompt pattern: B, C, photorealistic rendering, 3D render.
- Wrong negative_prompt pattern: photorealistic rendering, 3D render.

Sprite constraints:
- Include concise positive tags for one full-body character, static pose, centered composition, facing right, side view or three-quarter side view, readable silhouette, limited palette, strong contrast, and plain neutral background when they fit.
- Include concise negative tags for image-generation failures such as photorealistic rendering, 3D render, painterly rendering, blurry silhouette, dynamic pose, busy background, readable text, logo, watermark, and tiny clutter.
- Do not mention copyrighted characters, brands, logos, or living artists unless the user explicitly supplied them.
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
    temperature: float = 0.1
    timeout_s: int = DEFAULT_PROMPT_MODEL_TIMEOUT
    keep_alive: str | int = "0"
    ollama_num_gpu: int | None = DEFAULT_PROMPT_MODEL_NUM_GPU
    ollama_num_ctx: int | None = DEFAULT_PROMPT_MODEL_NUM_CTX
    ollama_num_predict: int | None = DEFAULT_PROMPT_MODEL_NUM_PREDICT
    think: bool = DEFAULT_PROMPT_MODEL_THINK

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.environ.get("SPRITEPIPE_LLM_PROVIDER", "none").strip().lower()
        model = os.environ.get("SPRITEPIPE_LLM_MODEL", "").strip()
        if provider == "ollama" and not model:
            model = DEFAULT_PROMPT_MODEL

        default_num_gpu = DEFAULT_PROMPT_MODEL_NUM_GPU if provider == "ollama" else None
        default_num_ctx = DEFAULT_PROMPT_MODEL_NUM_CTX if provider == "ollama" else None
        default_num_predict = DEFAULT_PROMPT_MODEL_NUM_PREDICT if provider == "ollama" else None

        return cls(
            provider=provider,
            model=model,
            endpoint=os.environ.get("SPRITEPIPE_LLM_ENDPOINT", "").strip(),
            api_key=os.environ.get("SPRITEPIPE_LLM_API_KEY", "").strip(),
            temperature=float(os.environ.get("SPRITEPIPE_LLM_TEMPERATURE", "0.1")),
            timeout_s=int(os.environ.get("SPRITEPIPE_LLM_TIMEOUT", str(DEFAULT_PROMPT_MODEL_TIMEOUT))),
            keep_alive=os.environ.get("SPRITEPIPE_LLM_KEEP_ALIVE", "0").strip(),
            ollama_num_gpu=_optional_int(os.environ.get("SPRITEPIPE_OLLAMA_NUM_GPU", ""), default=default_num_gpu),
            ollama_num_ctx=_optional_int(os.environ.get("SPRITEPIPE_OLLAMA_NUM_CTX", ""), default=default_num_ctx),
            ollama_num_predict=_optional_int(os.environ.get("SPRITEPIPE_OLLAMA_NUM_PREDICT", ""), default=default_num_predict),
            think=_bool_env(os.environ.get("SPRITEPIPE_LLM_THINK", ""), default=DEFAULT_PROMPT_MODEL_THINK),
        )


def compose_prompt(
    description: str | None = None,
    *,
    direct_prompt: str | None = None,
    feedback: str | None = None,
    previous_prompt: str | None = None,
    previous_negative_prompt: str | None = None,
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

    clean_feedback = _clean_one_line(feedback or "") or None

    config = llm_config or LLMConfig.from_env()
    if config.provider and config.provider != "none":
        try:
            return _compose_with_llm(
                clean_description,
                clean_feedback,
                previous_prompt,
                previous_negative_prompt,
                config,
            )
        except Exception as exc:  # noqa: BLE001 - fallback is an intentional UX feature.
            if not allow_fallback:
                raise
            fallback = _compose_fallback(clean_description, clean_feedback, previous_prompt)
            return PromptSpec(
                positive_prompt=fallback.positive_prompt,
                negative_prompt=fallback.negative_prompt,
                source="fallback",
                notes=f"LLM prompt rewrite failed, used deterministic fallback: {exc}",
            )

    return _compose_fallback(clean_description, clean_feedback, previous_prompt)


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
        notes="Deterministic prompt composer used; set SPRITEPIPE_LLM_PROVIDER for semantic prompt rewriting.",
    )


def _compose_with_llm(
    description: str,
    feedback: str | None,
    previous_prompt: str | None,
    previous_negative_prompt: str | None,
    config: LLMConfig,
) -> PromptSpec:
    user_payload = {
        "description": description,
        "previous_prompt": previous_prompt or "",
        "previous_negative_prompt": previous_negative_prompt or "",
        "feedback": feedback or "",
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
    negative = _enforce_negative_constraints(negative)
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
    options: dict[str, int | float] = {"temperature": config.temperature}
    if config.ollama_num_gpu is not None:
        options["num_gpu"] = config.ollama_num_gpu
    if config.ollama_num_ctx is not None:
        options["num_ctx"] = config.ollama_num_ctx
    if config.ollama_num_predict is not None:
        options["num_predict"] = config.ollama_num_predict
    try:
        response = requests.post(
            f"{base}/api/chat",
            json={
                "model": config.model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "keep_alive": _coerce_keep_alive(config.keep_alive),
                "think": config.think,
                "options": options,
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


def _enforce_negative_constraints(prompt: str) -> str:
    required_phrases = [
        "photorealistic rendering",
        "3D render",
        "painterly brush strokes",
        "blurry silhouette",
        "dynamic pose",
        "busy background",
        "excessive tiny details",
        "text",
        "logo",
        "watermark",
    ]
    lower = prompt.lower()
    additions = [phrase for phrase in required_phrases if phrase.lower() not in lower]
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


def _coerce_keep_alive(value: str | int) -> str | int:
    if isinstance(value, int):
        return value
    stripped = value.strip()
    if re.fullmatch(r"-?\d+", stripped):
        return int(stripped)
    return stripped


def _optional_int(value: str | None, *, default: int | None = None) -> int | None:
    stripped = (value or "").strip()
    if not stripped:
        return default
    return int(stripped)


def _bool_env(value: str | None, *, default: bool = False) -> bool:
    stripped = (value or "").strip().lower()
    if not stripped:
        return default
    return stripped in {"1", "true", "yes", "on"}
