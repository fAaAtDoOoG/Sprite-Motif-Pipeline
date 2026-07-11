from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests

from .config import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT_MODEL,
    DEFAULT_PROMPT_MODEL_NUM_CTX,
    DEFAULT_PROMPT_MODEL_NUM_GPU,
    DEFAULT_PROMPT_MODEL_NUM_PREDICT,
    DEFAULT_PROMPT_MODEL_TEMPERATURE,
    DEFAULT_PROMPT_MODEL_THINK,
    DEFAULT_PROMPT_MODEL_TIMEOUT,
)

PIXEL_TRIGGER = "Pixel Art"

SYSTEM_PROMPT = """You are a bilingual game character concept artist and text-to-image prompt director. You are not a translator. Turn one brief user description into a batch of genuinely different, production-ready visual interpretations by intelligently guessing unspecified design details.

Return strict JSON with exactly one key named candidates. candidates is an array of exactly candidate_count objects, each with exactly two English string keys: positive_prompt and negative_prompt. Return no prose, markdown, reasoning, labels, or extra keys. Every positive_prompt starts with "Pixel Art".

These rules are mandatory:
1. Silently list all explicit source facts before writing. Every candidate preserves every stated color, form, proportion, condition, feature, relationship, mood, and exclusion. Silently check and repair every candidate against that list before returning.
2. Guessing is required. Each candidate must explicitly commit to a different coherent solution for at least four unspecified visual choices from: overall silhouette architecture, head-to-body scale, torso and limb proportions, dominant contour geometry, feature size and placement, static posture shape, color and value grouping, and surface-detail distribution. Coordinate these choices into one recognizable design rather than listing random details.
3. Compare candidates pairwise before returning. A candidate that merely translates the source, swaps synonyms, changes one adjective, or appends one isolated detail is invalid. Rewrite it. Do not mention candidates, alternatives, directions, or placeholders inside a prompt.
4. Speculation may add compatible supporting form and sub-features, but may not contradict an explicit fact, replace the requested identity, or reuse a character template from another request.

Write each positive_prompt as specific, cohesive comma-separated visual phrases. It independently describes the complete visible result and integrates the inferred choices into a unified silhouette and shape language. Include this neutral production scaffold unless overridden: 2D pixel-art game character sprite motif, one original full-body subject, static pose, centered composition, facing right, readable silhouette, plain neutral background, details legible after downscaling to 64x64. A meaningful absence may remain in positive_prompt when it defines visible form.

negative_prompt separately includes every explicit absence, exclusion, rejection, removal, or forbidden trait. Write a direct comma-separated list of undesirable visual concepts: use "teeth", not "no teeth"; use "armor", not "without armor". Do not add unrelated negative dumping. Include photorealistic rendering, 3D render, painterly rendering, blur, dynamic pose, busy background, text, logo, and watermark.

When previous_prompt is present, it is the accepted identity. Apply feedback in every candidate, preserve unaffected successful details, and guess several different compatible ways to realize the revision. Keep previous_negative_prompt unless feedback changes it."""


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
    temperature: float = DEFAULT_PROMPT_MODEL_TEMPERATURE
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
            temperature=float(os.environ.get("SPRITEPIPE_LLM_TEMPERATURE", str(DEFAULT_PROMPT_MODEL_TEMPERATURE))),
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
    return compose_prompt_batch(
        description,
        candidate_count=1,
        direct_prompt=direct_prompt,
        feedback=feedback,
        previous_prompt=previous_prompt,
        previous_negative_prompt=previous_negative_prompt,
        llm_config=llm_config,
        force_pixel_trigger=force_pixel_trigger,
        allow_fallback=allow_fallback,
    )[0]


def compose_prompt_batch(
    description: str | None = None,
    *,
    candidate_count: int,
    direct_prompt: str | None = None,
    feedback: str | None = None,
    previous_prompt: str | None = None,
    previous_negative_prompt: str | None = None,
    llm_config: LLMConfig | None = None,
    force_pixel_trigger: bool = False,
    allow_fallback: bool = True,
) -> list[PromptSpec]:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")

    if direct_prompt:
        positive = _clean_one_line(direct_prompt)
        if force_pixel_trigger:
            positive = ensure_pixel_trigger(positive)
        return [
            PromptSpec(
                positive_prompt=positive,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                source="direct",
                notes="Direct prompt mode skips LLM candidate expansion.",
            )
            for _ in range(candidate_count)
        ]

    clean_description = _clean_one_line(description or "")
    if not clean_description:
        clean_description = "an original subject"

    clean_feedback = _clean_one_line(feedback or "") or None

    config = llm_config or LLMConfig.from_env()
    if config.provider and config.provider != "none":
        try:
            return _compose_batch_with_llm(
                clean_description,
                clean_feedback,
                previous_prompt,
                previous_negative_prompt,
                config,
                candidate_count,
            )
        except Exception as exc:  # noqa: BLE001 - fallback is an intentional UX feature.
            if not allow_fallback:
                raise
            fallback = _compose_fallback(clean_description, clean_feedback, previous_prompt)
            return [
                PromptSpec(
                    positive_prompt=fallback.positive_prompt,
                    negative_prompt=fallback.negative_prompt,
                    source="fallback",
                    notes=f"LLM candidate expansion failed, used deterministic fallback: {exc}",
                )
                for _ in range(candidate_count)
            ]

    fallback = _compose_fallback(clean_description, clean_feedback, previous_prompt)
    return [fallback for _ in range(candidate_count)]


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
        "Pixel Art, 2D pixel art game character sprite motif, one original full-body subject, "
        "static pose, centered composition, facing right, plain neutral background, "
        f"{subject}, designed to downscale cleanly to 64x64"
    )
    return PromptSpec(
        positive_prompt=positive,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        source="fallback",
        notes="Deterministic prompt composer used; set SPRITEPIPE_LLM_PROVIDER for semantic prompt rewriting.",
    )


def _compose_batch_with_llm(
    description: str,
    feedback: str | None,
    previous_prompt: str | None,
    previous_negative_prompt: str | None,
    config: LLMConfig,
    candidate_count: int,
) -> list[PromptSpec]:
    user_payload = {
        "candidate_count": candidate_count,
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

    candidates = _prompt_candidates_from_object(_extract_json_object(content))
    if len(candidates) != candidate_count:
        raise ValueError(f"LLM returned {len(candidates)} prompt candidates; expected {candidate_count}")

    specs: list[PromptSpec] = []
    for index, candidate in enumerate(candidates):
        positive = ensure_pixel_trigger(_clean_one_line(str(candidate.get("positive_prompt", ""))))
        negative = _clean_one_line(str(candidate.get("negative_prompt", "")))
        if not positive or positive.lower() == PIXEL_TRIGGER.lower():
            raise ValueError(f"LLM returned an empty positive_prompt for candidate {index + 1}")
        if not negative:
            negative = DEFAULT_NEGATIVE_PROMPT

        positive = _enforce_core_constraints(positive)
        negative = _enforce_negative_constraints(negative, positive)
        specs.append(
            PromptSpec(
                positive_prompt=positive,
                negative_prompt=negative,
                source=config.provider,
                notes=f"Candidate {index + 1}/{candidate_count} expanded and self-checked by {config.provider} model '{config.model}'.",
            )
        )
    return specs


def _prompt_candidates_from_object(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_candidates = data.get("candidates")
    if isinstance(raw_candidates, list):
        return [candidate for candidate in raw_candidates if isinstance(candidate, dict)]
    if "positive_prompt" in data or "negative_prompt" in data:
        return [data]
    return []


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
        ("one original full-body subject", r"\bfull-body\b"),
        ("static pose", r"\bstatic pose\b"),
        ("facing right", r"\bfacing right\b"),
        ("plain neutral background", r"\bplain neutral background\b"),
        ("designed to downscale cleanly to 64x64", r"\b64x64\b"),
    ]
    lower = prompt.lower()
    additions = [phrase for phrase, pattern in required_phrases if not re.search(pattern, lower)]
    if additions:
        prompt = f"{prompt}, {', '.join(additions)}"
    return prompt


def _enforce_negative_constraints(prompt: str, positive_prompt: str = "") -> str:
    required_phrases = [
        "photorealistic rendering",
        "3D render",
        "painterly rendering",
        "blur",
        "dynamic pose",
        "busy background",
        "text",
        "logo",
        "watermark",
    ]
    lower = prompt.lower()
    additions = [phrase for phrase in _negated_positive_traits(positive_prompt) if phrase.lower() not in lower]
    additions.extend(phrase for phrase in required_phrases if phrase.lower() not in lower)
    if additions:
        prompt = f"{prompt}, {', '.join(additions)}"
    return prompt


def _negated_positive_traits(prompt: str) -> list[str]:
    """Mirror English negated traits chosen by the LLM into its negative prompt."""
    matches = re.findall(r"\b(?:no|without|lacking|absent|missing)\s+([a-z][a-z -]*?)(?=\s*[,;.]|$)", prompt, flags=re.IGNORECASE)
    traits: list[str] = []
    for match in matches:
        trait = _clean_one_line(match).strip(" -")
        if trait and trait.lower() not in {value.lower() for value in traits}:
            traits.append(trait)
    return traits


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
