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
- Classify the user's input into desired visual traits and exclusions. Do not simply translate negated phrases into the positive prompt.
- Treat negative_constraints in the user payload as mandatory exclusions. Keep those concepts out of positive_prompt and include them in negative_prompt.
- Chinese or English phrases like "无牙齿", "不要牙齿", "without teeth", "avoid teeth", or unwanted exposed anatomy such as "可见肌肉" should become negative_prompt concepts like "teeth" or "visible muscles".
- Describe exactly one full-body character, static, centered, facing right in side view or three-quarter side view.
- Make the silhouette readable at 64x64: clean outline, limited palette, strong contrast, large simple shapes.
- Prefer a neutral plain background and no readable text.
- The negative prompt must be specific and useful. It must reject photorealism, 3D render style, painterly/soft rendering, blur, motion/action poses, busy backgrounds, readable text, logos, watermarks, and tiny clutter that would fail after 64x64 downscaling.
- Tailor the negative prompt to the description and feedback when needed, for example excluding unwanted props, backgrounds, poses, styles, or motifs that conflict with the requested sprite.
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
class PromptIntent:
    positive_text: str
    negative_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "none"
    model: str = ""
    endpoint: str = ""
    api_key: str = ""
    temperature: float = 0.2
    timeout_s: int = 60
    keep_alive: str | int = "0"

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
            keep_alive=os.environ.get("SPRITEPIPE_LLM_KEEP_ALIVE", "0").strip(),
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
    description_intent = split_prompt_intent(clean_description)
    clean_description = description_intent.positive_text
    if not clean_description:
        clean_description = "an original adventurer character"

    feedback_intent = split_prompt_intent(feedback or "")
    clean_feedback = feedback_intent.positive_text or None
    negative_constraints = _dedupe_terms([*description_intent.negative_terms, *feedback_intent.negative_terms])

    config = llm_config or LLMConfig.from_env()
    if config.provider and config.provider != "none":
        try:
            return _compose_with_llm(
                clean_description,
                clean_feedback,
                previous_prompt,
                previous_negative_prompt,
                config,
                negative_constraints,
                raw_description=_clean_one_line(description or ""),
                raw_feedback=_clean_one_line(feedback or ""),
            )
        except Exception as exc:  # noqa: BLE001 - fallback is an intentional UX feature.
            if not allow_fallback:
                raise
            fallback = _compose_fallback(clean_description, clean_feedback, previous_prompt, negative_constraints)
            return PromptSpec(
                positive_prompt=fallback.positive_prompt,
                negative_prompt=fallback.negative_prompt,
                source="fallback",
                notes=f"LLM prompt rewrite failed, used deterministic fallback: {exc}",
            )

    return _compose_fallback(clean_description, clean_feedback, previous_prompt, negative_constraints)


def split_prompt_intent(text: str | None) -> PromptIntent:
    clean = _clean_one_line(text or "")
    if not clean:
        return PromptIntent("")

    positive_parts: list[str] = []
    negative_terms: list[str] = []
    negative_context = False

    for raw_part in re.split(r"[,，;；、。.!?？\n\r]+", clean):
        part = _clean_clause(raw_part)
        if not part:
            continue

        positive_prefix, extracted_terms = _extract_negative_clause(part)
        if extracted_terms:
            if positive_prefix:
                positive_parts.append(positive_prefix)
            negative_terms.extend(extracted_terms)
            negative_context = True
            continue

        if _looks_like_negative_fragment(part, carry_context=negative_context):
            negative_terms.extend(_split_negative_targets(part))
            negative_context = True
            continue

        positive_parts.append(part)
        negative_context = False

    return PromptIntent(
        positive_text=_clean_one_line(", ".join(positive_parts)),
        negative_terms=tuple(_dedupe_terms(negative_terms)),
    )


def ensure_pixel_trigger(prompt: str) -> str:
    stripped = prompt.strip()
    if stripped.lower().startswith(PIXEL_TRIGGER.lower()):
        return stripped
    return f"{PIXEL_TRIGGER}, {stripped}"


def _compose_fallback(
    description: str,
    feedback: str | None,
    previous_prompt: str | None,
    negative_constraints: Iterable[str] = (),
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
        negative_prompt=_merge_negative_prompt(DEFAULT_NEGATIVE_PROMPT, negative_constraints),
        source="fallback",
        notes="Deterministic prompt composer used; set SPRITEPIPE_LLM_PROVIDER for LLM rewriting.",
    )


def _compose_with_llm(
    description: str,
    feedback: str | None,
    previous_prompt: str | None,
    previous_negative_prompt: str | None,
    config: LLMConfig,
    negative_constraints: Iterable[str] = (),
    *,
    raw_description: str = "",
    raw_feedback: str = "",
) -> PromptSpec:
    required_negative_terms = _dedupe_terms(negative_constraints)
    user_payload = {
        "description": description,
        "raw_user_description": raw_description or description,
        "previous_prompt": previous_prompt or "",
        "previous_negative_prompt": previous_negative_prompt or "",
        "feedback": feedback or "",
        "raw_feedback": raw_feedback,
        "negative_constraints": required_negative_terms,
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
    negative = _enforce_negative_constraints(_merge_negative_prompt(negative, required_negative_terms))
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
                "keep_alive": _coerce_keep_alive(config.keep_alive),
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


def _extract_negative_clause(part: str) -> tuple[str, list[str]]:
    chinese_markers = [
        "不要出现",
        "不能出现",
        "不能有",
        "不需要",
        "不要",
        "没有",
        "避免",
        "禁止",
        "去掉",
        "移除",
        "排除",
        "别出现",
        "别有",
        "别",
        "无",
    ]
    candidates: list[tuple[int, str]] = []
    for marker in chinese_markers:
        index = part.find(marker)
        if index < 0:
            continue
        if marker == "无" and index != 0:
            continue
        candidates.append((index, marker))

    english_match = re.search(
        r"\b(no visible|no|without|avoid|exclude|remove|not)\b\s+(?P<target>.+)$",
        part,
        flags=re.IGNORECASE,
    )
    if english_match:
        candidates.append((english_match.start(), english_match.group(1)))

    if not candidates:
        return "", []

    index, marker = min(candidates, key=lambda item: item[0])
    target_start = index + len(marker)
    positive_prefix = _clean_clause(part[:index])
    target = _clean_clause(part[target_start:])
    return positive_prefix, _split_negative_targets(target)


def _split_negative_targets(text: str) -> list[str]:
    clean = _clean_clause(text)
    if not clean:
        return []
    clean = re.sub(r"^(任何|任意|所有|明显的?)\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*(等|等等|之类|这类|这些|这种|etc\.?)$", "", clean, flags=re.IGNORECASE)
    pieces = re.split(r"\s*(?:、|/|或|或者|和|与|及|以及|\band\b|\bor\b)\s*", clean, flags=re.IGNORECASE)
    return _dedupe_terms(_normalize_negative_term(piece) for piece in pieces if _clean_clause(piece))


def _looks_like_negative_fragment(part: str, *, carry_context: bool) -> bool:
    lowered = part.lower()
    anatomy_keywords = [
        "可见肌肉",
        "暴露肌肉",
        "裸露肌肉",
        "外露肌肉",
        "肌肉组织",
        "血肉",
        "内脏",
        "肠子",
        "gore",
        "visible muscle",
        "exposed muscle",
        "muscle tissue",
        "organs",
        "intestines",
    ]
    if any(keyword in lowered for keyword in anatomy_keywords):
        return True
    carry_keywords = ["牙齿", "牙", "teeth", "tooth", "fangs"]
    return carry_context and any(keyword in lowered for keyword in carry_keywords)


def _normalize_negative_term(term: str) -> str:
    clean = _clean_clause(term)
    clean = re.sub(r"^(不要|不需要|不能出现|不能有|没有|避免|禁止|去掉|移除|排除|别出现|别有|别|无)\s*", "", clean)
    clean = re.sub(r"^(no visible|no|without|avoid|exclude|remove|not)\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*(等|等等|之类|这类|这些|这种|etc\.?)$", "", clean, flags=re.IGNORECASE)
    translations = {
        "牙": "teeth",
        "牙齿": "teeth",
        "尖牙": "fangs",
        "獠牙": "fangs",
        "可见肌肉": "visible muscles",
        "明显肌肉": "visible muscles",
        "暴露肌肉": "exposed muscles",
        "裸露肌肉": "exposed muscles",
        "外露肌肉": "exposed muscles",
        "肌肉组织": "muscle tissue",
        "血肉": "gore",
        "血液": "blood",
        "血": "blood",
        "内脏": "organs",
        "肠子": "intestines",
        "骨头": "bones",
        "骷髅": "skull",
    }
    return translations.get(clean, clean)


def _merge_negative_prompt(prompt: str, terms: Iterable[str]) -> str:
    additions = _dedupe_terms(terms)
    if not additions:
        return prompt
    lower_prompt = prompt.lower()
    missing = [term for term in additions if term.lower() not in lower_prompt]
    if missing:
        return f"{prompt}, {', '.join(missing)}"
    return prompt


def _dedupe_terms(terms: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for term in terms:
        clean = _clean_clause(str(term))
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(clean)
    return unique


def _clean_clause(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n,，;；、。.!?？:：()（）[]【】\"'")


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
