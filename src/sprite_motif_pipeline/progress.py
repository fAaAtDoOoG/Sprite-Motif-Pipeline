from __future__ import annotations

import re


def percent_from_message(message: str) -> int | None:
    match = re.search(r"(?<!\d)(100|[1-9]?\d)%", message)
    if match:
        return int(match.group(1))
    lowered = message.lower()
    if lowered.startswith("done ") or lowered.startswith("manifest="):
        return 100
    return None


def generation_percent(message: str, batch_size: int) -> int | None:
    if batch_size <= 0:
        return None
    if message.startswith("contact_sheet="):
        return 95
    if message.startswith("manifest="):
        return 100
    match = re.match(r"\[(\d+)\]\s+(.+)", message)
    if not match:
        return None
    index = int(match.group(1))
    stage = match.group(2)
    stage_fraction = 0.15
    if stage.startswith("prompt_id="):
        stage_fraction = 0.35
    elif stage.startswith("saved lowres=") or stage.startswith("dry-run"):
        stage_fraction = 0.95
    return int(((index + stage_fraction) / batch_size) * 90)


def short_status(message: str, limit: int = 100) -> str:
    return message if len(message) <= limit else message[: limit - 3] + "..."
