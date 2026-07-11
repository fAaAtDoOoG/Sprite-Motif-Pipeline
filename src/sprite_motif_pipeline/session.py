from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Size, format_size
from .prompting import PromptSpec


@dataclass
class UserInput:
    kind: str
    text: str
    created_at: str = ""
    selected_index: int | None = None


@dataclass
class Candidate:
    index: int
    seed: int
    positive_prompt: str
    negative_prompt: str
    prompt_id: str = ""
    highres_path: str = ""
    lowres_path: str = ""
    api_prompt_path: str = ""


@dataclass
class RunManifest:
    run_id: str
    description: str
    prompt_source: str
    prompt_notes: str
    high_res: str
    low_res: str
    image_backend: str = "qwen-comfy"
    image_model: str = ""
    image_endpoint: str = ""
    parent_run: str = ""
    selected_index: int | None = None
    feedback: str = ""
    user_inputs: list[UserInput] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n"


def new_run_dir(base_dir: Path, prefix: str = "run") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{prefix}_{timestamp}"
    path = base_dir / run_id
    counter = 1
    while path.exists():
        path = base_dir / f"{run_id}_{counter}"
        counter += 1
    path.mkdir(parents=True)
    return path


def save_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    path = run_dir / "manifest.json"
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def load_manifest(run_dir: Path) -> RunManifest:
    data: dict[str, Any] = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    candidates = [_from_dict(Candidate, candidate) for candidate in data.pop("candidates", [])]
    user_inputs = [_from_dict(UserInput, user_input) for user_input in data.pop("user_inputs", [])]
    allowed = {field.name for field in fields(RunManifest)}
    return RunManifest(**{key: value for key, value in data.items() if key in allowed}, candidates=candidates, user_inputs=user_inputs)


def make_user_input(kind: str, text: str, selected_index: int | None = None) -> UserInput:
    return UserInput(
        kind=kind,
        text=text,
        selected_index=selected_index,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )


def user_input_history(manifest: RunManifest) -> list[UserInput]:
    history = list(manifest.user_inputs)
    if not history and manifest.description.strip():
        history.append(make_user_input("description", manifest.description))
    return history


def create_manifest(
    *,
    run_dir: Path,
    description: str,
    prompt_spec: PromptSpec,
    high_res: Size,
    low_res: Size,
    parent_run: str = "",
    feedback: str = "",
    user_input_kind: str = "description",
    user_input_text: str | None = None,
    user_inputs: list[UserInput] | None = None,
    image_backend: str = "qwen-comfy",
    image_model: str = "",
    image_endpoint: str = "",
) -> RunManifest:
    history = list(user_inputs or [])
    if not history:
        text = (description if user_input_text is None else user_input_text).strip()
        if text:
            history.append(make_user_input(user_input_kind, text))
    return RunManifest(
        run_id=run_dir.name,
        description=description,
        prompt_source=prompt_spec.source,
        prompt_notes=prompt_spec.notes,
        high_res=format_size(high_res),
        low_res=format_size(low_res),
        image_backend=image_backend,
        image_model=image_model,
        image_endpoint=image_endpoint,
        parent_run=parent_run,
        feedback=feedback,
        user_inputs=history,
    )


def _from_dict(cls, data: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in allowed})
