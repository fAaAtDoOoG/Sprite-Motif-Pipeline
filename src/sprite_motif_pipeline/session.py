from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Size, format_size
from .prompting import PromptSpec


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
    parent_run: str = ""
    selected_index: int | None = None
    feedback: str = ""
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
    candidates = [Candidate(**candidate) for candidate in data.pop("candidates", [])]
    return RunManifest(**data, candidates=candidates)


def create_manifest(
    *,
    run_dir: Path,
    description: str,
    prompt_spec: PromptSpec,
    high_res: Size,
    low_res: Size,
    parent_run: str = "",
    feedback: str = "",
) -> RunManifest:
    return RunManifest(
        run_id=run_dir.name,
        description=description,
        prompt_source=prompt_spec.source,
        prompt_notes=prompt_spec.notes,
        high_res=format_size(high_res),
        low_res=format_size(low_res),
        parent_run=parent_run,
        feedback=feedback,
    )
