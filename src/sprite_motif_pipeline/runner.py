from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .comfy import ComfyClient
from .config import DEFAULT_HIGH_RES, DEFAULT_LOW_RES, DEFAULTS, Size, format_size, parse_size
from .postprocess import downscale_nearest, make_contact_sheet
from .prompting import PromptSpec
from .session import Candidate, create_manifest, new_run_dir, save_manifest
from .workflow import build_api_prompt, export_api_prompt

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class GenerationOptions:
    batch_size: int = 4
    high_res: str | Size = DEFAULT_HIGH_RES
    low_res: str | Size = DEFAULT_LOW_RES
    seed: int | None = None
    steps: int = DEFAULTS.steps
    cfg: float = DEFAULTS.cfg
    lora_name: str = DEFAULTS.pixel_lora
    lora_strength: float = DEFAULTS.pixel_lora_strength
    comfy_url: str = "http://127.0.0.1:8188"
    timeout: int = 900
    output_dir: Path = Path("runs")
    dry_run: bool = False


def generate_batch(
    prompt_spec: PromptSpec,
    *,
    description: str,
    options: GenerationOptions,
    parent_run: str = "",
    selected_index: int | None = None,
    feedback: str = "",
    progress: ProgressCallback | None = None,
) -> Path:
    high_res = _coerce_size(options.high_res, DEFAULT_HIGH_RES)
    low_res = _coerce_size(options.low_res, DEFAULT_LOW_RES)
    if options.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    run_dir = new_run_dir(options.output_dir)
    manifest = create_manifest(
        run_dir=run_dir,
        description=description,
        prompt_spec=prompt_spec,
        high_res=high_res,
        low_res=low_res,
        parent_run=parent_run,
        feedback=feedback,
    )
    manifest.selected_index = selected_index

    api_dir = run_dir / "api_prompts"
    high_dir = run_dir / "highres"
    low_dir = run_dir / "lowres"
    client = None if options.dry_run else ComfyClient(options.comfy_url)
    seeds = seeds_for_batch(options.seed, options.batch_size)

    _emit(progress, f"run={run_dir}")
    for index, seed in enumerate(seeds):
        stem = f"candidate_{index:02d}_seed_{seed}"
        api_prompt = build_api_prompt(
            positive_prompt=prompt_spec.positive_prompt,
            negative_prompt=prompt_spec.negative_prompt,
            width=high_res[0],
            height=high_res[1],
            seed=seed,
            filename_prefix=f"sprite_motif/{run_dir.name}/{stem}",
            lora_name=options.lora_name,
            lora_strength=options.lora_strength,
            steps=options.steps,
            cfg=options.cfg,
        )
        api_path = api_dir / f"{stem}.json"
        export_api_prompt(api_path, api_prompt)

        candidate = Candidate(
            index=index,
            seed=seed,
            positive_prompt=prompt_spec.positive_prompt,
            negative_prompt=prompt_spec.negative_prompt,
            api_prompt_path=str(api_path),
        )

        if client is not None:
            _emit(progress, f"[{index}] queue seed={seed}")
            prompt_id = client.queue_prompt(api_prompt)
            _emit(progress, f"[{index}] prompt_id={prompt_id}")
            history = client.wait_for_history(prompt_id, timeout_s=options.timeout)
            downloaded = client.download_images(history, high_dir, stem)
            high_path = downloaded[0]
            low_path = downscale_nearest(high_path, low_dir / f"{stem}_{format_size(low_res)}.png", low_res)
            candidate.prompt_id = prompt_id
            candidate.highres_path = str(high_path)
            candidate.lowres_path = str(low_path)
            _emit(progress, f"[{index}] saved lowres={low_path}")
        else:
            _emit(progress, f"[{index}] dry-run seed={seed} api_prompt={api_path}")

        manifest.candidates.append(candidate)
        save_manifest(run_dir, manifest)

    low_paths = [Path(candidate.lowres_path) for candidate in manifest.candidates if candidate.lowres_path]
    if low_paths:
        sheet = make_contact_sheet(low_paths, run_dir / "contact_sheet.png")
        _emit(progress, f"contact_sheet={sheet}")

    save_manifest(run_dir, manifest)
    _emit(progress, f"manifest={run_dir / 'manifest.json'}")
    return run_dir


def seeds_for_batch(base_seed: int | None, count: int) -> list[int]:
    if base_seed is None:
        rng = random.SystemRandom()
        return [rng.randrange(0, 2**63 - 1) for _ in range(count)]
    return [base_seed + offset for offset in range(count)]


def _coerce_size(value: str | Size, default: Size) -> Size:
    if isinstance(value, tuple):
        return value
    return parse_size(value, default)


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
