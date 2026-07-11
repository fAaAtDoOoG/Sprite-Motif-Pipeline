from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from .comfy import ComfyClient
from .config import DEFAULT_HIGH_RES, DEFAULT_LOW_RES, DEFAULTS, Size, format_size, parse_size
from .image_backends import (
    DEFAULT_OPENAI_IMAGE_MODEL,
    DEFAULT_OPENAI_IMAGES_ENDPOINT,
    IMAGE_BACKEND_CUSTOM_COMFY,
    IMAGE_BACKEND_OPENAI,
    IMAGE_BACKEND_QWEN,
    build_openai_image_request,
    generate_openai_image,
    load_custom_workflow,
    normalize_image_backend,
    public_endpoint,
    render_custom_workflow,
)
from .postprocess import downscale_nearest, make_contact_sheet
from .prompting import PromptSpec
from .session import Candidate, UserInput, create_manifest, new_run_dir, save_manifest
from .workflow import build_api_prompt, export_api_prompt

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class GenerationOptions:
    batch_size: int = 4
    high_res: str | Size = DEFAULT_HIGH_RES
    low_res: str | Size = DEFAULT_LOW_RES
    seed: int | None = None
    steps: int | None = None
    cfg: float | None = None
    lora_name: str = DEFAULTS.pixel_lora
    lora_strength: float = DEFAULTS.pixel_lora_strength
    image_backend: str = IMAGE_BACKEND_QWEN
    image_model: str = ""
    image_endpoint: str = DEFAULT_OPENAI_IMAGES_ENDPOINT
    image_api_key: str = ""
    custom_workflow: Path | None = None
    comfy_url: str = "http://127.0.0.1:8188"
    timeout: int = 900
    output_dir: Path = Path("runs")
    dry_run: bool = False


def generate_batch(
    prompt_spec: PromptSpec | Sequence[PromptSpec],
    *,
    description: str,
    options: GenerationOptions,
    parent_run: str = "",
    selected_index: int | None = None,
    feedback: str = "",
    user_input_kind: str = "description",
    user_input_text: str | None = None,
    user_inputs: list[UserInput] | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    high_res = _coerce_size(options.high_res, DEFAULT_HIGH_RES)
    low_res = _coerce_size(options.low_res, DEFAULT_LOW_RES)
    if options.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    prompt_specs = _prompt_specs_for_batch(prompt_spec, options.batch_size)

    image_backend = normalize_image_backend(options.image_backend)
    custom_template = load_custom_workflow(options.custom_workflow) if image_backend == IMAGE_BACKEND_CUSTOM_COMFY else None
    client = None
    if not options.dry_run and image_backend in {IMAGE_BACKEND_QWEN, IMAGE_BACKEND_CUSTOM_COMFY}:
        client = ComfyClient(options.comfy_url)
    steps = options.steps if options.steps is not None else DEFAULTS.steps
    cfg = options.cfg if options.cfg is not None else DEFAULTS.cfg
    image_model = options.image_model.strip()
    if not image_model and image_backend == IMAGE_BACKEND_QWEN:
        image_model = DEFAULTS.diffusion_model
    elif not image_model and image_backend == IMAGE_BACKEND_OPENAI:
        image_model = DEFAULT_OPENAI_IMAGE_MODEL
    qwen_defaults = replace(DEFAULTS, diffusion_model=image_model or DEFAULTS.diffusion_model)

    run_dir = new_run_dir(options.output_dir)
    manifest = create_manifest(
        run_dir=run_dir,
        description=description,
        prompt_spec=prompt_specs[0],
        high_res=high_res,
        low_res=low_res,
        parent_run=parent_run,
        feedback=feedback,
        user_input_kind=user_input_kind,
        user_input_text=user_input_text,
        user_inputs=user_inputs,
        image_backend=image_backend,
        image_model=image_model,
        image_endpoint=public_endpoint(options.image_endpoint if image_backend == IMAGE_BACKEND_OPENAI else options.comfy_url),
    )
    manifest.selected_index = selected_index

    api_dir = run_dir / "api_prompts"
    high_dir = run_dir / "highres"
    low_dir = run_dir / "lowres"
    seeds = seeds_for_batch(options.seed, options.batch_size)

    _emit(progress, f"run={run_dir}")
    for index, seed in enumerate(seeds):
        candidate_prompt = prompt_specs[index]
        stem = f"candidate_{index:02d}_seed_{seed}"
        filename_prefix = f"sprite_motif/{run_dir.name}/{stem}"
        if image_backend == IMAGE_BACKEND_QWEN:
            api_prompt = build_api_prompt(
                positive_prompt=candidate_prompt.positive_prompt,
                negative_prompt=candidate_prompt.negative_prompt,
                width=high_res[0],
                height=high_res[1],
                seed=seed,
                filename_prefix=filename_prefix,
                defaults=qwen_defaults,
                lora_name=options.lora_name,
                lora_strength=options.lora_strength,
                steps=steps,
                cfg=cfg,
            )
        elif image_backend == IMAGE_BACKEND_CUSTOM_COMFY:
            assert custom_template is not None
            api_prompt = render_custom_workflow(
                custom_template,
                positive_prompt=candidate_prompt.positive_prompt,
                negative_prompt=candidate_prompt.negative_prompt,
                width=high_res[0],
                height=high_res[1],
                seed=seed,
                steps=steps,
                cfg=cfg,
                filename_prefix=filename_prefix,
                model=image_model,
                lora_name=options.lora_name,
                lora_strength=options.lora_strength,
            )
        else:
            api_prompt = build_openai_image_request(
                positive_prompt=candidate_prompt.positive_prompt,
                negative_prompt=candidate_prompt.negative_prompt,
                width=high_res[0],
                height=high_res[1],
                model=image_model,
            )
        api_path = api_dir / f"{stem}.json"
        export_api_prompt(api_path, api_prompt)

        candidate = Candidate(
            index=index,
            seed=seed,
            positive_prompt=candidate_prompt.positive_prompt,
            negative_prompt=candidate_prompt.negative_prompt,
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
        elif not options.dry_run:
            _emit(progress, f"[{index}] request Images API seed={seed}")
            high_path = generate_openai_image(
                api_prompt,
                endpoint=options.image_endpoint,
                api_key=options.image_api_key,
                timeout_s=options.timeout,
                output_stem=high_dir / stem,
            )
            low_path = downscale_nearest(high_path, low_dir / f"{stem}_{format_size(low_res)}.png", low_res)
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


def _prompt_specs_for_batch(prompt_spec: PromptSpec | Sequence[PromptSpec], count: int) -> list[PromptSpec]:
    if isinstance(prompt_spec, PromptSpec):
        return [prompt_spec for _ in range(count)]
    specs = list(prompt_spec)
    if len(specs) == 1:
        return [specs[0] for _ in range(count)]
    if len(specs) != count:
        raise ValueError(f"prompt candidate count {len(specs)} does not match batch size {count}")
    return specs


def _coerce_size(value: str | Size, default: Size) -> Size:
    if isinstance(value, tuple):
        return value
    return parse_size(value, default)


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
