from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .comfy import ComfyClient, validate_model_assets, validate_required_nodes
from .config import DEFAULT_HIGH_RES, DEFAULT_LOW_RES, DEFAULTS, format_size, parse_size
from .postprocess import downscale_nearest, make_contact_sheet
from .prompting import LLMConfig, compose_prompt
from .session import Candidate, create_manifest, load_manifest, new_run_dir, save_manifest
from .workflow import build_api_prompt, export_api_prompt, required_node_types


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - command-line UX should show concise failure.
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spritepipe", description="Iterative txt2img pipeline for pixel game character motifs.")
    sub = parser.add_subparsers(required=True)

    prompt_cmd = sub.add_parser("prompt", help="Rewrite a simple description into a generation prompt.")
    _add_prompt_args(prompt_cmd)
    prompt_cmd.set_defaults(func=cmd_prompt)

    generate_cmd = sub.add_parser("generate", help="Generate a batch through ComfyUI.")
    _add_prompt_args(generate_cmd)
    _add_generation_args(generate_cmd)
    generate_cmd.set_defaults(func=cmd_generate)

    iterate_cmd = sub.add_parser("iterate", help="Select a candidate, apply feedback, and generate a new batch.")
    iterate_cmd.add_argument("run_dir", type=Path)
    iterate_cmd.add_argument("--index", type=int, required=True, help="Candidate index from the previous manifest.")
    iterate_cmd.add_argument("--feedback", required=True, help="Natural-language modification request.")
    _add_generation_args(iterate_cmd)
    iterate_cmd.set_defaults(func=cmd_iterate)

    inspect_cmd = sub.add_parser("inspect", help="Print candidates in a run manifest.")
    inspect_cmd.add_argument("run_dir", type=Path)
    inspect_cmd.set_defaults(func=cmd_inspect)

    workflow_cmd = sub.add_parser("workflow", help="Export a parameterized ComfyUI API workflow.")
    workflow_sub = workflow_cmd.add_subparsers(required=True)
    export_cmd = workflow_sub.add_parser("export", help="Write a sample ComfyUI API prompt JSON.")
    export_cmd.add_argument("--output", type=Path, default=Path("workflows/qwen_image_2512_pixel_sprite_api.json"))
    export_cmd.add_argument("--high-res", default=format_size(DEFAULT_HIGH_RES))
    export_cmd.add_argument("--seed", type=int, default=42)
    export_cmd.set_defaults(func=cmd_workflow_export)

    validate_cmd = sub.add_parser("validate-comfy", help="Check whether the active ComfyUI backend exposes required nodes.")
    validate_cmd.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    validate_cmd.set_defaults(func=cmd_validate_comfy)

    return parser


def _add_prompt_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--description", "-d", help="Simple character description for the prompt composer.")
    group.add_argument("--prompt", help="Direct positive prompt; skips prompt composer.")
    parser.add_argument("--force-pixel-trigger", action="store_true", help="Prepend 'Pixel Art' to direct prompts if missing.")
    parser.add_argument("--llm-provider", choices=["none", "openai", "openai-compatible", "ollama"], help="Override SPRITEPIPE_LLM_PROVIDER.")
    parser.add_argument("--llm-model", help="Override SPRITEPIPE_LLM_MODEL.")
    parser.add_argument("--llm-endpoint", help="Override SPRITEPIPE_LLM_ENDPOINT.")


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", "-n", type=int, default=4)
    parser.add_argument("--high-res", default=format_size(DEFAULT_HIGH_RES))
    parser.add_argument("--low-res", default=format_size(DEFAULT_LOW_RES))
    parser.add_argument("--seed", type=int, help="Base seed. Omit for random seeds.")
    parser.add_argument("--steps", type=int, default=DEFAULTS.steps)
    parser.add_argument("--cfg", type=float, default=DEFAULTS.cfg)
    parser.add_argument("--lora-name", default=DEFAULTS.pixel_lora)
    parser.add_argument("--lora-strength", type=float, default=DEFAULTS.pixel_lora_strength)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--dry-run", action="store_true", help="Write prompts/manifests without contacting ComfyUI.")


def cmd_prompt(args: argparse.Namespace) -> int:
    spec = _prompt_spec_from_args(args)
    print(json.dumps({"positive_prompt": spec.positive_prompt, "negative_prompt": spec.negative_prompt, "source": spec.source, "notes": spec.notes}, ensure_ascii=False, indent=2))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    spec = _prompt_spec_from_args(args)
    description = args.description or args.prompt or ""
    run_dir = _generate_batch(args, spec, description=description)
    print(f"run: {run_dir}")
    print(f"manifest: {run_dir / 'manifest.json'}")
    return 0


def cmd_iterate(args: argparse.Namespace) -> int:
    previous = load_manifest(args.run_dir)
    selected = _find_candidate(previous.candidates, args.index)
    spec = compose_prompt(
        previous.description,
        feedback=args.feedback,
        previous_prompt=selected.positive_prompt,
        llm_config=_llm_config_from_args(args),
    )
    args.description = previous.description
    args.prompt = None
    run_dir = _generate_batch(
        args,
        spec,
        description=previous.description,
        parent_run=str(args.run_dir),
        selected_index=args.index,
        feedback=args.feedback,
    )
    previous.selected_index = args.index
    previous.feedback = args.feedback
    save_manifest(args.run_dir, previous)
    print(f"new run: {run_dir}")
    print(f"previous selection recorded in: {args.run_dir / 'manifest.json'}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.run_dir)
    print(f"run_id: {manifest.run_id}")
    print(f"description: {manifest.description}")
    print(f"prompt_source: {manifest.prompt_source}")
    for candidate in manifest.candidates:
        print(f"[{candidate.index}] seed={candidate.seed} lowres={candidate.lowres_path} highres={candidate.highres_path}")
    return 0


def cmd_workflow_export(args: argparse.Namespace) -> int:
    width, height = parse_size(args.high_res, DEFAULT_HIGH_RES)
    prompt = build_api_prompt(
        positive_prompt="Pixel Art, 2D pixel art game character sprite motif, one original full-body character, static pose, centered, facing right, plain neutral background, no readable text.",
        negative_prompt="photorealistic rendering, blurry silhouette, text, logo, watermark",
        width=width,
        height=height,
        seed=args.seed,
        filename_prefix="sprite_motif_sample",
    )
    export_api_prompt(args.output, prompt)
    print(args.output)
    return 0


def cmd_validate_comfy(args: argparse.Namespace) -> int:
    client = ComfyClient(args.comfy_url)
    missing = validate_required_nodes(client, required_node_types())
    if missing:
        print("missing nodes:")
        for node in missing:
            print(f"- {node}")
        return 2
    missing_assets = validate_model_assets(client)
    if missing_assets:
        print("ComfyUI exposes all required core nodes, but these model files are missing from the backend lists:")
        for label, expected in missing_assets.items():
            print(f"- {label}: {expected}")
        return 3
    print("ComfyUI exposes all required core nodes and default model files.")
    return 0


def _generate_batch(
    args: argparse.Namespace,
    prompt_spec,
    *,
    description: str,
    parent_run: str = "",
    selected_index: int | None = None,
    feedback: str = "",
) -> Path:
    high_res = parse_size(args.high_res, DEFAULT_HIGH_RES)
    low_res = parse_size(args.low_res, DEFAULT_LOW_RES)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    run_dir = new_run_dir(args.output_dir)
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
    client = None if args.dry_run else ComfyClient(args.comfy_url)
    seeds = _seeds(args.seed, args.batch_size)

    for index, seed in enumerate(seeds):
        stem = f"candidate_{index:02d}_seed_{seed}"
        api_prompt = build_api_prompt(
            positive_prompt=prompt_spec.positive_prompt,
            negative_prompt=prompt_spec.negative_prompt,
            width=high_res[0],
            height=high_res[1],
            seed=seed,
            filename_prefix=f"sprite_motif/{run_dir.name}/{stem}",
            lora_name=args.lora_name,
            lora_strength=args.lora_strength,
            steps=args.steps,
            cfg=args.cfg,
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
            prompt_id = client.queue_prompt(api_prompt)
            history = client.wait_for_history(prompt_id, timeout_s=args.timeout)
            downloaded = client.download_images(history, high_dir, stem)
            high_path = downloaded[0]
            low_path = downscale_nearest(high_path, low_dir / f"{stem}_{format_size(low_res)}.png", low_res)
            candidate.prompt_id = prompt_id
            candidate.highres_path = str(high_path)
            candidate.lowres_path = str(low_path)
            print(f"[{index}] prompt_id={prompt_id} seed={seed} lowres={low_path}")
        else:
            print(f"[{index}] dry-run seed={seed} api_prompt={api_path}")

        manifest.candidates.append(candidate)
        save_manifest(run_dir, manifest)

    low_paths = [Path(candidate.lowres_path) for candidate in manifest.candidates if candidate.lowres_path]
    if low_paths:
        sheet = make_contact_sheet(low_paths, run_dir / "contact_sheet.png")
        print(f"contact_sheet={sheet}")
    save_manifest(run_dir, manifest)
    return run_dir


def _prompt_spec_from_args(args: argparse.Namespace):
    return compose_prompt(
        args.description,
        direct_prompt=args.prompt,
        llm_config=_llm_config_from_args(args),
        force_pixel_trigger=args.force_pixel_trigger,
    )


def _llm_config_from_args(args: argparse.Namespace) -> LLMConfig:
    config = LLMConfig.from_env()
    return LLMConfig(
        provider=(getattr(args, "llm_provider", None) or config.provider),
        model=(getattr(args, "llm_model", None) or config.model),
        endpoint=(getattr(args, "llm_endpoint", None) or config.endpoint),
        api_key=config.api_key,
        temperature=config.temperature,
        timeout_s=config.timeout_s,
    )


def _seeds(base_seed: int | None, count: int) -> list[int]:
    if base_seed is None:
        rng = random.SystemRandom()
        return [rng.randrange(0, 2**63 - 1) for _ in range(count)]
    return [base_seed + offset for offset in range(count)]


def _find_candidate(candidates: list[Candidate], index: int) -> Candidate:
    for candidate in candidates:
        if candidate.index == index:
            return candidate
    raise ValueError(f"candidate index {index} not found")


if __name__ == "__main__":
    raise SystemExit(main())
