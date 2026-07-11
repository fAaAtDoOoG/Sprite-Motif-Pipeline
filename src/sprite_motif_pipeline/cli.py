from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from .comfy import (
    ComfyClient,
    default_comfy_dir,
    temporary_comfyui_server,
    validate_model_assets,
    validate_required_nodes,
)
from .config import DEFAULT_HIGH_RES, DEFAULT_LOW_RES, DEFAULTS, format_size, parse_size
from .image_backends import (
    DEFAULT_OPENAI_IMAGE_MODEL,
    DEFAULT_OPENAI_IMAGES_ENDPOINT,
    IMAGE_BACKENDS,
    IMAGE_BACKEND_CUSTOM_COMFY,
    IMAGE_BACKEND_OPENAI,
    IMAGE_BACKEND_QWEN,
    custom_workflow_node_types,
    load_custom_workflow,
    normalize_image_backend,
)
from .model_assets import default_models_root
from .ollama import temporary_ollama_server, validate_ollama_model
from .prompting import LLMConfig, compose_prompt, compose_prompt_batch
from .runner import GenerationOptions, generate_batch
from .session import Candidate, load_manifest, make_user_input, save_manifest, user_input_history
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

    generate_cmd = sub.add_parser("generate", help="Generate a batch through the selected image backend.")
    _add_prompt_args(generate_cmd)
    _add_generation_args(generate_cmd)
    generate_cmd.set_defaults(func=cmd_generate)

    iterate_cmd = sub.add_parser("iterate", help="Select a candidate, apply feedback, and generate a new batch.")
    iterate_cmd.add_argument("run_dir", type=Path)
    iterate_cmd.add_argument("--index", type=int, required=True, help="Candidate index from the previous manifest.")
    iterate_cmd.add_argument("--feedback", required=True, help="Natural-language modification request.")
    _add_llm_args(iterate_cmd)
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
    validate_cmd.add_argument("--image-backend", choices=IMAGE_BACKENDS, default=IMAGE_BACKEND_QWEN)
    validate_cmd.add_argument("--image-model", default=DEFAULTS.diffusion_model)
    validate_cmd.add_argument("--custom-workflow", type=Path)
    validate_cmd.set_defaults(func=cmd_validate_comfy)

    return parser


def _add_prompt_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--description", "-d", help="Simple character description for the prompt composer.")
    group.add_argument("--prompt", help="Direct positive prompt; skips prompt composer.")
    parser.add_argument("--force-pixel-trigger", action="store_true", help="Prepend 'Pixel Art' to direct prompts if missing.")
    _add_llm_args(parser)


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-provider", choices=["none", "openai", "openai-compatible", "ollama"], help="Override SPRITEPIPE_LLM_PROVIDER.")
    parser.add_argument("--llm-model", help="Override SPRITEPIPE_LLM_MODEL.")
    parser.add_argument("--llm-endpoint", help="Override SPRITEPIPE_LLM_ENDPOINT.")
    parser.add_argument("--llm-api-key", help="Prompt API key. Prefer SPRITEPIPE_LLM_API_KEY to keep it out of shell history.")


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", "-n", type=int, default=4)
    parser.add_argument("--high-res", default=format_size(DEFAULT_HIGH_RES))
    parser.add_argument("--low-res", default=format_size(DEFAULT_LOW_RES))
    parser.add_argument("--seed", type=int, help="Base seed. Omit for random seeds.")
    parser.add_argument("--steps", type=int, help="Sampler steps. Uses the default when omitted.")
    parser.add_argument("--cfg", type=float, help="Sampler CFG. Uses the default when omitted.")
    parser.add_argument("--lora-name", default=DEFAULTS.pixel_lora)
    parser.add_argument("--lora-strength", type=float, default=DEFAULTS.pixel_lora_strength)
    parser.add_argument(
        "--image-backend",
        choices=IMAGE_BACKENDS,
        default=os.environ.get("SPRITEPIPE_IMAGE_BACKEND", IMAGE_BACKEND_QWEN),
    )
    parser.add_argument("--image-model", default=os.environ.get("SPRITEPIPE_IMAGE_MODEL", ""))
    parser.add_argument(
        "--image-endpoint",
        default=os.environ.get("SPRITEPIPE_IMAGE_ENDPOINT", DEFAULT_OPENAI_IMAGES_ENDPOINT),
        help="Full OpenAI-compatible Images API endpoint or API base URL.",
    )
    parser.add_argument(
        "--image-api-key",
        default=os.environ.get("SPRITEPIPE_IMAGE_API_KEY", ""),
        help="Images API key. Prefer SPRITEPIPE_IMAGE_API_KEY to keep it out of shell history.",
    )
    parser.add_argument("--custom-workflow", type=Path, help="ComfyUI API-format JSON with SpritePipe placeholders.")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-dir", default="", help="Local ComfyUI folder used for on-demand startup.")
    parser.add_argument("--models-root", type=Path, default=default_models_root(), help="ComfyUI models folder.")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--dry-run", action="store_true", help="Write requests and manifests without contacting an image backend.")


def cmd_prompt(args: argparse.Namespace) -> int:
    spec = _prompt_spec_from_args(args)
    print(json.dumps({"positive_prompt": spec.positive_prompt, "negative_prompt": spec.negative_prompt, "source": spec.source, "notes": spec.notes}, ensure_ascii=False, indent=2))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    specs = _prompt_specs_from_args(args, candidate_count=args.batch_size)
    description = args.description or args.prompt or ""
    run_dir = _generate_batch(args, specs, description=description)
    print(f"run: {run_dir}")
    print(f"manifest: {run_dir / 'manifest.json'}")
    return 0


def cmd_iterate(args: argparse.Namespace) -> int:
    previous = load_manifest(args.run_dir)
    selected = _find_candidate(previous.candidates, args.index)
    history = user_input_history(previous)
    feedback_input = make_user_input("feedback", args.feedback, selected_index=args.index)
    config = _llm_config_from_args(args)
    specs = _run_with_temporary_prompt_service(
        config,
        lambda: compose_prompt_batch(
            previous.description,
            candidate_count=args.batch_size,
            feedback=args.feedback,
            previous_prompt=selected.positive_prompt,
            previous_negative_prompt=selected.negative_prompt,
            llm_config=config,
        ),
    )
    args.description = previous.description
    args.prompt = None
    run_dir = _generate_batch(
        args,
        specs,
        description=previous.description,
        parent_run=str(args.run_dir),
        selected_index=args.index,
        feedback=args.feedback,
        user_inputs=[*history, feedback_input],
    )
    previous.selected_index = args.index
    previous.feedback = args.feedback
    if not previous.user_inputs:
        previous.user_inputs = history
    previous.user_inputs.append(feedback_input)
    save_manifest(args.run_dir, previous)
    print(f"new run: {run_dir}")
    print(f"previous selection recorded in: {args.run_dir / 'manifest.json'}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.run_dir)
    print(f"run_id: {manifest.run_id}")
    print(f"description: {manifest.description}")
    print(f"prompt_source: {manifest.prompt_source}")
    for index, user_input in enumerate(user_input_history(manifest), start=1):
        selected = "" if user_input.selected_index is None else f" candidate={user_input.selected_index}"
        print(f"user_input[{index}] {user_input.kind}{selected}: {user_input.text}")
    for candidate in manifest.candidates:
        print(f"[{candidate.index}] seed={candidate.seed} lowres={candidate.lowres_path} highres={candidate.highres_path}")
    return 0


def cmd_workflow_export(args: argparse.Namespace) -> int:
    width, height = parse_size(args.high_res, DEFAULT_HIGH_RES)
    positive = "Pixel Art, 2D pixel art game character sprite motif, one original full-body subject, static pose, centered, facing right, plain neutral background, no readable text."
    negative = "photorealistic rendering, 3D render, painterly rendering, blur, dynamic pose, busy background, text, logo, watermark"
    prompt = build_api_prompt(
        positive_prompt=positive,
        negative_prompt=negative,
        width=width,
        height=height,
        seed=args.seed,
        filename_prefix="sprite_motif_sample",
    )
    export_api_prompt(args.output, prompt)
    print(args.output)
    return 0


def cmd_validate_comfy(args: argparse.Namespace) -> int:
    backend = normalize_image_backend(args.image_backend)
    if backend == IMAGE_BACKEND_OPENAI:
        print("The selected Images API backend does not use ComfyUI.")
        return 0
    client = ComfyClient(args.comfy_url)
    if backend == IMAGE_BACKEND_CUSTOM_COMFY:
        template = load_custom_workflow(args.custom_workflow)
        required = custom_workflow_node_types(template)
    else:
        required = required_node_types()
    missing = validate_required_nodes(client, required)
    if missing:
        print("missing nodes:")
        for node in missing:
            print(f"- {node}")
        return 2
    if backend == IMAGE_BACKEND_CUSTOM_COMFY:
        print("ComfyUI exposes every node required by the custom workflow.")
        return 0
    defaults = replace(DEFAULTS, diffusion_model=args.image_model or DEFAULTS.diffusion_model)
    missing_assets = validate_model_assets(client, defaults)
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
    user_inputs=None,
) -> Path:
    input_kind = "direct_prompt" if getattr(args, "prompt", None) else "description"
    image_backend = normalize_image_backend(args.image_backend)
    image_model = str(args.image_model or "").strip()
    if not image_model and image_backend == IMAGE_BACKEND_QWEN:
        image_model = DEFAULTS.diffusion_model
    elif not image_model and image_backend == IMAGE_BACKEND_OPENAI:
        image_model = DEFAULT_OPENAI_IMAGE_MODEL
    options = GenerationOptions(
        batch_size=args.batch_size,
        high_res=args.high_res,
        low_res=args.low_res,
        seed=args.seed,
        steps=args.steps,
        cfg=args.cfg,
        lora_name=args.lora_name,
        lora_strength=args.lora_strength,
        image_backend=image_backend,
        image_model=image_model,
        image_endpoint=args.image_endpoint,
        image_api_key=args.image_api_key,
        custom_workflow=args.custom_workflow,
        comfy_url=args.comfy_url,
        timeout=args.timeout,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )

    def work() -> Path:
        return generate_batch(
            prompt_spec,
            description=description,
            options=options,
            parent_run=parent_run,
            selected_index=selected_index,
            feedback=feedback,
            user_input_kind=input_kind,
            user_input_text=description,
            user_inputs=user_inputs,
            progress=print,
        )

    if options.dry_run or options.image_backend == IMAGE_BACKEND_OPENAI:
        return work()

    comfy_dir = args.comfy_dir or str(default_comfy_dir(args.models_root))
    with temporary_comfyui_server(
        args.comfy_url,
        comfy_dir=comfy_dir,
        models_root=args.models_root,
        progress=print,
        timeout_s=max(180, args.timeout),
    ):
        client = ComfyClient(args.comfy_url)
        if options.image_backend == IMAGE_BACKEND_CUSTOM_COMFY:
            template = load_custom_workflow(options.custom_workflow)
            required = custom_workflow_node_types(template)
        else:
            required = required_node_types()
        missing_nodes = validate_required_nodes(client, required)
        if missing_nodes:
            raise RuntimeError(f"ComfyUI is missing required workflow nodes: {', '.join(missing_nodes)}")
        if options.image_backend == IMAGE_BACKEND_QWEN:
            defaults = replace(DEFAULTS, diffusion_model=options.image_model or DEFAULTS.diffusion_model)
            missing_assets = validate_model_assets(client, defaults)
            if missing_assets:
                raise RuntimeError(
                    "ComfyUI is missing required Qwen-Image-2512 model files: " + ", ".join(missing_assets.values())
                )
        return work()


def _prompt_spec_from_args(args: argparse.Namespace):
    config = _llm_config_from_args(args)
    return _run_with_temporary_prompt_service(
        config,
        lambda: compose_prompt(
            args.description,
            direct_prompt=args.prompt,
            llm_config=config,
            force_pixel_trigger=args.force_pixel_trigger,
        ),
        enabled=not args.prompt,
    )


def _prompt_specs_from_args(args: argparse.Namespace, *, candidate_count: int):
    config = _llm_config_from_args(args)
    return _run_with_temporary_prompt_service(
        config,
        lambda: compose_prompt_batch(
            args.description,
            candidate_count=candidate_count,
            direct_prompt=args.prompt,
            llm_config=config,
            force_pixel_trigger=args.force_pixel_trigger,
        ),
        enabled=not args.prompt,
    )


def _run_with_temporary_prompt_service(config: LLMConfig, work, *, enabled: bool = True):
    if not enabled or config.provider != "ollama":
        return work()
    with temporary_ollama_server(config.endpoint, model=config.model, progress=print):
        validation = validate_ollama_model(config.endpoint, config.model, auto_start=False)
        if not validation.model_present:
            raise RuntimeError(f"Prompt model '{config.model}' is not installed in Ollama.")
        return work()


def _llm_config_from_args(args: argparse.Namespace) -> LLMConfig:
    config = LLMConfig.from_env()
    return LLMConfig(
        provider=(getattr(args, "llm_provider", None) or config.provider),
        model=(getattr(args, "llm_model", None) or config.model),
        endpoint=(getattr(args, "llm_endpoint", None) or config.endpoint),
        api_key=(getattr(args, "llm_api_key", None) or config.api_key),
        temperature=config.temperature,
        timeout_s=config.timeout_s,
        keep_alive=config.keep_alive,
        ollama_num_gpu=config.ollama_num_gpu,
        ollama_num_ctx=config.ollama_num_ctx,
        ollama_num_predict=config.ollama_num_predict,
        think=config.think,
    )


def _find_candidate(candidates: list[Candidate], index: int) -> Candidate:
    for candidate in candidates:
        if candidate.index == index:
            return candidate
    raise ValueError(f"candidate index {index} not found")


if __name__ == "__main__":
    raise SystemExit(main())
