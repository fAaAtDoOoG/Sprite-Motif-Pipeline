# Sprite Motif Pipeline

[中文 README](README.md)

An open-source txt2img pipeline for 2D pixel-art game character motifs. It uses Qwen-Image-2512 with a Pixel Art LoRA, generates a high-resolution motif image first, defaults to `1024x1024`, then downscales it with nearest-neighbor sampling to `64x64` by default. The default target is one static full-body character facing right.

## Features

- Browser GUI by default: `spritepipe-gui` launches a local web control panel.
- Prompt composer: write a short description and the pipeline rewrites it into image-generation prompts for pixel character motifs.
- Optional prompt LLM: supports Ollama and OpenAI-compatible chat APIs. If the LLM is unavailable, the deterministic built-in composer is used as a fallback.
- Ollama validation and download: the web GUI can validate the local Ollama prompt model and pull the missing model.
- Progress-tracked prompt preview: prompt rewriting runs as a background job, updates the shared progress bar, and can unload the Ollama prompt model after use.
- ComfyUI model validation and download: the GUI validates required nodes and default `.safetensors` filenames, then can download missing files into the selected ComfyUI `models` folder.
- Direct prompt mode: bypasses prompt rewriting and sends your prompt straight into the pipeline.
- Batch generation: generates multiple candidates per run and saves high-res images, low-res sprites, ComfyUI API prompts, and a manifest.
- Iteration loop: select a candidate, enter revision feedback, and generate another batch using the previous design as context.
- Apache-2.0 project license. Model weights are not redistributed by this repository.

## Installation

```powershell
$env:PYTHONUTF8="1"
uv venv
uv pip install -e ".[dev]"
```

On Windows, keeping `$env:PYTHONUTF8="1"` is recommended if the project path contains non-ASCII characters.

Model setup is documented in [docs/model_setup.md](docs/model_setup.md). The default expected files are:

- `qwen_image_2512_fp8_e4m3fn.safetensors`
- `qwen_2.5_vl_7b_fp8_scaled.safetensors`
- `qwen_image_vae.safetensors`
- `Qwen-Image-2512-Master-Pixel-Art-LoRA.safetensors`

## Browser GUI

On Windows, you can double-click `start_gui.bat` in the project root. It sets UTF-8, checks for `uv`, and launches the default browser GUI. If something is already running on port 7865, it opens the existing page.

Launch the default GUI:

```powershell
$env:PYTHONUTF8="1"
uv run spritepipe-gui
```

This starts a local web UI and opens your browser at `http://127.0.0.1:7865/`.

You can also run it explicitly:

```powershell
uv run spritepipe-web --port 7865
```

The old Tk desktop GUI is still available as a fallback:

```powershell
uv run spritepipe-tk
```

The browser UI supports:

- ComfyUI validation and missing `.safetensors` download.
- Prompt model validation and Ollama model pull.
- Prompt preview with progress feedback.
- Batch generation with progress and logs.
- Candidate preview, file links, and iteration feedback.

## CLI Usage

Validate ComfyUI after starting the backend:

```powershell
spritepipe validate-comfy --comfy-url http://127.0.0.1:8188
```

Generate a batch from a simple description:

```powershell
spritepipe generate --description "red-haired woman knight, light armor, brave personality" --batch-size 4
```

Change resolutions:

```powershell
spritepipe generate --description "small forest mage, green cloak" --high-res 1328x1328 --low-res 96x96 --batch-size 6
```

Inspect a run:

```powershell
spritepipe inspect runs\run_YYYYMMDD_HHMMSS
```

Select candidate `2`, apply revision feedback, and create the next batch:

```powershell
spritepipe iterate runs\run_YYYYMMDD_HHMMSS --index 2 --feedback "lighter armor, shorter hair, rounder silhouette" --batch-size 4
```

Use a direct prompt:

```powershell
spritepipe generate --prompt "Pixel Art, 2D pixel art game character sprite motif, one original full-body desert rogue, static pose, centered, facing right, tan scarf, curved blade, plain neutral background, no readable text." --batch-size 4
```

Export or dry-run without generating images:

```powershell
spritepipe workflow export --output workflows\qwen_image_2512_pixel_sprite_api.json
spritepipe generate --description "ice mage girl, blue and white palette" --dry-run
```

## Prompt LLM

The project does not require an online LLM. To use an external prompt-rewriting model, configure environment variables.

OpenAI-compatible:

```powershell
$env:SPRITEPIPE_LLM_PROVIDER="openai-compatible"
$env:SPRITEPIPE_LLM_ENDPOINT="https://api.example.com/v1/chat/completions"
$env:SPRITEPIPE_LLM_API_KEY="..."
$env:SPRITEPIPE_LLM_MODEL="your-chat-model"
```

Ollama:

```powershell
$env:SPRITEPIPE_LLM_PROVIDER="ollama"
$env:SPRITEPIPE_LLM_ENDPOINT="http://127.0.0.1:11434"
$env:SPRITEPIPE_LLM_MODEL="qwen2.5:7b-instruct"
```

The browser GUI defaults to Ollama `qwen2.5:7b-instruct`. When `Provider = ollama`, the web UI validates the prompt model before previewing, generating, or iterating, so it will not silently use the deterministic fallback while you expect the LLM. If the model is missing, use `Validate Prompt Model` or `Download Prompt Model` in the web UI. To explicitly use the built-in deterministic composer, set `Provider` to `none`.

Ollama can keep recently used models in memory. This project sends `keep_alive=0` for prompt rewriting by default, so the local prompt model unloads after each preview/generation request. The browser GUI also has an `Unload Prompt Model` button. If you prefer faster repeated prompt rewrites and have enough RAM/VRAM, set a longer value before launching the GUI:

```powershell
$env:SPRITEPIPE_LLM_KEEP_ALIVE="5m"
```

This only controls the prompt LLM. ComfyUI may still keep Qwen-Image model weights loaded for image generation.

The built-in prompt curriculum is stored at `src/sprite_motif_pipeline/prompt_training_examples.jsonl`. It is used as few-shot runtime context and can also serve as starter data for future prompt-model fine-tuning.

## Output Layout

Every generation run writes to `runs/run_YYYYMMDD_HHMMSS/`:

- `manifest.json`: description, prompt, seed, selected candidate, and candidate metadata.
- `api_prompts/*.json`: the ComfyUI API prompt for each candidate.
- `highres/*.png`: generated ComfyUI images.
- `lowres/*.png`: nearest-neighbor downscaled sprite images.
- `contact_sheet.png`: a low-res candidate comparison sheet.

## License And Model Notes

This repository contains pipeline code, prompt examples, workflow templates, and documentation only. It does not redistribute model weights.

- Qwen-Image-2512: released by the Qwen team under Apache-2.0.
- Qwen-Image-2512 Pixel Art LoRA: `prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA`, Apache-2.0, trigger word `Pixel Art`.
- Project code: Apache-2.0.

Users are responsible for downloading models themselves and complying with the applicable terms for the base model, LoRA, ComfyUI, LLM services, and generated content in their jurisdiction.

## Contributors

- [@fAaAtDoOoG](https://github.com/fAaAtDoOoG): project initiator and requirements designer.
- Codex: collaborative development, implementation, and documentation.

## Development

```powershell
uv run pytest -q
```

The upstream ComfyUI workflow backup is stored at `workflows/upstream/image_qwen_Image_2512.upstream.json`. Runtime API prompts are generated by `src/sprite_motif_pipeline/workflow.py`.
