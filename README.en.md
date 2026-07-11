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
- Selectable image backend: keep the built-in Qwen-Image-2512 workflow, load any API-format ComfyUI txt2img workflow, or call an OpenAI-compatible Images API.
- Selectable prompt and image models: enter a local model identifier, API model name, endpoint, and transient API key directly in the browser GUI.
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
The top bar includes `Stop Server` for manually stopping the local web server. `Auto stop on close` is enabled by default, so the local web server exits shortly after the browser page is closed; normal refreshes usually reconnect before the timeout.

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
- Built-in Qwen, custom ComfyUI, and OpenAI-compatible Images API backends.
- Local server launch buttons for ComfyUI and Ollama.
- Prompt model validation and Ollama model pull.
- Prompt preview with progress feedback.
- Batch generation with progress, logs, and a default high-res/low-res comparison viewer.
- Candidate preview, file links, and iteration feedback.

The default remains the Qwen-Image-2512 + Pixel Art LoRA txt2img workflow.
Selecting another backend changes only how the generated prompts are submitted.
Selecting a candidate supplies its accepted prompts to the prompt LLM as
iteration context; the candidate image is never sent back to the image model.

Launching the browser GUI does not start ComfyUI or Ollama. `Generate` and
`Iterate Selected` first start Ollama, expand and self-check the complete set of
positive/negative prompts in one request, and then stop the Ollama process started by that job. Only then does the
pipeline starts ComfyUI for a built-in or custom ComfyUI backend and stops the
ComfyUI process started by that job. An Images API backend skips ComfyUI and
sends the request after the prompt phase. `Preview Prompt` runs only the Ollama
phase. Direct-prompt mode skips Ollama.

Description mode creates one independent positive/negative prompt pair per
candidate. A default batch of four therefore explores four meaningfully different,
source-compatible visual interpretations instead of changing only the seed.

## Image Backends

Choose an image backend in the browser's `Backend` section:

- `Built-in Qwen-Image-2512`: the default ComfyUI workflow and Pixel Art LoRA. The selected image model is the diffusion-model filename shown by ComfyUI.
- `Custom ComfyUI workflow`: any locally deployed txt2img model represented by a ComfyUI API-format JSON workflow.
- `OpenAI-compatible Images API`: a hosted API or local server implementing [`POST /v1/images/generations`](https://developers.openai.com/api/reference/resources/images/methods/generate). The client accepts `data[0].b64_json` and `data[0].url` responses.

For a custom ComfyUI backend, export the workflow with ComfyUI's **Save (API
Format)** option, then replace the values you want SpritePipe to control with
these placeholders:

`{{positive_prompt}}`, `{{negative_prompt}}`, `{{width}}`, `{{height}}`,
`{{seed}}`, `{{steps}}`, `{{cfg}}`, `{{filename_prefix}}`, and `{{model}}`.
Custom workflows may also use `{{lora_name}}` and `{{lora_strength}}`.

An exact placeholder preserves numeric types; for example, use
`"seed": "{{seed}}"`. A placeholder embedded inside a longer string is rendered
as text. The workflow must remain pure txt2img and include its own loader,
sampler, decoder, and save nodes. SpritePipe validates its node types against
the active ComfyUI backend before generation.

For an Images API, the web form keeps both image and prompt API keys only in the
current request. Keys are not written to manifests, exported request JSON, logs,
or browser storage. Environment variables are recommended for CLI use:

```powershell
$env:SPRITEPIPE_IMAGE_BACKEND="openai-images"
$env:SPRITEPIPE_IMAGE_ENDPOINT="https://api.example.com/v1/images/generations"
$env:SPRITEPIPE_IMAGE_API_KEY="..."
$env:SPRITEPIPE_IMAGE_MODEL="your-image-model"
spritepipe generate --description "small forest mage, green cloak"
```

The standard Images API has no separate negative-prompt field, so SpritePipe
adds the negative concepts to the submitted prompt as explicit avoid
instructions. OpenAI-compatible local servers vary; compatibility means they
follow the HTTP request and response shape above, not that every model supports
the same sizes or parameters.

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

Use a custom local ComfyUI model:

```powershell
spritepipe generate --description "small forest mage" --image-backend custom-comfy --custom-workflow workflows\my_model_api.json --image-model my_model.safetensors
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
$env:SPRITEPIPE_LLM_MODEL="qwen3:32b"
$env:SPRITEPIPE_OLLAMA_NUM_GPU="999"
$env:SPRITEPIPE_OLLAMA_NUM_CTX="4096"
$env:SPRITEPIPE_OLLAMA_NUM_PREDICT="1024"
```

The browser GUI defaults to Ollama `qwen3:32b` with `num_gpu=999`, `num_ctx=4096`, `num_predict=1024`, and `temperature=0.55`. If the model is missing, use `Validate Prompt Model` or `Download Prompt Model` in the web UI. To explicitly use the built-in deterministic composer, set `Provider` to `none`.

When enabled, the prompt LLM preserves explicit facts, decides which concepts belong in the positive and negative prompts, and speculates about compatible unspecified choices such as silhouette, proportion, shape rhythm, feature placement, value grouping, surface treatment, and low-resolution readability. It creates one coherent prompt pair per candidate and gives each candidate a distinct visual direction without contradicting or replacing the requested identity. The Python code contains no character example, source-specific vocabulary, or fixed keyword splitter.

After generation, selecting a candidate shows high-res on the left and the low-res sprite on the right, upscaled pixel-perfect to the same displayed size as the high-res image. The viewer supports mouse-wheel zoom, left-button drag panning, movement buttons, and zoom in/out controls.

The Backend section keeps `Start ComfyUI` and `Start Ollama` as manual diagnostic controls. The automatic job lifecycle terminates only processes it started. If either service was already running, the pipeline reuses it and unloads the relevant model afterward, but leaves the externally owned process running.

Prompt expansion and fact checking happen in one model request, which sends `keep_alive=0`. A pipeline-owned Ollama service is stopped when that phase ends; a reused external service remains running with the prompt model unloaded.

The prompt LLM timeout defaults to `SPRITEPIPE_LLM_TIMEOUT="900"` to tolerate `qwen3:32b` cold starts. The web GUI Prompt Model section also exposes GPU layers, context, max tokens, and thinking mode.

The prompt LLM receives only the current description, candidate count, selected candidate prompts, and user feedback. It expands them into distinct txt2img-friendly visual specifications without a character-specific training example or hardcoded vocabulary. During iteration, every variant preserves the selected identity and unaffected accepted details while exploring a different compatible realization of the feedback.

## Output Layout

Every generation run writes to `runs/run_YYYYMMDD_HHMMSS/`:

- `manifest.json`: description, original user input history, prompt, seed, selected candidate, and candidate metadata.
- `api_prompts/*.json`: the resolved ComfyUI prompt or Images API request for each candidate; credentials are excluded.
- `highres/*`: generated high-resolution images.
- `lowres/*.png`: nearest-neighbor downscaled sprite images.
- `contact_sheet.png`: a low-res candidate comparison sheet.

## License And Model Notes

This repository contains pipeline code, workflow templates, and documentation only. It does not redistribute model weights.

- Qwen-Image-2512: released by the Qwen team under Apache-2.0.
- Qwen-Image-2512 Pixel Art LoRA: `prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA`, Apache-2.0, trigger word `Pixel Art`.
- Project code: Apache-2.0.

Users download model weights themselves. Before use or redistribution, treat each
model card and its license text as the controlling source and comply with the
applicable terms for the base model, LoRA, ComfyUI, LLM services, and generated
content in their jurisdiction.

Custom models and external APIs are user-selected integrations and are not
endorsed, bundled, or relicensed by this project. Verify their model licenses,
API terms, commercial-use restrictions, privacy rules, and output rights before
use.

## Contributors

- [@fAaAtDoOoG](https://github.com/fAaAtDoOoG): project initiator and requirements designer.
- Codex: collaborative development, implementation, and documentation.

## Development

```powershell
uv run pytest -q
```

The upstream ComfyUI workflow backup is stored at `workflows/upstream/image_qwen_Image_2512.upstream.json`. Runtime API prompts are generated by `src/sprite_motif_pipeline/workflow.py`.
