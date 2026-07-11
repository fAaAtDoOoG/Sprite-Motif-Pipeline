# Model Setup

This project does not ship model weights. Put the files below in your local
ComfyUI model folders:

| File | Folder |
| --- | --- |
| `qwen_image_2512_fp8_e4m3fn.safetensors` | `ComfyUI/models/diffusion_models/` |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `ComfyUI/models/text_encoders/` |
| `qwen_image_vae.safetensors` | `ComfyUI/models/vae/` |
| `Qwen-Image-2512-Master-Pixel-Art-LoRA.safetensors` | `ComfyUI/models/loras/` |

The built-in/default workflow uses the FP8 Qwen-Image-2512 diffusion model plus
the Pixel Art LoRA listed above. Automatic downloads use these exact filenames.
You can enter another structurally compatible Qwen diffusion filename in the
web GUI. Other architectures should use a custom ComfyUI API workflow instead.

After placing files, start ComfyUI and run:

```powershell
spritepipe validate-comfy --comfy-url http://127.0.0.1:8188
```

If a required node is missing, update ComfyUI to a build that includes native
Qwen-Image support.

If the command reports missing model files, download those exact filenames into
the folders above and restart or refresh ComfyUI so `/object_info` sees the new
backend model lists.

The browser GUI has the same check. Its `Models` field defaults to a discovered
ComfyUI `models` folder, and `Validate ComfyUI` will offer to download missing
default `.safetensors` files into the correct subfolders. These files can be
large, so the GUI asks before downloading.

## Other Image Models

The web GUI has three image backends:

1. `Built-in Qwen-Image-2512` uses the files above.
2. `Custom ComfyUI workflow` uses any local txt2img model that has an API-format ComfyUI workflow.
3. `OpenAI-compatible Images API` sends prompts to a hosted or local `/v1/images/generations` endpoint.

For a custom ComfyUI workflow, export with **Save (API Format)** and replace
controllable values with `{{positive_prompt}}`, `{{negative_prompt}}`,
`{{width}}`, `{{height}}`, `{{seed}}`, `{{steps}}`, `{{cfg}}`,
`{{filename_prefix}}`, `{{model}}`, `{{lora_name}}`, or `{{lora_strength}}`.
Automatic model download is intentionally
limited to the built-in Qwen assets; install custom model files according to
their own model card and ComfyUI node documentation.

For API backends, configure the model identifier, full endpoint, and API key in
the browser, or use `SPRITEPIPE_IMAGE_MODEL`, `SPRITEPIPE_IMAGE_ENDPOINT`, and
`SPRITEPIPE_IMAGE_API_KEY`. Browser-entered keys are transient and are excluded
from run artifacts.

## Prompt Model

The default prompt-rewriting model is local Ollama `qwen3:32b` at
`http://127.0.0.1:11434`.

In the browser GUI, use `Validate Prompt Model` to check whether Ollama is
reachable and whether the selected model exists. If the model is missing, the UI
can pull it with Ollama. The same download is available through `Download Prompt
Model`.

Launching the GUI does not start Ollama or ComfyUI. `Preview Prompt` temporarily
starts only Ollama. Generation first completes that prompt phase and releases it,
then temporarily starts ComfyUI for Qwen-Image-2512 generation. Processes started
by the pipeline are stopped after their phase. A service that was already running
is reused and left running, while the model used by the pipeline is unloaded.

The prompt request includes the generation batch size. `qwen3:32b` returns one
distinct positive/negative prompt pair per candidate while preserving every
explicit source fact and speculating only where the user left visual choices open.
The default `num_predict` is `1024` so a four-candidate JSON response has enough
output budget.

You can also install it manually:

```powershell
ollama pull qwen3:32b
```
