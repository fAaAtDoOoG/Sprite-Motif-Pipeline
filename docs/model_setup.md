# Model Setup

This project does not ship model weights. Put the files below in your local
ComfyUI model folders:

| File | Folder |
| --- | --- |
| `qwen_image_2512_fp8_e4m3fn.safetensors` | `ComfyUI/models/diffusion_models/` |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `ComfyUI/models/text_encoders/` |
| `qwen_image_vae.safetensors` | `ComfyUI/models/vae/` |
| `Qwen-Image-2512-Master-Pixel-Art-LoRA.safetensors` | `ComfyUI/models/loras/` |

The default pipeline uses the FP8 Qwen-Image-2512 diffusion model for lower VRAM
pressure. If you have enough VRAM, you can pass another ComfyUI model filename
through the code defaults or by editing the generated API prompt JSON.

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

## Prompt Model

The default prompt-rewriting model is local Ollama `qwen2.5:7b-instruct` at
`http://127.0.0.1:11434`.

In the browser GUI, use `Validate Prompt Model` to check whether Ollama is
reachable and whether the selected model exists. If the model is missing, the UI
can pull it with Ollama. The same download is available through `Download Prompt
Model`.

`Preview Prompt` runs through the same background job/progress bar system as
generation. By default the project sends `keep_alive=0` to Ollama prompt-rewrite
requests, which unloads the prompt model after use and lowers memory pressure.
Use `Unload Prompt Model` in the browser GUI to explicitly free the selected
Ollama prompt model. If you prefer keeping it warm for repeated edits, set:

```powershell
$env:SPRITEPIPE_LLM_KEEP_ALIVE="5m"
```

This setting only affects the prompt LLM. ComfyUI may still retain image model
weights separately.

You can also install it manually:

```powershell
ollama pull qwen2.5:7b-instruct
```
