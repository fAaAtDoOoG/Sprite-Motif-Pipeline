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

The GUI has the same check. Its `Models` field defaults to a discovered ComfyUI
`models` folder, and `Validate` will offer to download missing default
`.safetensors` files into the correct subfolders. These files can be large, so
the GUI asks before downloading.
