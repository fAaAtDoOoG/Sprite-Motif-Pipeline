from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Size = Tuple[int, int]

DEFAULT_NEGATIVE_PROMPT = (
    "photorealistic rendering, 3D render, anime key visual, painterly brush strokes, "
    "soft airbrush, heavy antialiasing, blurry silhouette, muddy colors, noisy edges, "
    "cropped body, multiple characters, duplicate limbs, malformed hands, deformed face, "
    "dynamic action pose, weapon motion blur, complex scenery, readable text, logo, "
    "watermark, signature, UI frame, large background props, tiny details that disappear "
    "at 64x64"
)


@dataclass(frozen=True)
class ModelDefaults:
    diffusion_model: str = "qwen_image_2512_fp8_e4m3fn.safetensors"
    text_encoder: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    vae: str = "qwen_image_vae.safetensors"
    pixel_lora: str = "Qwen-Image-2512-Master-Pixel-Art-LoRA.safetensors"
    pixel_lora_strength: float = 0.9
    clip_type: str = "qwen_image"
    clip_device: str = "default"
    unet_weight_dtype: str = "default"
    aura_shift: float = 3.1
    steps: int = 50
    cfg: float = 4.0
    sampler_name: str = "euler"
    scheduler: str = "simple"
    denoise: float = 1.0


DEFAULTS = ModelDefaults()
DEFAULT_HIGH_RES: Size = (1024, 1024)
DEFAULT_LOW_RES: Size = (64, 64)
DEFAULT_PROMPT_MODEL = "qwen3:32b"
DEFAULT_PROMPT_MODEL_NUM_GPU = 999
DEFAULT_PROMPT_MODEL_NUM_CTX = 4096
DEFAULT_PROMPT_MODEL_NUM_PREDICT = 256
DEFAULT_PROMPT_MODEL_THINK = False
DEFAULT_PROMPT_MODEL_TIMEOUT = 900


def parse_size(value: str | int | None, default: Size) -> Size:
    if value is None:
        return default
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("size must be positive")
        return (value, value)

    raw = value.strip().lower().replace("*", "x").replace(",", "x")
    if raw.isdigit():
        number = int(raw)
        if number <= 0:
            raise ValueError("size must be positive")
        return (number, number)

    parts = [part.strip() for part in raw.split("x")]
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"invalid size '{value}', expected N or WIDTHxHEIGHT")

    width, height = (int(parts[0]), int(parts[1]))
    if width <= 0 or height <= 0:
        raise ValueError("size dimensions must be positive")
    return (width, height)


def format_size(size: Size) -> str:
    return f"{size[0]}x{size[1]}"
