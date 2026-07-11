from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DEFAULTS, ModelDefaults

Prompt = dict[str, dict[str, Any]]


def build_api_prompt(
    *,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    filename_prefix: str,
    defaults: ModelDefaults = DEFAULTS,
    lora_name: str | None = None,
    lora_strength: float | None = None,
    steps: int | None = None,
    cfg: float | None = None,
) -> Prompt:
    lora_name = defaults.pixel_lora if lora_name is None else lora_name
    lora_strength = defaults.pixel_lora_strength if lora_strength is None else lora_strength
    steps = defaults.steps if steps is None else steps
    cfg = defaults.cfg if cfg is None else cfg

    prompt: Prompt = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": defaults.diffusion_model,
                "weight_dtype": defaults.unet_weight_dtype,
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": defaults.text_encoder,
                "type": defaults.clip_type,
                "device": defaults.clip_device,
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": defaults.vae},
        },
    }

    model_node: list[Any]
    if lora_name:
        prompt["4"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": lora_name,
                "strength_model": lora_strength,
            },
        }
        model_node = ["4", 0]
    else:
        model_node = ["1", 0]

    prompt.update(
        {
            "5": {
                "class_type": "ModelSamplingAuraFlow",
                "inputs": {
                    "model": model_node,
                    "shift": defaults.aura_shift,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["2", 0],
                    "text": positive_prompt,
                },
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["2", 0],
                    "text": negative_prompt,
                },
            },
            "8": {
                "class_type": "EmptySD3LatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1,
                },
            },
            "9": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["5", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["8", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": defaults.sampler_name,
                    "scheduler": defaults.scheduler,
                    "denoise": defaults.denoise,
                },
            },
            "10": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["9", 0],
                    "vae": ["3", 0],
                },
            },
            "11": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["10", 0],
                    "filename_prefix": filename_prefix,
                },
            },
        }
    )
    return prompt


def export_api_prompt(path: Path, prompt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prompt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def required_node_types() -> set[str]:
    return {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "LoraLoaderModelOnly",
        "ModelSamplingAuraFlow",
        "CLIPTextEncode",
        "EmptySD3LatentImage",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    }
