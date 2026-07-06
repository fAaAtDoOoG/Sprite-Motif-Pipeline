from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from .config import Size


def downscale_nearest(input_path: Path, output_path: Path, size: Size) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as image:
        converted = image.convert("RGBA")
        resized = converted.resize(size, Image.Resampling.NEAREST)
        resized.save(output_path)
    return output_path


def make_contact_sheet(image_paths: Iterable[Path], output_path: Path, tile_size: int = 128) -> Path:
    paths = list(image_paths)
    if not paths:
        raise ValueError("cannot build contact sheet without images")

    margin = 12
    label_h = 22
    columns = min(4, len(paths))
    rows = (len(paths) + columns - 1) // columns
    width = columns * tile_size + (columns + 1) * margin
    height = rows * (tile_size + label_h) + (rows + 1) * margin
    sheet = Image.new("RGB", (width, height), (245, 245, 242))
    draw = ImageDraw.Draw(sheet)

    for index, path in enumerate(paths):
        row, col = divmod(index, columns)
        x = margin + col * (tile_size + margin)
        y = margin + row * (tile_size + label_h + margin)
        with Image.open(path) as image:
            tile = image.convert("RGBA").resize((tile_size, tile_size), Image.Resampling.NEAREST)
        sheet.paste(tile.convert("RGB"), (x, y))
        draw.text((x, y + tile_size + 4), f"{index}: {path.name}", fill=(25, 25, 25))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path
