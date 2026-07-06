from pathlib import Path

from PIL import Image

from sprite_motif_pipeline.postprocess import downscale_nearest, make_contact_sheet


def test_downscale_nearest(tmp_path: Path):
    source = tmp_path / "source.png"
    target = tmp_path / "low.png"
    Image.new("RGB", (16, 16), (255, 0, 0)).save(source)
    downscale_nearest(source, target, (4, 4))
    with Image.open(target) as image:
        assert image.size == (4, 4)


def test_contact_sheet(tmp_path: Path):
    image_path = tmp_path / "one.png"
    Image.new("RGB", (4, 4), (0, 255, 0)).save(image_path)
    sheet = make_contact_sheet([image_path], tmp_path / "sheet.png")
    assert sheet.exists()
