from pathlib import Path

from sprite_motif_pipeline.config import DEFAULTS
from sprite_motif_pipeline.model_assets import assets_for_filenames, missing_local_assets


def test_assets_for_filenames_maps_defaults():
    assets = assets_for_filenames([DEFAULTS.diffusion_model, DEFAULTS.text_encoder])
    assert [asset.filename for asset in assets] == [DEFAULTS.diffusion_model, DEFAULTS.text_encoder]


def test_missing_local_assets_checks_subdirectories(tmp_path: Path):
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae" / DEFAULTS.vae).write_bytes(b"ok")
    missing = missing_local_assets(tmp_path)
    filenames = {asset.filename for asset in missing}
    assert DEFAULTS.vae not in filenames
    assert DEFAULTS.diffusion_model in filenames
