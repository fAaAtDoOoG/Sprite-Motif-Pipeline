from sprite_motif_pipeline.progress import generation_percent, percent_from_message, short_status


def test_percent_from_message_parses_download_updates():
    assert percent_from_message("model.safetensors: 45% (1.0 GB / 2.2 GB)") == 45
    assert percent_from_message("done D:/AI/ComfyUI/models/model.safetensors") == 100


def test_generation_percent_tracks_candidate_stages():
    assert generation_percent("[0] queue seed=123", 4) == 3
    assert generation_percent("[1] saved lowres=out.png", 4) == 43
    assert generation_percent("manifest=runs/run/manifest.json", 4) == 100


def test_short_status_truncates_long_messages():
    assert short_status("x" * 110, limit=12) == "xxxxxxxxx..."
