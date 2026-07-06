from sprite_motif_pipeline.ollama import format_pull_progress, ollama_model_present


def test_ollama_model_present_accepts_exact_and_default_tag():
    models = ("qwen2.5:7b-instruct", "llama3.2:latest")

    assert ollama_model_present("qwen2.5:7b-instruct", models)
    assert ollama_model_present("llama3.2", models)
    assert not ollama_model_present("qwen2.5:14b", models)


def test_format_pull_progress_includes_percent_and_status():
    message = format_pull_progress("qwen2.5:7b-instruct", {"status": "downloading", "completed": 50, "total": 100})

    assert "50%" in message
    assert "downloading" in message
