from sprite_motif_pipeline import ollama
from sprite_motif_pipeline.ollama import OllamaServerLease, format_pull_progress, ollama_model_present


def test_ollama_model_present_accepts_exact_and_default_tag():
    models = ("qwen2.5:7b-instruct", "llama3.2:latest")

    assert ollama_model_present("qwen2.5:7b-instruct", models)
    assert ollama_model_present("llama3.2", models)
    assert not ollama_model_present("qwen2.5:14b", models)


def test_format_pull_progress_includes_percent_and_status():
    message = format_pull_progress("qwen2.5:7b-instruct", {"status": "downloading", "completed": 50, "total": 100})

    assert "50%" in message
    assert "downloading" in message


def test_stop_ollama_terminates_only_pipeline_owned_process(monkeypatch):
    process = object()
    calls = []
    monkeypatch.setattr(ollama, "terminate_process_tree", lambda value: calls.append(value))

    status = ollama.stop_ollama_server(
        OllamaServerLease("http://127.0.0.1:11434", process, True),  # type: ignore[arg-type]
    )

    assert status == "stopped"
    assert calls == [process]


def test_stop_ollama_unloads_but_keeps_preexisting_server(monkeypatch):
    calls = []
    monkeypatch.setattr(ollama, "ollama_version", lambda _endpoint: "test")
    monkeypatch.setattr(
        ollama,
        "unload_ollama_model",
        lambda endpoint, model, progress=None: calls.append((endpoint, model)),
    )

    status = ollama.stop_ollama_server(
        OllamaServerLease("http://127.0.0.1:11434", None, False),
        model="qwen3:32b",
    )

    assert status == "reused"
    assert calls == [("http://127.0.0.1:11434", "qwen3:32b")]
