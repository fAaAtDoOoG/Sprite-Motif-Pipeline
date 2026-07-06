from pathlib import Path

from sprite_motif_pipeline.comfy import build_comfy_launch_plan, resolve_comfyui_dir


def test_resolve_comfyui_dir_from_models_root(tmp_path: Path):
    comfy = tmp_path / "ComfyUI"
    (comfy / "models").mkdir(parents=True)
    (comfy / "main.py").write_text("", encoding="utf-8")

    assert resolve_comfyui_dir("", comfy / "models") == comfy.resolve()


def test_build_comfy_launch_plan_prefers_windows_batch(tmp_path: Path):
    (tmp_path / "run_nvidia_gpu.bat").write_text("@echo off\n", encoding="utf-8")

    plan = build_comfy_launch_plan(tmp_path, "http://127.0.0.1:8188")

    assert plan.label == "run_nvidia_gpu.bat"
    assert plan.command[:2] == ("cmd.exe", "/c")


def test_build_comfy_launch_plan_uses_comfy_virtualenv(tmp_path: Path):
    (tmp_path / "main.py").write_text("", encoding="utf-8")
    python = tmp_path / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    plan = build_comfy_launch_plan(tmp_path, "http://127.0.0.1:8188")

    assert plan.label == "main.py"
    assert plan.command[0] == str(python)
