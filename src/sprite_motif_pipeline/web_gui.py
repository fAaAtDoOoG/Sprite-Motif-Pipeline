from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import threading
import webbrowser
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlencode, urlparse

from .comfy import ComfyClient, validate_model_assets, validate_required_nodes
from .config import DEFAULT_HIGH_RES, DEFAULT_LOW_RES, DEFAULTS, format_size, parse_size
from .model_assets import assets_for_filenames, default_models_root, download_assets, missing_local_assets
from .ollama import DEFAULT_OLLAMA_ENDPOINT, pull_ollama_model, unload_ollama_model, validate_ollama_model
from .progress import generation_percent, percent_from_message, short_status
from .prompting import LLMConfig, compose_prompt
from .runner import GenerationOptions, generate_batch
from .session import Candidate, load_manifest, save_manifest
from .workflow import required_node_types


@dataclass
class JobSnapshot:
    id: int = 0
    active: bool = False
    label: str = "Ready"
    percent: int = 0
    logs: list[str] = field(default_factory=list)
    error: str = ""
    result: dict[str, Any] | None = None
    prompt: str = ""


class WebAppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.job = JobSnapshot()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return asdict(self.job)

    def start(self, label: str, work: Callable[[Callable[[str, int | None], None]], dict[str, Any]]) -> dict[str, Any]:
        with self.lock:
            if self.job.active:
                raise RuntimeError("A job is already running.")
            self.job = JobSnapshot(id=self.job.id + 1, active=True, label=label, logs=[label])
            job_id = self.job.id

        def progress(message: str, percent: int | None = None) -> None:
            with self.lock:
                if self.job.id != job_id:
                    return
                self.job.logs.append(message)
                if percent is not None:
                    self.job.percent = max(0, min(100, int(percent)))
                self.job.label = short_status(message)

        def target() -> None:
            try:
                result = work(progress)
            except Exception as exc:  # noqa: BLE001 - web UI should surface concise job failures.
                with self.lock:
                    self.job.error = str(exc)
                    self.job.active = False
                    self.job.label = "Failed"
            else:
                with self.lock:
                    self.job.result = result
                    self.job.percent = 100
                    self.job.active = False
                    self.job.label = "Done"

        threading.Thread(target=target, daemon=True).start()
        return {"job_id": job_id}

    def set_prompt(self, prompt: str) -> None:
        with self.lock:
            self.job.prompt = prompt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spritepipe-web", description="Launch the Sprite Motif browser UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    state = WebAppState()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    url = f"http://{args.host}:{args.port}/"
    print(f"Sprite Motif web GUI: {url}")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


def make_handler(state: WebAppState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "SpritePipeWeb/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_text(INDEX_HTML, "text/html; charset=utf-8")
                elif parsed.path == "/style.css":
                    self._send_text(STYLE_CSS, "text/css; charset=utf-8")
                elif parsed.path == "/app.js":
                    self._send_text(APP_JS, "application/javascript; charset=utf-8")
                elif parsed.path == "/api/defaults":
                    self._send_json(default_payload())
                elif parsed.path == "/api/job":
                    self._send_json(state.snapshot())
                elif parsed.path == "/api/latest-run":
                    output_dir = Path(_query(parsed, "output_dir", "runs"))
                    self._send_json(latest_run_response(output_dir))
                elif parsed.path == "/api/run":
                    self._send_json(run_response(Path(_query(parsed, "path", ""))))
                elif parsed.path == "/api/file":
                    self._send_file(Path(_query(parsed, "path", "")))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=500)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
                if parsed.path == "/api/validate-comfy":
                    self._send_json(validate_comfy_response(payload))
                elif parsed.path == "/api/download-models":
                    self._send_json(state.start("Downloading ComfyUI models", lambda progress: download_models_job(payload, progress)))
                elif parsed.path == "/api/validate-llm":
                    self._send_json(validate_llm_response(payload))
                elif parsed.path == "/api/download-llm":
                    self._send_json(state.start("Downloading prompt model", lambda progress: download_llm_job(payload, progress)))
                elif parsed.path == "/api/unload-llm":
                    self._send_json(state.start("Unloading prompt model", lambda progress: unload_llm_job(payload, progress)))
                elif parsed.path == "/api/prompt":
                    self._send_json(prompt_response(payload))
                elif parsed.path == "/api/preview-prompt":
                    self._send_json(state.start("Previewing prompt", lambda progress: preview_prompt_job(payload, state, progress)))
                elif parsed.path == "/api/generate":
                    self._send_json(state.start("Generating", lambda progress: generate_job(payload, state, progress)))
                elif parsed.path == "/api/iterate":
                    self._send_json(state.start("Iterating", lambda progress: iterate_job(payload, state, progress)))
                elif parsed.path == "/api/open-path":
                    open_local_path(Path(str(payload.get("path", ""))))
                    self._send_json({"ok": True})
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=500)

        def log_message(self, fmt: str, *args: object) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_text(self, text: str, content_type: str) -> None:
            data = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, path: Path) -> None:
            path = path.expanduser().resolve()
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def default_payload() -> dict[str, Any]:
    return {
        "comfy_url": "http://127.0.0.1:8188",
        "models_root": str(default_models_root()),
        "output_dir": "runs",
        "mode": "description",
        "description": "red-haired woman knight, light armor, brave personality",
        "batch_size": 4,
        "high_res": format_size(DEFAULT_HIGH_RES),
        "low_res": format_size(DEFAULT_LOW_RES),
        "seed": "",
        "steps": DEFAULTS.steps,
        "cfg": DEFAULTS.cfg,
        "lora_name": DEFAULTS.pixel_lora,
        "lora_strength": DEFAULTS.pixel_lora_strength,
        "timeout": 900,
        "dry_run": False,
        "llm_provider": "ollama",
        "llm_model": "qwen2.5:7b-instruct",
        "llm_endpoint": DEFAULT_OLLAMA_ENDPOINT,
    }


def validate_comfy_response(payload: dict[str, Any]) -> dict[str, Any]:
    client = ComfyClient(str(payload.get("comfy_url") or "http://127.0.0.1:8188"))
    missing_nodes = validate_required_nodes(client, required_node_types())
    if missing_nodes:
        return {"status": "missing_nodes", "missing_nodes": missing_nodes}

    missing_assets = validate_model_assets(client)
    if not missing_assets:
        return {"status": "ready"}

    models_root = Path(str(payload.get("models_root") or default_models_root()))
    local_assets = assets_for_filenames(missing_assets.values())
    local_missing = missing_local_assets(models_root, local_assets)
    return {
        "status": "missing_assets",
        "missing_assets": missing_assets,
        "models_root": str(models_root),
        "local_missing": [asdict(asset) for asset in local_missing],
    }


def download_models_job(payload: dict[str, Any], progress: Callable[[str, int | None], None]) -> dict[str, Any]:
    models_root = Path(str(payload.get("models_root") or default_models_root()))
    filenames = [str(name) for name in payload.get("filenames", []) if str(name).strip()]
    assets = assets_for_filenames(filenames) if filenames else missing_local_assets(models_root)
    paths = download_assets(models_root, assets, progress=lambda message: progress(message, percent_from_message(message)))
    return {"kind": "model_download", "paths": [str(path) for path in paths]}


def validate_llm_response(payload: dict[str, Any]) -> dict[str, Any]:
    config = llm_config_from_payload(payload)
    if config.provider != "ollama":
        return {"status": "unsupported", "message": "Automatic local model validation is available for Ollama providers."}
    result = validate_ollama_model(config.endpoint, config.model)
    return {"status": "ready" if result.model_present else "missing_model" if result.server_available else "server_unavailable", "result": asdict(result)}


def download_llm_job(payload: dict[str, Any], progress: Callable[[str, int | None], None]) -> dict[str, Any]:
    config = llm_config_from_payload(payload)
    if config.provider != "ollama":
        raise ValueError("Automatic local model download is available for Ollama providers.")
    model = pull_ollama_model(config.endpoint, config.model, progress=lambda message: progress(message, percent_from_message(message)))
    return {"kind": "llm_download", "model": model}


def unload_llm_job(payload: dict[str, Any], progress: Callable[[str, int | None], None]) -> dict[str, Any]:
    config = llm_config_from_payload(payload)
    if config.provider != "ollama":
        raise ValueError("Prompt model unload is available for Ollama providers.")
    model = unload_ollama_model(config.endpoint, config.model, progress=lambda message: progress(message, percent_from_message(message)))
    return {"kind": "llm_unload", "model": model}


def prompt_response(payload: dict[str, Any]) -> dict[str, Any]:
    spec = compose_from_payload(payload)
    return {
        "positive_prompt": spec.positive_prompt,
        "negative_prompt": spec.negative_prompt,
        "source": spec.source,
        "notes": spec.notes,
    }


def preview_prompt_job(payload: dict[str, Any], state: WebAppState, progress: Callable[[str, int | None], None]) -> dict[str, Any]:
    progress("Rewriting prompt with prompt model", 10)
    spec = compose_from_payload(payload)
    state.set_prompt(spec.positive_prompt)
    progress("Prompt preview ready", 95)
    return {
        "kind": "prompt",
        "positive_prompt": spec.positive_prompt,
        "negative_prompt": spec.negative_prompt,
        "source": spec.source,
        "notes": spec.notes,
    }


def generate_job(payload: dict[str, Any], state: WebAppState, progress: Callable[[str, int | None], None]) -> dict[str, Any]:
    spec = compose_from_payload(payload)
    state.set_prompt(spec.positive_prompt)
    options = generation_options_from_payload(payload)
    mode = str(payload.get("mode") or "description")
    text = str(payload.get("description") or payload.get("text") or "")
    description = text if mode == "description" else str(payload.get("prompt") or text)
    run_dir = generate_batch(
        spec,
        description=description,
        options=options,
        progress=lambda message: progress(message, generation_percent(message, options.batch_size)),
    )
    return {"kind": "run", "run": run_response(run_dir)}


def iterate_job(payload: dict[str, Any], state: WebAppState, progress: Callable[[str, int | None], None]) -> dict[str, Any]:
    run_dir = Path(str(payload.get("run_dir") or ""))
    manifest = load_manifest(run_dir)
    index = int(payload.get("selected_index", 0))
    candidate = selected_candidate(manifest.candidates, index)
    feedback = str(payload.get("feedback") or "").strip()
    if not feedback:
        raise ValueError("Feedback is empty.")

    config = llm_config_from_payload(payload)
    spec = compose_prompt(
        manifest.description,
        feedback=feedback,
        previous_prompt=candidate.positive_prompt,
        llm_config=config,
        allow_fallback=config.provider in {"", "none"},
    )
    state.set_prompt(spec.positive_prompt)
    options = generation_options_from_payload(payload)
    new_run_dir = generate_batch(
        spec,
        description=manifest.description,
        options=options,
        parent_run=str(run_dir),
        selected_index=candidate.index,
        feedback=feedback,
        progress=lambda message: progress(message, generation_percent(message, options.batch_size)),
    )
    manifest.selected_index = candidate.index
    manifest.feedback = feedback
    save_manifest(run_dir, manifest)
    return {"kind": "run", "run": run_response(new_run_dir)}


def compose_from_payload(payload: dict[str, Any]):
    mode = str(payload.get("mode") or "description")
    text = str(payload.get("description") or payload.get("text") or "")
    direct = text if mode == "prompt" else None
    description = text if mode == "description" else None
    config = llm_config_from_payload(payload)
    return compose_prompt(
        description,
        direct_prompt=direct,
        force_pixel_trigger=True,
        llm_config=config,
        allow_fallback=config.provider in {"", "none"},
    )


def generation_options_from_payload(payload: dict[str, Any]) -> GenerationOptions:
    seed_raw = str(payload.get("seed") or "").strip()
    seed = int(seed_raw) if seed_raw else None
    high_res = str(payload.get("high_res") or format_size(DEFAULT_HIGH_RES))
    low_res = str(payload.get("low_res") or format_size(DEFAULT_LOW_RES))
    parse_size(high_res, DEFAULT_HIGH_RES)
    parse_size(low_res, DEFAULT_LOW_RES)
    return GenerationOptions(
        batch_size=int(payload.get("batch_size") or 4),
        high_res=high_res,
        low_res=low_res,
        seed=seed,
        steps=int(payload.get("steps") or DEFAULTS.steps),
        cfg=float(payload.get("cfg") or DEFAULTS.cfg),
        lora_name=str(payload.get("lora_name") or DEFAULTS.pixel_lora),
        lora_strength=float(payload.get("lora_strength") or DEFAULTS.pixel_lora_strength),
        comfy_url=str(payload.get("comfy_url") or "http://127.0.0.1:8188"),
        timeout=int(payload.get("timeout") or 900),
        output_dir=Path(str(payload.get("output_dir") or "runs")),
        dry_run=bool(payload.get("dry_run")),
    )


def llm_config_from_payload(payload: dict[str, Any]) -> LLMConfig:
    env_config = LLMConfig.from_env()
    return LLMConfig(
        provider=str(payload.get("llm_provider") or env_config.provider or "ollama").lower(),
        model=str(payload.get("llm_model") or env_config.model or "qwen2.5:7b-instruct"),
        endpoint=str(payload.get("llm_endpoint") or env_config.endpoint or DEFAULT_OLLAMA_ENDPOINT),
        api_key=env_config.api_key,
        temperature=env_config.temperature,
        timeout_s=env_config.timeout_s,
        keep_alive=env_config.keep_alive,
    )


def latest_run_response(output_dir: Path) -> dict[str, Any]:
    runs = sorted((path for path in output_dir.glob("run_*") if (path / "manifest.json").exists()), key=lambda path: path.stat().st_mtime, reverse=True)
    if not runs:
        return {"status": "empty", "run": None}
    return {"status": "ready", "run": run_response(runs[0])}


def run_response(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    manifest = load_manifest(run_dir)
    data = asdict(manifest)
    data["run_dir"] = str(run_dir)
    data["contact_sheet_url"] = file_url(run_dir / "contact_sheet.png") if (run_dir / "contact_sheet.png").exists() else ""
    for candidate in data["candidates"]:
        for key in ("lowres_path", "highres_path", "api_prompt_path"):
            value = candidate.get(key) or ""
            candidate[f"{key}_url"] = file_url(Path(value)) if value and Path(value).exists() else ""
    return data


def selected_candidate(candidates: list[Candidate], index: int) -> Candidate:
    for candidate in candidates:
        if candidate.index == index:
            return candidate
    raise ValueError(f"candidate index {index} not found")


def file_url(path: Path) -> str:
    return "/api/file?" + urlencode({"path": str(path)})


def open_local_path(path: Path) -> None:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if hasattr(os, "startfile"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif os.name == "posix":
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, str(path)])
    else:
        webbrowser.open(path.as_uri())


def _query(parsed, key: str, default: str) -> str:
    values = parse_qs(parsed.query).get(key)
    return values[0] if values else default


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sprite Motif Pipeline</title>
  <link rel="stylesheet" href="/style.css?v=3">
</head>
<body>
  <header class="topbar">
    <div>
      <h1>Sprite Motif Pipeline</h1>
      <p id="statusText">Ready</p>
    </div>
    <div class="progress-wrap">
      <progress id="jobProgress" max="100" value="0"></progress>
      <span id="progressText">0%</span>
    </div>
  </header>

  <main class="workspace">
    <section class="panel controls">
      <div class="section-title">Backend</div>
      <div class="grid two">
        <label>ComfyUI<input id="comfyUrl"></label>
        <label>Output<input id="outputDir"></label>
        <label class="wide">Models<input id="modelsRoot"></label>
      </div>
      <div class="toolbar">
        <button id="validateComfy">Validate ComfyUI</button>
        <button id="downloadModels">Download Missing</button>
        <label class="check"><input id="dryRun" type="checkbox"> Dry run</label>
      </div>

      <div class="section-title">Input</div>
      <div class="segmented">
        <label><input type="radio" name="mode" value="description" checked> Description</label>
        <label><input type="radio" name="mode" value="prompt"> Direct prompt</label>
      </div>
      <textarea id="description" rows="6"></textarea>

      <div class="section-title">Generation</div>
      <div class="grid three">
        <label>Batch<input id="batchSize" type="number" min="1" max="32"></label>
        <label>Seed<input id="seed"></label>
        <label>Steps<input id="steps" type="number" min="1" max="100"></label>
        <label>High res<input id="highRes"></label>
        <label>Low res<input id="lowRes"></label>
        <label>CFG<input id="cfg" type="number" step="0.1"></label>
        <label class="wide">LoRA<input id="loraName"></label>
        <label>Strength<input id="loraStrength" type="number" step="0.05"></label>
        <label>Timeout<input id="timeout" type="number" min="30"></label>
      </div>

      <div class="section-title">Prompt Model</div>
      <div class="grid two">
        <label>Provider<select id="llmProvider"><option>ollama</option><option>none</option><option>openai-compatible</option><option>openai</option></select></label>
        <label>Model<input id="llmModel"></label>
        <label class="wide">Endpoint<input id="llmEndpoint"></label>
      </div>
      <div class="toolbar">
        <button id="validateLlm">Validate Prompt Model</button>
        <button id="downloadLlm">Download Prompt Model</button>
        <button id="unloadLlm">Unload Prompt Model</button>
      </div>

      <div class="toolbar primary">
        <button id="previewPrompt">Preview Prompt</button>
        <button id="generate">Generate</button>
      </div>
    </section>

    <section class="panel results">
      <div class="result-head">
        <div>
          <div class="section-title">Run</div>
          <input id="runPath" placeholder="Run directory">
        </div>
        <div class="toolbar">
          <button id="latestRun">Latest</button>
          <button id="loadRun">Load</button>
          <button id="openRun">Open Folder</button>
        </div>
      </div>

      <div class="preview-zone">
        <aside>
          <div class="section-title">Candidates</div>
          <div id="candidateList" class="candidate-list"></div>
        </aside>
        <figure class="image-stage">
          <img id="previewImage" alt="">
        </figure>
      </div>

      <div class="filebar">
        <a id="lowLink" target="_blank">Lowres</a>
        <a id="highLink" target="_blank">Highres</a>
        <a id="apiLink" target="_blank">API JSON</a>
      </div>

      <div class="section-title">Iteration</div>
      <textarea id="feedback" rows="3">lighter armor, shorter hair, rounder silhouette</textarea>
      <div class="toolbar">
        <button id="iterate">Iterate Selected</button>
      </div>

      <div class="split">
        <div>
          <div class="section-title">Prompt</div>
          <textarea id="promptPreview" rows="9" readonly></textarea>
        </div>
        <div>
          <div class="section-title">Log</div>
          <pre id="logBox"></pre>
        </div>
      </div>
    </section>
  </main>
  <script src="/app.js?v=3"></script>
</body>
</html>
"""


STYLE_CSS = """
:root {
  color-scheme: light;
  --bg: #f5f3ee;
  --panel: #ffffff;
  --ink: #242526;
  --muted: #666b70;
  --line: #d8d4ca;
  --accent: #1f7a6b;
  --accent-2: #b24d3e;
  --soft: #edf7f4;
  --shadow: 0 14px 40px rgba(42, 42, 37, 0.08);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  min-width: 360px;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 "Segoe UI", Arial, sans-serif;
}
.topbar {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 16px 22px;
  background: rgba(255, 255, 255, 0.94);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(8px);
}
h1 {
  margin: 0;
  font-size: 22px;
  font-weight: 720;
  letter-spacing: 0;
}
#statusText {
  margin: 2px 0 0;
  color: var(--muted);
  min-height: 20px;
}
.progress-wrap {
  display: grid;
  grid-template-columns: minmax(180px, 320px) 46px;
  gap: 10px;
  align-items: center;
}
progress {
  width: 100%;
  height: 14px;
  accent-color: var(--accent);
}
.workspace {
  display: grid;
  grid-template-columns: minmax(360px, 500px) minmax(420px, 1fr);
  gap: 18px;
  padding: 18px;
  align-items: start;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
  padding: 16px;
}
.section-title {
  margin: 14px 0 8px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}
.section-title:first-child { margin-top: 0; }
.grid {
  display: grid;
  gap: 10px;
}
.two { grid-template-columns: 1fr 1fr; }
.three { grid-template-columns: repeat(3, 1fr); }
.wide { grid-column: 1 / -1; }
label {
  display: grid;
  gap: 5px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 650;
}
input, select, textarea {
  width: 100%;
  min-height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px 9px;
  color: var(--ink);
  background: #fff;
  font: inherit;
}
textarea {
  resize: vertical;
  min-height: 80px;
}
button, .filebar a {
  min-height: 34px;
  border: 1px solid #bfc8c4;
  border-radius: 6px;
  padding: 7px 12px;
  background: #fff;
  color: var(--ink);
  font: inherit;
  font-weight: 650;
  cursor: pointer;
  text-align: center;
  text-decoration: none;
}
button:hover, .filebar a:hover {
  border-color: var(--accent);
  color: var(--accent);
}
.primary button:last-child {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin-top: 10px;
}
.check {
  display: flex;
  grid-auto-flow: column;
  align-items: center;
  gap: 6px;
}
.check input { width: auto; min-height: auto; }
.segmented {
  display: flex;
  width: max-content;
  border: 1px solid var(--line);
  border-radius: 6px;
  overflow: hidden;
  margin-bottom: 8px;
}
.segmented label {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 7px 10px;
  background: #fff;
  border-right: 1px solid var(--line);
}
.segmented label:last-child { border-right: 0; }
.segmented input { width: auto; min-height: auto; }
.result-head {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 12px;
  align-items: end;
}
.preview-zone {
  display: grid;
  grid-template-columns: 220px minmax(240px, 1fr);
  gap: 14px;
  margin-top: 14px;
}
.candidate-list {
  display: grid;
  gap: 6px;
  max-height: 460px;
  overflow: auto;
}
.candidate {
  display: grid;
  gap: 2px;
  padding: 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  cursor: pointer;
}
.candidate.active {
  background: var(--soft);
  border-color: var(--accent);
}
.candidate small {
  color: var(--muted);
  overflow-wrap: anywhere;
}
.image-stage {
  margin: 0;
  display: grid;
  place-items: center;
  min-height: 460px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfaf7;
}
.image-stage img {
  max-width: min(100%, 560px);
  max-height: 560px;
  image-rendering: pixelated;
}
.filebar {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin: 10px 0 12px;
}
.split {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
#logBox {
  height: 210px;
  margin: 0;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 9px;
  background: #222625;
  color: #e9eee9;
  white-space: pre-wrap;
}

@media (max-width: 980px) {
  .workspace, .preview-zone, .split, .result-head { grid-template-columns: 1fr; }
  .three, .two { grid-template-columns: 1fr 1fr; }
  .topbar { align-items: stretch; flex-direction: column; }
  .progress-wrap { grid-template-columns: 1fr 46px; }
}

@media (max-width: 580px) {
  .three, .two { grid-template-columns: 1fr; }
  .workspace { padding: 10px; }
  .panel { padding: 12px; }
  .filebar { grid-template-columns: 1fr; }
}
"""


APP_JS = """
const fields = [
  "comfyUrl", "modelsRoot", "outputDir", "description", "batchSize", "seed",
  "steps", "highRes", "lowRes", "cfg", "loraName", "loraStrength", "timeout",
  "llmProvider", "llmModel", "llmEndpoint"
];

let currentRun = null;
let selectedIndex = 0;
let handledJob = 0;

const $ = (id) => document.getElementById(id);

async function api(path, body = null) {
  const options = body === null ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || response.statusText);
  return data;
}

function payload() {
  const mode = document.querySelector("input[name='mode']:checked").value;
  return {
    comfy_url: $("comfyUrl").value,
    models_root: $("modelsRoot").value,
    output_dir: $("outputDir").value,
    mode,
    description: $("description").value,
    batch_size: Number($("batchSize").value || 4),
    seed: $("seed").value,
    steps: Number($("steps").value || 50),
    high_res: $("highRes").value,
    low_res: $("lowRes").value,
    cfg: Number($("cfg").value || 4),
    lora_name: $("loraName").value,
    lora_strength: Number($("loraStrength").value || 0.9),
    timeout: Number($("timeout").value || 900),
    dry_run: $("dryRun").checked,
    llm_provider: $("llmProvider").value,
    llm_model: $("llmModel").value,
    llm_endpoint: $("llmEndpoint").value
  };
}

function setStatus(text, percent = null) {
  $("statusText").textContent = text || "Ready";
  if (percent !== null) {
    $("jobProgress").value = percent;
    $("progressText").textContent = `${percent}%`;
  }
}

function setBusy(active) {
  for (const id of ["previewPrompt", "generate", "iterate", "downloadLlm", "unloadLlm", "downloadModels"]) {
    const element = $(id);
    if (element) element.disabled = active;
  }
}

function log(lines) {
  $("logBox").textContent = Array.isArray(lines) ? lines.join("\\n") : String(lines || "");
  $("logBox").scrollTop = $("logBox").scrollHeight;
}

function appendLog(message) {
  const box = $("logBox");
  box.textContent += `${message}\\n`;
  box.scrollTop = box.scrollHeight;
}

function applyDefaults(data) {
  $("comfyUrl").value = data.comfy_url;
  $("modelsRoot").value = data.models_root;
  $("outputDir").value = data.output_dir;
  $("description").value = data.description;
  $("batchSize").value = data.batch_size;
  $("seed").value = data.seed;
  $("steps").value = data.steps;
  $("highRes").value = data.high_res;
  $("lowRes").value = data.low_res;
  $("cfg").value = data.cfg;
  $("loraName").value = data.lora_name;
  $("loraStrength").value = data.lora_strength;
  $("timeout").value = data.timeout;
  $("dryRun").checked = data.dry_run;
  $("llmProvider").value = data.llm_provider;
  $("llmModel").value = data.llm_model;
  $("llmEndpoint").value = data.llm_endpoint;
}

async function validateComfy() {
  setStatus("Validating ComfyUI", 0);
  const result = await api("/api/validate-comfy", payload());
  if (result.status === "ready") {
    setStatus("ComfyUI ready", 100);
    appendLog("ComfyUI ready");
    return;
  }
  if (result.status === "missing_nodes") {
    alert(`Missing ComfyUI nodes:\\n${result.missing_nodes.join("\\n")}`);
    setStatus("Missing ComfyUI nodes", 0);
    return;
  }
  const missing = Object.values(result.missing_assets);
  appendLog(`Missing model files: ${missing.join(", ")}`);
  if (result.local_missing.length && confirm(`Download ${result.local_missing.length} missing model file(s)?`)) {
    setBusy(true);
    try {
      await api("/api/download-models", { ...payload(), filenames: missing });
    } catch (error) {
      setBusy(false);
      throw error;
    }
  }
}

async function validateLlm() {
  setStatus("Validating prompt model", 0);
  const result = await api("/api/validate-llm", payload());
  if (result.status === "ready") {
    setStatus("Prompt model ready", 100);
    appendLog(`Prompt model ready: ${result.result.model}`);
    return true;
  }
  if (result.status === "missing_model" && confirm(`Download prompt model ${result.result.model}?`)) {
    setBusy(true);
    try {
      await api("/api/download-llm", payload());
      setStatus("Downloading prompt model", 0);
      appendLog(`Downloading prompt model: ${result.result.model}`);
    } catch (error) {
      setBusy(false);
      throw error;
    }
    return false;
  }
  showPromptModelProblem(result);
  return false;
}

async function downloadPromptModel() {
  const result = await api("/api/validate-llm", payload());
  if (result.status === "ready") {
    setStatus("Prompt model ready", 100);
    alert(`Prompt model is already ready: ${result.result.model}`);
    return;
  }
  if (result.status === "missing_model") {
    if (confirm(`Download prompt model ${result.result.model}? This can be several GB.`)) {
      setBusy(true);
      try {
        await api("/api/download-llm", payload());
        setStatus("Downloading prompt model", 0);
        appendLog(`Downloading prompt model: ${result.result.model}`);
      } catch (error) {
        setBusy(false);
        throw error;
      }
    }
    return;
  }
  showPromptModelProblem(result);
}

async function unloadPromptModel() {
  const data = payload();
  if (data.llm_provider !== "ollama") return alert("Prompt model unload is available for Ollama providers.");
  if (!confirm(`Unload prompt model ${data.llm_model} from memory?`)) return;
  setBusy(true);
  setStatus("Unloading prompt model", 0);
  try {
    const result = await api("/api/unload-llm", data);
    handledJob = 0;
    appendLog(`job=${result.job_id}`);
  } catch (error) {
    setBusy(false);
    throw error;
  }
}

async function ensurePromptModelReady() {
  const data = payload();
  if (data.mode === "prompt" || data.llm_provider === "none" || data.llm_provider !== "ollama") return true;
  setBusy(true);
  setStatus("Validating prompt model", 0);
  const result = await api("/api/validate-llm", data);
  if (result.status === "ready") return true;
  if (result.status === "missing_model") {
    if (confirm(`Prompt model ${result.result.model} is missing. Download it now?`)) {
      try {
        await api("/api/download-llm", data);
        setStatus("Downloading prompt model", 0);
        appendLog(`Downloading prompt model: ${result.result.model}`);
        return false;
      } catch (error) {
        setBusy(false);
        throw error;
      }
    }
    setBusy(false);
    return false;
  }
  setBusy(false);
  showPromptModelProblem(result);
  return false;
}

function showPromptModelProblem(result) {
  const detail = result.result || {};
  if (result.status === "server_unavailable") {
    const endpoint = detail.endpoint || $("llmEndpoint").value;
    let message = `Ollama is not reachable at ${endpoint}.\\n\\n`;
    if (detail.cli_available) {
      message += "Ollama was found, but the local server could not be started automatically. Start Ollama manually, then click Validate Prompt Model again.";
    } else {
      message += "Ollama was not found on this machine. Install Ollama first, then click Validate Prompt Model again.\\n\\nhttps://ollama.com/download";
    }
    alert(message);
    appendLog(message);
    return;
  }
  alert(detail.message || result.message || "Prompt model is not ready.");
}

async function previewPrompt() {
  try {
    if (!(await ensurePromptModelReady())) return;
    setBusy(true);
    setStatus("Previewing prompt", 0);
    const result = await api("/api/preview-prompt", payload());
    handledJob = 0;
    appendLog(`job=${result.job_id}`);
  } catch (error) {
    setBusy(false);
    throw error;
  }
}

async function generate() {
  try {
    if (!(await ensurePromptModelReady())) return;
    setBusy(true);
    const result = await api("/api/generate", payload());
    handledJob = 0;
    appendLog(`job=${result.job_id}`);
  } catch (error) {
    setBusy(false);
    throw error;
  }
}

async function iterate() {
  if (!currentRun) return alert("Load a run first.");
  try {
    if (!(await ensurePromptModelReady())) return;
    setBusy(true);
    const body = {
      ...payload(),
      run_dir: currentRun.run_dir,
      selected_index: selectedIndex,
      feedback: $("feedback").value
    };
    const result = await api("/api/iterate", body);
    handledJob = 0;
    appendLog(`job=${result.job_id}`);
  } catch (error) {
    setBusy(false);
    throw error;
  }
}

async function loadLatest() {
  const response = await fetch(`/api/latest-run?${new URLSearchParams({ output_dir: $("outputDir").value })}`);
  const data = await response.json();
  if (data.error) throw new Error(data.error);
  if (!data.run) return alert("No runs found.");
  renderRun(data.run);
}

async function loadRunPath() {
  const response = await fetch(`/api/run?${new URLSearchParams({ path: $("runPath").value })}`);
  const data = await response.json();
  if (data.error) throw new Error(data.error);
  renderRun(data);
}

function renderRun(run) {
  currentRun = run;
  selectedIndex = run.candidates.length ? run.candidates[0].index : 0;
  $("runPath").value = run.run_dir;
  $("candidateList").innerHTML = "";
  for (const candidate of run.candidates) {
    const item = document.createElement("button");
    item.className = "candidate";
    item.dataset.index = candidate.index;
    item.innerHTML = `<strong>${candidate.index}: seed ${candidate.seed}</strong><small>${candidate.lowres_path || candidate.highres_path || "dry run"}</small>`;
    item.onclick = () => selectCandidate(candidate.index);
    $("candidateList").appendChild(item);
  }
  if (run.contact_sheet_url) $("previewImage").src = run.contact_sheet_url;
  selectCandidate(selectedIndex);
  appendLog(`loaded=${run.run_dir}`);
}

function selectCandidate(index) {
  if (!currentRun) return;
  selectedIndex = Number(index);
  document.querySelectorAll(".candidate").forEach((item) => item.classList.toggle("active", Number(item.dataset.index) === selectedIndex));
  const candidate = currentRun.candidates.find((item) => item.index === selectedIndex);
  if (!candidate) return;
  $("previewImage").src = candidate.lowres_path_url || candidate.highres_path_url || currentRun.contact_sheet_url || "";
  $("lowLink").href = candidate.lowres_path_url || "#";
  $("highLink").href = candidate.highres_path_url || "#";
  $("apiLink").href = candidate.api_prompt_path_url || "#";
  $("promptPreview").value = `${candidate.positive_prompt}\\n\\nNegative:\\n${candidate.negative_prompt}`;
}

async function openRun() {
  if (!currentRun) return;
  await api("/api/open-path", { path: currentRun.run_dir });
}

async function pollJob() {
  try {
    const job = await api("/api/job");
    setStatus(job.label, job.percent);
    if (job.logs && job.logs.length) log(job.logs);
    if (job.prompt) $("promptPreview").value = job.prompt;
    if (!job.active && job.id !== handledJob) {
      handledJob = job.id;
      setBusy(false);
      if (job.error) appendLog(`error=${job.error}`);
      if (job.result && job.result.kind === "prompt") {
        $("promptPreview").value = `${job.result.positive_prompt}\\n\\nNegative:\\n${job.result.negative_prompt}`;
        appendLog(job.result.notes || `Prompt source: ${job.result.source}`);
      }
      if (job.result && job.result.kind === "run") renderRun(job.result.run);
      if (job.result && job.result.kind === "llm_download") appendLog(`prompt_model=${job.result.model}`);
      if (job.result && job.result.kind === "llm_unload") appendLog(`unloaded_prompt_model=${job.result.model}`);
      if (job.result && job.result.kind === "model_download") appendLog(`downloaded=${job.result.paths.join(", ")}`);
    }
  } catch (error) {
    setStatus(error.message, 0);
  }
}

function bind() {
  $("validateComfy").onclick = () => validateComfy().catch((error) => alert(error.message));
  $("downloadModels").onclick = () => {
    if (confirm("Download missing default ComfyUI model files? These files can be large.")) {
      setBusy(true);
      api("/api/download-models", payload()).catch((error) => {
        setBusy(false);
        alert(error.message);
      });
    }
  };
  $("validateLlm").onclick = () => validateLlm().catch((error) => alert(error.message));
  $("downloadLlm").onclick = () => downloadPromptModel().catch((error) => alert(error.message));
  $("unloadLlm").onclick = () => unloadPromptModel().catch((error) => alert(error.message));
  $("previewPrompt").onclick = () => previewPrompt().catch((error) => alert(error.message));
  $("generate").onclick = () => generate().catch((error) => alert(error.message));
  $("iterate").onclick = () => iterate().catch((error) => alert(error.message));
  $("latestRun").onclick = () => loadLatest().catch((error) => alert(error.message));
  $("loadRun").onclick = () => loadRunPath().catch((error) => alert(error.message));
  $("openRun").onclick = () => openRun().catch((error) => alert(error.message));
}

async function start() {
  applyDefaults(await api("/api/defaults"));
  bind();
  setInterval(pollJob, 800);
}

start().catch((error) => alert(error.message));
"""


if __name__ == "__main__":
    raise SystemExit(main())
