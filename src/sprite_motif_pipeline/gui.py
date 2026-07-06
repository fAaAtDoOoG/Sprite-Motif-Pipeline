from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from PIL import Image, ImageTk

from .comfy import ComfyClient, validate_model_assets, validate_required_nodes
from .config import DEFAULT_HIGH_RES, DEFAULT_LOW_RES, DEFAULTS, format_size, parse_size
from .model_assets import assets_for_filenames, default_models_root, download_assets, missing_local_assets
from .ollama import DEFAULT_OLLAMA_ENDPOINT, OllamaValidation, pull_ollama_model, validate_ollama_model
from .progress import generation_percent, percent_from_message, short_status
from .prompting import LLMConfig, compose_prompt
from .runner import GenerationOptions, generate_batch
from .session import Candidate, RunManifest, load_manifest, save_manifest
from .workflow import required_node_types


def main() -> None:
    root = tk.Tk()
    SpritePipeApp(root)
    root.mainloop()


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget, padding: int = 0):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas, padding=padding)
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def bind_mousewheel_to_descendants(self) -> None:
        self._bind_tree(self.content)

    def _bind_tree(self, widget: tk.Widget) -> None:
        widget.bind("<Enter>", self._bind_mousewheel, add="+")
        widget.bind("<Leave>", self._unbind_mousewheel, add="+")
        for child in widget.winfo_children():
            self._bind_tree(child)

    def _on_content_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _bind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta:
            self.canvas.yview_scroll(int(-1 * (delta / 120)), "units")
            return
        if getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(3, "units")


class SpritePipeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sprite Motif Pipeline")
        self.root.geometry("1180x780")
        self.root.minsize(760, 520)

        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.current_run_dir: Path | None = None
        self.current_manifest: RunManifest | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.worker_active = False

        self._init_vars()
        self._build_ui()
        self._poll_events()

    def _init_vars(self) -> None:
        self.comfy_url_var = tk.StringVar(value="http://127.0.0.1:8188")
        self.models_root_var = tk.StringVar(value=str(default_models_root()))
        self.output_dir_var = tk.StringVar(value="runs")
        self.mode_var = tk.StringVar(value="description")
        self.batch_var = tk.IntVar(value=4)
        self.high_res_var = tk.StringVar(value=format_size(DEFAULT_HIGH_RES))
        self.low_res_var = tk.StringVar(value=format_size(DEFAULT_LOW_RES))
        self.seed_var = tk.StringVar(value="")
        self.steps_var = tk.IntVar(value=DEFAULTS.steps)
        self.cfg_var = tk.DoubleVar(value=DEFAULTS.cfg)
        self.lora_name_var = tk.StringVar(value=DEFAULTS.pixel_lora)
        self.lora_strength_var = tk.DoubleVar(value=DEFAULTS.pixel_lora_strength)
        self.timeout_var = tk.IntVar(value=900)
        self.dry_run_var = tk.BooleanVar(value=False)
        self.llm_provider_var = tk.StringVar(value="ollama")
        self.llm_model_var = tk.StringVar(value="qwen2.5:7b-instruct")
        self.llm_endpoint_var = tk.StringVar(value=DEFAULT_OLLAMA_ENDPOINT)
        self.status_var = tk.StringVar(value="Ready")

    def _build_ui(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.grid(row=0, column=0, sticky="nsew")

        left = ScrollableFrame(pane, padding=10)
        right = ScrollableFrame(pane, padding=10)
        pane.add(left, weight=3)
        pane.add(right, weight=4)

        self._build_left(left.content)
        self._build_right(right.content)
        left.bind_mousewheel_to_descendants()
        right.bind_mousewheel_to_descendants()
        self._build_log()

    def _build_left(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        backend = ttk.LabelFrame(parent, text="Backend", padding=8)
        backend.grid(row=0, column=0, sticky="ew")
        backend.columnconfigure(1, weight=1)
        ttk.Label(backend, text="ComfyUI").grid(row=0, column=0, sticky="w")
        ttk.Entry(backend, textvariable=self.comfy_url_var).grid(row=0, column=1, sticky="ew", padx=6)
        self.comfy_validate_button = ttk.Button(backend, text="Validate", command=self.validate_comfy)
        self.comfy_validate_button.grid(row=0, column=2)
        ttk.Label(backend, text="Models").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(backend, textvariable=self.models_root_var).grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        model_buttons = ttk.Frame(backend)
        model_buttons.grid(row=1, column=2, sticky="ew", pady=(6, 0))
        ttk.Button(model_buttons, text="Browse", command=self.browse_models_root).pack(side=tk.LEFT)
        self.model_download_button = ttk.Button(model_buttons, text="Download Missing", command=self.download_missing_models)
        self.model_download_button.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(backend, text="Output").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(backend, textvariable=self.output_dir_var).grid(row=2, column=1, sticky="ew", padx=6, pady=(6, 0))
        ttk.Button(backend, text="Browse", command=self.browse_output_dir).grid(row=2, column=2, pady=(6, 0))
        ttk.Checkbutton(backend, text="Dry run", variable=self.dry_run_var).grid(row=3, column=1, sticky="w", pady=(6, 0))

        source = ttk.LabelFrame(parent, text="Input", padding=8)
        source.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        source.columnconfigure(0, weight=1)
        mode_row = ttk.Frame(source)
        mode_row.grid(row=0, column=0, sticky="ew")
        ttk.Radiobutton(mode_row, text="Description", value="description", variable=self.mode_var).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Direct prompt", value="prompt", variable=self.mode_var).pack(side=tk.LEFT, padx=(12, 0))
        self.input_text = tk.Text(source, height=5, wrap=tk.WORD, undo=True)
        self.input_text.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.input_text.insert("1.0", "红发女骑士，轻甲，性格勇敢")

        generation = ttk.LabelFrame(parent, text="Generation", padding=8)
        generation.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        for column in (1, 3):
            generation.columnconfigure(column, weight=1)
        self._field(generation, 0, 0, "Batch", ttk.Spinbox(generation, from_=1, to=32, textvariable=self.batch_var, width=8))
        self._field(generation, 0, 2, "Seed", ttk.Entry(generation, textvariable=self.seed_var, width=12))
        self._field(generation, 1, 0, "High res", ttk.Entry(generation, textvariable=self.high_res_var, width=12))
        self._field(generation, 1, 2, "Low res", ttk.Entry(generation, textvariable=self.low_res_var, width=12))
        self._field(generation, 2, 0, "Steps", ttk.Spinbox(generation, from_=1, to=100, textvariable=self.steps_var, width=8))
        self._field(generation, 2, 2, "CFG", ttk.Entry(generation, textvariable=self.cfg_var, width=12))
        self._field(generation, 3, 0, "LoRA", ttk.Entry(generation, textvariable=self.lora_name_var, width=32), columnspan=3)
        self._field(generation, 4, 0, "Strength", ttk.Entry(generation, textvariable=self.lora_strength_var, width=12))
        self._field(generation, 4, 2, "Timeout", ttk.Spinbox(generation, from_=30, to=7200, textvariable=self.timeout_var, width=10))

        llm = ttk.LabelFrame(parent, text="Prompt model", padding=8)
        llm.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        llm.columnconfigure(1, weight=1)
        ttk.Label(llm, text="Provider").grid(row=0, column=0, sticky="w")
        ttk.Combobox(llm, textvariable=self.llm_provider_var, values=["none", "openai-compatible", "openai", "ollama"], state="readonly", width=18).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(llm, text="Model").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(llm, textvariable=self.llm_model_var).grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        ttk.Label(llm, text="Endpoint").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(llm, textvariable=self.llm_endpoint_var).grid(row=2, column=1, sticky="ew", padx=6, pady=(6, 0))
        llm_buttons = ttk.Frame(llm)
        llm_buttons.grid(row=3, column=1, sticky="ew", padx=6, pady=(8, 0))
        llm_buttons.columnconfigure((0, 1), weight=1)
        self.llm_validate_button = ttk.Button(llm_buttons, text="Validate", command=self.validate_prompt_model)
        self.llm_validate_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.llm_download_button = ttk.Button(llm_buttons, text="Download Model", command=self.download_prompt_model)
        self.llm_download_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        actions = ttk.Frame(parent)
        actions.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self.preview_button = ttk.Button(actions, text="Preview Prompt", command=self.preview_prompt)
        self.preview_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.generate_button = ttk.Button(actions, text="Generate", command=self.generate)
        self.generate_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        preview = ttk.LabelFrame(parent, text="Prompt", padding=8)
        preview.grid(row=5, column=0, sticky="nsew", pady=(10, 0))
        parent.rowconfigure(5, weight=1)
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)
        self.prompt_text = tk.Text(preview, height=8, wrap=tk.WORD)
        self.prompt_text.grid(row=0, column=0, sticky="nsew")

    def _build_right(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        run_frame = ttk.LabelFrame(parent, text="Run", padding=8)
        run_frame.grid(row=0, column=0, sticky="ew")
        run_frame.columnconfigure(1, weight=1)
        ttk.Button(run_frame, text="Latest", command=self.load_latest_run).grid(row=0, column=0, sticky="w")
        self.run_entry_var = tk.StringVar(value="")
        ttk.Entry(run_frame, textvariable=self.run_entry_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(run_frame, text="Browse", command=self.browse_run).grid(row=0, column=2)
        ttk.Button(run_frame, text="Open", command=self.open_current_run).grid(row=0, column=3, padx=(6, 0))

        results = ttk.Frame(parent)
        results.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        results.columnconfigure(1, weight=1)
        results.rowconfigure(0, weight=1)

        list_frame = ttk.LabelFrame(results, text="Candidates", padding=8)
        list_frame.grid(row=0, column=0, sticky="ns")
        self.candidate_list = tk.Listbox(list_frame, width=34, exportselection=False)
        self.candidate_list.pack(fill=tk.BOTH, expand=True)
        self.candidate_list.bind("<<ListboxSelect>>", lambda _event: self.show_selected_candidate())

        image_frame = ttk.LabelFrame(results, text="Preview", padding=8)
        image_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        self.image_label = ttk.Label(image_frame, anchor=tk.CENTER)
        self.image_label.grid(row=0, column=0, sticky="nsew")
        file_buttons = ttk.Frame(image_frame)
        file_buttons.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        file_buttons.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(file_buttons, text="Lowres", command=lambda: self.open_candidate_path("low")).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(file_buttons, text="Highres", command=lambda: self.open_candidate_path("high")).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(file_buttons, text="API JSON", command=lambda: self.open_candidate_path("api")).grid(row=0, column=2, sticky="ew", padx=(5, 0))

        iterate = ttk.LabelFrame(parent, text="Iteration", padding=8)
        iterate.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        iterate.columnconfigure(0, weight=1)
        self.feedback_text = tk.Text(iterate, height=4, wrap=tk.WORD)
        self.feedback_text.grid(row=0, column=0, sticky="ew")
        self.feedback_text.insert("1.0", "盔甲更轻，头发更短，轮廓更圆润")
        self.iterate_button = ttk.Button(iterate, text="Iterate Selected", command=self.iterate_selected)
        self.iterate_button.grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def _build_log(self) -> None:
        footer = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        footer.grid(row=1, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=160)
        self.progress.grid(row=0, column=1, sticky="e")
        log_frame = ttk.LabelFrame(footer, text="Log", padding=6)
        log_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=6, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky="ew")

    def _field(self, parent: ttk.Frame, row: int, col: int, label: str, widget: tk.Widget, columnspan: int = 1) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", pady=3)
        widget.grid(row=row, column=col + 1, columnspan=columnspan, sticky="ew", padx=(6, 10), pady=3)

    def validate_comfy(self) -> None:
        url = self.comfy_url_var.get().strip()
        models_root = Path(self.models_root_var.get().strip() or default_models_root())

        def work() -> dict[str, Any]:
            client = ComfyClient(url)
            missing_nodes = validate_required_nodes(client, required_node_types())
            if missing_nodes:
                return {"status": "missing_nodes", "missing_nodes": missing_nodes}
            missing_assets = validate_model_assets(client)
            if missing_assets:
                local_assets = assets_for_filenames(missing_assets.values())
                local_missing = missing_local_assets(models_root, local_assets)
                return {
                    "status": "missing_assets",
                    "missing_assets": missing_assets,
                    "local_missing": local_missing,
                    "models_root": models_root,
                }
            return {"status": "ready"}

        self._run_worker("Validating", work, self._handle_validation_result)

    def _handle_validation_result(self, result: dict[str, Any]) -> None:
        status = result["status"]
        if status == "missing_nodes":
            lines = ["Missing ComfyUI nodes:"]
            lines.extend(f"- {node}" for node in result["missing_nodes"])
            messagebox.showerror("ComfyUI", "\n".join(lines))
            return
        if status == "ready":
            messagebox.showinfo("ComfyUI", "ComfyUI is ready.")
            return

        missing_assets = result["missing_assets"]
        local_missing = result["local_missing"]
        models_root = result["models_root"]
        lines = ["Core nodes are available, but ComfyUI does not see these model files:"]
        lines.extend(f"- {label}: {name}" for label, name in missing_assets.items())
        if not local_missing:
            lines.append("")
            lines.append("The files exist in the selected model folder. Refresh or restart ComfyUI, then validate again.")
            messagebox.showinfo("ComfyUI", "\n".join(lines))
            return

        lines.append("")
        lines.append(f"Download missing files to:\n{models_root}")
        lines.append("")
        lines.append("These downloads can be large.")
        lines.append("")
        lines.extend(f"- {asset.subdir}/{asset.filename}" for asset in local_missing)
        should_download = messagebox.askyesno("Download models?", "\n".join(lines))
        if should_download:
            self._download_assets(local_missing, models_root)

    def preview_prompt(self) -> None:
        try:
            spec = self._compose_from_ui()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Prompt", str(exc))
            return
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert(tk.END, spec.positive_prompt)
        self.prompt_text.insert(tk.END, "\n\nNegative:\n")
        self.prompt_text.insert(tk.END, spec.negative_prompt)
        if spec.notes:
            self._log(spec.notes)

    def generate(self) -> None:
        try:
            mode, text, llm_config, options = self._collect_payload()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Generate", str(exc))
            return

        def work() -> Path:
            spec = compose_prompt(
                text if mode == "description" else None,
                direct_prompt=text if mode == "prompt" else None,
                force_pixel_trigger=True,
                llm_config=llm_config,
            )
            self.events.put(("prompt", spec.positive_prompt))
            return generate_batch(spec, description=text, options=options, progress=lambda message: self._queue_generation_progress(message, options.batch_size))

        self._run_worker("Generating", work, self.load_run, determinate=True)

    def iterate_selected(self) -> None:
        if self.current_run_dir is None or self.current_manifest is None:
            messagebox.showerror("Iteration", "Load a run first.")
            return
        candidate = self._selected_candidate()
        if candidate is None:
            messagebox.showerror("Iteration", "Select a candidate.")
            return
        feedback = self.feedback_text.get("1.0", tk.END).strip()
        if not feedback:
            messagebox.showerror("Iteration", "Feedback is empty.")
            return

        try:
            _mode, _text, llm_config, options = self._collect_payload(require_input=False)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Iteration", str(exc))
            return

        previous_run = self.current_run_dir
        previous_manifest = self.current_manifest

        def work() -> Path:
            spec = compose_prompt(
                previous_manifest.description,
                feedback=feedback,
                previous_prompt=candidate.positive_prompt,
                llm_config=llm_config,
            )
            self.events.put(("prompt", spec.positive_prompt))
            run_dir = generate_batch(
                spec,
                description=previous_manifest.description,
                options=options,
                parent_run=str(previous_run),
                selected_index=candidate.index,
                feedback=feedback,
                progress=lambda message: self._queue_generation_progress(message, options.batch_size),
            )
            previous_manifest.selected_index = candidate.index
            previous_manifest.feedback = feedback
            save_manifest(previous_run, previous_manifest)
            return run_dir

        self._run_worker("Iterating", work, self.load_run, determinate=True)

    def browse_output_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=str(Path(self.output_dir_var.get() or ".").resolve()))
        if path:
            self.output_dir_var.set(path)

    def browse_models_root(self) -> None:
        path = filedialog.askdirectory(initialdir=str(Path(self.models_root_var.get() or default_models_root()).resolve()))
        if path:
            self.models_root_var.set(path)

    def download_missing_models(self) -> None:
        models_root = Path(self.models_root_var.get().strip() or default_models_root())
        assets = missing_local_assets(models_root)
        if not assets:
            messagebox.showinfo("Models", "All default model files already exist in the selected folder.")
            return
        lines = [f"Download missing files to:\n{models_root}", "", "These downloads can be large.", ""]
        lines.extend(f"- {asset.subdir}/{asset.filename}" for asset in assets)
        if messagebox.askyesno("Download models?", "\n".join(lines)):
            self._download_assets(assets, models_root)

    def validate_prompt_model(self) -> None:
        config = self._llm_config_from_ui()
        if config.provider != "ollama":
            messagebox.showinfo("Prompt model", "Automatic local model validation is available for Ollama providers.")
            return
        if not config.model.strip():
            messagebox.showerror("Prompt model", "Ollama model name is empty.")
            return

        def work() -> OllamaValidation:
            return validate_ollama_model(
                config.endpoint,
                config.model,
                progress=lambda message: self.events.put(("log", message)),
            )

        self._run_worker("Validating prompt model", work, self._handle_prompt_model_validation)

    def _handle_prompt_model_validation(self, result: OllamaValidation) -> None:
        if not result.server_available:
            lines = [f"Ollama is not reachable at:\n{result.endpoint}", ""]
            if result.cli_available:
                lines.append("The Ollama executable was found, but the server could not be started automatically.")
            else:
                lines.append("The Ollama executable was not found. Install Ollama first, then click Validate again.")
                lines.append("Download page: https://ollama.com/download")
            messagebox.showerror("Prompt model", "\n".join(lines))
            return

        if result.model_present:
            messagebox.showinfo("Prompt model", f"Ollama is ready.\n\nVersion: {result.version}\nModel: {result.model}")
            return

        lines = [
            f"Ollama is running at:\n{result.endpoint}",
            "",
            f"Missing prompt model:\n{result.model}",
            "",
            "Download it now with Ollama?",
        ]
        if messagebox.askyesno("Download prompt model?", "\n".join(lines)):
            self._download_prompt_model(self._llm_config_from_ui())

    def download_prompt_model(self) -> None:
        config = self._llm_config_from_ui()
        if config.provider != "ollama":
            messagebox.showinfo("Prompt model", "Automatic local model download is available for Ollama providers.")
            return
        if not config.model.strip():
            messagebox.showerror("Prompt model", "Ollama model name is empty.")
            return
        lines = [
            f"Download Ollama model:\n{config.model}",
            "",
            f"Endpoint:\n{config.endpoint or DEFAULT_OLLAMA_ENDPOINT}",
            "",
            "This can be several GB.",
        ]
        if messagebox.askyesno("Download prompt model?", "\n".join(lines)):
            self._download_prompt_model(config)

    def _download_prompt_model(self, config: LLMConfig) -> None:
        def work() -> str:
            return pull_ollama_model(
                config.endpoint,
                config.model,
                progress=self._queue_download_progress,
            )

        def done(model: str) -> None:
            messagebox.showinfo("Prompt model", f"Ollama model is ready:\n{model}")

        self._run_worker("Downloading prompt model", work, done, determinate=True)

    def _download_assets(self, assets, models_root: Path) -> None:
        def work() -> list[Path]:
            return download_assets(models_root, assets, progress=self._queue_download_progress)

        def done(paths: list[Path]) -> None:
            messagebox.showinfo(
                "Models",
                "Download complete. Refresh or restart ComfyUI if Validate still cannot see the files.\n\n"
                + "\n".join(str(path) for path in paths),
            )

        self._run_worker("Downloading models", work, done, determinate=True)

    def browse_run(self) -> None:
        path = filedialog.askdirectory(initialdir=str(Path(self.output_dir_var.get() or ".").resolve()))
        if path:
            self.load_run(Path(path))

    def load_latest_run(self) -> None:
        base = Path(self.output_dir_var.get() or "runs")
        runs = sorted((path for path in base.glob("run_*") if (path / "manifest.json").exists()), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            messagebox.showinfo("Run", "No runs found.")
            return
        self.load_run(runs[0])

    def load_run(self, run_dir: Path) -> None:
        run_dir = Path(run_dir)
        manifest = load_manifest(run_dir)
        self.current_run_dir = run_dir
        self.current_manifest = manifest
        self.run_entry_var.set(str(run_dir))
        self.candidate_list.delete(0, tk.END)
        for candidate in manifest.candidates:
            suffix = "ready" if candidate.lowres_path or candidate.highres_path else "dry"
            self.candidate_list.insert(tk.END, f"{candidate.index}: seed {candidate.seed} [{suffix}]")
        if manifest.candidates:
            self.candidate_list.selection_set(0)
            self.show_selected_candidate()
        contact_sheet = run_dir / "contact_sheet.png"
        if contact_sheet.exists():
            self._show_image(contact_sheet)
        self.status_var.set(f"Loaded {run_dir.name}")
        self._log(f"loaded={run_dir}")

    def show_selected_candidate(self) -> None:
        candidate = self._selected_candidate()
        if candidate is None:
            return
        path = self._candidate_preview_path(candidate)
        if path is None:
            self.image_label.configure(text="No image", image="")
            self.preview_photo = None
            return
        self._show_image(path)

    def open_current_run(self) -> None:
        if self.current_run_dir is None:
            return
        self._open_path(self.current_run_dir)

    def open_candidate_path(self, kind: str) -> None:
        candidate = self._selected_candidate()
        if candidate is None:
            return
        path_value = {
            "low": candidate.lowres_path,
            "high": candidate.highres_path,
            "api": candidate.api_prompt_path,
        }.get(kind, "")
        if not path_value:
            return
        self._open_path(Path(path_value))

    def _collect_payload(self, require_input: bool = True) -> tuple[str, str, LLMConfig, GenerationOptions]:
        mode = self.mode_var.get()
        text = self.input_text.get("1.0", tk.END).strip()
        if require_input and not text:
            raise ValueError("Input is empty.")

        seed_raw = self.seed_var.get().strip()
        seed = int(seed_raw) if seed_raw else None
        parse_size(self.high_res_var.get(), DEFAULT_HIGH_RES)
        parse_size(self.low_res_var.get(), DEFAULT_LOW_RES)
        options = GenerationOptions(
            batch_size=int(self.batch_var.get()),
            high_res=self.high_res_var.get().strip(),
            low_res=self.low_res_var.get().strip(),
            seed=seed,
            steps=int(self.steps_var.get()),
            cfg=float(self.cfg_var.get()),
            lora_name=self.lora_name_var.get().strip(),
            lora_strength=float(self.lora_strength_var.get()),
            comfy_url=self.comfy_url_var.get().strip(),
            timeout=int(self.timeout_var.get()),
            output_dir=Path(self.output_dir_var.get().strip() or "runs"),
            dry_run=bool(self.dry_run_var.get()),
        )
        llm_config = self._llm_config_from_ui()
        return mode, text, llm_config, options

    def _llm_config_from_ui(self) -> LLMConfig:
        env_config = LLMConfig.from_env()
        return LLMConfig(
            provider=(self.llm_provider_var.get().strip() or env_config.provider).lower(),
            model=self.llm_model_var.get().strip() or env_config.model,
            endpoint=self.llm_endpoint_var.get().strip() or env_config.endpoint or DEFAULT_OLLAMA_ENDPOINT,
            api_key=env_config.api_key,
            temperature=env_config.temperature,
            timeout_s=env_config.timeout_s,
            keep_alive=env_config.keep_alive,
        )

    def _compose_from_ui(self):
        mode, text, llm_config, _options = self._collect_payload()
        return compose_prompt(
            text if mode == "description" else None,
            direct_prompt=text if mode == "prompt" else None,
            force_pixel_trigger=True,
            llm_config=llm_config,
        )

    def _run_worker(self, label: str, work, on_done, *, determinate: bool = False) -> None:
        if self.worker_active:
            return
        self.worker_active = True
        self._set_buttons(False)
        self.status_var.set(label)
        if determinate:
            self.progress.configure(mode="determinate", maximum=100, value=0)
            self.events.put(("progress", (0, label)))
        else:
            self.progress.configure(mode="indeterminate")
            self.progress.start(10)

        def target() -> None:
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001
                self.events.put(("error", exc))
            else:
                self.events.put(("done", (result, on_done)))

        threading.Thread(target=target, daemon=True).start()

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._log(str(payload))
            elif kind == "prompt":
                self.prompt_text.delete("1.0", tk.END)
                self.prompt_text.insert(tk.END, str(payload))
            elif kind == "progress":
                percent, label = payload
                if percent is not None:
                    self.progress.configure(mode="determinate", maximum=100, value=max(0, min(100, float(percent))))
                if label:
                    self.status_var.set(str(label))
            elif kind == "error":
                self._finish_worker()
                messagebox.showerror("Sprite Motif Pipeline", str(payload))
                self._log(f"error={payload}")
            elif kind == "done":
                result, on_done = payload
                self._finish_worker()
                on_done(result)
        self.root.after(120, self._poll_events)

    def _finish_worker(self) -> None:
        self.worker_active = False
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=100, value=0)
        self.status_var.set("Ready")
        self._set_buttons(True)

    def _set_buttons(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in (
            self.comfy_validate_button,
            self.model_download_button,
            self.llm_validate_button,
            self.llm_download_button,
            self.preview_button,
            self.generate_button,
            self.iterate_button,
        ):
            button.configure(state=state)

    def _selected_candidate(self) -> Candidate | None:
        if self.current_manifest is None:
            return None
        selection = self.candidate_list.curselection()
        if not selection:
            return None
        index = selection[0]
        if index >= len(self.current_manifest.candidates):
            return None
        return self.current_manifest.candidates[index]

    def _candidate_preview_path(self, candidate: Candidate) -> Path | None:
        for value in (candidate.lowres_path, candidate.highres_path):
            if value:
                path = Path(value)
                if path.exists():
                    return path
        return None

    def _show_image(self, path: Path) -> None:
        with Image.open(path) as image:
            image = image.convert("RGBA")
            image.thumbnail((460, 460), Image.Resampling.NEAREST)
            self.preview_photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.preview_photo, text="")
        self._log(f"preview={path}")

    def _open_path(self, path: Path) -> None:
        path = path.resolve()
        if not path.exists():
            messagebox.showerror("Open", f"Path not found:\n{path}")
            return
        if hasattr(os, "startfile"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif os.name == "posix":
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(path)])
        else:
            webbrowser.open(path.as_uri())

    def _log(self, message: str) -> None:
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def _queue_download_progress(self, message: str) -> None:
        self.events.put(("log", message))
        percent = percent_from_message(message)
        label = short_status(message)
        self.events.put(("progress", (percent, label)))

    def _queue_generation_progress(self, message: str, batch_size: int) -> None:
        self.events.put(("log", message))
        percent = generation_percent(message, batch_size)
        label = short_status(message)
        self.events.put(("progress", (percent, label)))


if __name__ == "__main__":
    main()
