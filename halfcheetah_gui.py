from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any

import numpy as np


def _load_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{filename} konnte nicht geladen werden.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = _load_module("workbench_base.py", "halfcheetah_workbench_base")
logic = _load_module("halfcheetah_logic.py", "halfcheetah_logic")


class HalfCheetahUI(base.WorkbenchUI):
    def __init__(self) -> None:
        super().__init__()
        self.title("Stable-Baselines3 Workbench - HalfCheetah-v5")
        self.geometry("1280x780")
        self.minsize(1080, 660)
        self.trainer = logic.HalfCheetahTrainer()
        self.configs = {method: logic.OnPolicyConfig(**logic.get_default_parameters_for_method(method)) for method in logic.SUPPORTED_METHODS}
        self.variables: dict[str, dict[str, tk.Variable]] = {method: {} for method in logic.SUPPORTED_METHODS}
        self.env_var = tk.StringVar(value="HalfCheetah-v5")
        self.method_var = tk.StringVar(value="PPO")
        self.selected_method = tk.StringVar(value="PPO")
        self.reward_mode_var = tk.StringVar(value="standard")
        self.total_timesteps_var = tk.StringVar(value="1000000")
        self.training_stop_mode_var = tk.StringVar(value="timesteps")
        self.target_episodes_var = tk.StringVar(value="5000")
        self.seed_var = tk.StringVar(value="123")
        self.evaluation_episodes_var = tk.StringVar(value="20")
        self.reward_history: dict[str, list[float]] = {method: [] for method in logic.SUPPORTED_METHODS}
        self.compare_vars = {method: tk.BooleanVar(value=True) for method in logic.SUPPORTED_METHODS}
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.demo_stop_event = threading.Event()
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.training_active = False
        self.training_paused = False
        self.demo_active = False
        self.compare_active = False
        self.running_workers = 0
        self._pending_animation_frame: np.ndarray | None = None
        self._animation_redraw_scheduled = False
        self._animation_frame_counter = 0
        self._animation_last_draw_time = 0.0
        self._animation_fps = 0.0
        self._comparison_placeholder_ready = False
        self._method_frames: dict[str, np.ndarray | None] = {method: None for method in logic.SUPPORTED_METHODS}
        self.build_workbench_layout(self._build_controls, self._build_actions)
        self.after(80, self._poll_events)

    def build_workbench_layout(self, build_controls, build_actions) -> None:
        self.title("Stable-Baselines3 Workbench - HalfCheetah-v5")
        self.geometry("1280x780")
        self.minsize(1080, 660)
        self._configure_lunarlander_style()
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=10, pady=(10, 6))
        ttk.Label(header, text="Stable-Baselines3 Workbench", style="Title.TLabel").pack(anchor="w", padx=(6, 0))
        ttk.Label(header, text="HalfCheetah-v5", style="Accent.TLabel").pack(anchor="w", padx=(6, 0), pady=(2, 0))
        main = ttk.PanedWindow(self, orient=tk.VERTICAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        top = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        main.add(top, weight=2)
        controls = self._scrollable(top)
        top.add(controls, weight=1)
        build_controls(controls.body)
        render_frame = ttk.LabelFrame(top, text="Animation", padding=6)
        top.add(render_frame, weight=1)
        self._build_render(render_frame)
        bottom = ttk.Frame(main)
        main.add(bottom, weight=4)
        self._build_lunarlander_bottom(bottom, build_actions)

    def _build_render(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent)
        header.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(header, text="Live-Animation", style="Section.TLabel").pack(side=tk.LEFT)
        self.animation_window_button = ttk.Button(header, text="Vollbild", command=self.toggle_animation_window, style="Secondary.TButton")
        self.animation_window_button.pack(side=tk.RIGHT, padx=(0, 6))
        self.fit_animation_button = ttk.Button(header, text="Anfenster", command=self.toggle_animation_fit, style="Secondary.TButton")
        self.fit_animation_button.pack(side=tk.RIGHT, padx=(0, 6))
        self.animation_button = ttk.Button(header, text="Animation EIN", command=self.toggle_training_animation, style="Secondary.TButton")
        self.animation_button.pack(side=tk.RIGHT)
        ttk.Label(parent, textvariable=self.animation_state_var, style="Status.TLabel").pack(anchor="w", padx=4, pady=(0, 4))
        self.render_figure = __import__('matplotlib').figure.Figure(figsize=(6.8, 3.9), dpi=120, facecolor=self.colors["panel"])
        self.render_axes = self.render_figure.add_subplot(111)
        self.render_figure.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04)
        self.render_axes.set_facecolor("#0b1220")
        self.render_image = None
        self._show_animation_placeholder("Training starten, um die Live-Animation zu sehen.")
        self.render_canvas = __import__('matplotlib.backends.backend_tkagg').backends.backend_tkagg.FigureCanvasTkAgg(self.render_figure, master=parent)
        self.render_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

    def _show_animation_placeholder(self, message: str) -> None:
        self.render_axes.clear()
        self.render_axes.set_facecolor("#0b1220")
        self.render_axes.set_axis_off()
        self.render_axes.set_title("Live-Animation", fontsize=12, pad=10, color="#e8f3ff")
        self.render_axes.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="#9db1c8", wrap=True)

    def toggle_training_animation(self) -> None:
        self.training_animation_visible = not self.training_animation_visible
        self.animation_button.configure(text="Animation EIN" if self.training_animation_visible else "Animation AUS")
        status = "EIN" if self.training_animation_visible else "AUS"
        if self.training_animation_visible:
            self.animation_state_var.set("Warte auf Trainingsframe")
            message = "Training starten, um die Live-Animation zu sehen."
        else:
            self.animation_state_var.set("Animation ausgeschaltet")
            message = "Animation ausgeschaltet"
        self.render_image = None
        self._show_animation_placeholder(message)
        self.render_canvas.draw_idle()
        self.status_var.set(f"Animation ist {status}")

    def toggle_animation_fit(self) -> None:
        self.animation_fit_mode = not self.animation_fit_mode
        self.fit_animation_button.configure(text="Anfenster" if self.animation_fit_mode else "Original")
        if self.render_image is not None:
            self.render_axes.set_aspect("auto" if self.animation_fit_mode else "equal")
            self.render_axes.set_position([0.04, 0.08, 0.92, 0.78] if self.animation_fit_mode else [0.12, 0.12, 0.76, 0.7])
            self.render_canvas.draw_idle()

    def toggle_animation_window(self) -> None:
        if self.animation_window is None or not self.animation_window.winfo_exists():
            self._open_animation_window()
        else:
            self._close_animation_window()

    def _open_animation_window(self) -> None:
        window = tk.Toplevel(self)
        window.title("Animation - Vollbild")
        window.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")
        try:
            window.state("zoomed")
        except tk.TclError:
            pass
        host = ttk.Frame(window)
        host.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        figure = __import__('matplotlib').figure.Figure(figsize=(14, 8), dpi=100, facecolor=self.colors["panel"])
        axes = figure.add_subplot(111)
        axes.set_facecolor(self.colors["background"])
        axes.set_axis_off()
        axes.set_title("HalfCheetah-Animation (Vollbild)", fontsize=14, pad=10, color=self.colors["text"])
        figure.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04)
        canvas = __import__('matplotlib.backends.backend_tkagg').backends.backend_tkagg.FigureCanvasTkAgg(figure, master=host)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.animation_window = window
        self.animation_window_figure = figure
        self.animation_window_axes = axes
        self.animation_window_canvas = canvas
        self.animation_window_button.configure(text="Fenster schließen")
        self.status_var.set("Animation-Fenster geöffnet")
        self._sync_animation_window_frame()
        window.protocol("WM_DELETE_WINDOW", self._close_animation_window)

    def _close_animation_window(self) -> None:
        if self.animation_window is not None and self.animation_window.winfo_exists():
            self.animation_window.destroy()
        self.animation_window = None
        self.animation_window_figure = None
        self.animation_window_axes = None
        self.animation_window_canvas = None
        self.animation_window_button.configure(text="Vollbild")
        self.status_var.set("Animation-Fenster geschlossen")

    def _sync_animation_window_frame(self) -> None:
        if self.animation_window_axes is None or self.animation_window_canvas is None:
            return
        self.animation_window_axes.clear()
        self.animation_window_axes.set_axis_off()
        if self.render_image is None:
            self.animation_window_axes.text(0.5, 0.5, "Animation noch nicht verfügbar", ha="center", va="center", fontsize=12, color=self.colors["text"])
        else:
            image = self.render_image.get_array()
            if image is not None:
                self.animation_window_axes.imshow(image, interpolation="bilinear")
            self.animation_window_axes.set_title("HalfCheetah-Animation (Vollbild)", fontsize=14, pad=10, color=self.colors["text"])
        self.animation_window_canvas.draw()
        self.animation_window_canvas.get_tk_widget().update_idletasks()

    def _build_lunarlander_bottom(self, parent: ttk.Frame, build_actions) -> None:
        ttk.Label(parent, text="Auswertung & Verlauf", style="Section.TLabel").pack(anchor="w", padx=4, pady=(6, 2))
        actions = ttk.Frame(parent)
        actions.pack(fill=tk.X, padx=4, pady=(0, 4))
        build_actions(actions)
        summary = ttk.LabelFrame(parent, text="Summary", padding=(8, 5))
        summary.pack(fill=tk.X, padx=4, pady=(2, 4))
        ttk.Label(summary, textvariable=self.summary_var, style="Status.TLabel", wraplength=1160, justify=tk.LEFT).pack(anchor="w", pady=(0, 4))
        progress = ttk.Frame(summary)
        progress.pack(fill=tk.X)
        ttk.Progressbar(progress, variable=self.progress_var, maximum=100, length=280, style="Red.Horizontal.TProgressbar").pack(side=tk.LEFT)
        ttk.Label(progress, text="  Fortschritt", style="Status.TLabel").pack(side=tk.LEFT, padx=(8, 0))
        plot = ttk.LabelFrame(parent, text="Reward Plot", padding=6)
        plot.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 2))
        plot_actions = ttk.Frame(plot)
        plot_actions.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(plot_actions, text="Plot speichern", command=lambda: self.save_plot(default_name="reward_verlauf.png"), style="Primary.TButton").pack(side=tk.RIGHT)
        self.figure = __import__('matplotlib').figure.Figure(figsize=(11.5, 3.3), dpi=120)
        self.axes = self.figure.add_subplot(111)
        self.canvas = __import__('matplotlib.backends.backend_tkagg').backends.backend_tkagg.FigureCanvasTkAgg(self.figure, master=plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.draw_reward_plot([], title="Reward-Verlauf")

    def _build_controls(self, parent: tk.Misc) -> None:
        ttk.Label(parent, text="Konfiguration", style="Section.TLabel").pack(anchor="w", padx=4, pady=(6, 2))
        general = ttk.LabelFrame(parent, text="Allgemein", padding=8)
        general.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(general, text="Env-ID").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(general, width=24, state="readonly", justify="left", textvariable=self.env_var).grid(row=0, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(general, text="Total Timesteps / Limit").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(general, textvariable=self.total_timesteps_var, width=24).grid(row=1, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(general, text="Reward-Modus").grid(row=2, column=0, sticky="w", padx=4, pady=3)
        reward_mode = ttk.Combobox(general, textvariable=self.reward_mode_var, values=logic.SUPPORTED_REWARD_MODES, state="readonly", width=21)
        reward_mode.grid(row=2, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(general, text="Trainings-Stopp").grid(row=3, column=0, sticky="w", padx=4, pady=3)
        stop_mode = ttk.Combobox(general, textvariable=self.training_stop_mode_var, values=("timesteps", "episodes"), state="readonly", width=21)
        stop_mode.grid(row=3, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(general, text="Ziel-Episoden (Training)").grid(row=4, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(general, textvariable=self.target_episodes_var, width=24).grid(row=4, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(general, text="Seed (leer = None)").grid(row=5, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(general, textvariable=self.seed_var, width=24).grid(row=5, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(general, text="Eval-Episoden").grid(row=6, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(general, textvariable=self.evaluation_episodes_var, width=24).grid(row=6, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(general, text="Aktive Methode").grid(row=7, column=0, sticky="w", padx=4, pady=3)
        ttk.Combobox(general, textvariable=self.selected_method, values=logic.SUPPORTED_METHODS, state="readonly", width=21).grid(row=7, column=1, sticky="w", padx=4, pady=3)
        methods = ttk.LabelFrame(parent, text="Methoden", padding=8)
        methods.pack(fill=tk.X, padx=4, pady=4)
        for method in logic.SUPPORTED_METHODS:
            ttk.Button(methods, text=f"{method} konfigurieren", command=lambda value=method: self._open_method_window(value), style="Secondary.TButton").pack(fill=tk.X, pady=3)
        ttk.Button(parent, text="Modell evaluieren", command=self.evaluate_selected, style="Secondary.TButton").pack(fill=tk.X, padx=4, pady=(6, 2))
        ttk.Button(parent, text="Theme wechseln", command=self.toggle_theme, style="Secondary.TButton").pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(parent, textvariable=self.status_var, style="Status.TLabel", wraplength=340).pack(anchor="w", padx=6, pady=6)

    def _build_actions(self, parent: ttk.Frame) -> None:
        for text, command, style in (
            ("Pausieren", self.pause_training, "Secondary.TButton"),
            ("Fortsetzen", self.resume_training, "Secondary.TButton"),
            ("Reset Plot", self.reset_plot, "Secondary.TButton"),
            ("Cancel", self.cancel_training, "Warning.TButton"),
            ("Vergleichen", self.show_compare_selection_window, "Primary.TButton"),
            ("Vollplot", lambda: self.open_plot_window(self.reward_history), "Secondary.TButton"),
            ("Plot speichern", self.save_plot, "Primary.TButton"),
            ("Modell speichern", self.save_selected_model, "Secondary.TButton"),
            ("Modell laden", self.load_selected_model, "Secondary.TButton"),
            ("Live Demo", self.run_live_demo, "Primary.TButton"),
        ):
            ttk.Button(parent, text=text, command=command, style=style).pack(side=tk.LEFT, padx=(0, 6))

    def show_compare_selection_window(self) -> None:
        window = tk.Toplevel(self)
        window.title("Methoden vergleichen")
        window.transient(self)
        window.resizable(False, False)
        window.grab_set()
        content = ttk.Frame(window, padding=14)
        content.pack(fill=tk.BOTH, expand=True)
        ttk.Label(content, text="Waehle mindestens zwei Methoden:", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        for method in logic.SUPPORTED_METHODS:
            ttk.Checkbutton(content, text=method, variable=self.compare_vars[method]).pack(anchor="w", pady=3)
        actions = ttk.Frame(content)
        actions.pack(fill=tk.X, pady=(14, 0))
        ttk.Button(actions, text="Abbrechen", command=window.destroy, style="Secondary.TButton").pack(side=tk.RIGHT)
        ttk.Button(actions, text="Vergleich starten", command=lambda: self.start_comparison(window), style="Primary.TButton").pack(side=tk.RIGHT, padx=(0, 6))

    def start_comparison(self, window: tk.Toplevel) -> None:
        if self.training_active:
            self.status_var.set("Training läuft bereits.")
            return
        methods = [method for method in logic.SUPPORTED_METHODS if self.compare_vars[method].get()]
        if len(methods) < 2:
            self.show_error("Bitte mindestens zwei Methoden auswaehlen.")
            return
        try:
            configs = [self._configured(method) for method in methods]
            for config in configs:
                config.validate()
        except Exception as error:
            self.show_error(error)
            return
        self.training_animation_visible = True
        self.animation_button.configure(text="Animation AUS")
        self.animation_state_var.set("Vergleich aktiv - Live-Animation eingeschaltet")
        self._comparison_placeholder_ready = True
        self._show_animation_placeholder("Vergleich läuft - Live-Animation aktiv")
        self.render_canvas.draw_idle()
        self.draw_sweep_plot({method: [] for method in methods}, axes=self.axes)
        self.canvas.draw_idle()
        window.destroy()
        self.compare_active = True
        self.training_active = True
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.running_workers = len(configs)
        self.progress_var.set(0)
        self.status_var.set(f"Vergleich läuft: {', '.join(methods)}")
        self.selected_method.set(methods[0])
        threading.Thread(target=self._comparison_worker, args=(configs,), daemon=True).start()

    def _comparison_worker(self, configs: list[logic.OnPolicyConfig]) -> None:
        barrier = threading.Barrier(len(configs))

        def train_method(config: logic.OnPolicyConfig) -> None:
            try:
                barrier.wait()
                self._training_worker(config)
            except Exception as error:
                self.events.put({"type": "error", "error": error})

        for config in configs:
            threading.Thread(target=train_method, args=(config,), daemon=True).start()

    def _field(self, parent: tk.Misc, method: str, name: str, value: Any, row: int) -> None:
        ttk.Label(parent, text=name).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        if isinstance(value, bool):
            variable: tk.Variable = tk.BooleanVar(value=value)
            widget = ttk.Checkbutton(parent, variable=variable)
        else:
            variable = tk.StringVar(value=str(value))
            widget = ttk.Entry(parent, textvariable=variable, width=25)
        widget.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        self.variables[method][name] = variable

    def _configured(self, method: str) -> logic.OnPolicyConfig:
        values = logic.get_default_parameters_for_method(method)
        for name, variable in self.variables[method].items():
            original, value = values[name], variable.get()
            values[name] = bool(value) if isinstance(original, bool) else int(value) if isinstance(original, int) else float(value) if isinstance(original, float) else value
        values["total_timesteps"] = int(self.total_timesteps_var.get())
        values["training_stop_mode"] = self.training_stop_mode_var.get()
        values["target_episodes"] = int(self.target_episodes_var.get())
        values["seed"] = int(self.seed_var.get()) if self.seed_var.get().strip() else None
        values["method"] = method
        config = logic.OnPolicyConfig(**values)
        self.configs[method] = config
        return config

    def _open_method_window(self, method: str) -> None:
        self.selected_method.set(method)
        window = tk.Toplevel(self)
        window.title(f"Hyperparameter - {method}")
        window.geometry("520x700")
        notebook = ttk.Notebook(window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        training, network = ttk.Frame(notebook, padding=8), ttk.Frame(notebook, padding=8)
        notebook.add(training, text="Training")
        notebook.add(network, text="Netzwerk")
        self.variables[method] = {}
        config = self.configs[method]
        shared = ("total_timesteps", "n_envs", "learning_rate", "gamma", "batch_size")
        if method == "PPO":
            create_fields = ("n_steps", "n_epochs", "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm")
        elif method == "CMA-ES":
            create_fields = ("learning_rate", "n_envs", "batch_size", "gamma", "buffer_size", "learning_starts", "ent_coef")
            ttk.Label(training, text="CMA-ES benötigt: pip install cma").grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 8))
        else:
            create_fields = ("buffer_size", "learning_starts", "tau", "train_freq", "gradient_steps", "policy_delay", "ent_coef")
        for row, name in enumerate(shared + create_fields):
            self._field(training, method, name, getattr(config, name), row)
        for row, name in enumerate(("net_arch_pi", "net_arch_vf", "activation_fn", "ortho_init", "device")):
            self._field(network, method, name, getattr(config, name), row)
        ttk.Button(training, text=f"{method} trainieren", command=lambda: self.start_training(method), style="Primary.TButton").grid(row=30, column=0, columnspan=2, sticky="ew", pady=12)

    def start_training(self, method: str) -> None:
        if self.training_active:
            self.status_var.set("Training läuft bereits.")
            return
        self.compare_active = False
        try:
            config = self._configured(method)
            config.validate()
        except Exception as error:
            self.show_error(error)
            return
        self.training_active = True
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.running_workers = 1
        self.status_var.set(f"{method} trainiert ...")
        threading.Thread(target=self._training_worker, args=(config,), daemon=True).start()

    def _training_worker(self, config: logic.OnPolicyConfig) -> None:
        try:
            result = self.trainer.train(config, self.stop_event, self.events.put, render_during_training=self.training_animation_visible, frame_capture_steps=self.training_frame_capture_steps, pause_event=self.pause_event)
            self.events.put({"type": "finished", "method": config.method, "result": result})
        except Exception as error:
            self.events.put({"type": "error", "error": error})

    def pause_training(self) -> None:
        if not self.training_active:
            self.status_var.set("Kein Training zum Pausieren aktiv.")
            return
        self.pause_event.set()
        self.training_paused = True
        self.status_var.set("Training pausiert")

    def resume_training(self) -> None:
        if not self.training_active or not self.training_paused:
            self.status_var.set("Kein pausiertes Training vorhanden.")
            return
        self.pause_event.clear()
        self.training_paused = False
        self.status_var.set("Training wird fortgesetzt")

    def reset_plot(self) -> None:
        self.reward_history = {method: [] for method in logic.SUPPORTED_METHODS}
        self.progress_var.set(0)
        self.draw_reward_plot([], title="Reward-Verlauf")
        self.summary_var.set("Plot zurückgesetzt")

    def cancel_training(self) -> None:
        if not self.training_active and not self.demo_active:
            self.status_var.set("Kein Training oder keine Live Demo aktiv.")
            return
        self.stop_event.set(); self.pause_event.clear(); self.demo_stop_event.set(); self.training_paused = False; self.status_var.set("Abbruch angefordert")

    def evaluate_selected(self) -> None:
        model = self.trainer.models.get(self.selected_method.get())
        if model is None:
            self.status_var.set("Bitte zuerst ein Modell trainieren oder laden.")
            return
        def evaluate() -> None:
            try:
                episodes = int(self.evaluation_episodes_var.get())
                seed = int(self.seed_var.get()) if self.seed_var.get().strip() else None
                rewards = self.trainer.evaluate(model, episodes=episodes, seed=seed)
                self.after(0, lambda: self.status_var.set(f"Evaluation: {sum(rewards) / len(rewards):.1f} Reward"))
            except Exception as error:
                self.after(0, lambda: self.show_error(error))
        threading.Thread(target=evaluate, daemon=True).start()

    def save_selected_model(self) -> None:
        method = self.selected_method.get()
        self.save_model_dialog(lambda path: self.trainer.save_model(method, path))

    def load_selected_model(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("SB3-Modell", "*.zip")])
        if path:
            try:
                self.trainer.load_model(self.selected_method.get(), path)
                self.status_var.set("Modell geladen")
            except Exception as error:
                self.show_error(error)

    def run_live_demo(self) -> None:
        if self.training_active or self.demo_active:
            self.status_var.set("Warte, bis das aktive Training oder die Demo beendet ist.")
            return
        method = self.selected_method.get()
        model = self.trainer.models.get(method)
        if model is None:
            self.status_var.set("Bitte zuerst ein Modell trainieren oder laden.")
            return
        config = self.trainer.configs.get(method, self.configs[method])
        self.demo_active = True
        self.demo_stop_event = threading.Event()
        self.status_var.set(f"Live Demo: {method}")
        threading.Thread(target=self._live_demo_worker, args=(model, config), daemon=True).start()

    def _live_demo_worker(self, model: Any, config: logic.OnPolicyConfig) -> None:
        environment = logic.make_halfcheetah_env(render_mode="rgb_array")
        try:
            observation, _ = environment.reset(seed=config.seed)
            while not self.demo_stop_event.is_set():
                frame = environment.render()
                if frame is not None:
                    self.after(0, lambda image=frame: self.show_training_frame(image))
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = environment.step(action)
                if terminated or truncated:
                    observation, _ = environment.reset()
                time.sleep(0.02)
        except Exception as error:
            self.after(0, lambda captured_error=error: self.show_error(captured_error))
        finally:
            environment.close()
            self.demo_active = False
            self.after(0, lambda: self.status_var.set("Live Demo beendet"))

    def show_training_frame(self, frame, method: str | None = None) -> None:
        target_method = method or self.selected_method.get()
        if frame is not None:
            try:
                image = np.asarray(frame)
            except Exception:
                return
            if image.size != 0:
                self._method_frames[target_method] = image
        if not self.training_animation_visible:
            return
        frame_to_draw = self._method_frames.get(target_method)
        if frame_to_draw is None and frame is not None:
            frame_to_draw = np.asarray(frame)
        if frame_to_draw is None:
            return
        try:
            image = np.asarray(frame_to_draw)
        except Exception:
            return
        if image.size == 0:
            return
        self._pending_animation_frame = image
        self._animation_frame_counter += 1
        if self._animation_redraw_scheduled:
            return
        self._animation_redraw_scheduled = True
        self.after(0, self._flush_animation_frame)

    def _flush_animation_frame(self) -> None:
        self._animation_redraw_scheduled = False
        image = self._pending_animation_frame
        self._pending_animation_frame = None
        if image is None:
            return
        now = time.perf_counter()
        if self._animation_last_draw_time:
            elapsed = now - self._animation_last_draw_time
            if elapsed < 0.03:
                self._pending_animation_frame = image
                self._animation_redraw_scheduled = True
                self.after(1, self._flush_animation_frame)
                return
        self._animation_last_draw_time = now
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        if image.ndim != 3 or image.shape[2] != 3:
            return
        if self.render_image is None or self.render_image.axes is None:
            self.render_axes.clear()
            self.render_image = self.render_axes.imshow(image, interpolation="bilinear", aspect="auto" if self.animation_fit_mode else "equal")
            self.render_axes.set_axis_off()
            self.render_axes.set_title("HalfCheetah Live", fontsize=13, pad=10, color="#e8f3ff")
        else:
            self.render_image.set_data(image)
        self.render_axes.set_aspect("auto" if self.animation_fit_mode else "equal")
        self.render_axes.set_position([0.04, 0.08, 0.92, 0.78] if self.animation_fit_mode else [0.12, 0.12, 0.76, 0.7])
        self.render_axes.set_xlim(0, image.shape[1] - 1)
        self.render_axes.set_ylim(image.shape[0] - 1, 0)
        self.render_axes.set_facecolor("#0b1220")
        self.render_axes.text(0.02, 0.985, f"Frame {self._animation_frame_counter}", transform=self.render_axes.transAxes, color="white", fontsize=9, ha="left", va="top", bbox={"boxstyle": "round,pad=0.25", "facecolor": "#0b1220", "edgecolor": "#4b89ff", "alpha": 0.7})
        self.render_axes.text(0.98, 0.985, f"{self._animation_fps:.1f} fps", transform=self.render_axes.transAxes, color="white", fontsize=9, ha="right", va="top", bbox={"boxstyle": "round,pad=0.25", "facecolor": "#0b1220", "edgecolor": "#4b89ff", "alpha": 0.7})
        self.render_canvas.draw()
        self.render_canvas.get_tk_widget().update_idletasks()
        self.animation_state_var.set("Live-Animation aktiv")
        self._sync_animation_window_frame()

    def _poll_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event["type"] == "episode":
                self.reward_history[event["method"]].append(float(event["reward"]))
                filtered = {name: rewards for name, rewards in self.reward_history.items() if rewards}
                self.draw_sweep_plot(filtered)
                self.refresh_full_plot(filtered)
                self._comparison_placeholder_ready = False
            elif event["type"] == "frame":
                method_name = event["method"]
                self._method_frames[method_name] = np.asarray(event["frame"])
                if self.compare_active:
                    self.selected_method.set(method_name)
                    if self.status_var.get().startswith("Vergleich läuft"):
                        self.status_var.set(f"Vergleich läuft: {method_name}")
                self.show_training_frame(event["frame"], method_name)
            elif event["type"] == "progress":
                self.progress_var.set(event["ratio"] * 100)
                if self.compare_active:
                    self._comparison_placeholder_ready = True
                    self.axes.set_title("Reward-Verlauf (Vergleich aktiv)", color=self.colors["text"])
                    self.axes.text(0.5, 0.5, "Training läuft ...", transform=self.axes.transAxes, ha="center", va="center", color=self.colors["muted"], fontsize=10)
                    self.canvas.draw_idle()
            elif event["type"] == "finished":
                self.running_workers -= 1
                result = event["result"]
                quality = result.get("quality", {})
                summary_lines = [f"Methode: {event['method']} | Trainierte Episoden: {len(result['rewards'])}"]
                if result.get("cancelled"):
                    summary_lines.append("Status: Training abgebrochen")
                elif quality:
                    training_status = "Erfolgreich trainiert" if quality.get("is_well_trained") else "Weitere Optimierung empfohlen"
                    summary_lines.append(f"Status: {training_status}")
                    summary_lines.append(f"Mean Reward: {quality.get('mean_reward', 0.0):.2f} | Std: {quality.get('std_reward', 0.0):.2f}")
                    if quality.get("message"):
                        summary_lines.append(f"Hinweis: {quality['message']}")
                if result.get("auto_saved_path"):
                    summary_lines.append(f"Gespeichert: {result['auto_saved_path']}")
                self.summary_var.set(" | ".join(summary_lines))
                if self.running_workers <= 0:
                    self.training_active = False
                    self.training_paused = False
                    self.compare_active = False
                    self.status_var.set("Training beendet")
            elif event["type"] == "error":
                self.training_active = False
                self.training_paused = False
                self.compare_active = False
                self.status_var.set("Training fehlgeschlagen")
                self.show_error(event["error"])
        self.after(80, self._poll_events)
