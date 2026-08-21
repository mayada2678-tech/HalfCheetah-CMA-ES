from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

THEMES = {
    "dark": {"background": "#000000", "panel": "#101010", "text": "#d4d4d4", "muted": "#9da1a6", "accent": "#0078d4", "grid": "#4d4d4d", "field": "#1a1a1a", "border": "#2b2b2b", "button": "#1a1a1a", "active": "#2a2a2a"},
    "light": {"background": "#f3f3f3", "panel": "#ffffff", "text": "#1f2328", "muted": "#57606a", "accent": "#0b57d0", "grid": "#57606a", "field": "#ffffff", "border": "#d0d7de", "button": "#f6f8fa", "active": "#eef2f6"},
}


def moving_average(values: Iterable[float], window_size: int = 10) -> list[float]:
    rewards = np.asarray(list(values), dtype=float)
    if rewards.size == 0:
        return []
    window_size = max(1, int(window_size))
    cumulative = np.concatenate(([0.0], np.cumsum(rewards)))
    ends = np.arange(1, rewards.size + 1)
    starts = np.maximum(0, ends - window_size)
    return ((cumulative[ends] - cumulative[starts]) / (ends - starts)).tolist()


class WorkbenchUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.theme_name = "dark"
        self.title("Stable-Baselines3 Workbench - HalfCheetah-v5")
        self.minsize(1100, 720)
        self.geometry("1320x860")
        self.figure = Figure(figsize=(7, 4.8), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.scroll_canvases: list[tk.Canvas] = []
        self.style = ttk.Style(self)
        self.status_var = tk.StringVar(value="Bereit")
        self.summary_var = tk.StringVar(value="Noch kein Training")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.training_animation_visible = True
        self.animation_fit_mode = True
        self.training_frame_capture_steps = 64
        self.animation_state_var = tk.StringVar(value="Warte auf Trainingsframe")
        self.animation_window: tk.Toplevel | None = None
        self.animation_window_figure: Figure | None = None
        self.animation_window_axes = None
        self.animation_window_canvas: FigureCanvasTkAgg | None = None
        self.full_plot_window: tk.Toplevel | None = None
        self.full_plot_figure: Figure | None = None
        self.full_plot_axes = None
        self.full_plot_canvas: FigureCanvasTkAgg | None = None
        self.apply_theme()

    @property
    def colors(self) -> dict[str, str]:
        return THEMES[self.theme_name]

    def _configure_lunarlander_style(self) -> None:
        colors = self.colors
        self.style.theme_use("clam")
        self.style.configure("TFrame", background=colors["panel"])
        self.style.configure("TLabelframe", background=colors["panel"], bordercolor=colors["border"], darkcolor=colors["border"], lightcolor=colors["border"], borderwidth=1, relief="solid")
        self.style.configure("TLabelframe.Label", background=colors["panel"], foreground=colors["text"], font=("Segoe UI Semibold", 10))
        self.style.configure("TLabel", background=colors["panel"], foreground=colors["text"])
        self.style.configure("Title.TLabel", background=colors["panel"], foreground="#4fc3f7", font=("Segoe UI Semibold", 14, "bold"))
        self.style.configure("Accent.TLabel", background=colors["panel"], foreground="#7ce7b5", font=("Segoe UI Semibold", 11, "bold"))
        self.style.configure("Section.TLabel", background=colors["panel"], foreground="#8ecbff", font=("Segoe UI Semibold", 10, "bold"))
        self.style.configure("Status.TLabel", background=colors["panel"], foreground="#dfe7f6", font=("Segoe UI", 9))
        self.style.configure("TButton", background=colors["button"], foreground=colors["text"], bordercolor=colors["border"], darkcolor=colors["border"], lightcolor=colors["border"], padding=(9, 6))
        self.style.map("TButton", background=[("pressed", colors["accent"]), ("active", colors["active"])], foreground=[("pressed", "#ffffff"), ("active", colors["text"])])
        self.style.configure("Primary.TButton", background="#1f9d8a", foreground="#ffffff", bordercolor="#1f9d8a", padding=(10, 6))
        self.style.map("Primary.TButton", background=[("pressed", "#177a6f"), ("active", "#2eb59f")], foreground=[("pressed", "#ffffff"), ("active", "#ffffff")])
        self.style.configure("Secondary.TButton", background=colors["button"], foreground=colors["text"], bordercolor=colors["border"], padding=(10, 6))
        self.style.map("Secondary.TButton", background=[("pressed", colors["active"]), ("active", colors["active"])], foreground=[("pressed", colors["text"]), ("active", colors["text"])])
        self.style.configure("Warning.TButton", background="#a05a00", foreground="#ffffff", bordercolor="#a05a00", padding=(10, 6))
        self.style.map("Warning.TButton", background=[("pressed", "#7d4700"), ("active", "#bb6d00")], foreground=[("pressed", "#ffffff"), ("active", "#ffffff")])
        self.style.configure("TCheckbutton", background=colors["panel"], foreground=colors["text"])
        self.style.configure("TEntry", fieldbackground=colors["field"], foreground=colors["text"], bordercolor=colors["border"], insertcolor=colors["text"], padding=(6, 4))
        self.style.configure("TCombobox", fieldbackground=colors["field"], foreground=colors["text"], bordercolor=colors["border"], arrowsize=14, padding=4)
        self.style.configure("TNotebook", background=colors["panel"], borderwidth=0)
        self.style.configure("TNotebook.Tab", background=colors["button"], foreground=colors["text"], padding=(10, 5))
        self.style.map("TNotebook.Tab", background=[("selected", colors["accent"])], foreground=[("selected", "#ffffff")])
        self.style.configure("Red.Horizontal.TProgressbar", troughcolor="#3a3a3a" if self.theme_name == "dark" else "#d0d7de", background="#d73027", lightcolor="#d73027", darkcolor="#d73027", bordercolor="#3a3a3a" if self.theme_name == "dark" else "#d0d7de")
        for canvas in self.scroll_canvases:
            canvas.configure(background=colors["panel"], highlightthickness=0)

    def apply_theme(self) -> None:
        colors = self.colors
        self.configure(bg=colors["background"])
        self.option_add("*Background", colors["background"])
        self.option_add("*Foreground", colors["text"])
        self.option_add("*insertBackground", colors["text"])
        self.figure.set_facecolor(colors["background"])
        self.axes.set_facecolor(colors["background"])
        self._configure_lunarlander_style()

    def toggle_theme(self) -> None:
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self.apply_theme()
        self.draw_reward_plot([], title="Reward-Verlauf")

    def _scrollable(self, parent: tk.Misc):
        frame = ttk.Frame(parent)
        canvas = tk.Canvas(frame, highlightthickness=0, bg=self.colors["background"])
        self.scroll_canvases.append(canvas)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        body = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfig(window, width=event.width))
        frame.body = body
        return frame

    def draw_reward_plot(self, rewards: Iterable[float], *, title: str = "Reward-Verlauf", label: str = "Reward") -> None:
        values = list(rewards)
        colors = self.colors
        self.axes.clear()
        self.figure.set_facecolor(colors["background"])
        self.axes.set_facecolor(colors["background"])
        self.axes.set_title(title, color=colors["text"])
        self.axes.set_xlabel("Episode", color=colors["muted"])
        self.axes.set_ylabel("Reward", color=colors["muted"])
        self.axes.tick_params(colors=colors["muted"])
        self.axes.grid(True, alpha=0.45, color=colors["grid"])
        if values:
            episodes = range(1, len(values) + 1)
            self.axes.plot(episodes, values, color=colors["accent"], alpha=0.35, linewidth=1, label=f"{label} Rohwerte")
            self.axes.plot(episodes, moving_average(values), color=colors["accent"], linewidth=2, label=f"{label} Durchschnitt (10)")
            self.axes.legend(facecolor=colors["panel"], edgecolor=colors["border"], labelcolor=colors["text"])
        if self.canvas:
            self.canvas.draw_idle()

    def draw_sweep_plot(self, curves: dict[str, list[float]], *, axes=None) -> None:
        target = axes or self.axes
        colors = self.colors
        target.clear()
        if hasattr(target.figure, "set_facecolor"):
            target.figure.set_facecolor(colors["background"])
        target.set_facecolor(colors["background"])
        target.set_title("Stable-Baselines3: Methodenvergleich Reward pro Episode", color=colors["text"])
        target.set_xlabel("Episode", color=colors["muted"])
        target.set_ylabel("Reward", color=colors["muted"])
        target.tick_params(colors=colors["muted"])
        target.grid(True, alpha=0.45, color=colors["grid"])
        plotted = False
        for label, rewards in curves.items():
            if not rewards:
                continue
            episodes = range(1, len(rewards) + 1)
            raw_line = target.plot(episodes, rewards, alpha=0.3, linewidth=1, label=f"{label} Rohwerte")[0]
            target.plot(episodes, moving_average(rewards), color=raw_line.get_color(), linewidth=2, label=f"{label} Durchschnitt (10)")
            plotted = True
        if plotted:
            target.legend(fontsize=8, facecolor=colors["panel"], edgecolor=colors["border"], labelcolor=colors["text"])
        if self.canvas and target is self.axes:
            self.canvas.draw_idle()

    def refresh_full_plot(self, curves: Optional[dict[str, list[float]]] = None) -> None:
        if self.full_plot_window is None or not self.full_plot_window.winfo_exists():
            return
        if self.full_plot_axes is None or self.full_plot_figure is None or self.full_plot_canvas is None:
            return
        data = curves if curves is not None else {}
        if not data:
            data = {"Reward": []}
        old_canvas = self.canvas
        self.canvas = None
        self.draw_sweep_plot(data, axes=self.full_plot_axes)
        self.canvas = old_canvas
        self.full_plot_canvas.draw_idle()

    def save_plot(self, figure: Optional[Figure] = None, default_name: str = "reward_plot.png") -> None:
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=default_name, filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf"), ("JPG", "*.jpg")])
        if path:
            (figure or self.figure).savefig(path, bbox_inches="tight")
            self.status_var.set(f"Plot gespeichert: {path}")

    def open_plot_window(self, curves: Optional[dict[str, list[float]]] = None, *, title: str = "Vollplot") -> None:
        if self.full_plot_window is not None and self.full_plot_window.winfo_exists():
            self.full_plot_window.focus_set()
            return

        window = tk.Toplevel(self)
        window.title(title)
        window.geometry("1000x720")
        window.configure(bg=self.colors["background"])
        figure = Figure(figsize=(10, 7), dpi=100, facecolor=self.colors["background"])
        axes = figure.add_subplot(111)
        axes.set_facecolor(self.colors["background"])
        if curves is None:
            curves = {}
        if not curves:
            curves = {"Reward": []}
        self.full_plot_window = window
        self.full_plot_figure = figure
        self.full_plot_axes = axes
        old_canvas = self.canvas
        self.canvas = None
        self.draw_sweep_plot(curves, axes=axes)
        self.canvas = old_canvas
        actions = ttk.Frame(window)
        actions.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Button(actions, text="Plot speichern", command=lambda: self.save_plot(figure, default_name="vollplot.png"), style="Primary.TButton").pack(side=tk.RIGHT)
        canvas = FigureCanvasTkAgg(figure, master=window)
        self.full_plot_canvas = canvas
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw_idle()
        window.protocol("WM_DELETE_WINDOW", self._close_full_plot_window)

    def _close_full_plot_window(self) -> None:
        if self.full_plot_window is not None and self.full_plot_window.winfo_exists():
            self.full_plot_window.destroy()
        self.full_plot_window = None
        self.full_plot_figure = None
        self.full_plot_axes = None
        self.full_plot_canvas = None

    def show_error(self, error: Exception | str) -> None:
        messagebox.showerror("HalfCheetah Workbench", str(error), parent=self)

    def save_model_dialog(self, save_action) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("SB3-Modell", "*.zip")])
        if path:
            save_action(Path(path))


__all__ = ["WorkbenchUI", "moving_average", "THEMES"]
