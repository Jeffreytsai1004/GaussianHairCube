"""
Log Window
==========

Independent floating window that mirrors the application's logging
output in real time.  Useful for debugging long-running ML / IO
operations and inspecting backend chatter.

Features
--------
- Real-time stream from the root logger via logging_setup listeners
- Replay of recent history (up to 2000 lines) on open
- Level filter dropdown (DEBUG / INFO / WARNING / ERROR)
- Colour coding per level
- Auto-scroll toggle, clear button, copy-all, open-log-folder
- Multiple windows safe (each maintains its own listener)
"""

import logging
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from src.config import logging_setup


_LEVEL_COLORS = {
    "DEBUG":    "#9e9e9e",
    "INFO":     "#e0e0e0",
    "WARNING":  "#ffb74d",
    "ERROR":    "#ef5350",
    "CRITICAL": "#d32f2f",
}

_LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}


class LogWindow(ctk.CTkToplevel):
    """A persistent, scrollable log viewer."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("📋 日志窗口")
        self.geometry("960x520")
        self.minsize(640, 320)

        self._min_level = "INFO"
        self._auto_scroll = True
        self._closed = False

        self._build_ui()
        self._populate_from_buffer()

        # Subscribe to new log records — wrapper marshals to the UI thread
        logging_setup.add_log_listener(self._on_new_log)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Toolbar ---
        toolbar = ctk.CTkFrame(self, fg_color="gray18", corner_radius=0, height=44)
        toolbar.grid(row=0, column=0, sticky="ew")
        for i in range(8):
            toolbar.grid_columnconfigure(i, weight=0)
        toolbar.grid_columnconfigure(4, weight=1)  # spacer

        ctk.CTkLabel(toolbar, text="级别:", font=ctk.CTkFont(size=11)).grid(
            row=0, column=0, padx=(12, 4), pady=8)
        self._level_menu = ctk.CTkOptionMenu(
            toolbar, values=["DEBUG", "INFO", "WARNING", "ERROR"],
            command=self._on_level_change, width=110,
        )
        self._level_menu.grid(row=0, column=1, padx=4, pady=8)
        self._level_menu.set(self._min_level)

        self._autoscroll_chk = ctk.CTkCheckBox(
            toolbar, text="自动滚动",
            command=self._on_autoscroll_toggle,
            font=ctk.CTkFont(size=11),
        )
        self._autoscroll_chk.grid(row=0, column=2, padx=(12, 4), pady=8)
        self._autoscroll_chk.select()

        self._line_count_label = ctk.CTkLabel(
            toolbar, text="0 行", font=ctk.CTkFont(size=11), text_color="gray60",
        )
        self._line_count_label.grid(row=0, column=3, padx=(12, 4), pady=8)

        ctk.CTkButton(
            toolbar, text="🗑 清空", width=66, height=26,
            fg_color="gray35", hover_color="gray25",
            command=self._on_clear, font=ctk.CTkFont(size=11),
        ).grid(row=0, column=5, padx=4, pady=8)

        ctk.CTkButton(
            toolbar, text="💾 保存到文件…", width=120, height=26,
            fg_color="gray35", hover_color="gray25",
            command=self._on_save, font=ctk.CTkFont(size=11),
        ).grid(row=0, column=6, padx=4, pady=8)

        ctk.CTkButton(
            toolbar, text="📂 打开日志目录", width=130, height=26,
            fg_color="gray35", hover_color="gray25",
            command=self._on_open_dir, font=ctk.CTkFont(size=11),
        ).grid(row=0, column=7, padx=(4, 12), pady=8)

        # --- Text area ---
        # Use CTkTextbox; underlying tkinter Text supports tag_config for colours.
        self._text = ctk.CTkTextbox(
            self, font=("Consolas", 11), wrap="none", fg_color="#1a1a1f",
            text_color="#e0e0e0",
        )
        self._text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 8))

        # Tag colours on the inner tk.Text widget
        inner = self._text._textbox  # CTkTextbox exposes the raw tk.Text
        for lvl, colour in _LEVEL_COLORS.items():
            inner.tag_config(lvl, foreground=colour)

        self._text.configure(state="disabled")

    # ------------------------------------------------------------------
    # State / actions
    # ------------------------------------------------------------------

    def _passes_filter(self, level: str) -> bool:
        return _LEVEL_ORDER.get(level, 1) >= _LEVEL_ORDER.get(self._min_level, 1)

    def _on_level_change(self, value: str):
        self._min_level = value
        # Re-render from buffer to apply new filter
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")
        self._populate_from_buffer()

    def _on_autoscroll_toggle(self):
        self._auto_scroll = bool(self._autoscroll_chk.get())

    def _on_clear(self):
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")
        self._update_line_count()

    def _on_save(self):
        path = filedialog.asksaveasfilename(
            title="保存日志到", defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
            initialfile="ghc_session.log",
        )
        if not path:
            return
        try:
            content = self._text.get("1.0", "end")
            Path(path).write_text(content, encoding="utf-8")
        except Exception:
            logging.getLogger(__name__).exception("Failed to save log")

    def _on_open_dir(self):
        d = logging_setup.get_log_dir()
        d.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(str(d))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(d)])
            else:
                subprocess.Popen(["xdg-open", str(d)])
        except Exception:
            logging.getLogger(__name__).exception("Failed to open log directory")

    # ------------------------------------------------------------------
    # Log ingestion
    # ------------------------------------------------------------------

    def _populate_from_buffer(self):
        """Replay the in-memory ring buffer on open or after filter change."""
        for msg, level in logging_setup.get_log_buffer():
            self._insert_line(msg, level)
        self._update_line_count()

    def _on_new_log(self, msg: str, level: str):
        """Called from logging thread — marshal to UI thread."""
        if self._closed:
            return
        try:
            self.after(0, lambda m=msg, l=level: self._insert_line(m, l, update_count=True))
        except Exception:
            # Window destroyed mid-call
            pass

    def _insert_line(self, msg: str, level: str, update_count: bool = False):
        if not self._passes_filter(level):
            return
        self._text.configure(state="normal")
        self._text.insert("end", msg + "\n", level)
        if self._auto_scroll:
            self._text.see("end")
        self._text.configure(state="disabled")
        if update_count:
            self._update_line_count()

    def _update_line_count(self):
        # Subtract 1 because tk.Text always has a trailing newline at end
        try:
            n = int(self._text._textbox.index("end-1c").split(".")[0]) - 1
        except Exception:
            n = 0
        self._line_count_label.configure(text=f"{max(n, 0)} 行")

    def _on_close(self):
        self._closed = True
        logging_setup.remove_log_listener(self._on_new_log)
        self.destroy()