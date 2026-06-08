"""
Modal dialog shown on first run when ML models need to be downloaded.
"""
import threading
import customtkinter as ctk


class ModelDownloadDialog(ctk.CTkToplevel):
    """
    Modal dialog for downloading HuggingFace model weights.

    Shows per-model progress, total size info, and allows cancellation.
    Automatically closes when download completes successfully.
    """

    def __init__(self, parent, models_to_download: list, on_complete=None, on_cancel=None,
                 auto_start: bool = False):
        """
        models_to_download: list of dicts with keys: display_name, repo_id, approx_size_mb
        on_complete: callback() called when download finishes successfully (main thread)
        on_cancel: callback() called when user cancels (main thread)
        auto_start: if True, kick off the download immediately after the dialog opens
        """
        super().__init__(parent)

        self._cancel_flag = False
        self._on_complete = on_complete
        self._on_cancel = on_cancel
        self._download_success = False

        # Calculate total size
        total_mb = sum(m.get("approx_size_mb", 0) for m in models_to_download)

        # Window setup
        self.title("首次运行 — 下载 AI 模型")
        self.geometry("520x380")
        self.resizable(False, False)
        self.grab_set()  # Modal
        self.protocol("WM_DELETE_WINDOW", self._on_user_close)

        # Center on parent
        self.after(10, self._center_on_parent)

        # --- UI ---

        # Header
        ctk.CTkLabel(
            self,
            text="首次运行需要下载 AI 模型",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(24, 4))

        ctk.CTkLabel(
            self,
            text=f"共需下载约 {total_mb} MB，将缓存到本地，之后无需重复下载。",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        ).pack(pady=(0, 16))

        # Model list
        models_frame = ctk.CTkFrame(self)
        models_frame.pack(fill="x", padx=24, pady=(0, 12))

        for m in models_to_download:
            cached = m.get("cached", False)
            row = ctk.CTkFrame(models_frame, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=4)
            name_text = f"• {m['display_name']}"
            if cached:
                name_text += "  ✓ 已缓存"
            ctk.CTkLabel(row, text=name_text, anchor="w").pack(side="left")
            size_text = f"~{m.get('approx_size_mb', '?')} MB"
            if cached:
                size_text = "已缓存"
            ctk.CTkLabel(
                row,
                text=size_text,
                text_color="gray",
                anchor="e",
            ).pack(side="right")

        # Progress bar
        self._progress_bar = ctk.CTkProgressBar(self, width=460)
        self._progress_bar.pack(padx=24, pady=(4, 4))
        self._progress_bar.set(0)

        # Status label
        self._status_label = ctk.CTkLabel(
            self,
            text="准备下载...",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self._status_label.pack(pady=(0, 16))

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=(0, 20))

        self._download_btn = ctk.CTkButton(
            btn_frame,
            text="开始下载",
            width=140,
            command=self._start_download,
        )
        self._download_btn.pack(side="left", padx=8)

        self._cancel_btn = ctk.CTkButton(
            btn_frame,
            text="取消",
            width=100,
            fg_color="gray",
            command=self._on_user_close,
        )
        self._cancel_btn.pack(side="left", padx=8)

        # Auto-start: schedule download right after the window has rendered
        if auto_start:
            self.after(150, self._start_download)

    def _center_on_parent(self):
        parent = self.master
        px = parent.winfo_x() + parent.winfo_width() // 2
        py = parent.winfo_y() + parent.winfo_height() // 2
        w, h = 520, 380
        self.geometry(f"{w}x{h}+{px - w // 2}+{py - h // 2}")

    def _start_download(self):
        self._download_btn.configure(state="disabled", text="下载中...")
        self._cancel_btn.configure(text="取消下载")
        self._cancel_flag = False

        thread = threading.Thread(target=self._download_worker, daemon=True)
        thread.start()

    def _download_worker(self):
        from src.core.model_manager import download_models

        success = download_models(
            progress_callback=self._thread_safe_progress,
            cancel_flag_getter=lambda: self._cancel_flag,
        )

        self._download_success = success
        self.after(0, self._on_download_finished)

    def _thread_safe_progress(self, progress: float, message: str):
        """Called from download thread — must use after() to update UI."""
        # Remember the most-recent message so failure shows the actual cause
        # (the previous version overwrote it with a generic "下载失败" string).
        self._last_message = message
        self.after(0, lambda p=progress, m=message: self._update_progress(p, m))

    def _update_progress(self, progress: float, message: str):
        self._progress_bar.set(min(progress, 1.0))
        self._status_label.configure(text=message)

    def _on_download_finished(self):
        if self._cancel_flag:
            return

        if self._download_success:
            self._update_progress(1.0, "✓ 所有模型下载完成！正在启动处理...")
            self.after(800, self._close_and_complete)
        else:
            # Preserve the actionable error from the last progress callback
            last = getattr(self, "_last_message", "") or "下载失败，请查看日志"
            if not last.startswith("✗"):
                last = "✗ " + last
            self._status_label.configure(
                text=last, text_color="#f44336", wraplength=460, justify="left",
            )
            self._progress_bar.set(0)
            self._download_btn.configure(state="normal", text="重试")
            self._cancel_btn.configure(text="关闭")
            self._show_failure_extras()

    def _show_failure_extras(self):
        """Add 查看日志 / 配置镜像 buttons after the first failure."""
        if getattr(self, "_extras_shown", False):
            return
        self._extras_shown = True

        extras = ctk.CTkFrame(self, fg_color="transparent")
        extras.pack(pady=(0, 8))

        def open_log():
            try:
                from src.ui.log_window import LogWindow
                LogWindow(self)
            except Exception:
                pass

        def open_mirror_settings():
            try:
                parent = self.master
                # Walk up to a window that owns _show_settings
                while parent is not None and not hasattr(parent, "_show_settings"):
                    parent = parent.master
                if parent is not None:
                    parent._show_settings()
            except Exception:
                pass

        ctk.CTkButton(
            extras, text="📋 查看日志", width=110, height=28,
            fg_color="gray35", hover_color="gray25",
            command=open_log,
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            extras, text="⚙ 配置镜像", width=110, height=28,
            fg_color="gray35", hover_color="gray25",
            command=open_mirror_settings,
        ).pack(side="left", padx=4)

    def _close_and_complete(self):
        self.grab_release()
        self.destroy()
        if self._on_complete:
            self._on_complete()

    def _on_user_close(self):
        self._cancel_flag = True
        self.grab_release()
        self.destroy()
        if self._on_cancel:
            self._on_cancel()
