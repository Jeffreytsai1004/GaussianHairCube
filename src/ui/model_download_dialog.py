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

    def __init__(self, parent, models_to_download: list, on_complete=None, on_cancel=None):
        """
        models_to_download: list of dicts with keys: display_name, repo_id, approx_size_mb
        on_complete: callback() called when download finishes successfully (main thread)
        on_cancel: callback() called when user cancels (main thread)
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
            self._update_progress(0.0, "✗ 下载失败，请检查网络连接后重试")
            self._download_btn.configure(state="normal", text="重试")
            self._cancel_btn.configure(text="关闭")

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
