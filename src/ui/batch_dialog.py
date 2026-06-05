"""
Batch Processing Dialog
=======================

A Toplevel window for managing a queue of image-set processing jobs.
"""

import threading
from pathlib import Path
from tkinter import filedialog
from typing import Optional

import customtkinter as ctk

from src.core.batch_processor import BatchJob, BatchProcessor, JobStatus


_STATUS_ICON = {
    JobStatus.PENDING:    ("⏳", "gray60"),
    JobStatus.PROCESSING: ("⚙️",  "#ff9800"),
    JobStatus.DONE:       ("✅", "#4caf50"),
    JobStatus.FAILED:     ("❌", "#f44336"),
    JobStatus.CANCELLED:  ("⛔", "gray50"),
}


class BatchDialog(ctk.CTkToplevel):
    """Batch job queue dialog."""

    def __init__(self, parent, processor: BatchProcessor):
        super().__init__(parent)
        self.processor = processor

        self.title("批量处理 / Batch Queue")
        self.geometry("640x560")
        self.resizable(True, True)
        self.minsize(520, 420)

        # Pending "add" state
        self._pending_images: list = []
        self._pending_output: str = ""

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_ui()
        self._refresh_job_list()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Header
        ctk.CTkLabel(
            self, text="批量处理队列 / Batch Queue",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, padx=16, pady=(14, 6), sticky="w")

        # Job list (scrollable)
        self.job_scroll = ctk.CTkScrollableFrame(self, fg_color="gray15", corner_radius=6)
        self.job_scroll.grid(row=1, column=0, padx=14, pady=4, sticky="nsew")
        self.job_scroll.grid_columnconfigure(0, weight=1)

        # ── Add-job panel ────────────────────────────────────────────
        add_frame = ctk.CTkFrame(self, fg_color="gray20", corner_radius=6)
        add_frame.grid(row=2, column=0, padx=14, pady=6, sticky="ew")
        add_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(add_frame, text="添加任务", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, columnspan=3, padx=10, pady=(8, 4), sticky="w")

        # Image selection
        ctk.CTkLabel(add_frame, text="输入图片:", font=ctk.CTkFont(size=11)).grid(
            row=1, column=0, padx=(10, 4), pady=4, sticky="w")
        self.images_label = ctk.CTkLabel(
            add_frame, text="未选择图片", font=ctk.CTkFont(size=11), text_color="gray60",
            anchor="w",
        )
        self.images_label.grid(row=1, column=1, padx=4, pady=4, sticky="ew")
        ctk.CTkButton(
            add_frame, text="选择…", width=70, height=28, command=self._browse_images,
        ).grid(row=1, column=2, padx=(4, 10), pady=4)

        # Output dir
        ctk.CTkLabel(add_frame, text="输出目录:", font=ctk.CTkFont(size=11)).grid(
            row=2, column=0, padx=(10, 4), pady=4, sticky="w")
        self.output_entry = ctk.CTkEntry(
            add_frame, placeholder_text="默认：与第一张图片相同目录"
        )
        self.output_entry.grid(row=2, column=1, padx=4, pady=4, sticky="ew")
        ctk.CTkButton(
            add_frame, text="浏览…", width=70, height=28, command=self._browse_output,
        ).grid(row=2, column=2, padx=(4, 10), pady=4)

        # Auto export
        ctk.CTkLabel(add_frame, text="自动导出:", font=ctk.CTkFont(size=11)).grid(
            row=3, column=0, padx=(10, 4), pady=(4, 10), sticky="w")
        self.export_menu = ctk.CTkOptionMenu(
            add_frame, values=["不导出", "FBX (Maya)", "GLB (Blender)"], width=160,
        )
        self.export_menu.grid(row=3, column=1, padx=4, pady=(4, 10), sticky="w")
        self.export_menu.set("不导出")

        ctk.CTkButton(
            add_frame, text="+ 添加到队列", command=self._add_job,
            fg_color="#1565c0", hover_color="#0d47a1", height=32,
        ).grid(row=3, column=2, padx=(4, 10), pady=(4, 10))

        # ── Bottom controls ──────────────────────────────────────────
        ctrl = ctk.CTkFrame(self, fg_color="transparent")
        ctrl.grid(row=3, column=0, padx=14, pady=(4, 14), sticky="ew")
        ctrl.grid_columnconfigure(2, weight=1)

        self.run_btn = ctk.CTkButton(
            ctrl, text="▶ 开始处理", command=self._run_batch,
            fg_color="#2e7d32", hover_color="#1b5e20", height=36,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.run_btn.grid(row=0, column=0, padx=(0, 6))

        self.cancel_btn = ctk.CTkButton(
            ctrl, text="⬛ 取消", command=self._cancel_batch,
            fg_color="#b71c1c", hover_color="#7f0000", height=36,
            state="disabled",
        )
        self.cancel_btn.grid(row=0, column=1, padx=(0, 6))

        ctk.CTkButton(
            ctrl, text="🗑 清除已完成", command=self._clear_finished,
            fg_color="gray40", hover_color="gray30", height=36,
        ).grid(row=0, column=2, sticky="e")

        # Progress
        self.progress_var = ctk.DoubleVar(value=0.0)
        self.progress_bar = ctk.CTkProgressBar(self, variable=self.progress_var)
        self.progress_bar.grid(row=4, column=0, padx=14, pady=(0, 4), sticky="ew")

        self.progress_label = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(size=10), text_color="gray60",
        )
        self.progress_label.grid(row=5, column=0, padx=14, pady=(0, 8))

    # ------------------------------------------------------------------
    # Job list rendering
    # ------------------------------------------------------------------

    def _refresh_job_list(self):
        for w in self.job_scroll.winfo_children():
            w.destroy()

        jobs = self.processor.jobs
        if not jobs:
            ctk.CTkLabel(
                self.job_scroll, text="队列为空 — 请先添加任务",
                font=ctk.CTkFont(size=12), text_color="gray55",
            ).pack(pady=30)
            return

        for job in jobs:
            self._add_job_card(job)

    def _add_job_card(self, job: BatchJob):
        icon, color = _STATUS_ICON.get(job.status, ("?", "white"))
        card = ctk.CTkFrame(self.job_scroll, fg_color="gray25", corner_radius=6)
        card.pack(fill="x", padx=4, pady=3)
        card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=16), width=28).grid(
            row=0, column=0, padx=(8, 4), pady=6)

        info_text = job.label
        if job.status == JobStatus.DONE:
            info_text += f"  →  {job.num_splats}spl / {job.num_strands}str"
        elif job.status == JobStatus.FAILED:
            info_text += f"  ❌ {job.error[:50] if job.error else ''}"

        ctk.CTkLabel(
            card, text=info_text, font=ctk.CTkFont(size=11),
            text_color=color, anchor="w",
        ).grid(row=0, column=1, padx=4, pady=6, sticky="ew")

        dir_text = str(Path(job.output_dir).name) if job.output_dir else ""
        if dir_text:
            ctk.CTkLabel(
                card, text=f"→ {dir_text}", font=ctk.CTkFont(size=10), text_color="gray55",
            ).grid(row=0, column=2, padx=(0, 8), pady=6)

        if job.status == JobStatus.PENDING:
            ctk.CTkButton(
                card, text="✕", width=22, height=22,
                fg_color="gray40", hover_color="#b71c1c",
                font=ctk.CTkFont(size=10),
                command=lambda jid=job.id: self._remove_job(jid),
            ).grid(row=0, column=3, padx=(0, 8), pady=6)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _browse_images(self):
        paths = filedialog.askopenfilenames(
            title="选择图片（至少3张）",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if paths:
            self._pending_images = list(paths)
            n = len(paths)
            label = f"{n} 张图片" if n > 1 else paths[0]
            if n > 0 and not self.output_entry.get().strip():
                self.output_entry.delete(0, "end")
                self.output_entry.insert(0, str(Path(paths[0]).parent))
            self.images_label.configure(text=label, text_color="white")

    def _browse_output(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, d)

    def _add_job(self):
        if not self._pending_images:
            self.images_label.configure(text="请先选择图片！", text_color="#f44336")
            return
        if len(self._pending_images) < 3:
            self.images_label.configure(
                text=f"只选了 {len(self._pending_images)} 张，需要至少 3 张", text_color="#f44336"
            )
            return

        out_dir = self.output_entry.get().strip()
        if not out_dir:
            out_dir = str(Path(self._pending_images[0]).parent)

        export_choice = self.export_menu.get()
        auto_export = {"FBX (Maya)": "fbx", "GLB (Blender)": "glb"}.get(export_choice)

        self.processor.add_job(self._pending_images, out_dir, auto_export=auto_export)

        # Reset add fields
        self._pending_images = []
        self.images_label.configure(text="未选择图片", text_color="gray60")
        self.output_entry.delete(0, "end")

        self._refresh_job_list()

    def _remove_job(self, job_id: str):
        self.processor.remove_job(job_id)
        self._refresh_job_list()

    def _clear_finished(self):
        self.processor.clear_finished()
        self._refresh_job_list()

    def _run_batch(self):
        if self.processor.is_running:
            return
        if self.processor.pending_count == 0:
            self.progress_label.configure(text="队列中没有待处理的任务")
            return

        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress_var.set(0.0)

        def on_job_update(job: BatchJob):
            self.after(0, self._refresh_job_list)

        def on_progress(job_idx: int, total: int, step: float, msg: str):
            overall = (job_idx + step) / max(total, 1)
            self.after(0, lambda: (
                self.progress_var.set(overall),
                self.progress_label.configure(text=msg),
            ))

        def run():
            self.processor.run_all(on_job_update=on_job_update, on_progress=on_progress)
            self.after(0, self._on_batch_done)

        threading.Thread(target=run, daemon=True).start()

    def _cancel_batch(self):
        self.processor.cancel()

    def _on_batch_done(self):
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        done = sum(1 for j in self.processor.jobs if j.status == JobStatus.DONE)
        failed = sum(1 for j in self.processor.jobs if j.status == JobStatus.FAILED)
        self.progress_label.configure(
            text=f"批处理完成 — {done} 成功，{failed} 失败",
            text_color="#4caf50" if failed == 0 else "#ff9800",
        )
        self._refresh_job_list()
