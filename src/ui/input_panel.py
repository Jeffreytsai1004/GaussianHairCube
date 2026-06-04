"""
Input Panel Module
==================

Multi-image input panel. Accepts at least 3 images for 3D reconstruction.
Supports bulk file selection and drag-and-drop.
"""

import re
from pathlib import Path
from typing import Callable, List, Optional

import customtkinter as ctk
import numpy as np
from PIL import Image
from tkinter import filedialog

try:
    from tkinterdnd2 import DND_FILES
    HAS_DND = True
except ImportError:
    HAS_DND = False

IMAGE_FORMATS = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg *.jpeg"),
    ("BMP", "*.bmp"),
    ("TIFF", "*.tiff"),
    ("WebP", "*.webp"),
    ("All files", "*.*"),
]

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
MIN_IMAGES = 3
_THUMB_SIZE = 78


class InputPanel(ctk.CTkFrame):
    """
    Multi-image input panel.

    Fires on_input_loaded({'type': 'images', 'images': [...], 'paths': [...]})
    every time the image list changes.
    """

    def __init__(self, parent, on_input_loaded: Optional[Callable] = None, **kwargs):
        super().__init__(parent, **kwargs)

        self.on_input_loaded = on_input_loaded

        self._images: List[np.ndarray] = []
        self._paths: List[str] = []
        self._thumb_refs: List = []  # prevent GC of CTkImage objects

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._create_widgets()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _create_widgets(self):
        ctk.CTkLabel(
            self, text="📁 Input Images",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        self.thumb_scroll = ctk.CTkScrollableFrame(self, fg_color="gray20")
        self.thumb_scroll.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.thumb_scroll.grid_columnconfigure((0, 1, 2), weight=1)

        self._placeholder = ctk.CTkLabel(
            self.thumb_scroll,
            text="Drop images here\nor click Add Images",
            font=ctk.CTkFont(size=13),
            text_color="gray60",
        )
        self._placeholder.grid(row=0, column=0, columnspan=3, padx=20, pady=40)

        self._setup_drag_drop()

        ctrl = ctk.CTkFrame(self, fg_color="transparent")
        ctrl.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        ctrl.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(
            ctrl, text="📤 Add Images",
            command=self._browse_files, height=36,
        ).grid(row=0, column=0, padx=(0, 5), pady=5, sticky="ew")

        ctk.CTkButton(
            ctrl, text="🗑️ Clear All",
            command=self._clear_all,
            fg_color="gray40", hover_color="gray30", height=36,
        ).grid(row=0, column=1, padx=(5, 0), pady=5, sticky="ew")

        self.status_label = ctk.CTkLabel(
            self,
            text=f"No images loaded (minimum {MIN_IMAGES} required)",
            font=ctk.CTkFont(size=11),
            text_color="gray60",
        )
        self.status_label.grid(row=3, column=0, padx=10, pady=(0, 5), sticky="w")

    def _setup_drag_drop(self):
        if not HAS_DND:
            return
        try:
            self.thumb_scroll.drop_target_register(DND_FILES)
            self.thumb_scroll.dnd_bind("<<Drop>>", self._on_drop)
            self.thumb_scroll.dnd_bind(
                "<<DragEnter>>", lambda e: self.thumb_scroll.configure(fg_color="gray30")
            )
            self.thumb_scroll.dnd_bind(
                "<<DragLeave>>", lambda e: self.thumb_scroll.configure(fg_color="gray20")
            )
        except Exception as exc:
            print(f"Warning: drag-and-drop setup failed: {exc}")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_drop(self, event):
        self.thumb_scroll.configure(fg_color="gray20")
        raw = event.data or ""
        # Parse space-separated paths; curly-braces wrap paths with spaces on Windows
        paths = re.findall(r"\{([^}]+)\}", raw)
        remainder = re.sub(r"\{[^}]+\}", "", raw).split()
        paths.extend(remainder)

        for p in paths:
            p = p.strip().strip("{}")
            if p and Path(p).exists() and Path(p).suffix.lower() in _IMAGE_EXTS:
                self._load_image(p)

    def _browse_files(self):
        filepaths = filedialog.askopenfilenames(
            title="Select Images (hold Ctrl/Shift for multiple)",
            filetypes=IMAGE_FORMATS,
        )
        for fp in filepaths:
            self._load_image(fp)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_image(self, filepath: str):
        try:
            pil_img = Image.open(filepath).convert("RGB")
            arr = np.array(pil_img)
            idx = len(self._images)
            self._images.append(arr)
            self._paths.append(filepath)
            self._add_thumbnail(pil_img, idx)
            self._update_status()
            self._fire_callback()
        except Exception as exc:
            self.status_label.configure(
                text=f"Error loading {Path(filepath).name}: {exc}",
                text_color="#f44336",
            )

    def _add_thumbnail(self, pil_img: Image.Image, idx: int):
        # Hide placeholder on first image
        try:
            if self._placeholder.winfo_ismapped():
                self._placeholder.grid_remove()
        except Exception:
            pass

        row, col = divmod(idx, 3)

        card = ctk.CTkFrame(self.thumb_scroll, fg_color="gray30", corner_radius=6)
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")

        thumb = pil_img.copy()
        thumb.thumbnail((_THUMB_SIZE, _THUMB_SIZE), Image.Resampling.LANCZOS)
        ctk_thumb = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=(_THUMB_SIZE, _THUMB_SIZE))
        self._thumb_refs.append(ctk_thumb)

        ctk.CTkLabel(card, image=ctk_thumb, text="").pack(padx=4, pady=(4, 0))

        name = Path(self._paths[idx]).name
        display_name = name if len(name) <= 12 else name[:9] + "..."
        ctk.CTkLabel(card, text=display_name, font=ctk.CTkFont(size=9), text_color="gray70").pack()

        ctk.CTkButton(
            card, text="✕", width=22, height=18,
            fg_color="gray50", hover_color="#b71c1c",
            font=ctk.CTkFont(size=9),
            command=lambda i=idx: self._remove_image(i),
        ).pack(pady=(0, 3))

    def _remove_image(self, idx: int):
        if idx >= len(self._images):
            return
        self._images.pop(idx)
        self._paths.pop(idx)
        self._rebuild_thumbnails()
        self._update_status()
        self._fire_callback()

    def _rebuild_thumbnails(self):
        for widget in self.thumb_scroll.winfo_children():
            widget.destroy()
        self._thumb_refs.clear()

        if not self._images:
            self._placeholder = ctk.CTkLabel(
                self.thumb_scroll,
                text="Drop images here\nor click Add Images",
                font=ctk.CTkFont(size=13),
                text_color="gray60",
            )
            self._placeholder.grid(row=0, column=0, columnspan=3, padx=20, pady=40)
            return

        for i, (arr, _) in enumerate(zip(self._images, self._paths)):
            pil_img = Image.fromarray(arr)
            self._add_thumbnail(pil_img, i)

    # ------------------------------------------------------------------
    # Status / callback
    # ------------------------------------------------------------------

    def _update_status(self):
        n = len(self._images)
        if n == 0:
            self.status_label.configure(
                text=f"No images loaded (minimum {MIN_IMAGES} required)",
                text_color="gray60",
            )
        elif n < MIN_IMAGES:
            self.status_label.configure(
                text=f"{n} image{'s' if n > 1 else ''} loaded — add {MIN_IMAGES - n} more to process",
                text_color="#ff9800",
            )
        else:
            self.status_label.configure(
                text=f"{n} images loaded ✓",
                text_color="#4caf50",
            )

    def _fire_callback(self):
        if self.on_input_loaded:
            self.on_input_loaded({
                "type": "images",
                "images": self._images.copy(),
                "paths": self._paths.copy(),
            })

    def _clear_all(self):
        self._images.clear()
        self._paths.clear()
        self._rebuild_thumbnails()
        self._update_status()
        if self.on_input_loaded:
            self.on_input_loaded({"type": "images", "images": [], "paths": []})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_images(self) -> List[np.ndarray]:
        return self._images.copy()

    def get_paths(self) -> List[str]:
        return self._paths.copy()

    def get_image_count(self) -> int:
        return len(self._images)

    def is_ready(self) -> bool:
        return len(self._images) >= MIN_IMAGES

    def get_current_image(self) -> Optional[np.ndarray]:
        """Return the first loaded image, or None."""
        return self._images[0] if self._images else None
