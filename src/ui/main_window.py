"""
Main Window Module
==================

Main application window integrating all UI components.
"""

import customtkinter as ctk
from typing import Optional
import threading
import numpy as np
from pathlib import Path
import os
import sys
from enum import Enum, auto


class AppState(Enum):
    IDLE = auto()
    INPUT_READY = auto()
    GENERATING = auto()
    GAUSSIANS_READY = auto()
    EXTRACTING = auto()
    DONE = auto()
    ERROR = auto()

# Try to import tkinterdnd2 for drag and drop support
try:
    from tkinterdnd2 import TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False
    print("Warning: tkinterdnd2 not available, drag and drop disabled")

# Import modules
from src.ui.input_panel import InputPanel, MIN_IMAGES
from src.ui.output_panel import OutputPanel
from src.ui.viewer_widget import ViewerWidget
from src.core.gaussian_generator import GaussianGenerator, GaussianCloud
from src.core.hair_strands import HairStrandsExtractor, HairStrandCollection
from src.core.geometry_controller import GeometryController, GeometryBrush, ControlMode
from src.core.batch_processor import BatchProcessor
from src.rendering.viewer_3d import ViewMode
from src.config import settings_manager


# Create a custom CTk class that supports drag and drop
class CTkDnD(ctk.CTk, TkinterDnD.DnDWrapper if HAS_DND else object):
    """CTk window with drag and drop support."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if HAS_DND:
            self.TkdndVersion = TkinterDnD._require(self)


class MainWindow(CTkDnD if HAS_DND else ctk.CTk):
    """
    Main application window.
    
    Layout:
    - Left: Input panel (image/video upload)
    - Center: 3D viewer (Gaussian/curves preview)
    - Right: Output panel (export controls)
    """
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        
        # Window configuration
        self.title("GaussianHairCube - Hair Reconstruction")
        self.geometry("1400x900")
        self.minsize(1000, 600)
        
        # Set window icon
        self._set_window_icon()
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Load persisted settings
        self.settings = settings_manager.load_settings()

        # Initialize processors
        self.gaussian_generator = GaussianGenerator()
        self.strand_extractor = HairStrandsExtractor()
        self.geometry_controller = GeometryController()
        self.batch_processor = BatchProcessor(self.gaussian_generator, self.strand_extractor)

        # State
        self.current_cloud: Optional[GaussianCloud] = None
        self.current_strands: Optional[HairStrandCollection] = None
        self.app_state: AppState = AppState.IDLE
        self._edit_mode: bool = False

        # Window close protection
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Configure grid
        self.grid_columnconfigure(0, weight=0, minsize=300)  # Input panel
        self.grid_columnconfigure(1, weight=1)  # Viewer
        self.grid_columnconfigure(2, weight=0, minsize=300)  # Output panel
        self.grid_rowconfigure(0, weight=0)  # Header
        self.grid_rowconfigure(1, weight=1)  # Main content
        self.grid_rowconfigure(2, weight=0)  # Footer
        self.grid_rowconfigure(3, weight=0)  # Model-status banner (first-launch)
        
        self._create_widgets()

        # First-launch model status banner (does not block — shows after main loop starts)
        self.after(800, self._maybe_show_model_banner)

    def _maybe_show_model_banner(self):
        """If AI models are not cached on first launch, show a non-modal banner
        with a one-click download CTA."""
        try:
            from src.core.model_manager import get_models_to_download
            missing = get_models_to_download()
        except Exception:
            return
        if not missing:
            return

        total_mb = sum(m.get("approx_size_mb", 0) for m in missing)
        banner = ctk.CTkFrame(self, fg_color="#5c3a00", corner_radius=0, height=44)
        banner.grid(row=3, column=0, columnspan=3, sticky="ew")
        banner.grid_columnconfigure(0, weight=1)
        banner.grid_propagate(False)

        msg = (
            f"⚠️  AI 模型未下载（共 ~{total_mb} MB）— 处理前需要先下载，"
            "或在 Settings → AI 模型 中配置镜像源"
        )
        ctk.CTkLabel(
            banner, text=msg, font=ctk.CTkFont(size=12),
            text_color="white", anchor="w",
        ).grid(row=0, column=0, padx=14, pady=8, sticky="w")

        def _open_and_close():
            banner.destroy()
            self._show_settings()

        ctk.CTkButton(
            banner, text="⬇ 立即下载", command=_open_and_close,
            height=28, width=110,
            fg_color="#ff9800", hover_color="#f57c00",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=1, padx=(8, 4), pady=8)

        ctk.CTkButton(
            banner, text="✕", command=banner.destroy,
            height=28, width=34,
            fg_color="transparent", hover_color="#7c4d00",
        ).grid(row=0, column=2, padx=(0, 8), pady=8)

    def _create_widgets(self):
        """Create main window widgets."""
        # Header
        self._create_header()
        
        # Input panel (left)
        self.input_panel = InputPanel(
            self,
            on_input_loaded=self._on_input_loaded
        )
        self.input_panel.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # Viewer (center)
        self.viewer_frame = ctk.CTkFrame(self)
        self.viewer_frame.grid(row=1, column=1, padx=5, pady=10, sticky="nsew")
        self.viewer_frame.grid_columnconfigure(0, weight=1)
        self.viewer_frame.grid_rowconfigure(0, weight=1)
        
        self.viewer = ViewerWidget(
            self.viewer_frame,
            width=800,
            height=600
        )
        self.viewer.grid(row=0, column=0, sticky="nsew")
        
        # Processing controls below viewer
        self._create_processing_controls()
        
        # Output panel (right)
        self.output_panel = OutputPanel(
            self,
            on_export_complete=self._on_export_complete,
            on_project_loaded=self._on_project_loaded,
        )
        self.output_panel.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")
        
        # Footer
        self._create_footer()
    
    def _create_header(self):
        """Create header section."""
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent", height=60)
        self.header_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 0), sticky="ew")
        self.header_frame.grid_columnconfigure(1, weight=1)
        
        # Logo/Title
        self.logo_label = ctk.CTkLabel(
            self.header_frame,
            text="💇 GaussianHairCube",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        # Subtitle
        self.subtitle_label = ctk.CTkLabel(
            self.header_frame,
            text="Hair Reconstruction with Strand-Aligned 3D Gaussians",
            font=ctk.CTkFont(size=12),
            text_color="gray60"
        )
        self.subtitle_label.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Batch processing button
        self.batch_btn = ctk.CTkButton(
            self.header_frame,
            text="📋 批量",
            width=72,
            height=36,
            font=ctk.CTkFont(size=12),
            fg_color="gray30",
            hover_color="gray20",
            command=self._show_batch_dialog,
        )
        self.batch_btn.grid(row=0, column=2, padx=(0, 4), pady=10)

        # Settings button
        self.settings_btn = ctk.CTkButton(
            self.header_frame,
            text="⚙️",
            width=40,
            height=40,
            font=ctk.CTkFont(size=18),
            fg_color="transparent",
            hover_color="gray30",
            command=self._show_settings
        )
        self.settings_btn.grid(row=0, column=3, padx=10, pady=10)
    
    def _create_processing_controls(self):
        """Create processing control buttons."""
        self.controls_frame = ctk.CTkFrame(self.viewer_frame, fg_color="transparent")
        self.controls_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.controls_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Generate Gaussians button
        self.generate_btn = ctk.CTkButton(
            self.controls_frame,
            text="🔮 Generate Gaussians",
            command=self._generate_gaussians,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            state="disabled"
        )
        self.generate_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Extract Curves button
        self.extract_btn = ctk.CTkButton(
            self.controls_frame,
            text="〰️ Extract Curves",
            command=self._extract_curves,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            state="disabled"
        )
        self.extract_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Auto Process button
        self.auto_btn = ctk.CTkButton(
            self.controls_frame,
            text="⚡ Auto Process",
            command=self._auto_process,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2d7d32",
            hover_color="#1b5e20",
            state="disabled"
        )
        self.auto_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        # Edit Gaussians button
        self.edit_btn = ctk.CTkButton(
            self.controls_frame,
            text="🖌️ Edit",
            command=self._toggle_edit_mode,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="gray40",
            hover_color="gray30",
            state="disabled"
        )
        self.edit_btn.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        # Geometry edit panel (hidden by default)
        self._create_geometry_edit_panel()
        
        # Progress frame
        self.progress_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        self.progress_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        self.progress_frame.grid_columnconfigure(0, weight=1)
        self.progress_frame.grid_remove()  # Hidden initially

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(5, 0))
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="",
            font=ctk.CTkFont(size=11)
        )
        self.progress_label.grid(row=1, column=0, pady=(2, 0))

        # Cancel button (shown only while processing)
        self.cancel_btn = ctk.CTkButton(
            self.progress_frame,
            text="Cancel",
            width=80,
            height=28,
            fg_color="#b71c1c",
            hover_color="#7f0000",
            font=ctk.CTkFont(size=11),
            command=self._cancel_processing
        )
        self.cancel_btn.grid(row=0, column=1, padx=(8, 0), pady=(5, 0))
        self.cancel_btn.grid_remove()  # Hidden initially

    def _create_geometry_edit_panel(self):
        """Create the geometry brush editing panel (hidden by default)."""
        self.geo_frame = ctk.CTkFrame(self.viewer_frame, fg_color="gray15", corner_radius=8)
        self.geo_frame.grid(row=3, column=0, padx=10, pady=(0, 6), sticky="ew")
        self.geo_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        self.geo_frame.grid_remove()  # Hidden initially

        # --- Row 0: Brush mode ---
        ctk.CTkLabel(self.geo_frame, text="笔刷模式:", font=ctk.CTkFont(size=11)).grid(
            row=0, column=0, padx=(8, 4), pady=6, sticky="w")

        self._brush_mode = ctk.StringVar(value="DENSITY")
        modes = [("密度", "DENSITY"), ("朝向", "ORIENTATION"), ("缩放", "SCALE"), ("颜色", "COLOR")]
        for col, (label, val) in enumerate(modes):
            ctk.CTkRadioButton(
                self.geo_frame, text=label, variable=self._brush_mode, value=val,
                font=ctk.CTkFont(size=11), width=60
            ).grid(row=0, column=col + 1, padx=4, pady=6)

        # --- Row 1: Radius ---
        ctk.CTkLabel(self.geo_frame, text="半径:", font=ctk.CTkFont(size=11)).grid(
            row=1, column=0, padx=(8, 4), pady=4, sticky="w")
        self._brush_radius_var = ctk.DoubleVar(value=0.15)
        self._radius_slider = ctk.CTkSlider(
            self.geo_frame, from_=0.01, to=0.5, variable=self._brush_radius_var,
            command=lambda v: self._radius_label.configure(text=f"{v:.2f}")
        )
        self._radius_slider.grid(row=1, column=1, columnspan=3, padx=4, pady=4, sticky="ew")
        self._radius_label = ctk.CTkLabel(self.geo_frame, text="0.15", font=ctk.CTkFont(size=11), width=36)
        self._radius_label.grid(row=1, column=4, padx=(0, 8))

        # --- Row 2: Strength ---
        ctk.CTkLabel(self.geo_frame, text="强度:", font=ctk.CTkFont(size=11)).grid(
            row=2, column=0, padx=(8, 4), pady=4, sticky="w")
        self._brush_strength_var = ctk.DoubleVar(value=0.3)
        self._strength_slider = ctk.CTkSlider(
            self.geo_frame, from_=0.0, to=1.0, variable=self._brush_strength_var,
            command=lambda v: self._strength_label.configure(text=f"{v:.2f}")
        )
        self._strength_slider.grid(row=2, column=1, columnspan=3, padx=4, pady=4, sticky="ew")
        self._strength_label = ctk.CTkLabel(self.geo_frame, text="0.30", font=ctk.CTkFont(size=11), width=36)
        self._strength_label.grid(row=2, column=4, padx=(0, 8))

        # --- Row 3: Action buttons ---
        btn_row = ctk.CTkFrame(self.geo_frame, fg_color="transparent")
        btn_row.grid(row=3, column=0, columnspan=5, padx=8, pady=(4, 8), sticky="ew")
        btn_row.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkButton(btn_row, text="↩ 撤销", command=self._geo_undo,
                      fg_color="gray40", hover_color="gray30",
                      height=30, font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=4, sticky="ew")
        ctk.CTkButton(btn_row, text="🔄 平滑", command=self._geo_smooth,
                      fg_color="gray40", hover_color="gray30",
                      height=30, font=ctk.CTkFont(size=12)).grid(row=0, column=1, padx=4, sticky="ew")
        ctk.CTkButton(btn_row, text="✅ 完成编辑", command=self._exit_edit_mode,
                      fg_color="#1565c0", hover_color="#0d47a1",
                      height=30, font=ctk.CTkFont(size=12)).grid(row=0, column=2, padx=4, sticky="ew")

        self._geo_status = ctk.CTkLabel(
            self.geo_frame, text="点击视口中的高斯点云以应用笔刷",
            font=ctk.CTkFont(size=10), text_color="gray60"
        )
        self._geo_status.grid(row=4, column=0, columnspan=5, padx=8, pady=(0, 6))

    def _toggle_edit_mode(self):
        """Enter or exit geometry edit mode."""
        if self._edit_mode:
            self._exit_edit_mode()
        else:
            self._enter_edit_mode()

    def _enter_edit_mode(self):
        """Switch to geometry brush editing mode."""
        if self.current_cloud is None:
            return
        self._edit_mode = True
        self.geometry_controller.set_cloud(self.current_cloud)
        self.edit_btn.configure(text="🖌️ 编辑中…", fg_color="#e65100", hover_color="#bf360c")
        self.geo_frame.grid()
        self.viewer.set_brush_callback(self._on_brush_click)
        self._apply_state(AppState.GAUSSIANS_READY)  # keep other buttons disabled-ish
        self.status_label.configure(text="编辑模式：在视口点击/拖动应用笔刷")

    def _exit_edit_mode(self):
        """Leave geometry brush editing mode and sync updated cloud back."""
        self._edit_mode = False
        updated = self.geometry_controller.get_cloud()
        if updated is not None:
            self.current_cloud = updated
            self.viewer.set_gaussian_data(updated)
        self.viewer.clear_brush_callback()
        self.geo_frame.grid_remove()
        self.edit_btn.configure(text="🖌️ Edit", fg_color="gray40", hover_color="gray30")
        self.status_label.configure(text=f"高斯点云已更新：{self.current_cloud.num_splats} 个点")

    def _on_brush_click(self, sx: int, sy: int):
        """Called when user clicks/drags in viewport during edit mode."""
        pos = self.viewer.viewer.pick_nearest_gaussian(float(sx), float(sy))
        if pos is None:
            self._geo_status.configure(text="未拾取到高斯点 —— 尝试点击点云内部")
            return
        mode_map = {
            "DENSITY": ControlMode.DENSITY,
            "ORIENTATION": ControlMode.ORIENTATION,
            "SCALE": ControlMode.SCALE,
            "COLOR": ControlMode.COLOR,
        }
        mode = mode_map.get(self._brush_mode.get(), ControlMode.DENSITY)
        brush = GeometryBrush(
            center=pos,
            radius=float(self._brush_radius_var.get()),
            falloff=0.5,
            strength=float(self._brush_strength_var.get()),
            mode=mode,
        )
        changed = self.geometry_controller.apply_brush(brush)
        if changed:
            updated = self.geometry_controller.get_cloud()
            self.viewer.viewer.set_gaussian_data(
                updated.get_positions(), updated.get_colors()
            )
            self.viewer._render()
            stats = self.geometry_controller.get_statistics()
            self._geo_status.configure(
                text=f"笔刷已应用 | {stats.get('num_splats', 0)} 个高斯点"
            )

    def _geo_undo(self):
        """Undo last brush stroke."""
        if self.geometry_controller.undo():
            updated = self.geometry_controller.get_cloud()
            self.viewer.viewer.set_gaussian_data(updated.get_positions(), updated.get_colors())
            self.viewer._render()
            self._geo_status.configure(text="已撤销上一步操作")
        else:
            self._geo_status.configure(text="没有可撤销的操作")

    def _geo_smooth(self):
        """Apply global smoothing to the entire Gaussian cloud."""
        if self.geometry_controller.current_cloud is None:
            return
        cloud = self.geometry_controller.current_cloud
        center = cloud.bounds_min + (cloud.bounds_max - cloud.bounds_min) * 0.5
        radius = float(np.linalg.norm(cloud.bounds_max - cloud.bounds_min) * 0.8)
        self.geometry_controller.smooth_region(center, radius, iterations=2)
        updated = self.geometry_controller.get_cloud()
        self.viewer.viewer.set_gaussian_data(updated.get_positions(), updated.get_colors())
        self.viewer._render()
        self._geo_status.configure(text="全局平滑已完成")

    def _create_footer(self):
        """Create footer section."""
        self.footer_frame = ctk.CTkFrame(self, fg_color="transparent", height=30)
        self.footer_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=(0, 5), sticky="ew")
        self.footer_frame.grid_columnconfigure(1, weight=1)
        
        # Status
        self.status_label = ctk.CTkLabel(
            self.footer_frame,
            text="Ready",
            font=ctk.CTkFont(size=11),
            text_color="gray60"
        )
        self.status_label.grid(row=0, column=0, padx=10)
        
        # Version
        self.version_label = ctk.CTkLabel(
            self.footer_frame,
            text="v1.0.0 | Based on GaussianHaircut",
            font=ctk.CTkFont(size=10),
            text_color="gray50"
        )
        self.version_label.grid(row=0, column=2, padx=10)
    
    def _on_input_loaded(self, data: dict):
        """Handle input loaded event."""
        images = data.get('images', [])
        n = len(images)

        if n == 0:
            self.status_label.configure(text="No images loaded")
            self._apply_state(AppState.IDLE)
        elif n < MIN_IMAGES:
            self.status_label.configure(text=f"{n} image{'s' if n > 1 else ''} loaded — need {MIN_IMAGES}")
            self._apply_state(AppState.IDLE)
        else:
            self.status_label.configure(text=f"{n} images ready")
            self._apply_state(AppState.INPUT_READY)
    
    def _check_and_download_models(self, on_ready):
        """
        Checks if ML models are cached locally.  If not, shows a download dialog.
        Calls on_ready() on the main thread once models are available (cached or
        just downloaded).  If the user cancels the dialog the app returns to
        INPUT_READY state.
        """
        try:
            from src.core.model_manager import get_models_to_download
            missing = get_models_to_download()
        except Exception:
            # If model_manager fails for any reason, proceed without the dialog
            on_ready()
            return

        if not missing:
            on_ready()
            return

        # Show download dialog — on_complete fires on the main thread via after()
        from src.ui.model_download_dialog import ModelDownloadDialog

        ModelDownloadDialog(
            parent=self,
            models_to_download=missing,
            on_complete=on_ready,
            on_cancel=lambda: self._apply_state(AppState.INPUT_READY),
        )

    def _generate_gaussians(self):
        """Generate Gaussian splats from input images."""
        if self.app_state in (AppState.GENERATING, AppState.EXTRACTING):
            return

        images = self.input_panel.get_images()
        if len(images) < MIN_IMAGES:
            return

        self._apply_state(AppState.GENERATING)
        self.status_label.configure(text="Generating Gaussians...")

        def _start_processing():
            """Called after models are confirmed available."""
            def generate():
                try:
                    def callback(progress, message):
                        self.after(0, lambda: self._update_progress(progress, message))

                    cloud = self.gaussian_generator.generate_from_images(images, callback=callback)
                    self.after(0, lambda: self._on_gaussians_generated(cloud))

                except Exception as e:
                    self.after(0, lambda: self._on_processing_error(str(e)))

            thread = threading.Thread(target=generate, daemon=True)
            thread.start()

        self._check_and_download_models(on_ready=_start_processing)
    
    def _extract_curves(self):
        """Extract hair curves from Gaussians."""
        if self.app_state in (AppState.GENERATING, AppState.EXTRACTING) or self.current_cloud is None:
            return

        self._apply_state(AppState.EXTRACTING)
        self.status_label.configure(text="Extracting curves...")

        def extract():
            try:
                def callback(progress, message):
                    self.after(0, lambda: self._update_progress(progress, message))
                
                strands = self.strand_extractor.extract(self.current_cloud, callback)
                
                self.after(0, lambda: self._on_curves_extracted(strands))
                
            except Exception as e:
                self.after(0, lambda: self._on_processing_error(str(e)))
        
        thread = threading.Thread(target=extract, daemon=True)
        thread.start()
    
    def _auto_process(self):
        """Automatically generate Gaussians and extract curves."""
        if self.app_state in (AppState.GENERATING, AppState.EXTRACTING):
            return

        images = self.input_panel.get_images()
        if len(images) < MIN_IMAGES:
            return

        self._apply_state(AppState.GENERATING)
        self.status_label.configure(text="Auto processing...")

        def _start_auto():
            """Called after models are confirmed available."""
            def process():
                try:
                    def callback(progress, message):
                        self.after(0, lambda: self._update_progress(progress * 0.6, message))

                    cloud = self.gaussian_generator.generate_from_images(images, callback=callback)

                    # Post intermediate viewer update (fire-and-forget), then continue in same thread
                    self.after(0, lambda c=cloud: self.viewer.set_gaussian_data(c))

                    def callback2(progress, message):
                        self.after(0, lambda: self._update_progress(0.6 + progress * 0.4, message))

                    strands = self.strand_extractor.extract(cloud, callback2)

                    self.after(0, lambda c=cloud, s=strands: self._on_auto_complete(c, s))

                except Exception as e:
                    self.after(0, lambda: self._on_processing_error(str(e)))

            thread = threading.Thread(target=process, daemon=True)
            thread.start()

        self._check_and_download_models(on_ready=_start_auto)

    def _on_auto_complete(self, cloud: GaussianCloud, strands: HairStrandCollection):
        """Handle completion of auto-process (called from main thread)."""
        self.current_cloud = cloud
        self.current_strands = strands

        # Update viewer and output panel
        self.viewer.set_gaussian_data(cloud)
        self.viewer.set_curve_data(strands)
        self.viewer._set_view_mode(ViewMode.CURVES)

        self.output_panel.set_gaussian_data(cloud)
        self.output_panel.set_strand_data(strands)

        self._apply_state(AppState.DONE)
        self.edit_btn.configure(state="normal")
        self.status_label.configure(
            text=f"Generated {cloud.num_splats} splats, {strands.num_strands} strands"
        )
    
    def _on_gaussians_generated(self, cloud: GaussianCloud, continue_processing: bool = False):
        """Handle Gaussians generated."""
        self.current_cloud = cloud

        # Update viewer
        self.viewer.set_gaussian_data(cloud)

        # Update output panel
        self.output_panel.set_gaussian_data(cloud)

        # Update status
        if not continue_processing:
            self._apply_state(AppState.GAUSSIANS_READY)

        self.extract_btn.configure(state="normal")
        self.edit_btn.configure(state="normal")

        if cloud.num_splats < 100:
            self.status_label.configure(
                text=f"警告：仅生成 {cloud.num_splats} 个高斯点（建议 ≥1000）"
            )
        else:
            self.status_label.configure(text=f"生成完成：{cloud.num_splats} 个高斯点")
    
    def _on_curves_extracted(self, strands: HairStrandCollection):
        """Handle curves extracted."""
        self.current_strands = strands

        self.output_panel.set_strand_data(strands)
        self._apply_state(AppState.DONE)

        if strands.num_strands == 0:
            # Friendly guidance instead of silent empty result
            self.status_label.configure(
                text="未提取到发丝 — 尝试调整 Settings 中的最小发丝长度或提取方法"
            )
            try:
                from CTkMessagebox import CTkMessagebox
                CTkMessagebox(
                    master=self,
                    title="发丝提取结果为空",
                    message="未提取到任何发丝曲线。\n\n建议：\n"
                            "• 降低 Settings → Min strand length\n"
                            "• 切换提取方法为 flow_field\n"
                            "• 确认高斯点云包含足够多的点",
                    icon="warning",
                )
            except Exception:
                pass
        else:
            self.viewer.set_curve_data(strands)
            self.viewer._set_view_mode(ViewMode.CURVES)
            self.status_label.configure(
                text=f"提取完成：{strands.num_strands} 条发丝，{strands.total_points} 个点"
            )
    
    def _on_processing_error(self, error: str):
        """Handle processing error."""
        self._apply_state(AppState.ERROR)
        self.status_label.configure(text=f"Error: {error}")
        try:
            from CTkMessagebox import CTkMessagebox
            CTkMessagebox(master=self, title="处理失败", message=str(error), icon="cancel")
        except Exception:
            pass  # CTkMessagebox not installed; status label already updated
    
    def _on_project_loaded(self, cloud, strands):
        """Restore viewer and state after a .ghc project is loaded."""
        if cloud is not None:
            self.current_cloud = cloud
            self.viewer.set_gaussian_data(cloud)
            self.edit_btn.configure(state="normal")

        if strands is not None:
            self.current_strands = strands
            self.viewer.set_curve_data(strands)

        if cloud is not None and strands is not None:
            self.viewer._set_view_mode(ViewMode.CURVES)
        elif cloud is not None:
            self.viewer._set_view_mode(ViewMode.GAUSSIANS)

        self._apply_state(AppState.DONE)
        n_s = cloud.num_splats if cloud else 0
        n_c = strands.num_strands if strands else 0
        self.status_label.configure(text=f"Project loaded — {n_s} splats, {n_c} strands")

    def _on_export_complete(self, success: bool, filepath: str):
        """Handle export complete."""
        if success:
            filename = Path(filepath).name
            self.status_label.configure(text=f"Exported: {filename}")
        else:
            self.status_label.configure(text="Export failed")
    
    def _show_progress(self):
        """Show progress bar."""
        self.progress_frame.grid()
        self.progress_bar.set(0)
        self.progress_label.configure(text="")
    
    def _hide_progress(self):
        """Hide progress bar."""
        self.progress_frame.grid_remove()
    
    def _update_progress(self, progress: float, message: str):
        """Update progress bar."""
        self.progress_bar.set(progress)
        self.progress_label.configure(text=message)
    
    def _set_buttons_state(self, state: str):
        """Set state of processing buttons."""
        if state == "disabled":
            self.generate_btn.configure(state="disabled")
            self.extract_btn.configure(state="disabled")
            self.auto_btn.configure(state="disabled")
            self.edit_btn.configure(state="disabled")
        else:
            # Only enable if we have enough images
            if self.input_panel.is_ready():
                self.generate_btn.configure(state="normal")
                self.auto_btn.configure(state="normal")

            if self.current_cloud is not None:
                self.extract_btn.configure(state="normal")
                self.edit_btn.configure(state="normal")

    def _apply_state(self, state: AppState):
        """Apply UI changes consistently based on the new AppState."""
        self.app_state = state

        if state == AppState.IDLE:
            self._hide_progress()
            self._set_buttons_state("normal")
            if hasattr(self, 'cancel_btn'):
                self.cancel_btn.grid_remove()

        elif state == AppState.INPUT_READY:
            self._hide_progress()
            self._set_buttons_state("normal")
            if hasattr(self, 'cancel_btn'):
                self.cancel_btn.grid_remove()

        elif state in (AppState.GENERATING, AppState.EXTRACTING):
            self._show_progress()
            self._set_buttons_state("disabled")
            if hasattr(self, 'cancel_btn'):
                self.cancel_btn.grid()
            if hasattr(self, 'viewer'):
                self.viewer.show_loading("Processing...")

        elif state == AppState.GAUSSIANS_READY:
            self._hide_progress()
            self._set_buttons_state("normal")
            if hasattr(self, 'cancel_btn'):
                self.cancel_btn.grid_remove()
            if hasattr(self, 'viewer'):
                self.viewer.hide_loading()

        elif state == AppState.DONE:
            self._hide_progress()
            self._set_buttons_state("normal")
            if hasattr(self, 'cancel_btn'):
                self.cancel_btn.grid_remove()
            if hasattr(self, 'viewer'):
                self.viewer.hide_loading()

        elif state == AppState.ERROR:
            self._hide_progress()
            self._set_buttons_state("normal")
            if hasattr(self, 'cancel_btn'):
                self.cancel_btn.grid_remove()
            if hasattr(self, 'viewer'):
                self.viewer.hide_loading()

    def _on_closing(self):
        """Handle window close — cancel any active work first."""
        self.gaussian_generator.cancel()
        if hasattr(self.strand_extractor, 'cancel'):
            self.strand_extractor.cancel()
        self.destroy()

    def _cancel_processing(self):
        """Cancel ongoing Gaussian generation or strand extraction."""
        self.gaussian_generator.cancel()
        if hasattr(self.strand_extractor, 'cancel'):
            self.strand_extractor.cancel()
        self.status_label.configure(text="Cancelled")
    
    def _set_window_icon(self):
        """Set the window icon for both title bar and taskbar."""
        try:
            from PIL import Image, ImageTk
            
            # Set Windows AppUserModelID for proper taskbar icon
            try:
                import ctypes
                myappid = 'gaussianhaircut.gaussianhaircube.app.1.0'
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except Exception:
                pass  # Not on Windows or failed
            
            # Determine base path (for PyInstaller or normal execution)
            if getattr(sys, 'frozen', False):
                # Running as compiled executable
                base_path = sys._MEIPASS
            else:
                # Running as script
                base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Try to load .ico file
            icon_ico_path = os.path.join(base_path, 'assets', 'icon.ico')
            icon_png_path = os.path.join(base_path, 'assets', 'icon.png')
            
            # Set iconbitmap for Windows title bar
            if os.path.exists(icon_ico_path):
                self.iconbitmap(default=icon_ico_path)
                self.iconbitmap(icon_ico_path)
            
            # Also set iconphoto for taskbar (more reliable on some systems)
            if os.path.exists(icon_png_path):
                icon_image = Image.open(icon_png_path)
                # Create multiple sizes for better display
                icon_sizes = []
                for size in [16, 32, 48, 64, 128, 256]:
                    if icon_image.width >= size and icon_image.height >= size:
                        resized = icon_image.resize((size, size), Image.Resampling.LANCZOS)
                        icon_sizes.append(ImageTk.PhotoImage(resized))
                
                if icon_sizes:
                    self.iconphoto(True, *icon_sizes)
                    self._icon_photos = icon_sizes  # Keep references
            elif os.path.exists(icon_ico_path):
                # Try to use .ico as fallback for iconphoto
                icon_image = Image.open(icon_ico_path)
                icon_photo = ImageTk.PhotoImage(icon_image)
                self.iconphoto(True, icon_photo)
                self._icon_photo = icon_photo
                
        except Exception as e:
            print(f"Warning: Could not set window icon: {e}")
    
    def _show_settings(self):
        """Show settings dialog."""
        settings_window = SettingsDialog(self)
        settings_window.grab_set()

    def _show_batch_dialog(self):
        """Open the batch processing queue dialog."""
        from src.ui.batch_dialog import BatchDialog
        dlg = BatchDialog(self, self.batch_processor)
        dlg.grab_set()


class SettingsDialog(ctk.CTkToplevel):
    """Tabbed settings dialog with per-tab Reset and a one-click model download."""

    _HELP = {
        'iterations':    "GPU 高斯优化的迭代次数；越大质量越好但越慢（默认 1000）",
        'points':        "每条发丝重采样为多少个控制点（默认 32）",
        'num_strands':   "提取的最大发丝数量上限（默认 2000）",
        'min_length':    "短于此长度的发丝会被丢弃；单位与点云空间相同（默认 0.05）",
        'method':        "clustering：图遍历，速度快；flow_field：RK4 流场积分，覆盖率高",
        'theme':         "Dark / Light / 跟随系统",
        'mirror':        "国内访问慢时可填 https://hf-mirror.com，留空使用官方源",
    }

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent

        self.title("Settings")
        self.geometry("560x600")
        self.minsize(520, 540)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._create_widgets()
        self._load_from_settings()

    def _create_widgets(self):
        ctk.CTkLabel(
            self, text="⚙️ 设置 / Settings",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=(16, 6), sticky="w")

        self.tabs = ctk.CTkTabview(self, anchor="nw")
        self.tabs.grid(row=1, column=0, padx=14, pady=4, sticky="nsew")
        self.tabs.add("生成")
        self.tabs.add("发丝")
        self.tabs.add("AI 模型")
        self.tabs.add("外观")

        self._build_generation_tab(self.tabs.tab("生成"))
        self._build_strands_tab(self.tabs.tab("发丝"))
        self._build_models_tab(self.tabs.tab("AI 模型"))
        self._build_appearance_tab(self.tabs.tab("外观"))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=2, column=0, padx=14, pady=(6, 14), sticky="ew")
        btn_row.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(
            btn_row, text="↺ 恢复默认", command=self._reset_current_tab,
            fg_color="gray30", hover_color="gray20", height=34, width=120,
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            btn_row, text="取消", command=self.destroy,
            fg_color="gray40", hover_color="gray30", height=34, width=90,
        ).grid(row=0, column=1, padx=(6, 6))

        ctk.CTkButton(
            btn_row, text="✓ 保存", command=self._save_settings,
            fg_color="#1565c0", hover_color="#0d47a1",
            height=34, width=110, font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=2)

    def _add_field(self, parent, row, label, key_help, widget):
        ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=12)).grid(
            row=row, column=0, padx=(8, 8), pady=(8, 0), sticky="w")
        widget.grid(row=row, column=1, padx=(0, 8), pady=(8, 0), sticky="e")
        ctk.CTkLabel(
            parent, text=self._HELP.get(key_help, ""),
            font=ctk.CTkFont(size=10), text_color="gray55", justify="left",
            wraplength=480,
        ).grid(row=row + 1, column=0, columnspan=2, padx=8, pady=(0, 6), sticky="w")

    def _build_generation_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        self.iter_entry = ctk.CTkEntry(tab, width=120)
        self._add_field(tab, 0, "迭代次数 (GPU only):", 'iterations', self.iter_entry)

    def _build_strands_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        self.pts_entry = ctk.CTkEntry(tab, width=120)
        self._add_field(tab, 0, "每条发丝点数:", 'points', self.pts_entry)
        self.num_entry = ctk.CTkEntry(tab, width=120)
        self._add_field(tab, 2, "最大发丝数:", 'num_strands', self.num_entry)
        self.minlen_entry = ctk.CTkEntry(tab, width=120)
        self._add_field(tab, 4, "最小发丝长度:", 'min_length', self.minlen_entry)
        self.method_menu = ctk.CTkOptionMenu(tab, values=["clustering", "flow_field"], width=140)
        self._add_field(tab, 6, "提取方法:", 'method', self.method_menu)

    def _build_models_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)

        status_card = ctk.CTkFrame(tab, fg_color="gray18", corner_radius=8)
        status_card.grid(row=0, column=0, padx=8, pady=(10, 6), sticky="ew")
        self._models_status_label = ctk.CTkLabel(
            status_card, text="正在检查本地缓存…",
            font=ctk.CTkFont(size=12), justify="left", anchor="w",
        )
        self._models_status_label.pack(padx=12, pady=10, fill="x")

        self.one_click_btn = ctk.CTkButton(
            tab, text="⬇  一键下载所有 AI 模型 (~184 MB)",
            command=self._one_click_download,
            height=48, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2e7d32", hover_color="#1b5e20",
        )
        self.one_click_btn.grid(row=1, column=0, padx=8, pady=(4, 6), sticky="ew")

        ctk.CTkLabel(
            tab, text="模型缓存到 ~/.cache/huggingface/hub，首次下载后无需重复下载。",
            font=ctk.CTkFont(size=10), text_color="gray55",
        ).grid(row=2, column=0, padx=8, pady=(0, 12), sticky="w")

        ctk.CTkLabel(tab, text="HuggingFace 镜像:", font=ctk.CTkFont(size=12)).grid(
            row=3, column=0, padx=8, pady=(8, 0), sticky="w")
        self.mirror_entry = ctk.CTkEntry(tab, placeholder_text="https://hf-mirror.com（推荐国内用户）")
        self.mirror_entry.grid(row=4, column=0, padx=8, pady=(2, 0), sticky="ew")
        ctk.CTkLabel(
            tab, text=self._HELP['mirror'],
            font=ctk.CTkFont(size=10), text_color="gray55",
            wraplength=480, justify="left",
        ).grid(row=5, column=0, padx=8, pady=(2, 10), sticky="w")

        self._refresh_models_status()

    def _build_appearance_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        self.theme_menu = ctk.CTkOptionMenu(
            tab, values=["Dark", "Light", "System"], command=self._change_theme, width=140,
        )
        self._add_field(tab, 0, "主题:", 'theme', self.theme_menu)

    def _load_from_settings(self):
        s = self.parent_window.settings
        self.iter_entry.insert(0, str(s.get('num_iterations', 1000)))
        self.pts_entry.insert(0, str(s.get('points_per_strand', 32)))
        self.num_entry.insert(0, str(s.get('num_strands', 2000)))
        self.minlen_entry.insert(0, str(s.get('min_strand_length', 0.05)))
        self.method_menu.set(s.get('extraction_method', 'clustering'))
        self.theme_menu.set(s.get('theme', 'dark').capitalize())
        if s.get('hf_endpoint', ''):
            self.mirror_entry.insert(0, s['hf_endpoint'])

    def _reset_current_tab(self):
        from src.config.settings_manager import DEFAULT_SETTINGS
        d = DEFAULT_SETTINGS
        current = self.tabs.get()
        if current == "生成":
            self.iter_entry.delete(0, "end")
            self.iter_entry.insert(0, str(d['num_iterations']))
        elif current == "发丝":
            self.pts_entry.delete(0, "end"); self.pts_entry.insert(0, str(d['points_per_strand']))
            self.num_entry.delete(0, "end"); self.num_entry.insert(0, str(d['num_strands']))
            self.minlen_entry.delete(0, "end"); self.minlen_entry.insert(0, str(d['min_strand_length']))
            self.method_menu.set(d['extraction_method'])
        elif current == "AI 模型":
            self.mirror_entry.delete(0, "end")
        elif current == "外观":
            self.theme_menu.set(d['theme'].capitalize())
            self._change_theme(d['theme'].capitalize())

    def _refresh_models_status(self):
        try:
            from src.core.model_manager import get_all_models_status
            models = get_all_models_status()
        except Exception:
            self._models_status_label.configure(
                text="无法读取模型状态（transformers 可能未安装）",
                text_color="#ff9800",
            )
            return
        if not models:
            self._models_status_label.configure(text="(无模型条目)", text_color="gray60")
            return
        lines = []
        all_cached = True
        for m in models:
            mark = "✅ 已缓存" if m.get('cached') else "⬜ 未下载"
            if not m.get('cached'):
                all_cached = False
            lines.append(f"{mark}   {m['display_name']}  (~{m.get('approx_size_mb', '?')} MB)")
        text = "\n".join(lines)
        color = "#4caf50" if all_cached else "#ff9800"
        self._models_status_label.configure(text=text, text_color=color)
        if all_cached:
            self.one_click_btn.configure(
                text="✓ 所有模型已缓存（点击重新下载）",
                fg_color="gray40", hover_color="gray30",
            )

    def _one_click_download(self):
        endpoint = self.mirror_entry.get().strip()
        if endpoint:
            import os
            os.environ['HF_ENDPOINT'] = endpoint
        try:
            from src.core.model_manager import get_models_to_download, get_all_models_status
            missing = get_models_to_download()
            all_models = get_all_models_status()
        except Exception as exc:
            self._models_status_label.configure(
                text=f"模型状态查询失败：{exc}", text_color="#f44336")
            return
        from src.ui.model_download_dialog import ModelDownloadDialog
        ModelDownloadDialog(
            parent=self,
            models_to_download=missing if missing else all_models,
            on_complete=self._refresh_models_status,
            on_cancel=None,
        )

    def _change_theme(self, value):
        ctk.set_appearance_mode(value.lower())

    def _save_settings(self):
        try:
            num_iterations = int(self.iter_entry.get())
            points_per_strand = int(self.pts_entry.get())
            num_strands = int(self.num_entry.get())
            min_strand_length = float(self.minlen_entry.get())
            if num_iterations <= 0 or points_per_strand <= 0 or num_strands <= 0:
                raise ValueError("数值必须为正")
            if min_strand_length < 0:
                raise ValueError("最小发丝长度必须 >= 0")
        except ValueError as exc:
            try:
                from CTkMessagebox import CTkMessagebox
                CTkMessagebox(master=self, title="输入无效",
                              message=f"请检查输入值。\n{exc}", icon="cancel")
            except Exception:
                pass
            return

        parent = self.parent_window
        method = self.method_menu.get()
        if hasattr(parent.gaussian_generator, 'set_parameters'):
            parent.gaussian_generator.set_parameters(num_iterations=num_iterations)
        if hasattr(parent.strand_extractor, 'set_parameters'):
            from src.core.hair_strands import StrandExtractionMethod
            method_enum = (StrandExtractionMethod.FLOW_FIELD if method == 'flow_field'
                           else StrandExtractionMethod.CLUSTERING)
            parent.strand_extractor.set_parameters(
                points_per_strand=points_per_strand,
                num_strands=num_strands,
                min_strand_length=min_strand_length,
                method=method_enum,
            )

        parent.settings['num_iterations'] = num_iterations
        parent.settings['points_per_strand'] = points_per_strand
        parent.settings['num_strands'] = num_strands
        parent.settings['min_strand_length'] = min_strand_length
        parent.settings['extraction_method'] = method
        parent.settings['theme'] = self.theme_menu.get().lower()
        parent.settings['hf_endpoint'] = self.mirror_entry.get().strip()

        try:
            from src.core.model_manager import apply_hf_mirror
            apply_hf_mirror()
        except Exception:
            pass

        settings_manager.save_settings(parent.settings)
        self.destroy()
