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

        # State
        self.current_cloud: Optional[GaussianCloud] = None
        self.current_strands: Optional[HairStrandCollection] = None
        self.app_state: AppState = AppState.IDLE

        # Window close protection
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Configure grid
        self.grid_columnconfigure(0, weight=0, minsize=300)  # Input panel
        self.grid_columnconfigure(1, weight=1)  # Viewer
        self.grid_columnconfigure(2, weight=0, minsize=300)  # Output panel
        self.grid_rowconfigure(0, weight=0)  # Header
        self.grid_rowconfigure(1, weight=1)  # Main content
        self.grid_rowconfigure(2, weight=0)  # Footer
        
        self._create_widgets()
    
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
            on_export_complete=self._on_export_complete
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
        self.settings_btn.grid(row=0, column=2, padx=10, pady=10)
    
    def _create_processing_controls(self):
        """Create processing control buttons."""
        self.controls_frame = ctk.CTkFrame(self.viewer_frame, fg_color="transparent")
        self.controls_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.controls_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
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
        self.status_label.configure(text=f"Generated {cloud.num_splats} Gaussians")
    
    def _on_curves_extracted(self, strands: HairStrandCollection):
        """Handle curves extracted."""
        self.current_strands = strands

        # Update viewer
        self.viewer.set_curve_data(strands)
        self.viewer._set_view_mode(ViewMode.CURVES)

        # Update output panel
        self.output_panel.set_strand_data(strands)

        # Update status
        self._apply_state(AppState.DONE)
        self.status_label.configure(
            text=f"Extracted {strands.num_strands} strands ({strands.total_points} points)"
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
        else:
            # Only enable if we have enough images
            if self.input_panel.is_ready():
                self.generate_btn.configure(state="normal")
                self.auto_btn.configure(state="normal")

            if self.current_cloud is not None:
                self.extract_btn.configure(state="normal")

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


class SettingsDialog(ctk.CTkToplevel):
    """Settings dialog window."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent

        self.title("Settings")
        self.geometry("400x660")
        self.resizable(False, False)

        self._create_widgets()
    
    def _create_widgets(self):
        """Create settings widgets."""
        # Title
        title = ctk.CTkLabel(
            self,
            text="⚙️ Settings",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(padx=20, pady=20)
        
        # Generation settings
        gen_frame = ctk.CTkFrame(self)
        gen_frame.pack(fill="x", padx=20, pady=10)
        
        gen_label = ctk.CTkLabel(
            gen_frame,
            text="Generation Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        gen_label.pack(padx=10, pady=10, anchor="w")
        
        # Iterations
        iter_frame = ctk.CTkFrame(gen_frame, fg_color="transparent")
        iter_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(iter_frame, text="Iterations:").pack(side="left")
        self.iter_entry = ctk.CTkEntry(iter_frame, width=100)
        self.iter_entry.pack(side="right")
        self.iter_entry.insert(0, str(self.parent_window.settings.get('num_iterations', 1000)))

        # Strand settings
        strand_frame = ctk.CTkFrame(self)
        strand_frame.pack(fill="x", padx=20, pady=10)

        strand_label = ctk.CTkLabel(
            strand_frame,
            text="Strand Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        strand_label.pack(padx=10, pady=10, anchor="w")

        # Points per strand
        pts_frame = ctk.CTkFrame(strand_frame, fg_color="transparent")
        pts_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(pts_frame, text="Points per strand:").pack(side="left")
        self.pts_entry = ctk.CTkEntry(pts_frame, width=100)
        self.pts_entry.pack(side="right")
        self.pts_entry.insert(0, str(self.parent_window.settings.get('points_per_strand', 32)))

        # Num strands
        num_frame = ctk.CTkFrame(strand_frame, fg_color="transparent")
        num_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(num_frame, text="Max strands:").pack(side="left")
        self.num_entry = ctk.CTkEntry(num_frame, width=100)
        self.num_entry.pack(side="right")
        self.num_entry.insert(0, str(self.parent_window.settings.get('num_strands', 10000)))
        
        # Appearance
        app_frame = ctk.CTkFrame(self)
        app_frame.pack(fill="x", padx=20, pady=10)
        
        app_label = ctk.CTkLabel(
            app_frame,
            text="Appearance",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        app_label.pack(padx=10, pady=10, anchor="w")
        
        # Theme
        theme_frame = ctk.CTkFrame(app_frame, fg_color="transparent")
        theme_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(theme_frame, text="Theme:").pack(side="left")
        self.theme_menu = ctk.CTkOptionMenu(
            theme_frame,
            values=["Dark", "Light", "System"],
            command=self._change_theme
        )
        self.theme_menu.pack(side="right")
        saved_theme = self.parent_window.settings.get('theme', 'dark')
        self.theme_menu.set(saved_theme.capitalize())
        
        # Pre-download models button
        models_frame = ctk.CTkFrame(self)
        models_frame.pack(fill="x", padx=20, pady=(0, 10))

        models_label = ctk.CTkLabel(
            models_frame,
            text="AI 模型",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        models_label.pack(padx=10, pady=(10, 6), anchor="w")

        predownload_btn = ctk.CTkButton(
            models_frame,
            text="预下载 AI 模型",
            command=self._predownload_models,
        )
        predownload_btn.pack(padx=10, pady=(0, 6), anchor="w")

        # HuggingFace mirror
        mirror_row = ctk.CTkFrame(models_frame, fg_color="transparent")
        mirror_row.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(mirror_row, text="镜像地址:", font=ctk.CTkFont(size=11)).pack(side="left")
        self.mirror_entry = ctk.CTkEntry(mirror_row, placeholder_text="https://hf-mirror.com（留空用官方）")
        self.mirror_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))
        saved_mirror = self.parent_window.settings.get('hf_endpoint', '')
        if saved_mirror:
            self.mirror_entry.insert(0, saved_mirror)

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(10, 20))

        save_btn = ctk.CTkButton(
            btn_frame,
            text="Save",
            command=self._save_settings
        )
        save_btn.pack(side="right", padx=5)

        cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            fg_color="gray40",
            command=self.destroy
        )
        cancel_btn.pack(side="right", padx=5)
    
    def _predownload_models(self):
        """Open a download dialog for all AI models (cached ones shown as already done)."""
        try:
            from src.core.model_manager import get_models_to_download
            missing = get_models_to_download()
        except Exception:
            missing = []

        if not missing:
            # All cached — show a status dialog using the download dialog (all cached)
            try:
                from src.core.model_manager import get_all_models_status
                all_models = get_all_models_status()
            except Exception:
                all_models = []

            if not all_models:
                try:
                    from CTkMessagebox import CTkMessagebox
                    CTkMessagebox(
                        master=self,
                        title="AI 模型",
                        message="所有模型均已缓存，无需重新下载。",
                        icon="check",
                    )
                except Exception:
                    pass
                return

            # Show dialog with all models marked as cached
            from src.ui.model_download_dialog import ModelDownloadDialog
            ModelDownloadDialog(
                parent=self,
                models_to_download=all_models,
                on_complete=None,
                on_cancel=None,
            )
        else:
            from src.ui.model_download_dialog import ModelDownloadDialog
            ModelDownloadDialog(
                parent=self,
                models_to_download=missing,
                on_complete=None,
                on_cancel=None,
            )

    def _change_theme(self, value):
        """Change application theme."""
        ctk.set_appearance_mode(value.lower())
    
    def _save_settings(self):
        """Validate, apply, persist settings, then close."""
        # Validate numeric fields
        try:
            num_iterations = int(self.iter_entry.get())
            points_per_strand = int(self.pts_entry.get())
            num_strands = int(self.num_entry.get())
            if num_iterations <= 0 or points_per_strand <= 0 or num_strands <= 0:
                raise ValueError("Values must be positive")
        except ValueError as exc:
            # Show inline error in status (avoid hard dependency on CTkMessagebox here)
            try:
                from CTkMessagebox import CTkMessagebox
                CTkMessagebox(
                    master=self,
                    title="Invalid Settings",
                    message=f"Please enter positive integers.\n{exc}",
                    icon="cancel"
                )
            except Exception:
                pass
            return

        parent = self.parent_window

        # Apply to processors if methods exist
        if hasattr(parent.gaussian_generator, 'set_parameters'):
            parent.gaussian_generator.set_parameters(num_iterations=num_iterations)

        if hasattr(parent.strand_extractor, 'set_parameters'):
            parent.strand_extractor.set_parameters(
                points_per_strand=points_per_strand,
                num_strands=num_strands
            )

        # Update in-memory settings dict
        parent.settings['num_iterations'] = num_iterations
        parent.settings['points_per_strand'] = points_per_strand
        parent.settings['num_strands'] = num_strands
        parent.settings['theme'] = self.theme_menu.get().lower()
        parent.settings['hf_endpoint'] = getattr(self, 'mirror_entry', None) and self.mirror_entry.get().strip() or ''

        # Apply mirror immediately so any subsequent download uses it
        try:
            from src.core.model_manager import apply_hf_mirror
            apply_hf_mirror()
        except Exception:
            pass

        # Persist to disk
        settings_manager.save_settings(parent.settings)

        self.destroy()