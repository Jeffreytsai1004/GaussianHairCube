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

# Try to import tkinterdnd2 for drag and drop support
try:
    from tkinterdnd2 import TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False
    print("Warning: tkinterdnd2 not available, drag and drop disabled")

# Import modules
from src.ui.input_panel import InputPanel
from src.ui.output_panel import OutputPanel
from src.ui.viewer_widget import ViewerWidget
from src.core.gaussian_generator import GaussianGenerator, GaussianCloud
from src.core.hair_strands import HairStrandsExtractor, HairStrandCollection
from src.rendering.viewer_3d import ViewMode


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
        
        # Initialize processors
        self.gaussian_generator = GaussianGenerator()
        self.strand_extractor = HairStrandsExtractor()
        
        # State
        self.current_cloud: Optional[GaussianCloud] = None
        self.current_strands: Optional[HairStrandCollection] = None
        self.is_processing = False
        
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
        input_type = data.get('type')
        
        if input_type == 'image':
            self.status_label.configure(text="Image loaded")
            self.generate_btn.configure(state="normal")
            self.auto_btn.configure(state="normal")
        elif input_type == 'video':
            frames = data.get('frames', [])
            if frames:
                self.status_label.configure(text=f"Video loaded: {len(frames)} frames")
                self.generate_btn.configure(state="normal")
                self.auto_btn.configure(state="normal")
            else:
                self.status_label.configure(text="Video loaded - extract frames to process")
    
    def _generate_gaussians(self):
        """Generate Gaussian splats from input."""
        if self.is_processing:
            return
        
        image = self.input_panel.get_current_image()
        frames = self.input_panel.get_video_frames()
        
        if image is None and not frames:
            return
        
        self.is_processing = True
        self._show_progress()
        self.status_label.configure(text="Generating Gaussians...")
        
        # Disable buttons
        self._set_buttons_state("disabled")
        
        def generate():
            try:
                def callback(progress, message):
                    self.after(0, lambda: self._update_progress(progress, message))
                
                if frames:
                    cloud = self.gaussian_generator.generate_from_video(frames, callback)
                else:
                    cloud = self.gaussian_generator.generate_from_image(image, callback=callback)
                
                self.after(0, lambda: self._on_gaussians_generated(cloud))
                
            except Exception as e:
                self.after(0, lambda: self._on_processing_error(str(e)))
        
        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
    
    def _extract_curves(self):
        """Extract hair curves from Gaussians."""
        if self.is_processing or self.current_cloud is None:
            return
        
        self.is_processing = True
        self._show_progress()
        self.status_label.configure(text="Extracting curves...")
        
        self._set_buttons_state("disabled")
        
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
        if self.is_processing:
            return
        
        image = self.input_panel.get_current_image()
        frames = self.input_panel.get_video_frames()
        
        if image is None and not frames:
            return
        
        self.is_processing = True
        self._show_progress()
        self.status_label.configure(text="Auto processing...")
        
        self._set_buttons_state("disabled")
        
        def process():
            try:
                def callback(progress, message):
                    self.after(0, lambda: self._update_progress(progress * 0.6, message))
                
                # Generate Gaussians
                if frames:
                    cloud = self.gaussian_generator.generate_from_video(frames, callback)
                else:
                    cloud = self.gaussian_generator.generate_from_image(image, callback=callback)
                
                self.after(0, lambda: self._on_gaussians_generated(cloud, continue_processing=True))
                
                # Extract curves
                def callback2(progress, message):
                    self.after(0, lambda: self._update_progress(0.6 + progress * 0.4, message))
                
                strands = self.strand_extractor.extract(cloud, callback2)
                
                self.after(0, lambda: self._on_curves_extracted(strands))
                
            except Exception as e:
                self.after(0, lambda: self._on_processing_error(str(e)))
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def _on_gaussians_generated(self, cloud: GaussianCloud, continue_processing: bool = False):
        """Handle Gaussians generated."""
        self.current_cloud = cloud
        
        # Update viewer
        self.viewer.set_gaussian_data(cloud)
        
        # Update output panel
        self.output_panel.set_gaussian_data(cloud)
        
        # Update status
        if not continue_processing:
            self.is_processing = False
            self._hide_progress()
            self._set_buttons_state("normal")
        
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
        self.is_processing = False
        self._hide_progress()
        self._set_buttons_state("normal")
        self.status_label.configure(
            text=f"Extracted {strands.num_strands} strands ({strands.total_points} points)"
        )
    
    def _on_processing_error(self, error: str):
        """Handle processing error."""
        self.is_processing = False
        self._hide_progress()
        self._set_buttons_state("normal")
        self.status_label.configure(text=f"Error: {error}")
    
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
            # Only enable if we have input
            if self.input_panel.get_current_image() is not None or self.input_panel.get_video_frames():
                self.generate_btn.configure(state="normal")
                self.auto_btn.configure(state="normal")
            
            if self.current_cloud is not None:
                self.extract_btn.configure(state="normal")
    
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
        
        self.title("Settings")
        self.geometry("400x500")
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
        self.iter_entry.insert(0, "1000")
        
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
        self.pts_entry.insert(0, "32")
        
        # Num strands
        num_frame = ctk.CTkFrame(strand_frame, fg_color="transparent")
        num_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(num_frame, text="Max strands:").pack(side="left")
        self.num_entry = ctk.CTkEntry(num_frame, width=100)
        self.num_entry.pack(side="right")
        self.num_entry.insert(0, "10000")
        
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
        self.theme_menu.set("Dark")
        
        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=20)
        
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
    
    def _change_theme(self, value):
        """Change application theme."""
        ctk.set_appearance_mode(value.lower())
    
    def _save_settings(self):
        """Save settings and close."""
        # In a real app, settings would be saved to a config file
        self.destroy()