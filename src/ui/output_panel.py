"""
Output Panel Module
===================

Panel for controlling output and export options.
Provides download functionality for FBX and GLB formats.
"""

import customtkinter as ctk
from tkinter import filedialog
from typing import Optional, Callable
from pathlib import Path
import threading

# Import export modules
from src.export.fbx_exporter import FBXExporter, FBXExportOptions
from src.export.glb_exporter import GLBExporter, GLBExportOptions
from src.core.hair_strands import HairStrandCollection
from src.core.gaussian_generator import GaussianCloud


class OutputPanel(ctk.CTkFrame):
    """
    Output panel for export controls.
    
    Features:
    - Export format selection
    - Export options configuration
    - Progress indication
    - Direct download
    """
    
    def __init__(
        self, 
        parent,
        on_export_complete: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize the output panel.
        
        Args:
            parent: Parent widget
            on_export_complete: Callback when export completes
            **kwargs: Additional arguments for CTkFrame
        """
        super().__init__(parent, **kwargs)
        
        self.on_export_complete = on_export_complete
        
        # Data references
        self.gaussian_cloud: Optional[GaussianCloud] = None
        self.hair_strands: Optional[HairStrandCollection] = None
        
        # Export state
        self.is_exporting = False
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create panel widgets."""
        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text="📤 Export",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        # Status frame
        self.status_frame = ctk.CTkFrame(self, fg_color="gray20")
        self.status_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.status_frame.grid_columnconfigure(1, weight=1)
        
        # Gaussian status
        self.gaussian_icon = ctk.CTkLabel(
            self.status_frame,
            text="⚪",
            font=ctk.CTkFont(size=14)
        )
        self.gaussian_icon.grid(row=0, column=0, padx=5, pady=5)
        
        self.gaussian_status = ctk.CTkLabel(
            self.status_frame,
            text="Gaussians: Not generated",
            font=ctk.CTkFont(size=11),
            text_color="gray60"
        )
        self.gaussian_status.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Curves status
        self.curves_icon = ctk.CTkLabel(
            self.status_frame,
            text="⚪",
            font=ctk.CTkFont(size=14)
        )
        self.curves_icon.grid(row=1, column=0, padx=5, pady=5)
        
        self.curves_status = ctk.CTkLabel(
            self.status_frame,
            text="Curves: Not extracted",
            font=ctk.CTkFont(size=11),
            text_color="gray60"
        )
        self.curves_status.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Export format section
        self.format_label = ctk.CTkLabel(
            self,
            text="Export Format:",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.format_label.grid(row=2, column=0, padx=10, pady=(15, 5), sticky="w")
        
        # Format selection
        self.format_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.format_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        self.format_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.format_var = ctk.StringVar(value="fbx")
        
        self.fbx_radio = ctk.CTkRadioButton(
            self.format_frame,
            text="Maya FBX Curves",
            variable=self.format_var,
            value="fbx",
            font=ctk.CTkFont(size=12)
        )
        self.fbx_radio.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.glb_radio = ctk.CTkRadioButton(
            self.format_frame,
            text="Blender GLB",
            variable=self.format_var,
            value="glb",
            font=ctk.CTkFont(size=12)
        )
        self.glb_radio.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Export options
        self.options_label = ctk.CTkLabel(
            self,
            text="Options:",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.options_label.grid(row=4, column=0, padx=10, pady=(15, 5), sticky="w")
        
        self.options_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.options_frame.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
        self.options_frame.grid_columnconfigure(1, weight=1)
        
        # Scale factor
        self.scale_label = ctk.CTkLabel(
            self.options_frame,
            text="Scale:",
            font=ctk.CTkFont(size=11)
        )
        self.scale_label.grid(row=0, column=0, padx=5, pady=3, sticky="w")
        
        self.scale_entry = ctk.CTkEntry(
            self.options_frame,
            width=80,
            placeholder_text="1.0"
        )
        self.scale_entry.grid(row=0, column=1, padx=5, pady=3, sticky="w")
        self.scale_entry.insert(0, "1.0")
        
        # Up axis
        self.axis_label = ctk.CTkLabel(
            self.options_frame,
            text="Up Axis:",
            font=ctk.CTkFont(size=11)
        )
        self.axis_label.grid(row=1, column=0, padx=5, pady=3, sticky="w")
        
        self.axis_var = ctk.StringVar(value="Y")
        self.axis_menu = ctk.CTkOptionMenu(
            self.options_frame,
            variable=self.axis_var,
            values=["Y", "Z"],
            width=80
        )
        self.axis_menu.grid(row=1, column=1, padx=5, pady=3, sticky="w")
        
        # Include color
        self.color_var = ctk.BooleanVar(value=True)
        self.color_check = ctk.CTkCheckBox(
            self.options_frame,
            text="Include Color",
            variable=self.color_var,
            font=ctk.CTkFont(size=11)
        )
        self.color_check.grid(row=2, column=0, columnspan=2, padx=5, pady=3, sticky="w")
        
        # Export button
        self.export_btn = ctk.CTkButton(
            self,
            text="💾 Export",
            command=self._export,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled"
        )
        self.export_btn.grid(row=6, column=0, padx=10, pady=15, sticky="ew")
        
        # Progress bar
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.grid(row=7, column=0, padx=10, pady=5, sticky="ew")
        self.progress_frame.grid_remove()  # Hidden initially
        
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        self.progress_bar.set(0)
        
        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="",
            font=ctk.CTkFont(size=10)
        )
        self.progress_label.grid(row=1, column=0, pady=(2, 0))
        
        # Quick export section
        self.quick_label = ctk.CTkLabel(
            self,
            text="Quick Export:",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.quick_label.grid(row=8, column=0, padx=10, pady=(15, 5), sticky="w")
        
        self.quick_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.quick_frame.grid(row=9, column=0, padx=10, pady=5, sticky="ew")
        self.quick_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.quick_fbx_btn = ctk.CTkButton(
            self.quick_frame,
            text="📁 FBX",
            command=lambda: self._quick_export("fbx"),
            height=32,
            state="disabled"
        )
        self.quick_fbx_btn.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="ew")
        
        self.quick_glb_btn = ctk.CTkButton(
            self.quick_frame,
            text="📁 GLB",
            command=lambda: self._quick_export("glb"),
            height=32,
            state="disabled"
        )
        self.quick_glb_btn.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="ew")
    
    def set_gaussian_data(self, cloud: GaussianCloud):
        """Set Gaussian cloud data."""
        self.gaussian_cloud = cloud
        
        # Update status
        self.gaussian_icon.configure(text="🟢")
        self.gaussian_status.configure(
            text=f"Gaussians: {cloud.num_splats} splats",
            text_color="white"
        )
        
        self._update_export_state()
    
    def set_strand_data(self, strands: HairStrandCollection):
        """Set hair strand data."""
        self.hair_strands = strands
        
        # Update status
        self.curves_icon.configure(text="🟢")
        self.curves_status.configure(
            text=f"Curves: {strands.num_strands} strands ({strands.total_points} points)",
            text_color="white"
        )
        
        self._update_export_state()
    
    def _update_export_state(self):
        """Update export button state based on available data."""
        if self.hair_strands is not None:
            self.export_btn.configure(state="normal")
            self.quick_fbx_btn.configure(state="normal")
            self.quick_glb_btn.configure(state="normal")
        else:
            self.export_btn.configure(state="disabled")
            self.quick_fbx_btn.configure(state="disabled")
            self.quick_glb_btn.configure(state="disabled")
    
    def _get_export_options(self):
        """Get current export options."""
        try:
            scale = float(self.scale_entry.get())
        except ValueError:
            scale = 1.0
        
        up_axis = self.axis_var.get()
        include_color = self.color_var.get()
        
        return {
            'scale_factor': scale,
            'up_axis': up_axis,
            'include_color': include_color
        }
    
    def _export(self):
        """Handle export button click."""
        if self.hair_strands is None:
            return
        
        format_type = self.format_var.get()
        
        # Ask for save location
        if format_type == "fbx":
            filepath = filedialog.asksaveasfilename(
                title="Export Maya FBX",
                defaultextension=".fbx",
                filetypes=[("FBX files", "*.fbx"), ("All files", "*.*")]
            )
        else:
            filepath = filedialog.asksaveasfilename(
                title="Export Blender GLB",
                defaultextension=".glb",
                filetypes=[("GLB files", "*.glb"), ("All files", "*.*")]
            )
        
        if filepath:
            self._do_export(filepath, format_type)
    
    def _quick_export(self, format_type: str):
        """Quick export with default filename."""
        if self.hair_strands is None:
            return
        
        # Ask for save location
        if format_type == "fbx":
            filepath = filedialog.asksaveasfilename(
                title="Export Maya FBX",
                defaultextension=".fbx",
                initialfile="hair_curves.fbx",
                filetypes=[("FBX files", "*.fbx"), ("All files", "*.*")]
            )
        else:
            filepath = filedialog.asksaveasfilename(
                title="Export Blender GLB",
                defaultextension=".glb",
                initialfile="hair_mesh.glb",
                filetypes=[("GLB files", "*.glb"), ("All files", "*.*")]
            )
        
        if filepath:
            self._do_export(filepath, format_type)
    
    def _do_export(self, filepath: str, format_type: str):
        """Perform the export operation."""
        if self.is_exporting:
            return
        
        self.is_exporting = True
        
        # Show progress
        self.progress_frame.grid()
        self.progress_bar.set(0)
        self.progress_label.configure(text="Starting export...")
        
        # Disable buttons
        self.export_btn.configure(state="disabled")
        self.quick_fbx_btn.configure(state="disabled")
        self.quick_glb_btn.configure(state="disabled")
        
        def export_thread():
            try:
                options = self._get_export_options()
                
                def progress_callback(progress: float, message: str):
                    self.after(0, lambda: self._update_progress(progress, message))
                
                if format_type == "fbx":
                    fbx_options = FBXExportOptions(
                        scale_factor=options['scale_factor'],
                        up_axis=options['up_axis'],
                        include_color=options['include_color']
                    )
                    exporter = FBXExporter(fbx_options)
                    success = exporter.export(self.hair_strands, filepath, progress_callback)
                else:
                    glb_options = GLBExportOptions(
                        scale_factor=options['scale_factor'],
                        up_axis=options['up_axis'],
                        include_color=options['include_color'],
                        export_type="mesh"
                    )
                    exporter = GLBExporter(glb_options)
                    success = exporter.export_strands(self.hair_strands, filepath, progress_callback)
                
                self.after(0, lambda: self._on_export_complete(success, filepath))
                
            except Exception as e:
                self.after(0, lambda: self._on_export_error(str(e)))
        
        thread = threading.Thread(target=export_thread, daemon=True)
        thread.start()
    
    def _update_progress(self, progress: float, message: str):
        """Update progress bar."""
        self.progress_bar.set(progress)
        self.progress_label.configure(text=message)
    
    def _on_export_complete(self, success: bool, filepath: str):
        """Called when export completes."""
        self.is_exporting = False
        
        if success:
            self.progress_label.configure(text=f"✓ Exported to {Path(filepath).name}")
        else:
            self.progress_label.configure(text="✗ Export failed")
        
        # Re-enable buttons
        self._update_export_state()
        
        # Hide progress after delay
        self.after(3000, self._hide_progress)
        
        # Callback
        if self.on_export_complete:
            self.on_export_complete(success, filepath)
    
    def _on_export_error(self, error: str):
        """Called when export fails."""
        self.is_exporting = False
        
        self.progress_label.configure(text=f"✗ Error: {error}")
        self._update_export_state()
        
        self.after(5000, self._hide_progress)
    
    def _hide_progress(self):
        """Hide progress bar."""
        if not self.is_exporting:
            self.progress_frame.grid_remove()
    
    def clear_data(self):
        """Clear all data."""
        self.gaussian_cloud = None
        self.hair_strands = None
        
        # Reset status
        self.gaussian_icon.configure(text="⚪")
        self.gaussian_status.configure(
            text="Gaussians: Not generated",
            text_color="gray60"
        )
        
        self.curves_icon.configure(text="⚪")
        self.curves_status.configure(
            text="Curves: Not extracted",
            text_color="gray60"
        )
        
        self._update_export_state()