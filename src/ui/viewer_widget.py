"""
3D Viewer Widget Module
=======================

Tkinter widget for embedding the 3D viewer.
Provides interactive 3D visualization with mouse controls.
"""

import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
from typing import Optional, Callable
import threading
import time

# Import rendering
from pathlib import Path
from src.rendering.viewer_3d import Viewer3D, ViewMode, ShadingMode
from src.rendering.gaussian_renderer import GaussianRenderer
from src.core.gaussian_generator import GaussianCloud
from src.core.hair_strands import HairStrandCollection


class ViewerWidget(ctk.CTkFrame):
    """
    Widget for 3D visualization.
    
    Embeds a 3D viewer in a Tkinter frame with:
    - Interactive camera controls
    - View mode switching
    - Real-time rendering updates
    """
    
    def __init__(
        self, 
        parent,
        width: int = 600,
        height: int = 400,
        **kwargs
    ):
        """
        Initialize the viewer widget.
        
        Args:
            parent: Parent widget
            width: Viewer width
            height: Viewer height
            **kwargs: Additional arguments for CTkFrame
        """
        super().__init__(parent, **kwargs)
        
        self.viewer_width = width
        self.viewer_height = height
        
        # Create viewer
        self.viewer = Viewer3D(width, height)
        
        # State
        self.view_mode = ViewMode.GAUSSIANS
        self.is_rendering = False
        self._render_thread = None
        self._stop_render = False
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self._create_widgets()
        self._bind_events()
    
    def _create_widgets(self):
        """Create viewer widgets."""
        # Canvas for rendering
        self.canvas = ctk.CTkCanvas(
            self,
            width=self.viewer_width,
            height=self.viewer_height,
            bg="#1a1a1f",
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Control overlay frame
        self.controls_frame = ctk.CTkFrame(
            self,
            fg_color=("gray85", "gray15"),
            corner_radius=8
        )
        self.controls_frame.place(relx=0.02, rely=0.02, anchor="nw")
        
        # View mode buttons
        self.mode_label = ctk.CTkLabel(
            self.controls_frame,
            text="View:",
            font=ctk.CTkFont(size=11)
        )
        self.mode_label.grid(row=0, column=0, padx=5, pady=3)
        
        self.mode_var = ctk.StringVar(value="gaussians")
        
        self.mode_gaussians_btn = ctk.CTkButton(
            self.controls_frame,
            text="Splats",
            width=60,
            height=24,
            font=ctk.CTkFont(size=11),
            command=lambda: self._set_view_mode(ViewMode.GAUSSIANS)
        )
        self.mode_gaussians_btn.grid(row=0, column=1, padx=2, pady=3)
        
        self.mode_curves_btn = ctk.CTkButton(
            self.controls_frame,
            text="Curves",
            width=60,
            height=24,
            font=ctk.CTkFont(size=11),
            fg_color="gray40",
            command=lambda: self._set_view_mode(ViewMode.CURVES)
        )
        self.mode_curves_btn.grid(row=0, column=2, padx=2, pady=3)
        
        self.mode_both_btn = ctk.CTkButton(
            self.controls_frame,
            text="Both",
            width=50,
            height=24,
            font=ctk.CTkFont(size=11),
            fg_color="gray40",
            command=lambda: self._set_view_mode(ViewMode.BOTH)
        )
        self.mode_both_btn.grid(row=0, column=3, padx=2, pady=3)
        
        # Reset view button
        self.reset_btn = ctk.CTkButton(
            self.controls_frame,
            text="⟲",
            width=30,
            height=24,
            font=ctk.CTkFont(size=14),
            fg_color="gray40",
            command=self._reset_view
        )
        self.reset_btn.grid(row=0, column=4, padx=(10, 5), pady=3)
        
        # Stats overlay
        self.stats_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="gray60",
            fg_color=("gray85", "gray15"),
            corner_radius=4
        )
        self.stats_label.place(relx=0.98, rely=0.98, anchor="se")
        
        # Loading indicator
        self.loading_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="white"
        )
        self.loading_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Placeholder image
        self._show_placeholder()
    
    def _bind_events(self):
        """Bind mouse and keyboard events."""
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.canvas.bind("<ButtonPress-2>", self._on_middle_press)
        self.canvas.bind("<ButtonPress-3>", self._on_right_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<ButtonRelease-2>", self._on_mouse_release)
        self.canvas.bind("<ButtonRelease-3>", self._on_mouse_release)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<B2-Motion>", self._on_mouse_drag)
        self.canvas.bind("<B3-Motion>", self._on_mouse_drag)
        self.canvas.bind("<MouseWheel>", self._on_mouse_scroll)
        
        # Resize
        self.bind("<Configure>", self._on_resize)
    
    def _show_placeholder(self):
        """Show placeholder when no data is loaded."""
        # Create placeholder image
        placeholder = np.ones((self.viewer_height, self.viewer_width, 3), dtype=np.uint8)
        placeholder[:, :] = [26, 26, 31]  # Dark gray
        
        # Draw centered text (simple approach)
        self._update_canvas(placeholder)
        
        self.loading_label.configure(text="Load an image to generate hair")
    
    def _update_canvas(self, image: np.ndarray):
        """Update canvas with new image."""
        # Convert to PIL
        if image.shape[2] == 4:
            pil_image = Image.fromarray(image, mode='RGBA')
        else:
            pil_image = Image.fromarray(image, mode='RGB')
        
        # Resize if needed
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            if pil_image.size != (canvas_width, canvas_height):
                pil_image = pil_image.resize(
                    (canvas_width, canvas_height),
                    Image.Resampling.LANCZOS
                )
        
        # Convert to PhotoImage
        self._photo = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
    
    def _set_view_mode(self, mode: ViewMode):
        """Set the view mode."""
        self.view_mode = mode
        self.viewer.set_view_mode(mode)
        
        # Update button states
        default_color = ctk.ThemeManager.theme["CTkButton"]["fg_color"]
        gray_color = "gray40"
        
        self.mode_gaussians_btn.configure(
            fg_color=default_color if mode == ViewMode.GAUSSIANS else gray_color
        )
        self.mode_curves_btn.configure(
            fg_color=default_color if mode == ViewMode.CURVES else gray_color
        )
        self.mode_both_btn.configure(
            fg_color=default_color if mode == ViewMode.BOTH else gray_color
        )
        
        # Re-render
        self._render()
    
    def _reset_view(self):
        """Reset camera to default view."""
        self.viewer._fit_camera_to_data()
        self._render()
    
    def _on_mouse_press(self, event):
        """Handle mouse press."""
        self.viewer.on_mouse_press(event.x, event.y, 0)
    
    def _on_middle_press(self, event):
        """Handle middle mouse press."""
        self.viewer.on_mouse_press(event.x, event.y, 2)
    
    def _on_right_press(self, event):
        """Handle right mouse press."""
        self.viewer.on_mouse_press(event.x, event.y, 1)
    
    def _on_mouse_release(self, event):
        """Handle mouse release."""
        self.viewer.on_mouse_release(0)
        self.viewer.on_mouse_release(1)
        self.viewer.on_mouse_release(2)
    
    def _on_mouse_drag(self, event):
        """Handle mouse drag."""
        self.viewer.on_mouse_move(event.x, event.y)
        self._render()
    
    def _on_mouse_scroll(self, event):
        """Handle mouse scroll."""
        delta = event.delta / 120  # Normalize
        self.viewer.on_mouse_scroll(delta)
        self._render()
    
    def _on_resize(self, event):
        """Handle widget resize."""
        if event.widget == self:
            new_width = event.width
            new_height = event.height - 50  # Account for controls
            
            if new_width > 100 and new_height > 100:
                self.viewer.width = new_width
                self.viewer.height = new_height
                self._render()
    
    def _render(self):
        """Render the current view."""
        try:
            image = self.viewer.render_to_image()
            self._update_canvas(image)
            self._update_stats()
        except Exception as e:
            print(f"Render error: {e}")
    
    def _update_stats(self):
        """Update stats display."""
        stats = self.viewer.get_render_stats()
        
        if stats.get('gaussian_count', 0) > 0 or stats.get('curve_count', 0) > 0:
            text = f"Splats: {stats.get('gaussian_count', 0)} | Curves: {stats.get('curve_count', 0)}"
            self.stats_label.configure(text=text)
            self.loading_label.configure(text="")
        else:
            self.stats_label.configure(text="")
    
    def set_gaussian_data(self, cloud: GaussianCloud):
        """
        Set Gaussian cloud data for visualization.
        
        Args:
            cloud: Gaussian cloud to visualize
        """
        positions = cloud.get_positions()
        colors = cloud.get_colors()
        scales = np.array([s.scale for s in cloud.splats])
        
        self.viewer.set_gaussian_data(positions, colors, scales)
        self._render()
    
    def set_curve_data(self, strands: HairStrandCollection):
        """
        Set hair strand data for visualization.
        
        Args:
            strands: Hair strand collection to visualize
        """
        curves = [s.points for s in strands.strands]
        colors = [s.colors for s in strands.strands]
        
        self.viewer.set_curve_data(curves, colors)
        self._render()
    
    def show_loading(self, message: str = "Processing..."):
        """Show loading indicator."""
        self.loading_label.configure(text=message)
    
    def hide_loading(self):
        """Hide loading indicator."""
        self.loading_label.configure(text="")
    
    def start_animation(self):
        """Start auto-rotation animation."""
        self._stop_render = False
        
        def animate():
            angle = 0
            while not self._stop_render:
                angle += 0.02
                self.viewer.camera.orbit(0.02, 0)
                self.after(0, self._render)
                time.sleep(1/30)  # 30 FPS
        
        self._render_thread = threading.Thread(target=animate, daemon=True)
        self._render_thread.start()
    
    def stop_animation(self):
        """Stop auto-rotation animation."""
        self._stop_render = True
        if self._render_thread:
            self._render_thread.join(timeout=0.5)
    
    def export_image(self, filepath: str):
        """Export current view to image file."""
        image = self.viewer.render_to_image()
        pil_image = Image.fromarray(image)
        pil_image.save(filepath)
    
    def get_current_view_mode(self) -> ViewMode:
        """Get current view mode."""
        return self.view_mode