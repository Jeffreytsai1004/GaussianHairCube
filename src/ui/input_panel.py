"""
Input Panel Module
==================

Left panel for uploading images or videos.
Supports multiple formats including common image and video formats.
"""

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from typing import Optional, Callable, List
from pathlib import Path
import cv2
import threading

# Try to import tkinterdnd2 for drag and drop support
try:
    from tkinterdnd2 import DND_FILES
    HAS_DND = True
except ImportError:
    HAS_DND = False
    print("Warning: tkinterdnd2 not available, drag and drop disabled")


# Supported file formats
IMAGE_FORMATS = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg *.jpeg"),
    ("BMP", "*.bmp"),
    ("TIFF", "*.tiff"),
    ("WebP", "*.webp"),
    ("All files", "*.*")
]

VIDEO_FORMATS = [
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
    ("MP4", "*.mp4"),
    ("AVI", "*.avi"),
    ("MOV", "*.mov"),
    ("MKV", "*.mkv"),
    ("WebM", "*.webm"),
    ("All files", "*.*")
]

ALL_FORMATS = [
    ("All supported", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp *.mp4 *.avi *.mov *.mkv *.webm"),
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
    ("All files", "*.*")
]


class InputPanel(ctk.CTkFrame):
    """
    Input panel for image/video upload.
    
    Features:
    - Drag and drop support
    - Image preview
    - Video frame extraction
    - Format validation
    """
    
    def __init__(
        self, 
        parent, 
        on_input_loaded: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize the input panel.
        
        Args:
            parent: Parent widget
            on_input_loaded: Callback when input is loaded
            **kwargs: Additional arguments for CTkFrame
        """
        super().__init__(parent, **kwargs)
        
        self.on_input_loaded = on_input_loaded
        
        # State
        self.current_image: Optional[np.ndarray] = None
        self.current_video_path: Optional[str] = None
        self.video_frames: List[np.ndarray] = []
        self.is_video = False
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create panel widgets."""
        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text="📁 Input",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        # Preview area
        self.preview_frame = ctk.CTkFrame(self, fg_color="gray20")
        self.preview_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(0, weight=1)
        
        # Preview label with placeholder
        self.preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="Drop image/video here\nor click Upload",
            font=ctk.CTkFont(size=14),
            text_color="gray60"
        )
        self.preview_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Bind click to preview area
        self.preview_frame.bind("<Button-1>", lambda e: self._browse_file())
        self.preview_label.bind("<Button-1>", lambda e: self._browse_file())
        
        # Setup drag and drop if available
        self._setup_drag_drop()
        
        # Control buttons frame
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        self.controls_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Upload button
        self.upload_btn = ctk.CTkButton(
            self.controls_frame,
            text="📤 Upload",
            command=self._browse_file,
            height=36
        )
        self.upload_btn.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="ew")
        
        # Clear button
        self.clear_btn = ctk.CTkButton(
            self.controls_frame,
            text="🗑️ Clear",
            command=self._clear_input,
            fg_color="gray40",
            hover_color="gray30",
            height=36
        )
        self.clear_btn.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="ew")
        
        # File info
        self.info_label = ctk.CTkLabel(
            self,
            text="No file loaded",
            font=ctk.CTkFont(size=11),
            text_color="gray60"
        )
        self.info_label.grid(row=3, column=0, padx=10, pady=(0, 5), sticky="w")
        
        # Video controls (hidden by default)
        self.video_controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.video_controls_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        self.video_controls_frame.grid_columnconfigure(0, weight=1)
        self.video_controls_frame.grid_remove()  # Hide initially
        
        # Frame slider
        self.frame_slider = ctk.CTkSlider(
            self.video_controls_frame,
            from_=0,
            to=100,
            command=self._on_frame_slider_change
        )
        self.frame_slider.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.frame_label = ctk.CTkLabel(
            self.video_controls_frame,
            text="Frame: 0 / 0",
            font=ctk.CTkFont(size=11)
        )
        self.frame_label.grid(row=1, column=0, padx=5, pady=(0, 5))
        
        # Extract frames button
        self.extract_btn = ctk.CTkButton(
            self.video_controls_frame,
            text="🎬 Extract Frames",
            command=self._extract_video_frames,
            height=32
        )
        self.extract_btn.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
    
    def _setup_drag_drop(self):
        """Setup drag and drop functionality."""
        if not HAS_DND:
            return
        
        try:
            # Register the preview frame as a drop target
            self.preview_frame.drop_target_register(DND_FILES)
            self.preview_frame.dnd_bind('<<Drop>>', self._on_drop)
            self.preview_frame.dnd_bind('<<DragEnter>>', self._on_drag_enter)
            self.preview_frame.dnd_bind('<<DragLeave>>', self._on_drag_leave)
            
            # Also register the preview label
            self.preview_label.drop_target_register(DND_FILES)
            self.preview_label.dnd_bind('<<Drop>>', self._on_drop)
            self.preview_label.dnd_bind('<<DragEnter>>', self._on_drag_enter)
            self.preview_label.dnd_bind('<<DragLeave>>', self._on_drag_leave)
            
            print("Drag and drop enabled")
        except Exception as e:
            print(f"Warning: Could not setup drag and drop: {e}")
    
    def _on_drop(self, event):
        """Handle file drop event."""
        # Get the dropped file path
        filepath = event.data
        
        # Clean up the path (remove curly braces if present on Windows)
        if filepath.startswith('{') and filepath.endswith('}'):
            filepath = filepath[1:-1]
        
        # Handle multiple files (take first one)
        if ' ' in filepath and not Path(filepath).exists():
            # Try to split by space and take first valid path
            parts = filepath.split()
            for part in parts:
                clean_part = part.strip('{}')
                if Path(clean_part).exists():
                    filepath = clean_part
                    break
        
        # Reset visual feedback
        self._on_drag_leave(None)
        
        # Load the file
        if filepath and Path(filepath).exists():
            self._load_file(filepath)
    
    def _on_drag_enter(self, event):
        """Handle drag enter event - visual feedback."""
        self.preview_frame.configure(fg_color="gray30")
        if not self.current_image:
            self.preview_label.configure(text="Drop to load file")
    
    def _on_drag_leave(self, event):
        """Handle drag leave event - reset visual feedback."""
        self.preview_frame.configure(fg_color="gray20")
        if not self.current_image:
            self.preview_label.configure(text="Drop image/video here\nor click Upload")
    
    def _browse_file(self):
        """Open file browser dialog."""
        filepath = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=ALL_FORMATS
        )
        
        if filepath:
            self._load_file(filepath)
    
    def _load_file(self, filepath: str):
        """Load image or video file."""
        path = Path(filepath)
        extension = path.suffix.lower()
        
        # Check if image or video
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        if extension in image_extensions:
            self._load_image(filepath)
        elif extension in video_extensions:
            self._load_video(filepath)
        else:
            self.info_label.configure(text="Unsupported file format")
    
    def _load_image(self, filepath: str):
        """Load and display an image."""
        try:
            # Load with PIL
            pil_image = Image.open(filepath)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Store as numpy array
            self.current_image = np.array(pil_image)
            self.is_video = False
            self.current_video_path = None
            self.video_frames = []
            
            # Display preview
            self._display_preview(pil_image)
            
            # Update info
            h, w = self.current_image.shape[:2]
            filename = Path(filepath).name
            self.info_label.configure(text=f"📷 {filename} ({w}×{h})")
            
            # Hide video controls
            self.video_controls_frame.grid_remove()
            
            # Callback
            if self.on_input_loaded:
                self.on_input_loaded({
                    'type': 'image',
                    'image': self.current_image,
                    'path': filepath
                })
                
        except Exception as e:
            self.info_label.configure(text=f"Error loading image: {str(e)}")
    
    def _load_video(self, filepath: str):
        """Load and display a video."""
        try:
            # Open video
            cap = cv2.VideoCapture(filepath)
            
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Read first frame
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_image = frame_rgb
                
                # Display preview
                pil_image = Image.fromarray(frame_rgb)
                self._display_preview(pil_image)
            
            cap.release()
            
            # Store video info
            self.is_video = True
            self.current_video_path = filepath
            self.video_frames = []
            
            # Update info
            filename = Path(filepath).name
            duration = frame_count / fps if fps > 0 else 0
            self.info_label.configure(
                text=f"🎥 {filename} ({width}×{height}, {frame_count} frames, {duration:.1f}s)"
            )
            
            # Show video controls
            self.frame_slider.configure(to=max(1, frame_count - 1))
            self.frame_slider.set(0)
            self.frame_label.configure(text=f"Frame: 1 / {frame_count}")
            self.video_controls_frame.grid()
            
        except Exception as e:
            self.info_label.configure(text=f"Error loading video: {str(e)}")
    
    def _display_preview(self, pil_image: Image.Image):
        """Display image preview in the panel."""
        # Get preview area size
        self.update_idletasks()
        preview_width = self.preview_frame.winfo_width() - 20
        preview_height = self.preview_frame.winfo_height() - 20
        
        if preview_width <= 0:
            preview_width = 300
        if preview_height <= 0:
            preview_height = 200
        
        # Resize image to fit
        img_width, img_height = pil_image.size
        scale = min(preview_width / img_width, preview_height / img_height)
        
        new_width = max(1, int(img_width * scale))
        new_height = max(1, int(img_height * scale))
        
        resized = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to CTkImage
        self._preview_image = ctk.CTkImage(
            light_image=resized,
            dark_image=resized,
            size=(new_width, new_height)
        )
        
        # Update label
        self.preview_label.configure(image=self._preview_image, text="")
    
    def _on_frame_slider_change(self, value):
        """Handle frame slider change."""
        if not self.current_video_path:
            return
        
        frame_idx = int(value)
        
        # Load frame from video
        try:
            cap = cv2.VideoCapture(self.current_video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_image = frame_rgb
                
                pil_image = Image.fromarray(frame_rgb)
                self._display_preview(pil_image)
                
                frame_count = int(self.frame_slider.cget("to")) + 1
                self.frame_label.configure(text=f"Frame: {frame_idx + 1} / {frame_count}")
                
        except Exception as e:
            print(f"Error loading frame: {e}")
    
    def _extract_video_frames(self):
        """Extract frames from video in background."""
        if not self.current_video_path:
            return
        
        self.extract_btn.configure(state="disabled", text="Extracting...")
        
        def extract():
            try:
                cap = cv2.VideoCapture(self.current_video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Sample frames (limit to max 100 frames)
                max_frames = 100
                step = max(1, frame_count // max_frames)
                
                frames = []
                for i in range(0, frame_count, step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)
                
                cap.release()
                
                self.video_frames = frames
                
                # Update UI from main thread
                self.after(0, lambda: self._on_frames_extracted(len(frames)))
                
            except Exception as e:
                self.after(0, lambda: self._on_extract_error(str(e)))
        
        thread = threading.Thread(target=extract, daemon=True)
        thread.start()
    
    def _on_frames_extracted(self, count: int):
        """Called when frame extraction completes."""
        self.extract_btn.configure(state="normal", text=f"✓ {count} Frames Extracted")
        self.info_label.configure(
            text=self.info_label.cget("text") + f" - {count} frames extracted"
        )
        
        # Callback
        if self.on_input_loaded:
            self.on_input_loaded({
                'type': 'video',
                'frames': self.video_frames,
                'path': self.current_video_path
            })
    
    def _on_extract_error(self, error: str):
        """Called when frame extraction fails."""
        self.extract_btn.configure(state="normal", text="🎬 Extract Frames")
        self.info_label.configure(text=f"Error: {error}")
    
    def _clear_input(self):
        """Clear current input."""
        self.current_image = None
        self.current_video_path = None
        self.video_frames = []
        self.is_video = False
        
        # Reset preview
        self.preview_label.configure(
            image=None,
            text="Drop image/video here\nor click Upload"
        )
        
        # Reset info
        self.info_label.configure(text="No file loaded")
        
        # Hide video controls
        self.video_controls_frame.grid_remove()
        self.extract_btn.configure(state="normal", text="🎬 Extract Frames")
    
    def get_current_image(self) -> Optional[np.ndarray]:
        """Get the currently loaded image."""
        return self.current_image
    
    def get_video_frames(self) -> List[np.ndarray]:
        """Get extracted video frames."""
        return self.video_frames
    
    def is_video_input(self) -> bool:
        """Check if current input is a video."""
        return self.is_video