"""
3D Viewer Module
================

Interactive 3D viewer for visualizing Gaussian splats and hair curves.
Uses OpenGL for hardware-accelerated rendering with support for
interactive camera controls.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import math

# Import types
from pathlib import Path


class ViewMode(Enum):
    """Display modes for the 3D viewer."""
    GAUSSIANS = "gaussians"
    CURVES = "curves"
    BOTH = "both"


class ShadingMode(Enum):
    """Shading modes for rendering."""
    SOLID = "solid"
    VERTEX_COLOR = "vertex_color"
    NORMAL = "normal"
    DEPTH = "depth"


@dataclass
class Camera:
    """Camera for 3D viewing."""
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov: float = 45.0
    near: float = 0.01
    far: float = 100.0
    
    def get_view_matrix(self) -> np.ndarray:
        """Compute view matrix (look-at)."""
        forward = self.target - self.position
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        right = np.cross(forward, self.up)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        up = np.cross(right, forward)
        
        rotation = np.array([
            [right[0], right[1], right[2], 0],
            [up[0], up[1], up[2], 0],
            [-forward[0], -forward[1], -forward[2], 0],
            [0, 0, 0, 1]
        ])
        
        translation = np.array([
            [1, 0, 0, -self.position[0]],
            [0, 1, 0, -self.position[1]],
            [0, 0, 1, -self.position[2]],
            [0, 0, 0, 1]
        ])
        
        return rotation @ translation
    
    def get_projection_matrix(self, aspect: float) -> np.ndarray:
        """Compute perspective projection matrix."""
        fov_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fov_rad / 2)
        
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (self.far + self.near) / (self.near - self.far),
             2 * self.far * self.near / (self.near - self.far)],
            [0, 0, -1, 0]
        ])
    
    def orbit(self, delta_azimuth: float, delta_elevation: float):
        """Orbit camera around target."""
        # Get current spherical coordinates
        offset = self.position - self.target
        distance = np.linalg.norm(offset)
        
        # Current angles
        azimuth = math.atan2(offset[0], offset[2])
        elevation = math.asin(np.clip(offset[1] / distance, -1, 1))
        
        # Apply deltas
        azimuth += delta_azimuth
        elevation = np.clip(elevation + delta_elevation, -math.pi/2 + 0.01, math.pi/2 - 0.01)
        
        # Convert back to Cartesian
        self.position = self.target + np.array([
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
            distance * math.cos(elevation) * math.cos(azimuth)
        ])
    
    def zoom(self, factor: float):
        """Zoom camera (move closer/farther from target)."""
        offset = self.position - self.target
        distance = np.linalg.norm(offset)
        new_distance = max(0.1, distance * factor)
        self.position = self.target + offset / distance * new_distance
    
    def pan(self, delta_x: float, delta_y: float):
        """Pan camera (move target and position together)."""
        forward = self.target - self.position
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.cross(forward, self.up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        
        offset = right * delta_x + up * delta_y
        self.position += offset
        self.target += offset


class Viewer3D:
    """
    Interactive 3D viewer for hair visualization.
    
    Provides:
    - Real-time rendering of Gaussian splats
    - Hair curve visualization
    - Interactive camera controls (orbit, zoom, pan)
    - Multiple shading modes
    """
    
    def __init__(self, width: int = 800, height: int = 600):
        """
        Initialize the 3D viewer.
        
        Args:
            width: Viewport width
            height: Viewport height
        """
        self.width = width
        self.height = height
        
        # Camera setup
        self.camera = Camera(
            position=np.array([0.0, 0.0, 3.0]),
            target=np.array([0.0, 0.0, 0.0]),
            up=np.array([0.0, 1.0, 0.0])
        )
        
        # Display settings
        self.view_mode = ViewMode.GAUSSIANS
        self.shading_mode = ShadingMode.VERTEX_COLOR
        self.background_color = np.array([0.1, 0.1, 0.15, 1.0])
        self.point_size = 3.0
        self.line_width = 1.0
        
        # Data
        self.gaussian_positions: Optional[np.ndarray] = None
        self.gaussian_colors: Optional[np.ndarray] = None
        self.gaussian_scales: Optional[np.ndarray] = None
        
        self.curve_positions: Optional[List[np.ndarray]] = None
        self.curve_colors: Optional[List[np.ndarray]] = None
        
        # OpenGL state (initialized later)
        self._gl_initialized = False
        self._vao_gaussians = None
        self._vao_curves = None
        self._shader_program = None
        
        # Interaction state
        self._last_mouse_pos = None
        self._is_rotating = False
        self._is_panning = False
        self._is_zooming = False
    
    def set_gaussian_data(
        self,
        positions: np.ndarray,
        colors: Optional[np.ndarray] = None,
        scales: Optional[np.ndarray] = None
    ):
        """
        Set Gaussian splat data for rendering.
        
        Args:
            positions: (N, 3) array of positions
            colors: Optional (N, 3) array of RGB colors
            scales: Optional (N, 3) array of scales
        """
        self.gaussian_positions = positions.astype(np.float32)
        
        if colors is not None:
            self.gaussian_colors = np.clip(colors, 0, 1).astype(np.float32)
        else:
            self.gaussian_colors = np.ones((len(positions), 3), dtype=np.float32) * 0.5
        
        if scales is not None:
            self.gaussian_scales = scales.astype(np.float32)
        else:
            self.gaussian_scales = np.ones((len(positions), 3), dtype=np.float32) * 0.01
        
        # Update OpenGL buffers if initialized
        if self._gl_initialized:
            self._update_gaussian_buffers()
        
        # Auto-fit camera to data
        self._fit_camera_to_data()
    
    def set_curve_data(
        self,
        curves: List[np.ndarray],
        colors: Optional[List[np.ndarray]] = None
    ):
        """
        Set hair curve data for rendering.
        
        Args:
            curves: List of (N, 3) arrays, one per curve
            colors: Optional list of (N, 3) color arrays
        """
        self.curve_positions = [c.astype(np.float32) for c in curves]
        
        if colors is not None:
            self.curve_colors = [np.clip(c, 0, 1).astype(np.float32) for c in colors]
        else:
            self.curve_colors = [np.ones((len(c), 3), dtype=np.float32) * 0.3 for c in curves]
        
        # Update OpenGL buffers if initialized
        if self._gl_initialized:
            self._update_curve_buffers()
        
        # Auto-fit camera to data
        self._fit_camera_to_data()
    
    def set_view_mode(self, mode: ViewMode):
        """Set the display mode."""
        self.view_mode = mode
    
    def set_shading_mode(self, mode: ShadingMode):
        """Set the shading mode."""
        self.shading_mode = mode
    
    def _fit_camera_to_data(self):
        """Automatically fit camera to view all data."""
        all_positions = []
        
        if self.gaussian_positions is not None:
            all_positions.append(self.gaussian_positions)
        
        if self.curve_positions is not None:
            all_positions.extend(self.curve_positions)
        
        if not all_positions:
            return
        
        combined = np.vstack(all_positions)
        center = np.mean(combined, axis=0)
        
        # Compute bounding sphere radius
        distances = np.linalg.norm(combined - center, axis=1)
        radius = np.max(distances) if len(distances) > 0 else 1.0
        
        # Position camera to view entire scene
        self.camera.target = center
        self.camera.position = center + np.array([0, 0, radius * 2.5])
    
    def on_mouse_press(self, x: float, y: float, button: int):
        """Handle mouse press event."""
        self._last_mouse_pos = (x, y)
        
        if button == 0:  # Left button
            self._is_rotating = True
        elif button == 1:  # Right button
            self._is_panning = True
        elif button == 2:  # Middle button
            self._is_zooming = True
    
    def on_mouse_release(self, button: int):
        """Handle mouse release event."""
        self._is_rotating = False
        self._is_panning = False
        self._is_zooming = False
        self._last_mouse_pos = None
    
    def on_mouse_move(self, x: float, y: float):
        """Handle mouse move event."""
        if self._last_mouse_pos is None:
            return
        
        dx = x - self._last_mouse_pos[0]
        dy = y - self._last_mouse_pos[1]
        
        if self._is_rotating:
            self.camera.orbit(dx * 0.01, -dy * 0.01)
        elif self._is_panning:
            # Scale pan speed based on distance to target
            dist = np.linalg.norm(self.camera.position - self.camera.target)
            self.camera.pan(-dx * dist * 0.001, dy * dist * 0.001)
        elif self._is_zooming:
            self.camera.zoom(1.0 - dy * 0.01)
        
        self._last_mouse_pos = (x, y)
    
    def on_mouse_scroll(self, delta: float):
        """Handle mouse scroll event."""
        self.camera.zoom(1.0 - delta * 0.1)
    
    def render_to_image(self) -> np.ndarray:
        """
        Render scene to numpy array (for embedding in Tkinter).
        
        Returns:
            RGBA image as (H, W, 4) numpy array
        """
        # Software rendering fallback
        return self._software_render()
    
    def _software_render(self) -> np.ndarray:
        """Software renderer for preview."""
        # Create image buffer
        image = np.ones((self.height, self.width, 4), dtype=np.uint8)
        image[:, :, :3] = (np.array(self.background_color[:3]) * 255).astype(np.uint8)
        image[:, :, 3] = 255
        
        # Get view and projection matrices
        aspect = self.width / self.height
        view_matrix = self.camera.get_view_matrix()
        proj_matrix = self.camera.get_projection_matrix(aspect)
        mvp = proj_matrix @ view_matrix
        
        # Render based on mode
        if self.view_mode in [ViewMode.GAUSSIANS, ViewMode.BOTH]:
            self._render_gaussians_software(image, mvp)
        
        if self.view_mode in [ViewMode.CURVES, ViewMode.BOTH]:
            self._render_curves_software(image, mvp)
        
        return image
    
    def _project_point(self, point: np.ndarray, mvp: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """Project 3D point to screen coordinates."""
        p = np.append(point, 1.0)
        clip = mvp @ p
        
        if clip[3] <= 0:
            return None
        
        ndc = clip[:3] / clip[3]
        
        if np.any(np.abs(ndc[:2]) > 1.5):
            return None
        
        screen_x = int((ndc[0] + 1) * 0.5 * self.width)
        screen_y = int((1 - ndc[1]) * 0.5 * self.height)
        depth = ndc[2]
        
        return (screen_x, screen_y, depth)
    
    def _render_gaussians_software(self, image: np.ndarray, mvp: np.ndarray):
        """Vectorized software render for Gaussian points."""
        if self.gaussian_positions is None or len(self.gaussian_positions) == 0:
            return

        n = len(self.gaussian_positions)
        colors = self.gaussian_colors if self.gaussian_colors is not None else np.full((n, 3), 0.5, dtype=np.float32)

        # Batch homogeneous projection: (N, 4) @ (4, 4)^T = (N, 4)
        pos_h = np.ones((n, 4), dtype=np.float32)
        pos_h[:, :3] = self.gaussian_positions
        clip = (mvp @ pos_h.T).T  # (N, 4)

        # Keep points in front of camera
        valid = clip[:, 3] > 1e-6
        if not np.any(valid):
            return

        clip_v = clip[valid]
        colors_v = colors[valid]

        w = clip_v[:, 3:4]
        ndc = clip_v[:, :3] / w  # (M, 3)

        # Frustum cull (slightly relaxed to catch partially-visible points)
        in_view = np.all(np.abs(ndc[:, :2]) <= 1.5, axis=1)
        ndc = ndc[in_view]
        colors_v = colors_v[in_view]
        if len(ndc) == 0:
            return

        # Screen-space integer coords (will be clipped when writing)
        sx = ((ndc[:, 0] + 1) * 0.5 * self.width).astype(np.int32)
        sy = ((1.0 - ndc[:, 1]) * 0.5 * self.height).astype(np.int32)

        # Back-to-front depth sort
        order = np.argsort(-ndc[:, 2])
        sx = sx[order]
        sy = sy[order]
        rgb = (np.clip(colors_v[order], 0.0, 1.0) * 255).astype(np.uint8)

        ps = max(1, int(self.point_size))
        if ps == 1:
            # Fast path: one pixel per point
            in_bounds = (sx >= 0) & (sx < self.width) & (sy >= 0) & (sy < self.height)
            image[sy[in_bounds], sx[in_bounds], :3] = rgb[in_bounds]
        else:
            # Precompute disc offsets once
            r = ps
            dy_arr, dx_arr = np.mgrid[-r:r+1, -r:r+1]
            disc = dy_arr*dy_arr + dx_arr*dx_arr <= r*r
            offsets = np.column_stack([dy_arr[disc], dx_arr[disc]])
            for ody, odx in offsets:
                psx = sx + odx
                psy = sy + ody
                in_bounds = (psx >= 0) & (psx < self.width) & (psy >= 0) & (psy < self.height)
                image[psy[in_bounds], psx[in_bounds], :3] = rgb[in_bounds]

    def _render_curves_software(self, image: np.ndarray, mvp: np.ndarray):
        """Batch-project ALL curve points in one matrix multiply, draw with cv2.polylines."""
        if not self.curve_positions:
            return

        try:
            import cv2
            _cv2 = True
        except ImportError:
            _cv2 = False

        bgr = image[:, :, 2::-1].copy() if _cv2 else None

        # --- Single batch projection for all curves ---
        lengths = [len(c) for c in self.curve_positions]
        total = sum(lengths)
        if total == 0:
            return

        all_pts = np.ones((total, 4), dtype=np.float32)
        all_pts[:, :3] = np.vstack(self.curve_positions)
        clip = (mvp @ all_pts.T).T  # (total, 4)

        w = clip[:, 3]
        valid_w = w > 1e-6
        safe_w = np.where(valid_w, w, 1.0)
        ndc = clip[:, :3] / safe_w[:, None]
        in_view = valid_w & (np.abs(ndc[:, 0]) <= 1.5) & (np.abs(ndc[:, 1]) <= 1.5)

        sx = ((ndc[:, 0] + 1.0) * 0.5 * self.width).astype(np.int32)
        sy = ((1.0 - ndc[:, 1]) * 0.5 * self.height).astype(np.int32)

        # --- Split back per curve and draw ---
        offset = 0
        for ci, length in enumerate(lengths):
            end = offset + length
            if length < 2:
                offset = end
                continue

            s_iv = in_view[offset:end]
            s_sx = sx[offset:end]
            s_sy = sy[offset:end]
            colors = (
                self.curve_colors[ci]
                if self.curve_colors
                else np.full((length, 3), 0.3, dtype=np.float32)
            )
            avg_col = (np.mean(colors, axis=0) * 255).clip(0, 255).astype(np.uint8)

            if _cv2:
                bgr_col = (int(avg_col[2]), int(avg_col[1]), int(avg_col[0]))
                if s_iv.all():
                    # Fast path: entire curve in view — single polylines call
                    pts = np.stack([s_sx, s_sy], axis=1).reshape(-1, 1, 2).astype(np.int32)
                    cv2.polylines(bgr, [pts], False, bgr_col, 1, cv2.LINE_AA)
                else:
                    # Slow path: find contiguous valid runs
                    vi = s_iv.astype(np.int8)
                    diff = np.diff(np.concatenate([[0], vi, [0]]))
                    for s, e in zip(np.where(diff == 1)[0], np.where(diff == -1)[0]):
                        if e - s < 2:
                            continue
                        pts = np.stack([s_sx[s:e], s_sy[s:e]], axis=1).reshape(-1, 1, 2).astype(np.int32)
                        cv2.polylines(bgr, [pts], False, bgr_col, 1, cv2.LINE_AA)
            else:
                prev_i = None
                for i in range(length):
                    if s_iv[i]:
                        if prev_i is not None:
                            self._draw_line(image, int(s_sx[prev_i]), int(s_sy[prev_i]),
                                            int(s_sx[i]), int(s_sy[i]), colors[prev_i], colors[i])
                        prev_i = i
                    else:
                        prev_i = None

            offset = end

        if _cv2:
            image[:, :, :3] = bgr[:, :, ::-1]
    
    def _draw_line(
        self,
        image: np.ndarray,
        x0: int, y0: int,
        x1: int, y1: int,
        color0: np.ndarray,
        color1: np.ndarray
    ):
        """Draw a line using Bresenham's algorithm with color interpolation."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        err = dx - dy
        
        length = max(dx, dy)
        if length == 0:
            length = 1
        
        x, y = x0, y0
        step = 0
        
        while True:
            if 0 <= x < self.width and 0 <= y < self.height:
                t = step / length
                color = color0 * (1 - t) + color1 * t
                rgb = (color * 255).astype(np.uint8)
                image[y, x, :3] = rgb
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x += sx
            
            if e2 < dx:
                err += dx
                y += sy
            
            step += 1
    
    def screen_to_world_ray(
        self, sx: float, sy: float
    ) -> tuple:
        """Convert screen pixel (sx, sy) to a world-space ray (origin, direction)."""
        aspect = self.width / max(self.height, 1)
        mvp = self.camera.get_projection_matrix(aspect) @ self.camera.get_view_matrix()
        try:
            inv_mvp = np.linalg.inv(mvp)
        except np.linalg.LinAlgError:
            return None, None

        nx = (sx / self.width) * 2.0 - 1.0
        ny = 1.0 - (sy / self.height) * 2.0

        near_w = inv_mvp @ np.array([nx, ny, -1.0, 1.0])
        far_w  = inv_mvp @ np.array([nx, ny,  1.0, 1.0])
        near_w = near_w[:3] / near_w[3]
        far_w  = far_w[:3]  / far_w[3]

        direction = far_w - near_w
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return None, None
        return near_w.copy(), direction / norm

    def pick_nearest_gaussian(self, sx: float, sy: float, max_dist: float = 0.5):
        """Return 3D position of Gaussian nearest to the screen-click ray, or None."""
        if self.gaussian_positions is None or len(self.gaussian_positions) == 0:
            return None
        origin, direction = self.screen_to_world_ray(sx, sy)
        if origin is None:
            return None
        diff = self.gaussian_positions - origin           # (N, 3)
        perp = np.cross(diff, direction)                  # (N, 3)
        dists = np.linalg.norm(perp, axis=1)             # (N,)
        idx = int(np.argmin(dists))
        return self.gaussian_positions[idx].copy() if dists[idx] <= max_dist else None

    def get_render_stats(self) -> Dict[str, Any]:
        """Get rendering statistics."""
        stats = {
            "width": self.width,
            "height": self.height,
            "view_mode": self.view_mode.value,
            "shading_mode": self.shading_mode.value,
            "gaussian_count": len(self.gaussian_positions) if self.gaussian_positions is not None else 0,
            "curve_count": len(self.curve_positions) if self.curve_positions is not None else 0,
        }
        
        if self.curve_positions is not None:
            total_points = sum(len(c) for c in self.curve_positions)
            stats["curve_point_count"] = total_points
        
        return stats


# OpenGL-based renderer (when available)
class OpenGLViewer3D(Viewer3D):
    """
    OpenGL-accelerated 3D viewer.
    
    Provides hardware-accelerated rendering using ModernGL.
    """
    
    def __init__(self, width: int = 800, height: int = 600):
        super().__init__(width, height)
        self._ctx = None
        self._fbo = None
        
    def initialize_gl(self):
        """Initialize OpenGL context and resources."""
        try:
            import moderngl
            
            # Create standalone context
            self._ctx = moderngl.create_standalone_context()
            
            # Create framebuffer
            self._color_texture = self._ctx.texture((self.width, self.height), 4)
            self._depth_texture = self._ctx.depth_texture((self.width, self.height))
            self._fbo = self._ctx.framebuffer(
                color_attachments=[self._color_texture],
                depth_attachment=self._depth_texture
            )
            
            # Compile shaders
            self._compile_shaders()
            
            self._gl_initialized = True
            
        except ImportError:
            print("ModernGL not available, using software rendering")
            self._gl_initialized = False
        except Exception as e:
            print(f"OpenGL initialization failed: {e}")
            self._gl_initialized = False
    
    def _compile_shaders(self):
        """Compile shader programs."""
        vertex_shader = """
        #version 330
        
        in vec3 in_position;
        in vec3 in_color;
        
        out vec3 v_color;
        
        uniform mat4 u_mvp;
        uniform float u_point_size;
        
        void main() {
            gl_Position = u_mvp * vec4(in_position, 1.0);
            gl_PointSize = u_point_size;
            v_color = in_color;
        }
        """
        
        fragment_shader = """
        #version 330
        
        in vec3 v_color;
        out vec4 fragColor;
        
        void main() {
            fragColor = vec4(v_color, 1.0);
        }
        """
        
        self._shader_program = self._ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
    
    def render_to_image(self) -> np.ndarray:
        """Render to image using OpenGL if available."""
        if not self._gl_initialized:
            return self._software_render()
        
        try:
            return self._gl_render()
        except Exception:
            return self._software_render()
    
    def _gl_render(self) -> np.ndarray:
        """OpenGL rendering implementation."""
        # Bind framebuffer
        self._fbo.use()
        
        # Clear
        self._ctx.clear(
            self.background_color[0],
            self.background_color[1],
            self.background_color[2],
            self.background_color[3]
        )
        
        # Enable depth test
        self._ctx.enable(self._ctx.DEPTH_TEST)
        
        # Compute MVP matrix
        aspect = self.width / self.height
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix(aspect)
        mvp = proj @ view
        
        # Set uniforms
        self._shader_program['u_mvp'].write(mvp.astype('f4').tobytes())
        self._shader_program['u_point_size'].value = self.point_size
        
        # Render Gaussians
        if self.view_mode in [ViewMode.GAUSSIANS, ViewMode.BOTH]:
            if self._vao_gaussians is not None:
                self._vao_gaussians.render(mode=self._ctx.POINTS)
        
        # Render Curves
        if self.view_mode in [ViewMode.CURVES, ViewMode.BOTH]:
            if self._vao_curves is not None:
                self._vao_curves.render(mode=self._ctx.LINES)
        
        # Read pixels
        data = self._fbo.read(components=4)
        image = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4)
        
        # Flip vertically (OpenGL has Y up)
        image = np.flipud(image)
        
        return image
    
    def _update_gaussian_buffers(self):
        """Update OpenGL buffers for Gaussians."""
        if not self._gl_initialized or self.gaussian_positions is None:
            return
        
        # Create VBO
        positions = self.gaussian_positions.flatten()
        colors = self.gaussian_colors.flatten() if self.gaussian_colors is not None else np.ones(len(positions))
        
        vbo_positions = self._ctx.buffer(positions.astype('f4').tobytes())
        vbo_colors = self._ctx.buffer(colors.astype('f4').tobytes())
        
        self._vao_gaussians = self._ctx.vertex_array(
            self._shader_program,
            [
                (vbo_positions, '3f', 'in_position'),
                (vbo_colors, '3f', 'in_color'),
            ]
        )
    
    def _update_curve_buffers(self):
        """Update OpenGL buffers for curves."""
        if not self._gl_initialized or self.curve_positions is None:
            return
        
        # Combine all curves into line segments
        all_positions = []
        all_colors = []
        
        for i, curve in enumerate(self.curve_positions):
            colors = self.curve_colors[i] if self.curve_colors else np.ones((len(curve), 3)) * 0.3
            
            for j in range(len(curve) - 1):
                all_positions.extend([curve[j], curve[j + 1]])
                all_colors.extend([colors[j], colors[j + 1]])
        
        if not all_positions:
            return
        
        positions = np.array(all_positions, dtype=np.float32).flatten()
        colors = np.array(all_colors, dtype=np.float32).flatten()
        
        vbo_positions = self._ctx.buffer(positions.astype('f4').tobytes())
        vbo_colors = self._ctx.buffer(colors.astype('f4').tobytes())
        
        self._vao_curves = self._ctx.vertex_array(
            self._shader_program,
            [
                (vbo_positions, '3f', 'in_position'),
                (vbo_colors, '3f', 'in_color'),
            ]
        )