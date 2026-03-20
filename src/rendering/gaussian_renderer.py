"""
Gaussian Renderer Module
========================

Specialized renderer for 3D Gaussian Splatting visualization.
Implements view-dependent splat rendering with proper depth sorting
and alpha blending.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import math

# Import types
from pathlib import Path
from src.core.gaussian_generator import GaussianCloud, GaussianSplat


@dataclass
class RenderSettings:
    """Settings for Gaussian rendering."""
    width: int = 800
    height: int = 600
    background_color: Tuple[float, float, float] = (0.1, 0.1, 0.15)
    opacity_multiplier: float = 1.0
    scale_multiplier: float = 1.0
    max_splats_per_pixel: int = 32
    enable_antialiasing: bool = True
    show_ellipsoids: bool = False


class GaussianRenderer:
    """
    Renderer for 3D Gaussian Splatting visualization.
    
    Implements:
    - View-dependent Gaussian projection
    - Depth-sorted alpha blending
    - Efficient tile-based rendering
    """
    
    def __init__(self, settings: Optional[RenderSettings] = None):
        """
        Initialize the Gaussian renderer.
        
        Args:
            settings: Render settings
        """
        self.settings = settings or RenderSettings()
        self.cloud: Optional[GaussianCloud] = None
        
        # Camera parameters
        self.camera_position = np.array([0.0, 0.0, 3.0])
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.camera_up = np.array([0.0, 1.0, 0.0])
        self.camera_fov = 45.0
        
    def set_cloud(self, cloud: GaussianCloud):
        """Set the Gaussian cloud to render."""
        self.cloud = cloud
        self._auto_fit_camera()
    
    def set_camera(
        self,
        position: np.ndarray,
        target: np.ndarray,
        up: Optional[np.ndarray] = None
    ):
        """Set camera parameters."""
        self.camera_position = position
        self.camera_target = target
        if up is not None:
            self.camera_up = up
    
    def _auto_fit_camera(self):
        """Auto-fit camera to view entire cloud."""
        if self.cloud is None:
            return
        
        center = (self.cloud.bounds_min + self.cloud.bounds_max) / 2
        size = np.max(self.cloud.bounds_max - self.cloud.bounds_min)
        
        self.camera_target = center
        self.camera_position = center + np.array([0, 0, size * 2])
    
    def render(self) -> np.ndarray:
        """
        Render the Gaussian cloud.
        
        Returns:
            RGBA image as (H, W, 4) uint8 numpy array
        """
        if self.cloud is None or len(self.cloud.splats) == 0:
            return self._create_empty_image()
        
        # Compute view and projection matrices
        view_matrix = self._compute_view_matrix()
        proj_matrix = self._compute_projection_matrix()
        
        # Project all Gaussians to screen space
        projected_splats = self._project_gaussians(view_matrix, proj_matrix)
        
        # Sort by depth (back to front)
        projected_splats.sort(key=lambda x: -x['depth'])
        
        # Render with alpha blending
        image = self._render_splats(projected_splats)
        
        return image
    
    def _create_empty_image(self) -> np.ndarray:
        """Create empty image with background color."""
        image = np.ones((self.settings.height, self.settings.width, 4), dtype=np.uint8)
        image[:, :, 0] = int(self.settings.background_color[0] * 255)
        image[:, :, 1] = int(self.settings.background_color[1] * 255)
        image[:, :, 2] = int(self.settings.background_color[2] * 255)
        image[:, :, 3] = 255
        return image
    
    def _compute_view_matrix(self) -> np.ndarray:
        """Compute view matrix (look-at)."""
        forward = self.camera_target - self.camera_position
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        right = np.cross(forward, self.camera_up)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        up = np.cross(right, forward)
        
        rotation = np.array([
            [right[0], right[1], right[2], 0],
            [up[0], up[1], up[2], 0],
            [-forward[0], -forward[1], -forward[2], 0],
            [0, 0, 0, 1]
        ])
        
        translation = np.array([
            [1, 0, 0, -self.camera_position[0]],
            [0, 1, 0, -self.camera_position[1]],
            [0, 0, 1, -self.camera_position[2]],
            [0, 0, 0, 1]
        ])
        
        return rotation @ translation
    
    def _compute_projection_matrix(self) -> np.ndarray:
        """Compute perspective projection matrix."""
        aspect = self.settings.width / self.settings.height
        fov_rad = math.radians(self.camera_fov)
        f = 1.0 / math.tan(fov_rad / 2)
        near, far = 0.01, 100.0
        
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0]
        ])
    
    def _project_gaussians(
        self,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Project all Gaussians to screen space."""
        projected = []
        
        mvp = proj_matrix @ view_matrix
        
        for i, splat in enumerate(self.cloud.splats):
            # Project center
            pos_h = np.append(splat.position, 1.0)
            clip = mvp @ pos_h
            
            if clip[3] <= 0:
                continue
            
            ndc = clip[:3] / clip[3]
            
            # Clip to view frustum
            if np.any(np.abs(ndc[:2]) > 1.2):
                continue
            
            # Screen coordinates
            screen_x = (ndc[0] + 1) * 0.5 * self.settings.width
            screen_y = (1 - ndc[1]) * 0.5 * self.settings.height
            
            # Compute screen-space size based on scale and distance
            view_pos = view_matrix @ pos_h
            distance = abs(view_pos[2])
            
            # Project covariance to 2D
            cov_2d = self._project_covariance(
                splat.covariance, 
                view_matrix[:3, :3],
                distance
            )
            
            # Compute screen-space radius
            eigenvalues = np.linalg.eigvalsh(cov_2d)
            radius = np.sqrt(np.max(eigenvalues)) * self.settings.scale_multiplier
            
            # Scale by FOV and distance
            fov_scale = self.settings.height / (2 * math.tan(math.radians(self.camera_fov) / 2))
            screen_radius = max(1, int(radius * fov_scale / distance))
            
            projected.append({
                'index': i,
                'screen_x': screen_x,
                'screen_y': screen_y,
                'screen_radius': screen_radius,
                'cov_2d': cov_2d,
                'depth': ndc[2],
                'color': splat.color,
                'opacity': splat.opacity * self.settings.opacity_multiplier
            })
        
        return projected
    
    def _project_covariance(
        self,
        cov_3d: np.ndarray,
        rotation: np.ndarray,
        distance: float
    ) -> np.ndarray:
        """Project 3D covariance to 2D screen space."""
        # Rotate covariance to view space
        cov_view = rotation @ cov_3d @ rotation.T
        
        # Take upper-left 2x2 (x, y components in view space)
        cov_2d = cov_view[:2, :2]
        
        return cov_2d
    
    def _render_splats(self, projected_splats: List[Dict[str, Any]]) -> np.ndarray:
        """Render projected splats with alpha blending."""
        # Initialize image with background
        image = np.zeros((self.settings.height, self.settings.width, 4), dtype=np.float32)
        image[:, :, :3] = self.settings.background_color
        image[:, :, 3] = 1.0
        
        # Accumulated alpha for proper compositing
        accum_alpha = np.zeros((self.settings.height, self.settings.width), dtype=np.float32)
        
        for splat_data in projected_splats:
            x = splat_data['screen_x']
            y = splat_data['screen_y']
            radius = splat_data['screen_radius']
            color = splat_data['color']
            opacity = splat_data['opacity']
            cov_2d = splat_data['cov_2d']
            
            # Compute bounding box
            x0 = max(0, int(x - radius * 3))
            x1 = min(self.settings.width, int(x + radius * 3) + 1)
            y0 = max(0, int(y - radius * 3))
            y1 = min(self.settings.height, int(y + radius * 3) + 1)
            
            if x1 <= x0 or y1 <= y0:
                continue
            
            # Create coordinate grid
            yy, xx = np.mgrid[y0:y1, x0:x1]
            
            # Compute Gaussian weights
            dx = xx - x
            dy = yy - y
            
            # Use isotropic approximation for simplicity
            sigma_sq = radius * radius / 4
            weights = np.exp(-(dx*dx + dy*dy) / (2 * sigma_sq))
            
            # Apply opacity
            alpha = weights * opacity
            alpha = np.clip(alpha, 0, 1)
            
            # Check accumulated alpha
            remaining_alpha = 1.0 - accum_alpha[y0:y1, x0:x1]
            effective_alpha = alpha * remaining_alpha
            
            # Blend color
            for c in range(3):
                image[y0:y1, x0:x1, c] = (
                    image[y0:y1, x0:x1, c] * (1 - effective_alpha) +
                    color[c] * effective_alpha
                )
            
            # Update accumulated alpha
            accum_alpha[y0:y1, x0:x1] += effective_alpha
            accum_alpha = np.clip(accum_alpha, 0, 1)
        
        # Convert to uint8
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        return image_uint8
    
    def render_depth(self) -> np.ndarray:
        """
        Render depth map.
        
        Returns:
            Depth image as (H, W) float32 array
        """
        if self.cloud is None:
            return np.zeros((self.settings.height, self.settings.width), dtype=np.float32)
        
        view_matrix = self._compute_view_matrix()
        proj_matrix = self._compute_projection_matrix()
        
        depth_image = np.ones((self.settings.height, self.settings.width), dtype=np.float32)
        
        mvp = proj_matrix @ view_matrix
        
        for splat in self.cloud.splats:
            pos_h = np.append(splat.position, 1.0)
            clip = mvp @ pos_h
            
            if clip[3] <= 0:
                continue
            
            ndc = clip[:3] / clip[3]
            
            if np.any(np.abs(ndc[:2]) > 1.0):
                continue
            
            x = int((ndc[0] + 1) * 0.5 * self.settings.width)
            y = int((1 - ndc[1]) * 0.5 * self.settings.height)
            
            if 0 <= x < self.settings.width and 0 <= y < self.settings.height:
                depth = (ndc[2] + 1) * 0.5  # Map to [0, 1]
                depth_image[y, x] = min(depth_image[y, x], depth)
        
        return depth_image
    
    def render_normals(self) -> np.ndarray:
        """
        Render normal map based on Gaussian orientations.
        
        Returns:
            Normal image as (H, W, 3) uint8 array
        """
        if self.cloud is None:
            return np.zeros((self.settings.height, self.settings.width, 3), dtype=np.uint8)
        
        view_matrix = self._compute_view_matrix()
        proj_matrix = self._compute_projection_matrix()
        
        normal_image = np.zeros((self.settings.height, self.settings.width, 3), dtype=np.float32)
        normal_image[:, :] = [0.5, 0.5, 1.0]  # Default: facing camera
        
        mvp = proj_matrix @ view_matrix
        rotation = view_matrix[:3, :3]
        
        for splat in self.cloud.splats:
            pos_h = np.append(splat.position, 1.0)
            clip = mvp @ pos_h
            
            if clip[3] <= 0:
                continue
            
            ndc = clip[:3] / clip[3]
            
            if np.any(np.abs(ndc[:2]) > 1.0):
                continue
            
            x = int((ndc[0] + 1) * 0.5 * self.settings.width)
            y = int((1 - ndc[1]) * 0.5 * self.settings.height)
            
            if 0 <= x < self.settings.width and 0 <= y < self.settings.height:
                # Get principal direction from covariance
                eigenvalues, eigenvectors = np.linalg.eigh(splat.covariance)
                normal = eigenvectors[:, np.argmax(eigenvalues)]
                
                # Transform to view space
                normal_view = rotation @ normal
                
                # Map to [0, 1] for visualization
                normal_vis = (normal_view + 1) * 0.5
                normal_image[y, x] = normal_vis
        
        return (normal_image * 255).astype(np.uint8)
    
    def get_splat_at_pixel(self, x: int, y: int) -> Optional[int]:
        """
        Get the index of the splat at the given pixel.
        
        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            
        Returns:
            Splat index or None if no splat at pixel
        """
        if self.cloud is None:
            return None
        
        view_matrix = self._compute_view_matrix()
        proj_matrix = self._compute_projection_matrix()
        
        projected = self._project_gaussians(view_matrix, proj_matrix)
        
        # Sort by depth (front to back for picking)
        projected.sort(key=lambda s: s['depth'])
        
        for splat_data in projected:
            sx = splat_data['screen_x']
            sy = splat_data['screen_y']
            radius = splat_data['screen_radius']
            
            # Check if pixel is within splat
            dist = math.sqrt((x - sx)**2 + (y - sy)**2)
            if dist < radius * 2:
                return splat_data['index']
        
        return None