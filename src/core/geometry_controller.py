"""
Geometry Controller Module
==========================

This module provides geometry-aware splatter control for hair reconstruction.
It manages the relationship between Gaussian splats and their geometric properties,
enabling fine-grained control over the reconstruction process.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

from src.core.gaussian_generator import GaussianCloud, GaussianSplat


class ControlMode(Enum):
    """Control modes for geometry manipulation."""
    DENSITY = "density"
    ORIENTATION = "orientation"
    SCALE = "scale"
    COLOR = "color"
    ALL = "all"


@dataclass
class GeometryBrush:
    """Brush for interactive geometry editing."""
    center: np.ndarray
    radius: float
    falloff: float  # 0 = hard, 1 = soft
    strength: float
    mode: ControlMode


class GeometryController:
    """
    Controls geometry-aware properties of Gaussian splats.
    
    Provides tools for:
    - Density control (add/remove splats)
    - Orientation alignment
    - Scale adjustment
    - Color manipulation
    - Interactive brush editing
    """
    
    def __init__(self):
        """Initialize the geometry controller."""
        self.current_cloud: Optional[GaussianCloud] = None
        self.edit_history: List[Dict[str, Any]] = []
        self.max_history = 50
        
        # Control parameters
        self.params = {
            'density_threshold': 0.01,
            'orientation_smoothness': 0.5,
            'scale_uniformity': 0.3,
            'color_coherence': 0.7,
        }
    
    def set_cloud(self, cloud: GaussianCloud):
        """Set the current Gaussian cloud to control."""
        self.current_cloud = cloud
        self.edit_history.clear()
    
    def get_cloud(self) -> Optional[GaussianCloud]:
        """Get the current controlled cloud."""
        return self.current_cloud
    
    def _save_state(self):
        """Save current state for undo."""
        if self.current_cloud is None:
            return
        
        state = {
            'splats': [
                GaussianSplat(
                    position=s.position.copy(),
                    covariance=s.covariance.copy(),
                    color=s.color.copy(),
                    opacity=s.opacity,
                    scale=s.scale.copy(),
                    rotation=s.rotation.copy()
                )
                for s in self.current_cloud.splats
            ],
            'bounds_min': self.current_cloud.bounds_min.copy(),
            'bounds_max': self.current_cloud.bounds_max.copy()
        }
        
        self.edit_history.append(state)
        
        # Limit history size
        if len(self.edit_history) > self.max_history:
            self.edit_history.pop(0)
    
    def undo(self) -> bool:
        """Undo last operation."""
        if not self.edit_history:
            return False
        
        state = self.edit_history.pop()
        self.current_cloud = GaussianCloud(
            splats=state['splats'],
            bounds_min=state['bounds_min'],
            bounds_max=state['bounds_max']
        )
        return True
    
    def apply_brush(
        self,
        brush: GeometryBrush,
        target_value: Optional[Any] = None
    ) -> bool:
        """
        Apply brush operation to the cloud.
        
        Args:
            brush: Brush parameters
            target_value: Optional target value for the operation
            
        Returns:
            True if operation succeeded
        """
        if self.current_cloud is None:
            return False
        
        self._save_state()
        
        positions = self.current_cloud.get_positions()
        
        # Compute distances from brush center
        distances = np.linalg.norm(positions - brush.center, axis=1)
        
        # Compute influence weights
        weights = np.zeros(len(distances))
        mask = distances < brush.radius
        
        if not np.any(mask):
            return False
        
        # Apply falloff
        normalized_dist = distances[mask] / brush.radius
        if brush.falloff > 0:
            weights[mask] = 1.0 - (normalized_dist ** (1.0 / brush.falloff))
        else:
            weights[mask] = 1.0
        
        weights = weights * brush.strength
        
        # Apply based on mode
        if brush.mode == ControlMode.DENSITY:
            self._apply_density_brush(weights, target_value)
        elif brush.mode == ControlMode.ORIENTATION:
            self._apply_orientation_brush(weights, target_value)
        elif brush.mode == ControlMode.SCALE:
            self._apply_scale_brush(weights, target_value)
        elif brush.mode == ControlMode.COLOR:
            self._apply_color_brush(weights, target_value)
        elif brush.mode == ControlMode.ALL:
            self._apply_all_brush(weights, target_value)
        
        return True
    
    def _apply_density_brush(
        self,
        weights: np.ndarray,
        target_density: Optional[float] = None
    ):
        """Modify splat density based on brush."""
        if target_density is None:
            target_density = 1.0
        
        # Adjust opacity based on density target
        for i, splat in enumerate(self.current_cloud.splats):
            if weights[i] > 0:
                current_opacity = splat.opacity
                target_opacity = current_opacity * target_density
                splat.opacity = current_opacity + weights[i] * (target_opacity - current_opacity)
                splat.opacity = np.clip(splat.opacity, 0.01, 1.0)
    
    def _apply_orientation_brush(
        self,
        weights: np.ndarray,
        target_direction: Optional[np.ndarray] = None
    ):
        """Align splat orientations based on brush."""
        if target_direction is None:
            target_direction = np.array([0, -1, 0])  # Default: downward
        
        target_direction = target_direction / (np.linalg.norm(target_direction) + 1e-6)
        
        for i, splat in enumerate(self.current_cloud.splats):
            if weights[i] > 0:
                # Get current principal direction from covariance
                eigenvalues, eigenvectors = np.linalg.eigh(splat.covariance)
                principal_idx = np.argmax(eigenvalues)
                current_dir = eigenvectors[:, principal_idx]
                
                # Interpolate towards target
                new_dir = current_dir + weights[i] * (target_direction - current_dir)
                new_dir = new_dir / (np.linalg.norm(new_dir) + 1e-6)
                
                # Reconstruct covariance with new orientation
                # Simple approximation: rotate the covariance
                scale = np.sqrt(eigenvalues)
                
                # Create rotation from current to new direction
                axis = np.cross(current_dir, new_dir)
                axis_norm = np.linalg.norm(axis)
                
                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                    angle = np.arccos(np.clip(np.dot(current_dir, new_dir), -1, 1))
                    
                    # Rodrigues' rotation formula
                    K = np.array([
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]
                    ])
                    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
                    
                    # Rotate covariance
                    splat.covariance = R @ splat.covariance @ R.T
    
    def _apply_scale_brush(
        self,
        weights: np.ndarray,
        scale_factor: Optional[float] = None
    ):
        """Modify splat scales based on brush."""
        if scale_factor is None:
            scale_factor = 1.0
        
        for i, splat in enumerate(self.current_cloud.splats):
            if weights[i] > 0:
                factor = 1.0 + weights[i] * (scale_factor - 1.0)
                splat.scale = splat.scale * factor
                splat.scale = np.clip(splat.scale, 0.0001, 0.5)
                splat.covariance = splat.covariance * (factor ** 2)
    
    def _apply_color_brush(
        self,
        weights: np.ndarray,
        target_color: Optional[np.ndarray] = None
    ):
        """Modify splat colors based on brush."""
        if target_color is None:
            target_color = np.array([0.3, 0.2, 0.1])  # Default: brown
        
        for i, splat in enumerate(self.current_cloud.splats):
            if weights[i] > 0:
                splat.color = splat.color + weights[i] * (target_color - splat.color)
                splat.color = np.clip(splat.color, 0, 1)
    
    def _apply_all_brush(
        self,
        weights: np.ndarray,
        params: Optional[Dict[str, Any]] = None
    ):
        """Apply all control modes based on brush."""
        if params is None:
            params = {}
        
        if 'density' in params:
            self._apply_density_brush(weights, params['density'])
        if 'direction' in params:
            self._apply_orientation_brush(weights, params['direction'])
        if 'scale' in params:
            self._apply_scale_brush(weights, params['scale'])
        if 'color' in params:
            self._apply_color_brush(weights, params['color'])
    
    def smooth_region(
        self,
        center: np.ndarray,
        radius: float,
        iterations: int = 3
    ) -> bool:
        """
        Apply smoothing to a region of the cloud.
        
        Args:
            center: Center of smoothing region
            radius: Radius of smoothing region
            iterations: Number of smoothing iterations
            
        Returns:
            True if smoothing was applied
        """
        if self.current_cloud is None:
            return False
        
        self._save_state()
        
        from scipy.spatial import cKDTree
        
        positions = self.current_cloud.get_positions()
        tree = cKDTree(positions)
        
        # Find splats in region
        indices = tree.query_ball_point(center, radius)
        
        if not indices:
            return False
        
        for _ in range(iterations):
            new_positions = []
            new_colors = []
            
            for i in indices:
                # Find neighbors
                neighbors = tree.query_ball_point(positions[i], radius * 0.3)
                
                if len(neighbors) > 1:
                    # Average position
                    neighbor_positions = positions[neighbors]
                    new_pos = np.mean(neighbor_positions, axis=0)
                    
                    # Blend with original
                    alpha = self.params['orientation_smoothness']
                    new_positions.append(
                        alpha * new_pos + (1 - alpha) * positions[i]
                    )
                    
                    # Average color
                    neighbor_colors = np.array([
                        self.current_cloud.splats[j].color for j in neighbors
                    ])
                    new_color = np.mean(neighbor_colors, axis=0)
                    new_colors.append(
                        alpha * new_color + (1 - alpha) * self.current_cloud.splats[i].color
                    )
                else:
                    new_positions.append(positions[i])
                    new_colors.append(self.current_cloud.splats[i].color)
            
            # Apply updates
            for idx, i in enumerate(indices):
                self.current_cloud.splats[i].position = new_positions[idx]
                self.current_cloud.splats[i].color = new_colors[idx]
            
            # Rebuild tree for next iteration
            positions = self.current_cloud.get_positions()
            tree = cKDTree(positions)
        
        return True
    
    def align_to_surface(
        self,
        surface_normals: np.ndarray,
        surface_points: np.ndarray,
        influence_radius: float = 0.1
    ) -> bool:
        """
        Align splat orientations to a reference surface.
        
        Args:
            surface_normals: Normal vectors of reference surface
            surface_points: Points on reference surface
            influence_radius: Radius of influence for alignment
            
        Returns:
            True if alignment was applied
        """
        if self.current_cloud is None:
            return False
        
        self._save_state()
        
        from scipy.spatial import cKDTree
        
        surface_tree = cKDTree(surface_points)
        positions = self.current_cloud.get_positions()
        
        for i, pos in enumerate(positions):
            # Find closest surface point
            dist, idx = surface_tree.query(pos)
            
            if dist < influence_radius:
                # Get surface normal at closest point
                normal = surface_normals[idx]
                
                # Hair should be perpendicular to scalp, pointing away
                target_dir = normal
                if target_dir[1] < 0:  # Ensure pointing generally upward/outward
                    target_dir = -target_dir
                
                # Blend based on distance
                blend = 1.0 - (dist / influence_radius)
                
                # Apply orientation change
                brush = GeometryBrush(
                    center=pos,
                    radius=0.01,
                    falloff=0,
                    strength=blend,
                    mode=ControlMode.ORIENTATION
                )
                weights = np.zeros(len(positions))
                weights[i] = blend
                
                # Get current direction
                eigenvalues, eigenvectors = np.linalg.eigh(
                    self.current_cloud.splats[i].covariance
                )
                principal_idx = np.argmax(eigenvalues)
                current_dir = eigenvectors[:, principal_idx]
                
                # Interpolate
                new_dir = current_dir + blend * (target_dir - current_dir)
                new_dir = new_dir / (np.linalg.norm(new_dir) + 1e-6)
                
                # Update covariance orientation (simplified)
                scale = np.sqrt(eigenvalues[principal_idx])
                self.current_cloud.splats[i].covariance = \
                    np.outer(new_dir, new_dir) * (scale ** 2) + \
                    np.eye(3) * 0.0001
        
        return True
    
    def densify_region(
        self,
        center: np.ndarray,
        radius: float,
        factor: float = 2.0
    ) -> bool:
        """
        Increase splat density in a region by subdivision.
        
        Args:
            center: Center of densification region
            radius: Radius of region
            factor: Densification factor (2 = double density)
            
        Returns:
            True if densification was applied
        """
        if self.current_cloud is None:
            return False
        
        self._save_state()
        
        positions = self.current_cloud.get_positions()
        
        # Find splats in region
        distances = np.linalg.norm(positions - center, axis=1)
        mask = distances < radius
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            return False
        
        new_splats = list(self.current_cloud.splats)
        
        # Add new splats around existing ones
        num_new_per_splat = int(factor - 1)
        
        for i in indices:
            original = self.current_cloud.splats[i]
            
            for _ in range(num_new_per_splat):
                # Random offset within splat scale
                offset = np.random.randn(3) * original.scale * 0.5
                
                new_splat = GaussianSplat(
                    position=original.position + offset,
                    covariance=original.covariance * 0.5,
                    color=original.color + np.random.randn(3) * 0.02,
                    opacity=original.opacity * 0.9,
                    scale=original.scale * 0.7,
                    rotation=original.rotation.copy()
                )
                new_splat.color = np.clip(new_splat.color, 0, 1)
                new_splats.append(new_splat)
        
        # Update cloud
        all_positions = np.array([s.position for s in new_splats])
        self.current_cloud = GaussianCloud(
            splats=new_splats,
            bounds_min=np.min(all_positions, axis=0),
            bounds_max=np.max(all_positions, axis=0)
        )
        
        return True
    
    def prune_region(
        self,
        center: np.ndarray,
        radius: float,
        keep_ratio: float = 0.5
    ) -> bool:
        """
        Remove splats in a region.
        
        Args:
            center: Center of pruning region
            radius: Radius of region
            keep_ratio: Fraction of splats to keep (0-1)
            
        Returns:
            True if pruning was applied
        """
        if self.current_cloud is None:
            return False
        
        self._save_state()
        
        positions = self.current_cloud.get_positions()
        distances = np.linalg.norm(positions - center, axis=1)
        
        new_splats = []
        
        for i, splat in enumerate(self.current_cloud.splats):
            if distances[i] >= radius:
                # Outside region, keep
                new_splats.append(splat)
            elif np.random.random() < keep_ratio:
                # Inside region, randomly keep based on ratio
                new_splats.append(splat)
        
        if not new_splats:
            return False
        
        # Update cloud
        all_positions = np.array([s.position for s in new_splats])
        self.current_cloud = GaussianCloud(
            splats=new_splats,
            bounds_min=np.min(all_positions, axis=0),
            bounds_max=np.max(all_positions, axis=0)
        )
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current cloud."""
        if self.current_cloud is None:
            return {}
        
        positions = self.current_cloud.get_positions()
        opacities = self.current_cloud.get_opacities()
        scales = np.array([s.scale for s in self.current_cloud.splats])
        
        return {
            'num_splats': self.current_cloud.num_splats,
            'bounds_min': self.current_cloud.bounds_min.tolist(),
            'bounds_max': self.current_cloud.bounds_max.tolist(),
            'mean_position': np.mean(positions, axis=0).tolist(),
            'mean_opacity': float(np.mean(opacities)),
            'mean_scale': np.mean(scales, axis=0).tolist(),
            'std_position': np.std(positions, axis=0).tolist(),
            'std_scale': np.std(scales, axis=0).tolist(),
        }