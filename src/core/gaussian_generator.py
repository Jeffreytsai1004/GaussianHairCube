"""
Gaussian Generator Module
=========================

This module implements the 3D Gaussian Splatting generation for hair reconstruction.
Based on the GaussianHaircut approach, it generates strand-aligned 3D Gaussians
from input images or video frames.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import threading
from enum import Enum


class GaussianStatus(Enum):
    """Status of the Gaussian generation process."""
    IDLE = "idle"
    PREPROCESSING = "preprocessing"
    GENERATING = "generating"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class GaussianSplat:
    """Represents a single 3D Gaussian splat."""
    position: np.ndarray  # (3,) xyz coordinates
    covariance: np.ndarray  # (3, 3) covariance matrix
    color: np.ndarray  # (3,) RGB color
    opacity: float  # Alpha value
    scale: np.ndarray  # (3,) scale factors
    rotation: np.ndarray  # (4,) quaternion rotation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'position': self.position.tolist(),
            'covariance': self.covariance.tolist(),
            'color': self.color.tolist(),
            'opacity': self.opacity,
            'scale': self.scale.tolist(),
            'rotation': self.rotation.tolist()
        }


@dataclass 
class GaussianCloud:
    """Collection of Gaussian splats representing hair geometry."""
    splats: List[GaussianSplat]
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    
    @property
    def num_splats(self) -> int:
        return len(self.splats)
    
    def get_positions(self) -> np.ndarray:
        """Get all splat positions as (N, 3) array."""
        return np.array([s.position for s in self.splats])
    
    def get_colors(self) -> np.ndarray:
        """Get all splat colors as (N, 3) array."""
        return np.array([s.color for s in self.splats])
    
    def get_opacities(self) -> np.ndarray:
        """Get all opacities as (N,) array."""
        return np.array([s.opacity for s in self.splats])


class GaussianGenerator:
    """
    Generates 3D Gaussian splats for hair reconstruction.
    
    This class implements the core algorithm for converting input images
    or video frames into strand-aligned 3D Gaussian representations.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize the Gaussian generator.
        
        Args:
            device: Computing device ('cuda' or 'cpu')
        """
        self.device = device
        self.status = GaussianStatus.IDLE
        self.progress = 0.0
        self.current_cloud: Optional[GaussianCloud] = None
        self._lock = threading.Lock()
        self._cancel_flag = False
        
        # Generation parameters
        self.params = {
            'num_iterations': 30000,
            'learning_rate': 0.001,
            'densification_interval': 100,
            'opacity_reset_interval': 3000,
            'percent_dense': 0.01,
            'lambda_dssim': 0.2,
            'sh_degree': 3,
        }
        
    def set_parameters(self, **kwargs):
        """Update generation parameters."""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
                
    def generate_from_image(
        self, 
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        callback: Optional[callable] = None
    ) -> GaussianCloud:
        """
        Generate Gaussian splats from a single image.
        
        For single image input, we use a simplified approach that estimates
        depth and generates an initial Gaussian cloud.
        
        Args:
            image: Input image as (H, W, 3) numpy array
            mask: Optional hair mask as (H, W) boolean array
            callback: Progress callback function
            
        Returns:
            GaussianCloud containing the generated splats
        """
        with self._lock:
            self.status = GaussianStatus.PREPROCESSING
            self.progress = 0.0
            self._cancel_flag = False
        
        try:
            # Step 1: Preprocess image
            if callback:
                callback(0.1, "Preprocessing image...")
            processed_image = self._preprocess_image(image)
            
            if mask is None:
                if callback:
                    callback(0.15, "Generating hair mask...")
                mask = self._generate_hair_mask(processed_image)
            
            # Step 2: Estimate depth
            if callback:
                callback(0.25, "Estimating depth...")
            depth_map = self._estimate_depth(processed_image)
            
            # Step 3: Initialize point cloud from depth
            if callback:
                callback(0.35, "Initializing point cloud...")
            points, colors = self._depth_to_points(processed_image, depth_map, mask)
            
            with self._lock:
                self.status = GaussianStatus.GENERATING
            
            # Step 4: Generate initial Gaussians
            if callback:
                callback(0.5, "Generating Gaussians...")
            splats = self._initialize_gaussians(points, colors)
            
            # Step 5: Optimize Gaussians (simplified for single image)
            if callback:
                callback(0.7, "Optimizing Gaussians...")
            optimized_splats = self._optimize_gaussians_single_view(
                splats, processed_image, mask, callback
            )
            
            # Step 6: Create final cloud
            if callback:
                callback(0.95, "Finalizing...")
            
            bounds_min = np.min([s.position for s in optimized_splats], axis=0)
            bounds_max = np.max([s.position for s in optimized_splats], axis=0)
            
            cloud = GaussianCloud(
                splats=optimized_splats,
                bounds_min=bounds_min,
                bounds_max=bounds_max
            )
            
            with self._lock:
                self.current_cloud = cloud
                self.status = GaussianStatus.COMPLETED
                self.progress = 1.0
                
            if callback:
                callback(1.0, "Complete!")
                
            return cloud
            
        except Exception as e:
            with self._lock:
                self.status = GaussianStatus.ERROR
            raise RuntimeError(f"Gaussian generation failed: {str(e)}")
    
    def generate_from_video(
        self,
        frames: List[np.ndarray],
        callback: Optional[callable] = None
    ) -> GaussianCloud:
        """
        Generate Gaussian splats from video frames.
        
        Uses multi-view reconstruction approach similar to GaussianHaircut.
        
        Args:
            frames: List of video frames as (H, W, 3) numpy arrays
            callback: Progress callback function
            
        Returns:
            GaussianCloud containing the generated splats
        """
        with self._lock:
            self.status = GaussianStatus.PREPROCESSING
            self.progress = 0.0
            self._cancel_flag = False
        
        try:
            num_frames = len(frames)
            
            # Step 1: Extract camera poses
            if callback:
                callback(0.05, "Extracting camera poses...")
            camera_poses = self._estimate_camera_poses(frames)
            
            # Step 2: Generate masks for all frames
            if callback:
                callback(0.15, "Generating hair masks...")
            masks = [self._generate_hair_mask(f) for f in frames]
            
            # Step 3: Build initial point cloud from multi-view
            if callback:
                callback(0.25, "Building point cloud...")
            points, colors = self._multiview_reconstruction(frames, masks, camera_poses)
            
            with self._lock:
                self.status = GaussianStatus.GENERATING
            
            # Step 4: Initialize Gaussians
            if callback:
                callback(0.35, "Initializing Gaussians...")
            splats = self._initialize_gaussians(points, colors)
            
            # Step 5: Optimize with multi-view supervision
            if callback:
                callback(0.4, "Optimizing Gaussians...")
            optimized_splats = self._optimize_gaussians_multiview(
                splats, frames, masks, camera_poses, callback
            )
            
            # Step 6: Densification and pruning
            if callback:
                callback(0.9, "Densification and pruning...")
            final_splats = self._densify_and_prune(optimized_splats)
            
            # Step 7: Create final cloud
            if callback:
                callback(0.98, "Finalizing...")
            
            bounds_min = np.min([s.position for s in final_splats], axis=0)
            bounds_max = np.max([s.position for s in final_splats], axis=0)
            
            cloud = GaussianCloud(
                splats=final_splats,
                bounds_min=bounds_min,
                bounds_max=bounds_max
            )
            
            with self._lock:
                self.current_cloud = cloud
                self.status = GaussianStatus.COMPLETED
                self.progress = 1.0
                
            if callback:
                callback(1.0, "Complete!")
                
            return cloud
            
        except Exception as e:
            with self._lock:
                self.status = GaussianStatus.ERROR
            raise RuntimeError(f"Gaussian generation failed: {str(e)}")
    
    def cancel(self):
        """Cancel ongoing generation."""
        with self._lock:
            self._cancel_flag = True
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image (normalize, resize if needed)."""
        # Ensure float format in [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        return image
    
    def _generate_hair_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate hair segmentation mask using simple color-based approach."""
        # Simplified hair segmentation based on color
        # In production, this would use a trained segmentation model
        
        # Convert to grayscale for simple thresholding
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
            
        # Simple threshold-based mask (dark regions likely hair)
        # This is a placeholder - real implementation would use ML
        threshold = 0.4
        mask = gray < threshold
        
        # Clean up mask with morphological operations
        from scipy import ndimage
        mask = ndimage.binary_opening(mask, iterations=2)
        mask = ndimage.binary_closing(mask, iterations=2)
        
        return mask
    
    def _estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth map from single image."""
        # Simplified depth estimation
        # In production, this would use a trained depth estimation model (e.g., MiDaS)
        
        h, w = image.shape[:2]
        
        # Create a simple depth gradient (placeholder)
        # Real implementation would use neural depth estimation
        y_coords = np.linspace(0, 1, h)[:, np.newaxis]
        x_coords = np.linspace(0, 1, w)[np.newaxis, :]
        
        # Simple radial depth
        center_y, center_x = 0.5, 0.5
        dist = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        depth = 1.0 - dist * 0.5
        
        # Add some variation based on image intensity
        if len(image.shape) == 3:
            intensity = np.mean(image, axis=2)
        else:
            intensity = image
        
        depth = depth + intensity * 0.2
        depth = np.clip(depth, 0.1, 1.0)
        
        return depth.astype(np.float32)
    
    def _depth_to_points(
        self, 
        image: np.ndarray, 
        depth: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert depth map to 3D point cloud."""
        h, w = depth.shape
        
        # Create pixel coordinates
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Normalize to [-1, 1] range
        x_norm = (x_coords / w - 0.5) * 2
        y_norm = (y_coords / h - 0.5) * 2
        z_norm = depth * 2 - 1
        
        # Apply mask
        valid_mask = mask.astype(bool)
        
        # Stack coordinates
        points = np.stack([
            x_norm[valid_mask],
            -y_norm[valid_mask],  # Flip y
            z_norm[valid_mask]
        ], axis=1)
        
        # Get colors
        if len(image.shape) == 3:
            colors = image[valid_mask]
        else:
            colors = np.stack([image[valid_mask]] * 3, axis=1)
        
        return points.astype(np.float32), colors.astype(np.float32)
    
    def _initialize_gaussians(
        self, 
        points: np.ndarray, 
        colors: np.ndarray
    ) -> List[GaussianSplat]:
        """Initialize Gaussian splats from point cloud."""
        splats = []
        
        # Estimate point cloud density for scale initialization
        if len(points) > 1:
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            distances, _ = tree.query(points, k=min(4, len(points)))
            mean_dist = np.mean(distances[:, 1:]) if distances.shape[1] > 1 else 0.01
        else:
            mean_dist = 0.01
        
        # Initial scale based on point density
        init_scale = mean_dist * 0.5
        
        for i in range(len(points)):
            splat = GaussianSplat(
                position=points[i],
                covariance=np.eye(3) * (init_scale ** 2),
                color=colors[i] if i < len(colors) else np.array([0.5, 0.5, 0.5]),
                opacity=0.9,
                scale=np.array([init_scale, init_scale, init_scale]),
                rotation=np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
            )
            splats.append(splat)
        
        return splats
    
    def _optimize_gaussians_single_view(
        self,
        splats: List[GaussianSplat],
        target_image: np.ndarray,
        mask: np.ndarray,
        callback: Optional[callable] = None
    ) -> List[GaussianSplat]:
        """Optimize Gaussians for single view (simplified)."""
        # Simplified optimization - in production this would use differentiable rendering
        # For now, just return the initial splats with some adjustments
        
        optimized = []
        for i, splat in enumerate(splats):
            # Adjust scale based on local density
            new_scale = splat.scale * np.random.uniform(0.8, 1.2)
            
            # Slight position jitter for variety
            new_pos = splat.position + np.random.randn(3) * 0.01
            
            optimized.append(GaussianSplat(
                position=new_pos,
                covariance=np.eye(3) * np.mean(new_scale ** 2),
                color=splat.color,
                opacity=splat.opacity,
                scale=new_scale,
                rotation=splat.rotation
            ))
            
            if callback and i % 1000 == 0:
                progress = 0.7 + 0.2 * (i / len(splats))
                callback(progress, f"Optimizing Gaussian {i}/{len(splats)}")
        
        return optimized
    
    def _estimate_camera_poses(
        self, 
        frames: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Estimate camera poses from video frames."""
        # Simplified - return identity transforms
        # Production would use COLMAP or similar SfM
        poses = []
        for i in range(len(frames)):
            # Create simple rotation around Y axis
            angle = (i / len(frames)) * 2 * np.pi * 0.5  # 180 degree arc
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            pose = np.array([
                [cos_a, 0, sin_a, sin_a * 2],
                [0, 1, 0, 0],
                [-sin_a, 0, cos_a, cos_a * 2 + 2],
                [0, 0, 0, 1]
            ])
            poses.append(pose)
        
        return poses
    
    def _multiview_reconstruction(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        poses: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build point cloud from multiple views."""
        all_points = []
        all_colors = []
        
        for i, (frame, mask, pose) in enumerate(zip(frames, masks, poses)):
            # Estimate depth for this view
            depth = self._estimate_depth(frame)
            
            # Get points in camera space
            points, colors = self._depth_to_points(frame, depth, mask)
            
            # Transform to world space
            points_h = np.hstack([points, np.ones((len(points), 1))])
            world_points = (pose @ points_h.T).T[:, :3]
            
            all_points.append(world_points)
            all_colors.append(colors)
        
        # Concatenate and subsample
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        
        # Subsample if too many points
        max_points = 100000
        if len(all_points) > max_points:
            indices = np.random.choice(len(all_points), max_points, replace=False)
            all_points = all_points[indices]
            all_colors = all_colors[indices]
        
        return all_points, all_colors
    
    def _optimize_gaussians_multiview(
        self,
        splats: List[GaussianSplat],
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        poses: List[np.ndarray],
        callback: Optional[callable] = None
    ) -> List[GaussianSplat]:
        """Optimize Gaussians with multi-view supervision."""
        # Simplified multi-view optimization
        # Production would use differentiable Gaussian splatting
        
        num_iterations = min(self.params['num_iterations'], 1000)  # Reduce for demo
        
        for iteration in range(num_iterations):
            if self._cancel_flag:
                break
                
            # Simple update step (placeholder)
            for splat in splats:
                # Random perturbation
                splat.position += np.random.randn(3) * 0.001
                splat.scale *= (1 + np.random.randn(3) * 0.01)
                splat.scale = np.clip(splat.scale, 0.001, 0.1)
            
            if callback and iteration % 100 == 0:
                progress = 0.4 + 0.5 * (iteration / num_iterations)
                callback(progress, f"Iteration {iteration}/{num_iterations}")
        
        return splats
    
    def _densify_and_prune(
        self, 
        splats: List[GaussianSplat]
    ) -> List[GaussianSplat]:
        """Densify under-reconstructed regions and prune invalid Gaussians."""
        # Remove splats with very low opacity
        pruned = [s for s in splats if s.opacity > 0.1]
        
        # Remove splats that are too large
        max_scale = 0.2
        pruned = [s for s in pruned if np.max(s.scale) < max_scale]
        
        return pruned