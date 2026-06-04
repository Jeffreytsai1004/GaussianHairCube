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
import logging
from enum import Enum

logger = logging.getLogger(__name__)


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

        # Lazy-loaded ML models
        self._seg_model = None
        self._seg_processor = None
        self._depth_model = None
        self._depth_processor = None

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

    # ------------------------------------------------------------------
    # Model management helpers
    # ------------------------------------------------------------------

    def _load_seg_model(self):
        """Lazily load the jonathandinu/face-parsing segmentation model."""
        if self._seg_model is not None:
            return
        try:
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            logger.info("Loading face-parsing segmentation model…")
            self._seg_processor = SegformerImageProcessor.from_pretrained(
                "jonathandinu/face-parsing"
            )
            self._seg_model = SegformerForSemanticSegmentation.from_pretrained(
                "jonathandinu/face-parsing"
            )
            self._seg_model.to(self.device)
            self._seg_model.eval()
            logger.info("Segmentation model loaded.")
        except ImportError:
            logger.warning("transformers not installed — hair segmentation will use colour thresholding.")
        except Exception as exc:
            logger.warning("Failed to load segmentation model (%s) — falling back.", exc)
            self._seg_model = None
            self._seg_processor = None

    def _load_depth_model(self):
        """Lazily load the Depth-Anything-V2-Small depth estimation model."""
        if self._depth_model is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            logger.info("Loading Depth-Anything-V2-Small depth model…")
            self._depth_processor = AutoImageProcessor.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf"
            )
            self._depth_model = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf"
            )
            self._depth_model.to(self.device)
            self._depth_model.eval()
            logger.info("Depth model loaded.")
        except ImportError:
            logger.warning("transformers not installed — depth estimation will use gradient heuristic.")
        except Exception as exc:
            logger.warning("Failed to load depth model (%s) — falling back.", exc)
            self._depth_model = None
            self._depth_processor = None

    def _ensure_models_loaded(self, callback=None):
        """
        Trigger lazy loading of both ML models from the local HuggingFace cache.

        By the time this is called the download dialog has already ensured the
        model weights are present on disk, so these loads are disk-only and
        relatively fast.  Progress messages are emitted so the main progress bar
        shows meaningful text while the model graphs are being built.
        """
        if self._seg_model is None:
            if callback:
                callback(0.03, "正在加载发丝分割模型…")
            self._load_seg_model()

        if self._depth_model is None:
            if callback:
                callback(0.07, "正在加载深度估计模型…")
            self._load_depth_model()

        if callback:
            callback(0.10, "模型已就绪，开始处理…")

    # ------------------------------------------------------------------
    # Public generation entry-points
    # ------------------------------------------------------------------

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
            # Step 0: Ensure models are available
            self._ensure_models_loaded(callback)

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
                splats, processed_image, mask, depth_map, callback
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

    def generate_from_images(
        self,
        images: List[np.ndarray],
        callback: Optional[callable] = None
    ) -> GaussianCloud:
        """
        Generate Gaussian splats from multiple images via multi-view reconstruction.

        Args:
            images: List of input images as (H, W, 3) numpy arrays (minimum 3)
            callback: Progress callback function

        Returns:
            GaussianCloud containing the generated splats
        """
        frames = images  # internal alias
        with self._lock:
            self.status = GaussianStatus.PREPROCESSING
            self.progress = 0.0
            self._cancel_flag = False

        try:
            # Step 0: Ensure models are available
            self._ensure_models_loaded(callback)

            # ------------------------------------------------------------------
            # Steps 1-4 (NEW): Real multi-view reconstruction via
            #   src.core.multiview_reconstruction.reconstruct_from_frames
            #   4-tier fallback: pycolmap SfM → OpenCV chain → arc placeholder
            # ------------------------------------------------------------------
            from src.core.multiview_reconstruction import reconstruct_from_frames

            # Generate hair masks for all frames (used as hints by the SfM module)
            if callback:
                callback(0.10, "Generating hair masks...")
            hair_masks = [self._generate_hair_mask(self._preprocess_image(f)) for f in frames]

            if callback:
                callback(0.15, "Starting multi-view reconstruction...")

            sfm_points, sfm_colors, sfm_tier = reconstruct_from_frames(
                frames=[self._preprocess_image(f) for f in frames],
                hair_masks=hair_masks,
                callback=callback,
                cancel_flag_getter=lambda: self._cancel_flag,
            )

            tier_names = {1: "完整SfM", 2: "部分SfM", 3: "OpenCV", 4: "弧形占位符"}
            if callback:
                callback(
                    0.70,
                    f"三维重建完成 [Tier {sfm_tier}: {tier_names.get(sfm_tier, '')}]，"
                    f"{len(sfm_points)} 个点",
                )

            with self._lock:
                self.status = GaussianStatus.GENERATING

            # Initialize Gaussians from the reconstructed point cloud
            cloud_from_sfm = self._initialize_gaussians_from_points(
                sfm_points, sfm_colors, callback
            )
            splats = cloud_from_sfm.splats

            # Build camera_poses placeholder so the multiview optimiser below
            # still has something to iterate over (it is skipped on CPU anyway).
            camera_poses = self._estimate_camera_poses(frames)

            # Step 5: Optimize with multi-view supervision
            if callback:
                callback(0.4, "Optimizing Gaussians...")
            optimized_splats = self._optimize_gaussians_multiview(
                splats, frames, camera_poses, callback
            )

            # Step 6: Densification and pruning
            if callback:
                callback(0.9, "Densification and pruning...")
            final_splats = self._densify_and_prune(optimized_splats)

            # Step 7: Create final cloud
            if callback:
                callback(0.98, "Finalizing...")

            if final_splats:
                bounds_min = np.min([s.position for s in final_splats], axis=0)
                bounds_max = np.max([s.position for s in final_splats], axis=0)
            else:
                bounds_min = np.zeros(3, dtype=np.float32)
                bounds_max = np.zeros(3, dtype=np.float32)

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

    # ------------------------------------------------------------------
    # Core processing pipeline
    # ------------------------------------------------------------------

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image (normalize, resize if needed)."""
        # Ensure float format in [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        return image

    def _generate_hair_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate hair segmentation mask.

        Uses jonathandinu/face-parsing (SegFormer trained on CelebAMask-HQ)
        when transformers is available; falls back to colour thresholding.

        Args:
            image: float32 HxWx3 in [0, 1]

        Returns:
            bool numpy array HxW  (True = hair pixel)
        """
        orig_h, orig_w = image.shape[:2]

        # --- ML path ---
        try:
            if self._seg_model is None or self._seg_processor is None:
                raise RuntimeError("Segmentation model not loaded")

            import torch
            import torch.nn.functional as F
            from PIL import Image as PILImage

            # Convert to PIL (uint8)
            pil_img = PILImage.fromarray((image * 255).clip(0, 255).astype(np.uint8))

            # Preprocess
            inputs = self._seg_processor(images=pil_img, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)

            with torch.no_grad():
                outputs = self._seg_model(pixel_values=pixel_values)

            # Upsample logits to original resolution
            logits = outputs.logits  # 1 x num_labels x H' x W'
            upsampled = F.interpolate(
                logits,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            )
            label_map = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()  # HxW

            # Class 13 = hair in jonathandinu/face-parsing
            hair_mask = (label_map == 13)

            # Morphological cleanup
            from scipy import ndimage
            hair_mask = ndimage.binary_fill_holes(hair_mask)
            struct = np.ones((3, 3), dtype=bool)
            hair_mask = ndimage.binary_dilation(hair_mask, structure=struct)

            return hair_mask.astype(bool)

        except Exception as exc:
            logger.debug("ML hair segmentation failed (%s), using colour fallback.", exc)

        # --- Colour-based fallback ---
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        threshold = 0.4
        mask = gray < threshold

        from scipy import ndimage
        mask = ndimage.binary_opening(mask, iterations=2)
        mask = ndimage.binary_closing(mask, iterations=2)

        return mask

    def _estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from a single image.

        Uses depth-anything/Depth-Anything-V2-Small-hf when transformers is
        available; falls back to a gradient-based heuristic.

        Depth-Anything returns inverse depth (larger = closer), so the output
        is inverted to match _depth_to_points() convention (larger = further).

        Args:
            image: float32 HxWx3 in [0, 1]

        Returns:
            float32 HxW depth map in [0.1, 1.0]  (larger = further)
        """
        orig_h, orig_w = image.shape[:2]

        # --- ML path ---
        try:
            if self._depth_model is None or self._depth_processor is None:
                raise RuntimeError("Depth model not loaded")

            import torch
            import torch.nn.functional as F
            from PIL import Image as PILImage

            pil_img = PILImage.fromarray((image * 255).clip(0, 255).astype(np.uint8))

            inputs = self._depth_processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._depth_model(**inputs)

            predicted_depth = outputs.predicted_depth  # 1xH'xW' or HxW

            # Ensure shape [1, 1, H', W'] for interpolation
            if predicted_depth.dim() == 2:
                predicted_depth = predicted_depth.unsqueeze(0).unsqueeze(0)
            elif predicted_depth.dim() == 3:
                predicted_depth = predicted_depth.unsqueeze(1)

            # Upsample to original resolution
            depth_up = F.interpolate(
                predicted_depth,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            )
            depth = depth_up.squeeze().cpu().numpy().astype(np.float32)

            # Normalize to [0.1, 1.0]
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth = (depth - d_min) / (d_max - d_min) * 0.9 + 0.1
            else:
                depth = np.full_like(depth, 0.5)

            # Depth-Anything outputs inverse depth (large = close).
            # _depth_to_points() treats large depth as far (maps to high z).
            # Invert so that large values mean far.
            depth = 1.1 - depth  # maps [0.1,1.0] → [0.1,1.0]
            depth = np.clip(depth, 0.1, 1.0).astype(np.float32)

            return depth

        except Exception as exc:
            logger.debug("ML depth estimation failed (%s), using gradient fallback.", exc)

        # --- Gradient-based fallback ---
        h, w = image.shape[:2]

        y_coords = np.linspace(0, 1, h)[:, np.newaxis]
        x_coords = np.linspace(0, 1, w)[np.newaxis, :]

        center_y, center_x = 0.5, 0.5
        dist = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
        depth = 1.0 - dist * 0.5

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

        # Subsample to a manageable size — full-res images produce 50k-200k points
        # which makes downstream Python loops unusably slow.
        MAX_POINTS = 8000
        if len(points) > MAX_POINTS:
            idx = np.random.choice(len(points), MAX_POINTS, replace=False)
            points = points[idx]
            colors = colors[idx]

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

    def _initialize_gaussians_from_points(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        callback: Optional[callable] = None
    ) -> 'GaussianCloud':
        """
        Geometry-aware Gaussian initialization from a 3D point cloud.

        Uses local PCA (k=8 nearest neighbours) to orient each Gaussian's
        covariance matrix along the local fibre direction rather than using
        an isotropic identity covariance.

        Args:
            points:   float32 (N, 3) world-space positions
            colors:   float32 (N, 3) RGB colors in [0, 1]
            callback: Optional progress callback(progress, message)

        Returns:
            GaussianCloud with one splat per input point.
        """
        from scipy.spatial import cKDTree

        n = len(points)

        if n == 0:
            dummy_pos = np.zeros(3, dtype=np.float32)
            return GaussianCloud(
                splats=[],
                bounds_min=dummy_pos,
                bounds_max=dummy_pos,
            )

        if callback:
            callback(0.75, f"从 {n} 个点初始化高斯点云...")

        # Build KD-tree for local neighbourhoods
        tree = cKDTree(points)
        k_neighbors = min(8, n - 1)

        splats = []
        for i in range(n):
            if self._cancel_flag:
                break

            pos = points[i]
            color = colors[i]

            dists, idxs = tree.query(pos, k=k_neighbors + 1)
            idxs = idxs[1:]   # exclude self
            dists = dists[1:]

            init_scale = float(np.mean(dists) * 0.5) if len(dists) > 0 else 0.01
            init_scale = max(init_scale, 1e-4)

            # Local PCA for covariance orientation
            if len(idxs) >= 3:
                neighbors = points[idxs]
                centered = neighbors - pos
                if centered.shape[0] > 1:
                    cov = np.cov(centered.T).astype(np.float32)
                else:
                    cov = np.eye(3, dtype=np.float32) * (init_scale ** 2)
            else:
                cov = np.eye(3, dtype=np.float32) * (init_scale ** 2)

            splat = GaussianSplat(
                position=pos.copy().astype(np.float32),
                covariance=cov,
                color=color.clip(0, 1).astype(np.float32),
                opacity=0.9,
                scale=np.array([init_scale, init_scale, init_scale], dtype=np.float32),
                rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
            splats.append(splat)

            if callback and i % 1000 == 0 and n > 0:
                callback(0.75 + 0.10 * (i / n), f"初始化高斯: {i}/{n}")

        if len(splats) == 0:
            dummy_pos = np.zeros(3, dtype=np.float32)
            return GaussianCloud(
                splats=[],
                bounds_min=dummy_pos,
                bounds_max=dummy_pos,
            )

        all_pos = np.stack([s.position for s in splats])
        cloud = GaussianCloud(
            splats=splats,
            bounds_min=all_pos.min(axis=0),
            bounds_max=all_pos.max(axis=0),
        )

        if callback:
            callback(0.85, f"高斯初始化完成: {len(splats)} 个高斯点")

        return cloud

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _optimize_gaussians_single_view(
        self,
        splats: List[GaussianSplat],
        target_image: np.ndarray,
        mask: np.ndarray,
        depth: np.ndarray,
        callback: Optional[callable] = None
    ) -> List[GaussianSplat]:
        """
        Optimize Gaussians for a single view using differentiable rendering (GPU)
        or return unchanged on CPU.

        Args:
            splats:       Initial list of GaussianSplat objects
            target_image: float32 HxWx3 in [0,1]
            mask:         bool HxW — hair pixels
            depth:        float32 HxW depth map (unused directly, kept for API)
            callback:     Progress callback(progress: float, message: str)
        """
        try:
            import torch
        except ImportError:
            if callback:
                callback(1.0, "torch 未安装：跳过高斯优化")
            return splats

        if self.device == "cpu" or not torch.cuda.is_available():
            if callback:
                callback(1.0, "CPU 模式：跳过高斯优化（仅 GPU 支持完整优化）")
            return splats

        # ---- GPU optimisation path ----
        import torch.nn.functional as F

        device = self.device
        H, W = target_image.shape[:2]
        N = len(splats)

        if N == 0:
            return splats

        # Build tensors from splat list
        # scale: (N, 3)
        scales_np = np.stack([s.scale for s in splats], axis=0)           # Nx3
        positions_np = np.stack([s.position for s in splats], axis=0)     # Nx3
        colors_np = np.stack([s.color for s in splats], axis=0)           # Nx3
        opacities_np = np.array([s.opacity for s in splats])               # N

        # Clamp values to safe ranges before log/logit transforms
        scales_np = np.clip(scales_np, 1e-6, None)
        opacities_np = np.clip(opacities_np, 1e-4, 1 - 1e-4)

        positions = torch.tensor(positions_np, dtype=torch.float32, device=device, requires_grad=True)
        log_scales = torch.tensor(np.log(scales_np), dtype=torch.float32, device=device, requires_grad=True)
        raw_opacities = torch.tensor(
            np.log(opacities_np / (1.0 - opacities_np)),
            dtype=torch.float32, device=device, requires_grad=True
        )
        colors = torch.tensor(
            np.log(np.clip(colors_np, 1e-6, 1 - 1e-6) /
                   (1.0 - np.clip(colors_np, 1e-6, 1 - 1e-6))),
            dtype=torch.float32, device=device, requires_grad=True
        )

        target_t = torch.tensor(target_image, dtype=torch.float32, device=device)   # HxWx3
        hair_mask_t = torch.tensor(mask.astype(np.float32), dtype=torch.float32, device=device)  # HxW

        optimizer = torch.optim.Adam([
            {'params': [positions],     'lr': 1e-3},
            {'params': [log_scales],    'lr': 5e-3},
            {'params': [raw_opacities], 'lr': 5e-2},
            {'params': [colors],        'lr': 2.5e-3},
        ])

        num_iterations = getattr(self, 'num_iterations', self.params.get('num_iterations', 1000))
        num_iterations = min(num_iterations, 1000)  # Cap for single-view

        render_size = (128, 128)
        h, w = render_size

        target_small = F.interpolate(
            target_t.permute(2, 0, 1).unsqueeze(0),
            size=render_size, mode='bilinear', align_corners=False
        ).squeeze(0)  # 3xhxw

        mask_small = F.interpolate(
            hair_mask_t.unsqueeze(0).unsqueeze(0),
            size=render_size, mode='bilinear', align_corners=False
        ).squeeze()  # hxw

        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )

        def _project(pos3d):
            """Orthographic projection to screen coords."""
            pos_min = pos3d.min(dim=0).values
            pos_max = pos3d.max(dim=0).values
            pos_range = (pos_max - pos_min).clamp(min=1e-6)
            pos_norm = (pos3d - pos_min) / pos_range * 2 - 1  # Nx3 in [-1,1]
            sx = ((pos_norm[:, 0] + 1) / 2 * w).clamp(0, w - 1)
            sy = ((pos_norm[:, 1] + 1) / 2 * h).clamp(0, h - 1)
            dz = pos_norm[:, 2]
            return sx, sy, dz

        for i in range(num_iterations):
            if self._cancel_flag:
                break

            optimizer.zero_grad()

            scales_exp = torch.exp(log_scales).clamp(1e-6, 10.0)   # Nx3
            opacities_sig = torch.sigmoid(raw_opacities).clamp(1e-4, 1 - 1e-4)  # N
            cols_sig = torch.sigmoid(colors).clamp(0.0, 1.0)       # Nx3

            sx, sy, dz = _project(positions)

            # Sort back-to-front
            sort_idx = torch.argsort(dz, descending=True)
            sx = sx[sort_idx];  sy = sy[sort_idx]
            op = opacities_sig[sort_idx]
            col = cols_sig[sort_idx]
            sc = scales_exp[sort_idx].mean(dim=1)  # N — avg scale for 2D radius

            rendered = torch.zeros(3, h, w, device=device)
            alpha_acc = torch.zeros(h, w, device=device)

            batch_sz = min(500, N)
            max_splats = min(N, 2000)

            for b_start in range(0, max_splats, batch_sz):
                b_end = min(b_start + batch_sz, max_splats)
                b_sx  = sx[b_start:b_end]
                b_sy  = sy[b_start:b_end]
                b_op  = op[b_start:b_end]
                b_col = col[b_start:b_end]
                b_sc  = sc[b_start:b_end].clamp(0.5, max(h, w) / 4)

                dx = xx.unsqueeze(0) - b_sx.view(-1, 1, 1)   # BxHxW
                dy = yy.unsqueeze(0) - b_sy.view(-1, 1, 1)   # BxHxW
                sigma = b_sc.view(-1, 1, 1)
                gauss_w = torch.exp(-0.5 * (dx ** 2 + dy ** 2) / (sigma ** 2 + 1e-6))
                gauss_w = gauss_w * b_op.view(-1, 1, 1)       # BxHxW

                for c in range(3):
                    rendered[c] += (gauss_w * b_col[:, c].view(-1, 1, 1)).sum(dim=0)
                alpha_acc += gauss_w.sum(dim=0)

            alpha_safe = alpha_acc.clamp(min=1e-6)
            rendered = (rendered / alpha_safe).clamp(0.0, 1.0)

            # L1 loss on hair region
            m = mask_small.unsqueeze(0)
            loss = F.l1_loss(rendered * m, target_small * m)

            # Opacity entropy regularisation (encourages decisive opacities)
            loss = loss + 0.01 * (-(opacities_sig * torch.log(opacities_sig + 1e-6))).mean()

            loss.backward()
            optimizer.step()

            if i % 50 == 0 and callback:
                prog = 0.5 + (i / num_iterations) * 0.4
                callback(prog, f"优化高斯: 迭代 {i}/{num_iterations}, 损失: {loss.item():.4f}")

        # Write back optimised parameters to splat list
        with torch.no_grad():
            final_positions  = positions.detach().cpu().numpy()        # Nx3
            final_scales     = torch.exp(log_scales).detach().cpu().numpy()  # Nx3
            final_opacities  = torch.sigmoid(raw_opacities).detach().cpu().numpy()  # N
            final_colors     = torch.sigmoid(colors).detach().cpu().numpy()          # Nx3

        new_splats: List[GaussianSplat] = []
        for j, s in enumerate(splats):
            new_s = GaussianSplat(
                position=final_positions[j],
                covariance=np.diag(final_scales[j] ** 2),
                color=final_colors[j],
                opacity=float(final_opacities[j]),
                scale=final_scales[j],       # ndarray (3,)
                rotation=s.rotation
            )
            new_splats.append(new_s)

        if callback:
            callback(0.95, f"高斯优化完成，{len(new_splats)} 个高斯点")

        return new_splats

    def _estimate_camera_poses(
        self,
        frames: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Estimate camera poses from video frames."""
        # Simplified — return identity transforms
        # Production would use COLMAP or similar SfM
        poses = []
        for i in range(len(frames)):
            angle = (i / len(frames)) * 2 * np.pi * 0.5
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
            depth = self._estimate_depth(frame)
            points, colors = self._depth_to_points(frame, depth, mask)

            points_h = np.hstack([points, np.ones((len(points), 1))])
            world_points = (pose @ points_h.T).T[:, :3]

            all_points.append(world_points)
            all_colors.append(colors)

        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)

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
        camera_poses: List[np.ndarray],
        callback: Optional[callable] = None
    ) -> List[GaussianSplat]:
        """
        Optimize Gaussians with multi-view supervision.

        GPU: iterates over views applying differentiable optimisation each round.
        CPU: returns splats unchanged.
        """
        try:
            import torch
        except ImportError:
            if callback:
                callback(1.0, "torch 未安装：跳过多视角高斯优化")
            return splats

        if self.device == "cpu" or not torch.cuda.is_available():
            if callback:
                callback(1.0, "CPU 模式：跳过多视角高斯优化（仅 GPU 支持完整优化）")
            return splats

        # On GPU: apply single-view optimisation round-robin over frames
        num_rounds = max(1, min(len(frames), 5))  # cap to 5 rounds for speed
        for round_idx in range(num_rounds):
            if self._cancel_flag:
                break

            frame = frames[round_idx % len(frames)]
            frame_processed = self._preprocess_image(frame)
            frame_mask = self._generate_hair_mask(frame_processed)
            frame_depth = self._estimate_depth(frame_processed)

            def _round_cb(prog, msg):
                if callback:
                    overall = 0.4 + (round_idx / num_rounds) * 0.5 + (prog * 0.5 / num_rounds)
                    callback(overall, f"[视角 {round_idx + 1}/{num_rounds}] {msg}")

            splats = self._optimize_gaussians_single_view(
                splats, frame_processed, frame_mask, frame_depth, _round_cb
            )

        if callback:
            callback(0.9, f"多视角优化完成，{len(splats)} 个高斯点")

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
