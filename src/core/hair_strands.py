"""
Hair Strands Extractor Module
=============================

This module implements the extraction of hair strand curves from Gaussian splats.
It converts the unstructured Gaussian representation into polyline strands
suitable for export to DCC applications like Maya.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import heapq

from src.core.gaussian_generator import GaussianCloud, GaussianSplat


class StrandExtractionMethod(Enum):
    """Methods for extracting strands from Gaussians."""
    CLUSTERING = "clustering"
    FLOW_FIELD = "flow_field"
    NEURAL = "neural"


@dataclass
class HairStrand:
    """Represents a single hair strand as a polyline."""
    points: np.ndarray  # (N, 3) control points
    radii: np.ndarray   # (N,) radius at each point
    colors: np.ndarray  # (N, 3) color at each point
    
    @property
    def num_points(self) -> int:
        return len(self.points)
    
    @property
    def length(self) -> float:
        """Compute total strand length."""
        if self.num_points < 2:
            return 0.0
        segments = np.diff(self.points, axis=0)
        return float(np.sum(np.linalg.norm(segments, axis=1)))
    
    def resample(self, num_points: int) -> 'HairStrand':
        """Resample strand to have specified number of points."""
        if self.num_points < 2:
            return self
        
        # Compute cumulative arc length
        segments = np.diff(self.points, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        cumulative = np.zeros(self.num_points)
        cumulative[1:] = np.cumsum(segment_lengths)
        total_length = cumulative[-1]
        
        if total_length < 1e-6:
            return self
        
        # Uniform parameter values
        t_new = np.linspace(0, total_length, num_points)
        
        # Interpolate
        new_points = np.zeros((num_points, 3))
        new_radii = np.zeros(num_points)
        new_colors = np.zeros((num_points, 3))
        
        for i, t in enumerate(t_new):
            # Find segment
            idx = np.searchsorted(cumulative, t, side='right') - 1
            idx = max(0, min(idx, self.num_points - 2))
            
            # Local parameter
            segment_start = cumulative[idx]
            segment_len = segment_lengths[idx] if idx < len(segment_lengths) else 1.0
            local_t = (t - segment_start) / segment_len if segment_len > 0 else 0
            local_t = np.clip(local_t, 0, 1)
            
            # Interpolate
            new_points[i] = (1 - local_t) * self.points[idx] + local_t * self.points[idx + 1]
            new_radii[i] = (1 - local_t) * self.radii[idx] + local_t * self.radii[idx + 1]
            new_colors[i] = (1 - local_t) * self.colors[idx] + local_t * self.colors[idx + 1]
        
        return HairStrand(
            points=new_points,
            radii=new_radii,
            colors=new_colors
        )
    
    def smooth(self, iterations: int = 3, factor: float = 0.5) -> 'HairStrand':
        """Apply Laplacian smoothing to the strand."""
        if self.num_points < 3:
            return self
        
        points = self.points.copy()
        
        for _ in range(iterations):
            new_points = points.copy()
            for i in range(1, self.num_points - 1):
                laplacian = (points[i-1] + points[i+1]) / 2 - points[i]
                new_points[i] = points[i] + factor * laplacian
            points = new_points
        
        return HairStrand(
            points=points,
            radii=self.radii.copy(),
            colors=self.colors.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'points': self.points.tolist(),
            'radii': self.radii.tolist(),
            'colors': self.colors.tolist()
        }


@dataclass
class HairStrandCollection:
    """Collection of hair strands."""
    strands: List[HairStrand]
    
    @property
    def num_strands(self) -> int:
        return len(self.strands)
    
    @property
    def total_points(self) -> int:
        return sum(s.num_points for s in self.strands)
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of all strands."""
        if not self.strands:
            return np.zeros(3), np.zeros(3)
        
        all_points = np.vstack([s.points for s in self.strands])
        return np.min(all_points, axis=0), np.max(all_points, axis=0)
    
    def filter_by_length(self, min_length: float = 0.01) -> 'HairStrandCollection':
        """Filter out strands shorter than minimum length."""
        filtered = [s for s in self.strands if s.length >= min_length]
        return HairStrandCollection(strands=filtered)
    
    def resample_all(self, num_points: int) -> 'HairStrandCollection':
        """Resample all strands to have same number of points."""
        resampled = [s.resample(num_points) for s in self.strands]
        return HairStrandCollection(strands=resampled)
    
    def smooth_all(self, iterations: int = 3) -> 'HairStrandCollection':
        """Smooth all strands."""
        smoothed = [s.smooth(iterations) for s in self.strands]
        return HairStrandCollection(strands=smoothed)


class HairStrandsExtractor:
    """
    Extracts hair strand curves from Gaussian splat clouds.
    
    Implements multiple extraction methods including clustering-based
    and flow-field based approaches.
    """
    
    def __init__(self):
        """Initialize the strand extractor."""
        self.params = {
            'method': StrandExtractionMethod.CLUSTERING,
            'min_strand_length': 0.05,
            'points_per_strand': 32,
            'num_strands': 10000,
            'clustering_eps': 0.02,
            'flow_smoothness': 0.5,
        }
        
    def set_parameters(self, **kwargs):
        """Update extraction parameters."""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
    
    def extract(
        self,
        gaussian_cloud: GaussianCloud,
        callback: Optional[callable] = None
    ) -> HairStrandCollection:
        """
        Extract hair strands from Gaussian cloud.
        
        Args:
            gaussian_cloud: Input Gaussian splat cloud
            callback: Progress callback function
            
        Returns:
            HairStrandCollection containing extracted strands
        """
        method = self.params['method']
        
        if method == StrandExtractionMethod.CLUSTERING:
            return self._extract_clustering(gaussian_cloud, callback)
        elif method == StrandExtractionMethod.FLOW_FIELD:
            return self._extract_flow_field(gaussian_cloud, callback)
        else:
            return self._extract_clustering(gaussian_cloud, callback)
    
    def _extract_clustering(
        self,
        cloud: GaussianCloud,
        callback: Optional[callable] = None
    ) -> HairStrandCollection:
        """Extract strands using clustering-based method."""
        if callback:
            callback(0.1, "Preparing Gaussian data...")
        
        # Get positions and properties from Gaussians
        positions = cloud.get_positions()
        colors = cloud.get_colors()
        
        if len(positions) == 0:
            return HairStrandCollection(strands=[])
        
        if callback:
            callback(0.2, "Computing principal directions...")
        
        # Compute principal directions for each Gaussian
        directions = self._compute_principal_directions(cloud)
        
        if callback:
            callback(0.3, "Building strand graph...")
        
        # Build connectivity graph based on direction coherence
        adjacency = self._build_direction_graph(positions, directions)
        
        if callback:
            callback(0.5, "Tracing strands...")
        
        # Trace strands through the graph
        strands = self._trace_strands_from_graph(
            positions, colors, adjacency, callback
        )
        
        if callback:
            callback(0.9, "Post-processing strands...")
        
        # Post-process strands
        collection = HairStrandCollection(strands=strands)
        collection = collection.filter_by_length(self.params['min_strand_length'])
        collection = collection.smooth_all(iterations=2)
        collection = collection.resample_all(self.params['points_per_strand'])
        
        if callback:
            callback(1.0, "Complete!")
        
        return collection
    
    def _extract_flow_field(
        self,
        cloud: GaussianCloud,
        callback: Optional[callable] = None
    ) -> HairStrandCollection:
        """Extract strands using flow field method."""
        if callback:
            callback(0.1, "Building flow field...")
        
        positions = cloud.get_positions()
        colors = cloud.get_colors()
        
        if len(positions) == 0:
            return HairStrandCollection(strands=[])
        
        # Compute flow field from Gaussian orientations
        if callback:
            callback(0.2, "Computing flow vectors...")
        
        flow_field = self._compute_flow_field(cloud)
        
        if callback:
            callback(0.4, "Seeding strand roots...")
        
        # Sample seed points on scalp region
        seed_points = self._sample_seed_points(positions, self.params['num_strands'])
        
        if callback:
            callback(0.5, "Integrating strands...")
        
        # Integrate strands through flow field
        strands = []
        for i, seed in enumerate(seed_points):
            strand = self._integrate_strand(seed, flow_field, positions, colors)
            if strand is not None:
                strands.append(strand)
            
            if callback and i % 500 == 0:
                progress = 0.5 + 0.4 * (i / len(seed_points))
                callback(progress, f"Integrating strand {i}/{len(seed_points)}")
        
        if callback:
            callback(0.95, "Post-processing...")
        
        collection = HairStrandCollection(strands=strands)
        collection = collection.filter_by_length(self.params['min_strand_length'])
        collection = collection.smooth_all(iterations=2)
        
        if callback:
            callback(1.0, "Complete!")
        
        return collection
    
    def _compute_principal_directions(
        self, 
        cloud: GaussianCloud
    ) -> np.ndarray:
        """Compute principal direction for each Gaussian based on covariance."""
        directions = []
        
        for splat in cloud.splats:
            # Get eigenvectors of covariance
            eigenvalues, eigenvectors = np.linalg.eigh(splat.covariance)
            
            # Principal direction is eigenvector with largest eigenvalue
            principal_idx = np.argmax(eigenvalues)
            direction = eigenvectors[:, principal_idx]
            
            # Ensure consistent orientation (point generally downward for hair)
            if direction[1] > 0:  # y is up
                direction = -direction
            
            directions.append(direction)
        
        return np.array(directions)
    
    def _build_direction_graph(
        self,
        positions: np.ndarray,
        directions: np.ndarray,
        k_neighbors: int = 10
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Build graph connecting Gaussians with coherent directions."""
        from scipy.spatial import cKDTree
        
        n = len(positions)
        tree = cKDTree(positions)
        
        adjacency = {i: [] for i in range(n)}
        
        # Find k nearest neighbors for each point
        distances, indices = tree.query(positions, k=min(k_neighbors + 1, n))
        
        for i in range(n):
            for j_idx in range(1, len(indices[i])):  # Skip self
                j = indices[i][j_idx]
                dist = distances[i][j_idx]
                
                # Check direction coherence
                dir_i = directions[i]
                dir_j = directions[j]
                
                # Compute direction from i to j
                edge_dir = positions[j] - positions[i]
                edge_len = np.linalg.norm(edge_dir)
                if edge_len > 1e-6:
                    edge_dir = edge_dir / edge_len
                else:
                    continue
                
                # Check if directions are coherent with edge
                coherence_i = abs(np.dot(dir_i, edge_dir))
                coherence_j = abs(np.dot(dir_j, edge_dir))
                coherence = (coherence_i + coherence_j) / 2
                
                if coherence > 0.5:  # Threshold for coherence
                    weight = dist * (2 - coherence)  # Prefer coherent connections
                    adjacency[i].append((j, weight))
        
        return adjacency
    
    def _trace_strands_from_graph(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        adjacency: Dict[int, List[Tuple[int, float]]],
        callback: Optional[callable] = None
    ) -> List[HairStrand]:
        """Trace strands by following paths through the graph."""
        n = len(positions)
        visited = np.zeros(n, dtype=bool)
        strands = []
        
        # Find potential root nodes (high y coordinate = near scalp)
        y_coords = positions[:, 1]
        root_threshold = np.percentile(y_coords, 90)  # Top 10%
        root_candidates = np.where(y_coords >= root_threshold)[0]
        
        for root_idx, root in enumerate(root_candidates):
            if visited[root]:
                continue
            
            # Trace strand from this root
            strand_indices = self._trace_single_strand(
                root, positions, adjacency, visited
            )
            
            if len(strand_indices) >= 3:
                strand_points = positions[strand_indices]
                strand_colors = colors[strand_indices] if len(colors) > 0 else np.ones((len(strand_indices), 3)) * 0.3
                strand_radii = np.ones(len(strand_indices)) * 0.001  # Default radius
                
                strand = HairStrand(
                    points=strand_points,
                    radii=strand_radii,
                    colors=strand_colors
                )
                strands.append(strand)
            
            if callback and root_idx % 100 == 0:
                progress = 0.5 + 0.4 * (root_idx / len(root_candidates))
                callback(progress, f"Tracing strand {root_idx}/{len(root_candidates)}")
        
        return strands
    
    def _trace_single_strand(
        self,
        root: int,
        positions: np.ndarray,
        adjacency: Dict[int, List[Tuple[int, float]]],
        visited: np.ndarray,
        max_length: int = 100
    ) -> List[int]:
        """Trace a single strand starting from root node."""
        strand = [root]
        visited[root] = True
        current = root
        
        for _ in range(max_length):
            # Find best next node
            neighbors = adjacency.get(current, [])
            best_next = None
            best_score = float('inf')
            
            for neighbor, weight in neighbors:
                if not visited[neighbor]:
                    # Prefer nodes going downward (hair grows down)
                    dy = positions[current][1] - positions[neighbor][1]
                    if dy > 0:  # Going downward
                        score = weight - dy * 0.5
                        if score < best_score:
                            best_score = score
                            best_next = neighbor
            
            if best_next is None:
                break
            
            strand.append(best_next)
            visited[best_next] = True
            current = best_next
        
        return strand
    
    def _compute_flow_field(self, cloud: GaussianCloud) -> Dict[str, Any]:
        """Compute vector flow field from Gaussian orientations."""
        positions = cloud.get_positions()
        directions = self._compute_principal_directions(cloud)
        
        # Build KD-tree for spatial queries
        from scipy.spatial import cKDTree
        tree = cKDTree(positions)
        
        return {
            'positions': positions,
            'directions': directions,
            'tree': tree,
            'bounds_min': cloud.bounds_min,
            'bounds_max': cloud.bounds_max
        }
    
    def _sample_seed_points(
        self,
        positions: np.ndarray,
        num_seeds: int
    ) -> np.ndarray:
        """Sample seed points for strand generation."""
        # Find scalp region (top of head)
        y_coords = positions[:, 1]
        scalp_threshold = np.percentile(y_coords, 85)
        scalp_mask = y_coords >= scalp_threshold
        scalp_points = positions[scalp_mask]
        
        if len(scalp_points) < num_seeds:
            # Use all scalp points and add some random jitter
            seeds = scalp_points.copy()
            while len(seeds) < num_seeds:
                idx = np.random.randint(len(scalp_points))
                jitter = np.random.randn(3) * 0.01
                seeds = np.vstack([seeds, scalp_points[idx] + jitter])
        else:
            # Random subsample
            indices = np.random.choice(len(scalp_points), num_seeds, replace=False)
            seeds = scalp_points[indices]
        
        return seeds[:num_seeds]
    
    def _integrate_strand(
        self,
        seed: np.ndarray,
        flow_field: Dict[str, Any],
        positions: np.ndarray,
        colors: np.ndarray,
        step_size: float = 0.005,
        max_steps: int = 100
    ) -> Optional[HairStrand]:
        """Integrate a strand through the flow field using RK4."""
        tree = flow_field['tree']
        field_positions = flow_field['positions']
        field_directions = flow_field['directions']
        
        points = [seed.copy()]
        point_colors = []
        
        # Get initial color
        _, idx = tree.query(seed, k=1)
        if idx < len(colors):
            point_colors.append(colors[idx])
        else:
            point_colors.append(np.array([0.3, 0.2, 0.1]))
        
        current = seed.copy()
        
        for _ in range(max_steps):
            # Query nearest Gaussians
            distances, indices = tree.query(current, k=5)
            
            if len(indices) == 0 or distances[0] > 0.1:
                break
            
            # Interpolate direction from nearby Gaussians
            weights = 1.0 / (distances + 1e-6)
            weights = weights / np.sum(weights)
            
            direction = np.zeros(3)
            for i, idx in enumerate(indices):
                direction += weights[i] * field_directions[idx]
            
            # Normalize and ensure downward tendency
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-6:
                break
            direction = direction / direction_norm
            
            # Add gravity influence
            direction[1] -= 0.1  # Slight downward bias
            direction = direction / np.linalg.norm(direction)
            
            # RK4 integration step
            k1 = direction * step_size
            
            mid1 = current + k1 / 2
            _, mid_indices = tree.query(mid1, k=3)
            k2_dir = np.mean(field_directions[mid_indices], axis=0)
            k2_dir = k2_dir / (np.linalg.norm(k2_dir) + 1e-6)
            k2 = k2_dir * step_size
            
            mid2 = current + k2 / 2
            _, mid_indices = tree.query(mid2, k=3)
            k3_dir = np.mean(field_directions[mid_indices], axis=0)
            k3_dir = k3_dir / (np.linalg.norm(k3_dir) + 1e-6)
            k3 = k3_dir * step_size
            
            end = current + k3
            _, end_indices = tree.query(end, k=3)
            k4_dir = np.mean(field_directions[end_indices], axis=0)
            k4_dir = k4_dir / (np.linalg.norm(k4_dir) + 1e-6)
            k4 = k4_dir * step_size
            
            # Update position
            current = current + (k1 + 2*k2 + 2*k3 + k4) / 6
            
            # Check bounds
            if np.any(current < flow_field['bounds_min'] - 0.1) or \
               np.any(current > flow_field['bounds_max'] + 0.1):
                break
            
            points.append(current.copy())
            
            # Get color at new position
            _, idx = tree.query(current, k=1)
            if idx < len(colors):
                point_colors.append(colors[idx])
            else:
                point_colors.append(point_colors[-1])
        
        if len(points) < 3:
            return None
        
        return HairStrand(
            points=np.array(points),
            radii=np.ones(len(points)) * 0.001,
            colors=np.array(point_colors)
        )