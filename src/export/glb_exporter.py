"""
GLB/GLTF Exporter Module
========================

Exports hair strands and Gaussian splats as Blender-compatible GLB files.
This module generates both mesh-based and curve-based representations
suitable for import into Blender and other 3D applications.
"""

import json
import struct
import base64
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

# Import types
from src.core.hair_strands import HairStrand, HairStrandCollection
from src.core.gaussian_generator import GaussianCloud


@dataclass
class GLBExportOptions:
    """Options for GLB export."""
    export_type: str = "curves"  # "curves", "mesh", or "both"
    include_color: bool = True
    include_width: bool = True
    tube_segments: int = 4  # For mesh export
    tube_radius: float = 0.001
    up_axis: str = "Z"  # Blender uses Z-up
    scale_factor: float = 1.0
    embed_textures: bool = False


class GLBExporter:
    """
    Exports hair data to GLB format for Blender.
    
    Supports multiple export modes:
    - Curves: Polyline representations
    - Mesh: Tube mesh representations
    - Both: Combined export
    """
    
    def __init__(self, options: Optional[GLBExportOptions] = None):
        """
        Initialize the GLB exporter.
        
        Args:
            options: Export options
        """
        self.options = options or GLBExportOptions()
        
    def export_strands(
        self,
        strands: HairStrandCollection,
        output_path: str,
        callback: Optional[callable] = None
    ) -> bool:
        """
        Export hair strands to GLB file.
        
        Args:
            strands: Hair strand collection to export
            output_path: Output file path
            callback: Progress callback function
            
        Returns:
            True if export succeeded
        """
        try:
            if callback:
                callback(0.1, "Preparing strand data...")
            
            if self.options.export_type == "curves":
                gltf_data = self._create_curves_gltf(strands, callback)
            elif self.options.export_type == "mesh":
                gltf_data = self._create_mesh_gltf(strands, callback)
            else:  # both
                gltf_data = self._create_combined_gltf(strands, callback)
            
            if callback:
                callback(0.9, "Writing GLB file...")
            
            self._write_glb(gltf_data, output_path)
            
            if callback:
                callback(1.0, "Export complete!")
            
            return True
            
        except Exception as e:
            print(f"GLB export error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def export_gaussians(
        self,
        cloud: GaussianCloud,
        output_path: str,
        callback: Optional[callable] = None
    ) -> bool:
        """
        Export Gaussian cloud to GLB file as point cloud.
        
        Args:
            cloud: Gaussian cloud to export
            output_path: Output file path
            callback: Progress callback function
            
        Returns:
            True if export succeeded
        """
        try:
            if callback:
                callback(0.1, "Preparing Gaussian data...")
            
            gltf_data = self._create_gaussians_gltf(cloud, callback)
            
            if callback:
                callback(0.9, "Writing GLB file...")
            
            self._write_glb(gltf_data, output_path)
            
            if callback:
                callback(1.0, "Export complete!")
            
            return True
            
        except Exception as e:
            print(f"GLB export error: {e}")
            return False
    
    def _transform_coordinates(self, points: np.ndarray) -> np.ndarray:
        """Transform coordinates for Blender's coordinate system."""
        transformed = points.copy() * self.options.scale_factor
        
        if self.options.up_axis == "Z":
            # Swap Y and Z for Blender
            transformed = transformed[:, [0, 2, 1]]
            transformed[:, 1] = -transformed[:, 1]
        
        return transformed.astype(np.float32)
    
    def _create_curves_gltf(
        self,
        strands: HairStrandCollection,
        callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Create GLTF data with polyline primitives."""
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "GaussianHairCube"
            },
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "HairCurves"}],
            "meshes": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": []
        }
        
        # Collect all curve data
        all_positions = []
        all_colors = []
        all_indices = []
        
        current_index = 0
        num_strands = strands.num_strands
        
        for i, strand in enumerate(strands.strands):
            if callback and i % 500 == 0:
                progress = 0.1 + 0.6 * (i / num_strands)
                callback(progress, f"Processing strand {i}/{num_strands}")
            
            # Transform points
            points = self._transform_coordinates(strand.points)
            all_positions.append(points)
            
            # Colors
            if self.options.include_color and len(strand.colors) > 0:
                colors = np.clip(strand.colors, 0, 1).astype(np.float32)
                all_colors.append(colors)
            
            # Line strip indices
            n = len(points)
            for j in range(n - 1):
                all_indices.extend([current_index + j, current_index + j + 1])
            
            current_index += n
        
        # Combine data
        positions = np.vstack(all_positions)
        
        # Compute bounds
        pos_min = positions.min(axis=0).tolist()
        pos_max = positions.max(axis=0).tolist()
        
        # Build binary buffer
        buffer_data = bytearray()
        
        # Positions
        position_offset = len(buffer_data)
        position_data = positions.astype(np.float32).tobytes()
        buffer_data.extend(position_data)
        
        # Pad to 4-byte alignment
        while len(buffer_data) % 4 != 0:
            buffer_data.append(0)
        
        # Colors
        color_offset = None
        if all_colors:
            colors = np.vstack(all_colors)
            color_offset = len(buffer_data)
            color_data = colors.astype(np.float32).tobytes()
            buffer_data.extend(color_data)
            
            while len(buffer_data) % 4 != 0:
                buffer_data.append(0)
        
        # Indices
        indices = np.array(all_indices, dtype=np.uint32)
        indices_offset = len(buffer_data)
        indices_data = indices.tobytes()
        buffer_data.extend(indices_data)
        
        # Build GLTF structure
        buffer_views = []
        accessors = []
        
        # Position buffer view and accessor
        buffer_views.append({
            "buffer": 0,
            "byteOffset": position_offset,
            "byteLength": len(position_data),
            "target": 34962  # ARRAY_BUFFER
        })
        accessors.append({
            "bufferView": 0,
            "componentType": 5126,  # FLOAT
            "count": len(positions),
            "type": "VEC3",
            "min": pos_min,
            "max": pos_max
        })
        
        # Color buffer view and accessor
        if color_offset is not None:
            buffer_views.append({
                "buffer": 0,
                "byteOffset": color_offset,
                "byteLength": len(color_data),
                "target": 34962
            })
            accessors.append({
                "bufferView": 1,
                "componentType": 5126,
                "count": len(colors),
                "type": "VEC3"
            })
        
        # Index buffer view and accessor
        indices_view_idx = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": indices_offset,
            "byteLength": len(indices_data),
            "target": 34963  # ELEMENT_ARRAY_BUFFER
        })
        accessors.append({
            "bufferView": indices_view_idx,
            "componentType": 5125,  # UNSIGNED_INT
            "count": len(indices),
            "type": "SCALAR"
        })
        
        # Build mesh
        primitive = {
            "mode": 1,  # LINES
            "attributes": {"POSITION": 0},
            "indices": len(accessors) - 1
        }
        
        if color_offset is not None:
            primitive["attributes"]["COLOR_0"] = 1
        
        gltf["meshes"] = [{"primitives": [primitive], "name": "HairCurves"}]
        gltf["accessors"] = accessors
        gltf["bufferViews"] = buffer_views
        gltf["buffers"] = [{"byteLength": len(buffer_data)}]
        
        return {"gltf": gltf, "buffer": bytes(buffer_data)}
    
    def _create_mesh_gltf(
        self,
        strands: HairStrandCollection,
        callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Create GLTF data with tube mesh representations."""
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "GaussianHairCube"
            },
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "HairMesh"}],
            "meshes": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": []
        }
        
        all_vertices = []
        all_normals = []
        all_colors = []
        all_indices = []
        
        current_vertex = 0
        num_strands = strands.num_strands
        segments = self.options.tube_segments
        
        for i, strand in enumerate(strands.strands):
            if callback and i % 200 == 0:
                progress = 0.1 + 0.6 * (i / num_strands)
                callback(progress, f"Creating mesh {i}/{num_strands}")
            
            points = self._transform_coordinates(strand.points)
            n_points = len(points)
            
            if n_points < 2:
                continue
            
            # Get radius for each point
            if self.options.include_width:
                radii = strand.radii * self.options.scale_factor
            else:
                radii = np.ones(n_points) * self.options.tube_radius * self.options.scale_factor
            
            # Generate tube vertices
            vertices, normals, indices = self._create_tube_geometry(
                points, radii, segments, current_vertex
            )
            
            all_vertices.append(vertices)
            all_normals.append(normals)
            all_indices.append(indices)
            
            # Colors
            if self.options.include_color and len(strand.colors) > 0:
                # Repeat colors for each ring
                expanded_colors = np.repeat(strand.colors, segments, axis=0)
                all_colors.append(np.clip(expanded_colors, 0, 1).astype(np.float32))
            
            current_vertex += len(vertices)
        
        if not all_vertices:
            # Return empty mesh
            return self._create_empty_gltf()
        
        # Combine data
        vertices = np.vstack(all_vertices)
        normals = np.vstack(all_normals)
        indices = np.concatenate(all_indices)
        
        pos_min = vertices.min(axis=0).tolist()
        pos_max = vertices.max(axis=0).tolist()
        
        # Build binary buffer
        buffer_data = bytearray()
        
        # Vertices
        vertex_offset = len(buffer_data)
        vertex_data = vertices.astype(np.float32).tobytes()
        buffer_data.extend(vertex_data)
        
        # Normals
        normal_offset = len(buffer_data)
        normal_data = normals.astype(np.float32).tobytes()
        buffer_data.extend(normal_data)
        
        # Colors
        color_offset = None
        if all_colors:
            colors = np.vstack(all_colors)
            color_offset = len(buffer_data)
            color_data = colors.astype(np.float32).tobytes()
            buffer_data.extend(color_data)
        
        # Indices
        indices_offset = len(buffer_data)
        indices_data = indices.astype(np.uint32).tobytes()
        buffer_data.extend(indices_data)
        
        # Build GLTF structure
        buffer_views = []
        accessors = []
        
        # Position
        buffer_views.append({
            "buffer": 0,
            "byteOffset": vertex_offset,
            "byteLength": len(vertex_data),
            "target": 34962
        })
        accessors.append({
            "bufferView": 0,
            "componentType": 5126,
            "count": len(vertices),
            "type": "VEC3",
            "min": pos_min,
            "max": pos_max
        })
        
        # Normal
        buffer_views.append({
            "buffer": 0,
            "byteOffset": normal_offset,
            "byteLength": len(normal_data),
            "target": 34962
        })
        accessors.append({
            "bufferView": 1,
            "componentType": 5126,
            "count": len(normals),
            "type": "VEC3"
        })
        
        # Color
        color_accessor_idx = None
        if color_offset is not None:
            buffer_views.append({
                "buffer": 0,
                "byteOffset": color_offset,
                "byteLength": len(color_data),
                "target": 34962
            })
            color_accessor_idx = len(accessors)
            accessors.append({
                "bufferView": len(buffer_views) - 1,
                "componentType": 5126,
                "count": len(colors),
                "type": "VEC3"
            })
        
        # Indices
        buffer_views.append({
            "buffer": 0,
            "byteOffset": indices_offset,
            "byteLength": len(indices_data),
            "target": 34963
        })
        indices_accessor_idx = len(accessors)
        accessors.append({
            "bufferView": len(buffer_views) - 1,
            "componentType": 5125,
            "count": len(indices),
            "type": "SCALAR"
        })
        
        # Build mesh
        primitive = {
            "mode": 4,  # TRIANGLES
            "attributes": {
                "POSITION": 0,
                "NORMAL": 1
            },
            "indices": indices_accessor_idx
        }
        
        if color_accessor_idx is not None:
            primitive["attributes"]["COLOR_0"] = color_accessor_idx
        
        gltf["meshes"] = [{"primitives": [primitive], "name": "HairMesh"}]
        gltf["accessors"] = accessors
        gltf["bufferViews"] = buffer_views
        gltf["buffers"] = [{"byteLength": len(buffer_data)}]
        
        return {"gltf": gltf, "buffer": bytes(buffer_data)}
    
    def _create_tube_geometry(
        self,
        points: np.ndarray,
        radii: np.ndarray,
        segments: int,
        vertex_offset: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create tube mesh geometry for a strand."""
        n_points = len(points)
        n_vertices = n_points * segments
        
        vertices = np.zeros((n_vertices, 3), dtype=np.float32)
        normals = np.zeros((n_vertices, 3), dtype=np.float32)
        
        # Compute tangents
        tangents = np.zeros((n_points, 3))
        tangents[:-1] = points[1:] - points[:-1]
        tangents[-1] = tangents[-2]
        tangents = tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8)
        
        # Create ring vertices
        angle_step = 2 * np.pi / segments
        
        for i in range(n_points):
            tangent = tangents[i]
            
            # Create perpendicular vectors
            if abs(tangent[0]) < 0.9:
                up = np.array([1, 0, 0])
            else:
                up = np.array([0, 1, 0])
            
            right = np.cross(tangent, up)
            right = right / (np.linalg.norm(right) + 1e-8)
            up = np.cross(right, tangent)
            
            radius = radii[i]
            
            for j in range(segments):
                angle = j * angle_step
                normal = np.cos(angle) * right + np.sin(angle) * up
                
                idx = i * segments + j
                vertices[idx] = points[i] + radius * normal
                normals[idx] = normal
        
        # Create indices (triangle strip converted to triangles)
        indices = []
        for i in range(n_points - 1):
            for j in range(segments):
                # Current and next ring indices
                curr = vertex_offset + i * segments + j
                next_j = vertex_offset + i * segments + ((j + 1) % segments)
                curr_next_ring = vertex_offset + (i + 1) * segments + j
                next_j_next_ring = vertex_offset + (i + 1) * segments + ((j + 1) % segments)
                
                # Two triangles per quad
                indices.extend([curr, curr_next_ring, next_j])
                indices.extend([next_j, curr_next_ring, next_j_next_ring])
        
        return vertices, normals, np.array(indices, dtype=np.uint32)
    
    def _create_gaussians_gltf(
        self,
        cloud: GaussianCloud,
        callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Create GLTF data for Gaussian cloud as points."""
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "GaussianHairCube"
            },
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "GaussianSplats"}],
            "meshes": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": []
        }
        
        if callback:
            callback(0.3, "Processing Gaussians...")
        
        # Get positions and colors
        positions = self._transform_coordinates(cloud.get_positions())
        colors = cloud.get_colors().astype(np.float32)
        
        pos_min = positions.min(axis=0).tolist()
        pos_max = positions.max(axis=0).tolist()
        
        # Build binary buffer
        buffer_data = bytearray()
        
        # Positions
        position_offset = len(buffer_data)
        position_data = positions.tobytes()
        buffer_data.extend(position_data)
        
        # Colors
        color_offset = len(buffer_data)
        color_data = colors.tobytes()
        buffer_data.extend(color_data)
        
        # Build GLTF structure
        gltf["bufferViews"] = [
            {
                "buffer": 0,
                "byteOffset": position_offset,
                "byteLength": len(position_data),
                "target": 34962
            },
            {
                "buffer": 0,
                "byteOffset": color_offset,
                "byteLength": len(color_data),
                "target": 34962
            }
        ]
        
        gltf["accessors"] = [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": pos_min,
                "max": pos_max
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": len(colors),
                "type": "VEC3"
            }
        ]
        
        gltf["meshes"] = [{
            "primitives": [{
                "mode": 0,  # POINTS
                "attributes": {
                    "POSITION": 0,
                    "COLOR_0": 1
                }
            }],
            "name": "GaussianSplats"
        }]
        
        gltf["buffers"] = [{"byteLength": len(buffer_data)}]
        
        return {"gltf": gltf, "buffer": bytes(buffer_data)}
    
    def _create_combined_gltf(
        self,
        strands: HairStrandCollection,
        callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Create GLTF with both curves and mesh."""
        # For simplicity, just create mesh version
        # A full implementation would combine both
        return self._create_mesh_gltf(strands, callback)
    
    def _create_empty_gltf(self) -> Dict[str, Any]:
        """Create empty GLTF structure."""
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "GaussianHairCube"
            },
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": [],
            "meshes": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": [{"byteLength": 0}]
        }
        return {"gltf": gltf, "buffer": b""}
    
    def _write_glb(self, gltf_data: Dict[str, Any], output_path: str):
        """Write GLB binary file."""
        gltf = gltf_data["gltf"]
        binary_buffer = gltf_data["buffer"]
        
        # Serialize JSON
        json_str = json.dumps(gltf, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')
        
        # Pad JSON to 4-byte alignment
        json_padding = (4 - len(json_bytes) % 4) % 4
        json_bytes += b' ' * json_padding
        
        # Pad binary to 4-byte alignment
        binary_padding = (4 - len(binary_buffer) % 4) % 4
        binary_buffer += b'\x00' * binary_padding
        
        # Calculate total length
        header_length = 12
        json_chunk_header = 8
        binary_chunk_header = 8
        
        total_length = (
            header_length +
            json_chunk_header + len(json_bytes) +
            binary_chunk_header + len(binary_buffer)
        )
        
        # Write file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            # GLB Header
            f.write(b'glTF')  # Magic
            f.write(struct.pack('<I', 2))  # Version
            f.write(struct.pack('<I', total_length))  # Total length
            
            # JSON Chunk
            f.write(struct.pack('<I', len(json_bytes)))  # Chunk length
            f.write(b'JSON')  # Chunk type
            f.write(json_bytes)  # Chunk data
            
            # Binary Chunk
            f.write(struct.pack('<I', len(binary_buffer)))  # Chunk length
            f.write(b'BIN\x00')  # Chunk type
            f.write(binary_buffer)  # Chunk data


def export_strands_to_glb(
    strands: HairStrandCollection,
    output_path: str,
    options: Optional[GLBExportOptions] = None,
    callback: Optional[callable] = None
) -> bool:
    """
    Convenience function to export strands to GLB.
    
    Args:
        strands: Hair strands to export
        output_path: Output file path
        options: Export options
        callback: Progress callback
        
    Returns:
        True if export succeeded
    """
    exporter = GLBExporter(options)
    return exporter.export_strands(strands, output_path, callback)


def export_gaussians_to_glb(
    cloud: GaussianCloud,
    output_path: str,
    options: Optional[GLBExportOptions] = None,
    callback: Optional[callable] = None
) -> bool:
    """
    Convenience function to export Gaussians to GLB.
    
    Args:
        cloud: Gaussian cloud to export
        output_path: Output file path
        options: Export options
        callback: Progress callback
        
    Returns:
        True if export succeeded
    """
    exporter = GLBExporter(options)
    return exporter.export_gaussians(cloud, output_path, callback)