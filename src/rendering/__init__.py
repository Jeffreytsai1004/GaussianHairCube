"""
Rendering module for 3D visualization.
"""

__all__ = [
    'Viewer3D',
    'ViewMode',
    'ShadingMode',
]

def __getattr__(name):
    if name in ('Viewer3D', 'ViewMode', 'ShadingMode'):
        from src.rendering.viewer_3d import Viewer3D, ViewMode, ShadingMode
        return {'Viewer3D': Viewer3D, 'ViewMode': ViewMode, 'ShadingMode': ShadingMode}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")