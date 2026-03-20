"""
Export module for hair data.
"""

__all__ = [
    'FBXExporter',
    'FBXExportOptions',
    'GLBExporter',
    'GLBExportOptions'
]

def __getattr__(name):
    if name in ('FBXExporter', 'FBXExportOptions'):
        from src.export.fbx_exporter import FBXExporter, FBXExportOptions
        return {'FBXExporter': FBXExporter, 'FBXExportOptions': FBXExportOptions}[name]
    elif name in ('GLBExporter', 'GLBExportOptions'):
        from src.export.glb_exporter import GLBExporter, GLBExportOptions
        return {'GLBExporter': GLBExporter, 'GLBExportOptions': GLBExportOptions}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")