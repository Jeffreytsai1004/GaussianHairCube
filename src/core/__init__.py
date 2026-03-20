"""
Core module for hair reconstruction algorithms.
"""

__all__ = [
    'GaussianGenerator',
    'GaussianCloud',
    'GaussianSplat',
    'HairStrandsExtractor',
    'HairStrandCollection',
    'HairStrand',
    'GeometryController'
]

def __getattr__(name):
    if name in ('GaussianGenerator', 'GaussianCloud', 'GaussianSplat'):
        from src.core.gaussian_generator import GaussianGenerator, GaussianCloud, GaussianSplat
        return {'GaussianGenerator': GaussianGenerator, 'GaussianCloud': GaussianCloud, 'GaussianSplat': GaussianSplat}[name]
    elif name in ('HairStrandsExtractor', 'HairStrandCollection', 'HairStrand'):
        from src.core.hair_strands import HairStrandsExtractor, HairStrandCollection, HairStrand
        return {'HairStrandsExtractor': HairStrandsExtractor, 'HairStrandCollection': HairStrandCollection, 'HairStrand': HairStrand}[name]
    elif name == 'GeometryController':
        from src.core.geometry_controller import GeometryController
        return GeometryController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")