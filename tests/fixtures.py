"""Shared test fixtures: synthetic Gaussian clouds and hair strands."""

import os
import sys

# Ensure repo root is on sys.path so `from src...` works in any test runner
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.core.gaussian_generator import GaussianCloud, GaussianSplat
from src.core.hair_strands import HairStrand, HairStrandCollection


def make_cloud(n: int = 50, seed: int = 0) -> GaussianCloud:
    """Build a deterministic synthetic GaussianCloud for tests."""
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((n, 3)).astype(np.float32) * 0.3
    splats = [
        GaussianSplat(
            position=pos[i],
            covariance=np.eye(3, dtype=np.float32) * 0.01,
            color=np.array([0.5, 0.4, 0.3], dtype=np.float32),
            opacity=0.9,
            scale=np.array([0.02, 0.02, 0.02], dtype=np.float32),
            rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        for i in range(n)
    ]
    return GaussianCloud(splats=splats, bounds_min=pos.min(0), bounds_max=pos.max(0))


def make_strands(n_strands: int = 5, n_pts: int = 16, seed: int = 1) -> HairStrandCollection:
    """Build a deterministic HairStrandCollection."""
    rng = np.random.default_rng(seed)
    strands = []
    for _ in range(n_strands):
        pts = (rng.standard_normal((n_pts, 3)) * 0.1).astype(np.float32)
        pts[:, 1] = np.linspace(0, -0.3, n_pts, dtype=np.float32)  # downward
        strands.append(HairStrand(
            points=pts,
            radii=np.ones(n_pts, dtype=np.float32) * 0.001,
            colors=np.ones((n_pts, 3), dtype=np.float32) * 0.4,
        ))
    return HairStrandCollection(strands=strands)
