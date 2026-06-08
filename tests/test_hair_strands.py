"""Test hair strand extraction behavior."""

import unittest

import numpy as np

from tests.fixtures import make_cloud
from src.core.gaussian_generator import GaussianCloud, GaussianSplat
from src.core.hair_strands import (
    HairStrand, HairStrandCollection,
    HairStrandsExtractor, StrandExtractionMethod,
)


def _build_structured_hair(n_strands: int = 20, pts_per: int = 15, seed: int = 0):
    """Build a synthetic Gaussian cloud where points follow downward strand-shapes."""
    rng = np.random.default_rng(seed)
    all_pos, all_cov = [], []
    for _ in range(n_strands):
        theta = rng.uniform(0, np.pi / 2)
        phi = rng.uniform(0, 2 * np.pi)
        root = np.array([
            np.sin(theta) * np.cos(phi),
            np.cos(theta),
            np.sin(theta) * np.sin(phi),
        ]) * 0.8
        d = np.array([rng.uniform(-0.1, 0.1), -1.0, rng.uniform(-0.1, 0.1)])
        d = d / np.linalg.norm(d)
        for t in range(pts_per):
            pos = root + d * (t * 0.025) + rng.standard_normal(3) * 0.005
            cov = np.eye(3) * 0.0001 + np.outer(d, d) * 0.001
            all_pos.append(pos.astype(np.float32))
            all_cov.append(cov.astype(np.float32))

    all_pos_arr = np.array(all_pos)
    splats = [
        GaussianSplat(
            position=all_pos_arr[i], covariance=all_cov[i],
            color=np.array([0.5, 0.4, 0.3], dtype=np.float32),
            opacity=0.9,
            scale=np.array([0.01] * 3, dtype=np.float32),
            rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        for i in range(len(all_pos_arr))
    ]
    return GaussianCloud(
        splats=splats,
        bounds_min=all_pos_arr.min(0),
        bounds_max=all_pos_arr.max(0),
    )


class TestHairStrandExtraction(unittest.TestCase):

    def test_empty_cloud_raises(self):
        ext = HairStrandsExtractor()
        empty = GaussianCloud(
            splats=[], bounds_min=np.zeros(3, dtype=np.float32),
            bounds_max=np.zeros(3, dtype=np.float32),
        )
        with self.assertRaises(ValueError):
            ext.extract(empty)

    def test_extract_recovers_structured_strands(self):
        """On 20 well-formed synthetic strands, expect ≥ 40% recovery."""
        cloud = _build_structured_hair(n_strands=20, pts_per=15)
        ext = HairStrandsExtractor()
        ext.set_parameters(
            num_strands=30, points_per_strand=10,
            min_strand_length=0.02,
            method=StrandExtractionMethod.CLUSTERING,
        )
        result = ext.extract(cloud, callback=lambda p, m: None)
        self.assertGreater(result.num_strands, 8,
                           f"Only recovered {result.num_strands}/20 strands")

    def test_flow_field_method(self):
        """Flow field method should also return strands without crashing."""
        cloud = _build_structured_hair(n_strands=10, pts_per=15)
        ext = HairStrandsExtractor()
        ext.set_parameters(
            num_strands=20, points_per_strand=10,
            min_strand_length=0.02,
            method=StrandExtractionMethod.FLOW_FIELD,
        )
        result = ext.extract(cloud, callback=lambda p, m: None)
        self.assertIsInstance(result, HairStrandCollection)

    def test_set_parameters(self):
        ext = HairStrandsExtractor()
        ext.set_parameters(num_strands=42, points_per_strand=17, min_strand_length=0.123)
        self.assertEqual(ext.params["num_strands"], 42)
        self.assertEqual(ext.params["points_per_strand"], 17)
        self.assertAlmostEqual(ext.params["min_strand_length"], 0.123)

    def test_unknown_parameter_ignored(self):
        ext = HairStrandsExtractor()
        ext.set_parameters(nonexistent_param=999)
        self.assertNotIn("nonexistent_param", ext.params)


class TestHairStrand(unittest.TestCase):

    def test_strand_length(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32)
        s = HairStrand(pts, np.zeros(3), np.zeros((3, 3)))
        self.assertAlmostEqual(s.length, 2.0)

    def test_strand_resample(self):
        pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        s = HairStrand(pts,
                       np.array([0.001, 0.001], dtype=np.float32),
                       np.zeros((2, 3), dtype=np.float32))
        s2 = s.resample(5)
        self.assertEqual(s2.num_points, 5)

    def test_filter_by_length(self):
        long_pts = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32)
        short_pts = np.array([[0, 0, 0], [0.001, 0, 0]], dtype=np.float32)
        collection = HairStrandCollection(strands=[
            HairStrand(long_pts, np.zeros(2), np.zeros((2, 3))),
            HairStrand(short_pts, np.zeros(2), np.zeros((2, 3))),
        ])
        filtered = collection.filter_by_length(min_length=0.1)
        self.assertEqual(filtered.num_strands, 1)


if __name__ == "__main__":
    unittest.main()
