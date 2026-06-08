"""Test save_project / load_project round-trip."""

import os
import tempfile
import unittest

import numpy as np

from tests.fixtures import make_cloud, make_strands
from src.core.project_io import save_project, load_project, is_ghc_file


class TestProjectIO(unittest.TestCase):

    def test_roundtrip_full(self):
        """Cloud + strands → save → load → identical arrays."""
        cloud = make_cloud(n=80)
        strands = make_strands(n_strands=6, n_pts=20)

        with tempfile.NamedTemporaryFile(suffix=".ghc", delete=False) as f:
            path = f.name
        try:
            self.assertTrue(save_project(path, cloud, strands,
                                         image_paths=["/img/a.png", "/img/b.png"]))
            self.assertTrue(os.path.getsize(path) > 0)

            cloud2, strands2, meta = load_project(path)

            # Cloud
            self.assertEqual(cloud2.num_splats, cloud.num_splats)
            np.testing.assert_array_equal(
                np.stack([s.position for s in cloud.splats]),
                np.stack([s.position for s in cloud2.splats]),
            )
            np.testing.assert_array_equal(
                np.stack([s.color for s in cloud.splats]),
                np.stack([s.color for s in cloud2.splats]),
            )

            # Strands
            self.assertEqual(strands2.num_strands, strands.num_strands)
            for s_orig, s_loaded in zip(strands.strands, strands2.strands):
                np.testing.assert_array_equal(s_orig.points, s_loaded.points)
                np.testing.assert_array_equal(s_orig.colors, s_loaded.colors)

            # Meta
            self.assertEqual(meta["version"], "1.0")
            self.assertEqual(meta["num_splats"], cloud.num_splats)
            self.assertEqual(meta["num_strands"], strands.num_strands)
            self.assertEqual(meta["image_paths"], ["/img/a.png", "/img/b.png"])
        finally:
            os.unlink(path)

    def test_roundtrip_cloud_only(self):
        """Saving without strands still works; load returns None for strands."""
        cloud = make_cloud(n=10)
        with tempfile.NamedTemporaryFile(suffix=".ghc", delete=False) as f:
            path = f.name
        try:
            self.assertTrue(save_project(path, cloud, None))
            cloud2, strands2, meta = load_project(path)
            self.assertEqual(cloud2.num_splats, 10)
            self.assertIsNone(strands2)
            self.assertEqual(meta["num_strands"], 0)
        finally:
            os.unlink(path)

    def test_roundtrip_strands_only(self):
        """Saving without cloud still works; load returns None for cloud."""
        strands = make_strands(n_strands=3, n_pts=8)
        with tempfile.NamedTemporaryFile(suffix=".ghc", delete=False) as f:
            path = f.name
        try:
            self.assertTrue(save_project(path, None, strands))
            cloud2, strands2, meta = load_project(path)
            self.assertIsNone(cloud2)
            self.assertEqual(strands2.num_strands, 3)
        finally:
            os.unlink(path)

    def test_load_missing_file_returns_none(self):
        cloud, strands, meta = load_project("/nonexistent/path/file.ghc")
        self.assertIsNone(cloud)
        self.assertIsNone(strands)
        self.assertEqual(meta, {})

    def test_is_ghc_file(self):
        self.assertTrue(is_ghc_file("foo.ghc"))
        self.assertTrue(is_ghc_file("foo.GHC"))
        self.assertFalse(is_ghc_file("foo.fbx"))
        self.assertFalse(is_ghc_file("foo"))


if __name__ == "__main__":
    unittest.main()
