"""Test FBX and GLB exporters for correctness of file structure."""

import os
import struct
import tempfile
import unittest

import numpy as np

from tests.fixtures import make_strands
from src.core.hair_strands import HairStrandCollection
from src.export.fbx_exporter import FBXExporter, FBXExportOptions
from src.export.glb_exporter import GLBExporter, GLBExportOptions


class TestFBXExporter(unittest.TestCase):

    def test_export_creates_valid_fbx_header(self):
        strands = make_strands(n_strands=3, n_pts=10)
        with tempfile.NamedTemporaryFile(suffix=".fbx", delete=False) as f:
            path = f.name
        try:
            ok = FBXExporter().export(strands, path)
            self.assertTrue(ok)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            # Required FBX ASCII tokens
            self.assertIn("FBXHeaderExtension", content)
            self.assertIn("FBXVersion: 7700", content)
            self.assertIn("Geometry::HairCurve_", content)
            self.assertIn("Connections", content)
        finally:
            os.unlink(path)

    def test_control_points_format(self):
        """Multi-chunk control points must terminate non-last lines with ','."""
        # 50 points → 200 values → 25 chunks of 8 values
        strands = make_strands(n_strands=1, n_pts=50)
        with tempfile.NamedTemporaryFile(suffix=".fbx", delete=False) as f:
            path = f.name
        try:
            FBXExporter().export(strands, path)
            with open(path, encoding="utf-8") as f:
                content = f.read()

            pts_start = content.find("Points: *200")
            pts_end = content.find("}", pts_start) + 1
            block = content[pts_start:pts_end]
            data_lines = [l.rstrip() for l in block.split("\n")][1:-1]

            self.assertEqual(len(data_lines), 25)
            # Intermediate lines should all end with ','
            self.assertTrue(all(l.endswith(",") for l in data_lines[:-1]))
            # Last data line must NOT end with ','
            self.assertFalse(data_lines[-1].endswith(","))
        finally:
            os.unlink(path)

    def test_empty_collection_does_not_crash(self):
        empty = HairStrandCollection(strands=[])
        with tempfile.NamedTemporaryFile(suffix=".fbx", delete=False) as f:
            path = f.name
        try:
            # Should succeed (writes minimal FBX) — not crash
            FBXExporter().export(empty, path)
            self.assertTrue(os.path.exists(path))
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_scale_factor_applied(self):
        strands = make_strands(n_strands=1, n_pts=4)
        with tempfile.NamedTemporaryFile(suffix=".fbx", delete=False) as f:
            path = f.name
        try:
            FBXExporter(FBXExportOptions(scale_factor=10.0)).export(strands, path)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("UnitScaleFactor", content)
            self.assertIn("10.0", content)
        finally:
            os.unlink(path)


class TestGLBExporter(unittest.TestCase):

    def test_glb_magic_and_version(self):
        """GLB binary must start with 'glTF' magic + version 2."""
        strands = make_strands(n_strands=3, n_pts=8)
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            path = f.name
        try:
            ok = GLBExporter().export_strands(strands, path)
            self.assertTrue(ok)
            with open(path, "rb") as f:
                header = f.read(12)
            magic = header[:4]
            version = struct.unpack("<I", header[4:8])[0]
            total_length = struct.unpack("<I", header[8:12])[0]
            self.assertEqual(magic, b"glTF")
            self.assertEqual(version, 2)
            self.assertEqual(total_length, os.path.getsize(path))
        finally:
            os.unlink(path)

    def test_glb_json_chunk_valid(self):
        """The JSON chunk should be parseable and contain a single mesh."""
        import json
        strands = make_strands(n_strands=2, n_pts=6)
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            path = f.name
        try:
            GLBExporter().export_strands(strands, path)
            with open(path, "rb") as f:
                f.seek(12)  # skip header
                json_len = struct.unpack("<I", f.read(4))[0]
                json_type = f.read(4)
                json_bytes = f.read(json_len)
            self.assertEqual(json_type, b"JSON")
            data = json.loads(json_bytes.decode("utf-8"))
            self.assertEqual(data["asset"]["version"], "2.0")
            self.assertEqual(len(data["meshes"]), 1)
        finally:
            os.unlink(path)

    def test_z_up_swap_applied(self):
        """Z-up swap should rearrange axes."""
        strands = make_strands(n_strands=1, n_pts=4)
        # First export with Y-up (no swap)
        exp_y = GLBExporter(GLBExportOptions(up_axis="Y"))
        bgr_y = exp_y._transform_coordinates(strands.strands[0].points.copy())

        # Then Z-up (swap Y/Z and flip Y)
        exp_z = GLBExporter(GLBExportOptions(up_axis="Z"))
        bgr_z = exp_z._transform_coordinates(strands.strands[0].points.copy())

        # The two should differ unless Y==Z everywhere
        self.assertFalse(np.allclose(bgr_y, bgr_z))


if __name__ == "__main__":
    unittest.main()
