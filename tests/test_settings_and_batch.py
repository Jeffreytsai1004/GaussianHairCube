"""Test settings persistence and batch processor lifecycle."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.fixtures import make_cloud
from src.config import settings_manager
from src.core.batch_processor import BatchJob, BatchProcessor, JobStatus
from src.core.gaussian_generator import GaussianCloud
from src.core.hair_strands import HairStrandsExtractor


class TestSettings(unittest.TestCase):

    def test_defaults_have_required_keys(self):
        required = [
            "num_iterations", "points_per_strand", "num_strands",
            "min_strand_length", "extraction_method",
            "theme", "scale_factor", "up_axis", "hf_endpoint",
        ]
        for key in required:
            self.assertIn(key, settings_manager.DEFAULT_SETTINGS,
                          f"Missing default key: {key}")

    def test_load_missing_file_returns_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            fake_path = Path(tmpdir) / "nonexistent.json"
            with patch.object(settings_manager, "get_settings_path", return_value=fake_path):
                loaded = settings_manager.load_settings()
            self.assertEqual(loaded, settings_manager.DEFAULT_SETTINGS)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = os.path.join(tmpdir, "settings.json")
            with patch.object(settings_manager, "get_settings_path",
                              return_value=__import__("pathlib").Path(fake_path)):
                data = {"num_iterations": 500, "theme": "light", "hf_endpoint": "https://mirror.example/"}
                settings_manager.save_settings(data)
                loaded = settings_manager.load_settings()
            # Loaded should contain saved values plus all defaults
            self.assertEqual(loaded["num_iterations"], 500)
            self.assertEqual(loaded["theme"], "light")
            self.assertEqual(loaded["hf_endpoint"], "https://mirror.example/")
            self.assertIn("points_per_strand", loaded)  # default merged in

    def test_corrupt_file_returns_defaults(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{not valid json")
            path = f.name
        try:
            with patch.object(settings_manager, "get_settings_path",
                              return_value=__import__("pathlib").Path(path)):
                loaded = settings_manager.load_settings()
            self.assertEqual(loaded, settings_manager.DEFAULT_SETTINGS)
        finally:
            os.unlink(path)


class _FakeGenerator:
    """Mock GaussianGenerator that returns a fixed cloud without ML calls."""
    def cancel(self): pass
    def generate_from_images(self, images, callback=None):
        if callback:
            callback(0.5, "fake gen")
        return make_cloud(n=30)


class TestBatchProcessor(unittest.TestCase):

    def setUp(self):
        from PIL import Image as PILImage
        self.tmpdir = tempfile.mkdtemp()
        self.img_paths = []
        for i in range(4):
            p = os.path.join(self.tmpdir, f"img{i}.png")
            PILImage.fromarray(
                np.zeros((32, 32, 3), dtype=np.uint8)
            ).save(p)
            self.img_paths.append(p)

        ext = HairStrandsExtractor()
        ext.set_parameters(num_strands=10, points_per_strand=8, min_strand_length=0.01)
        self.proc = BatchProcessor(_FakeGenerator(), ext)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_and_remove_jobs(self):
        out = os.path.join(self.tmpdir, "out")
        j1 = self.proc.add_job(self.img_paths, out)
        j2 = self.proc.add_job(self.img_paths, out)
        self.assertEqual(len(self.proc.jobs), 2)
        self.assertEqual(self.proc.pending_count, 2)

        self.proc.remove_job(j1.id)
        self.assertEqual(len(self.proc.jobs), 1)
        self.assertEqual(self.proc.jobs[0].id, j2.id)

    def test_run_all_succeeds(self):
        out = os.path.join(self.tmpdir, "out")
        self.proc.add_job(self.img_paths, out)

        updates = []
        self.proc.run_all(on_job_update=lambda j: updates.append(j.status))

        self.assertEqual(self.proc.jobs[0].status, JobStatus.DONE)
        self.assertGreater(self.proc.jobs[0].num_splats, 0)
        self.assertTrue(os.path.exists(self.proc.jobs[0].result_path))

    def test_too_few_images_fails(self):
        out = os.path.join(self.tmpdir, "out")
        self.proc.add_job(self.img_paths[:1], out)  # only 1 image
        self.proc.run_all()
        self.assertEqual(self.proc.jobs[0].status, JobStatus.FAILED)
        self.assertIn("3", self.proc.jobs[0].error or "")  # error mentions 3-image requirement

    def test_clear_finished_keeps_pending(self):
        out = os.path.join(self.tmpdir, "out")
        self.proc.add_job(self.img_paths, out)
        self.proc.run_all()  # 1 done
        self.proc.add_job(self.img_paths, out)  # 1 pending
        self.proc.clear_finished()
        self.assertEqual(len(self.proc.jobs), 1)
        self.assertEqual(self.proc.jobs[0].status, JobStatus.PENDING)

    def test_batch_job_label(self):
        job = BatchJob(image_paths=["/tmp/foo.png", "/tmp/bar.png"], output_dir="")
        self.assertIn("foo", job.label)
        self.assertIn("2", job.label)


if __name__ == "__main__":
    unittest.main()
