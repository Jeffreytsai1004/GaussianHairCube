"""Test model_manager helpers: diagnose_env, error summariser."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core import model_manager


class TestDiagnoseEnv(unittest.TestCase):

    def test_returns_expected_shape(self):
        info = model_manager.diagnose_env()
        self.assertIn("python", info)
        self.assertIn("packages", info)
        self.assertIn("pip_hint", info)

        py = info["python"]
        self.assertEqual(py["executable"], sys.executable)
        self.assertTrue(py["version"])
        self.assertTrue(py["prefix"])

    def test_lists_all_required_packages(self):
        info = model_manager.diagnose_env()
        names = {p["name"] for p in info["packages"]}
        for required in ("transformers", "torch", "accelerate",
                         "huggingface_hub", "safetensors"):
            self.assertIn(required, names)

    def test_pip_hint_uses_current_python(self):
        info = model_manager.diagnose_env()
        # If anything is missing, the hint should reference sys.executable
        any_missing = any(
            not p.get("installed") or p.get("error") for p in info["packages"]
        )
        if any_missing:
            self.assertIn(sys.executable, info["pip_hint"])
            self.assertIn("-m pip install", info["pip_hint"])

    def test_installed_package_has_version_and_origin(self):
        info = model_manager.diagnose_env()
        # numpy is required by everything else, so use that as a sentinel
        # ... but diagnose_env only inspects the ML stack. Check whichever
        # of its packages happens to be installed in this environment.
        for p in info["packages"]:
            if p.get("installed") and not p.get("error"):
                self.assertIn("version", p)
                self.assertIn("origin", p)
                return
        # If none installed, that's also fine for this environment


class TestErrorSummariser(unittest.TestCase):

    def test_connection_error_suggests_mirror(self):
        # Simulate typical requests connection error text
        exc = ConnectionError("Max retries exceeded with url: ...")
        # Force endpoint to default to trigger mirror suggestion
        with patch.dict(os.environ, {"HF_ENDPOINT": "https://huggingface.co"}):
            msg = model_manager._summarise_download_error(exc)
        self.assertIn("hf-mirror.com", msg)

    def test_ssl_error_suggests_certifi(self):
        exc = Exception("SSL: CERTIFICATE_VERIFY_FAILED")
        msg = model_manager._summarise_download_error(exc)
        self.assertIn("certifi", msg)

    def test_403_error_recognised(self):
        exc = Exception("403 Forbidden")
        msg = model_manager._summarise_download_error(exc)
        self.assertIn("403", msg)

    def test_disk_full_recognised(self):
        exc = OSError("No space left on device")
        msg = model_manager._summarise_download_error(exc)
        self.assertIn("磁盘", msg)

    def test_unknown_error_falls_through(self):
        exc = Exception("totally unexpected weirdness")
        msg = model_manager._summarise_download_error(exc)
        self.assertIn("totally unexpected weirdness", msg)


class TestApplyHFMirror(unittest.TestCase):

    def test_env_var_not_overwritten(self):
        with patch.dict(os.environ, {"HF_ENDPOINT": "https://my.mirror/"}):
            model_manager.apply_hf_mirror()
            self.assertEqual(os.environ["HF_ENDPOINT"], "https://my.mirror/")

    def test_reads_from_settings_when_env_empty(self):
        """apply_hf_mirror should pick up hf_endpoint from settings when HF_ENDPOINT is unset."""
        from src.config import settings_manager
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            fake = Path(tmp) / "settings.json"
            fake.write_text('{"hf_endpoint": "https://mirror.from-settings/"}', encoding="utf-8")
            env = {k: v for k, v in os.environ.items() if k != "HF_ENDPOINT"}
            with patch.dict(os.environ, env, clear=True):
                with patch.object(settings_manager, "get_settings_path", return_value=fake):
                    model_manager.apply_hf_mirror()
                    self.assertEqual(
                        os.environ.get("HF_ENDPOINT"),
                        "https://mirror.from-settings/",
                    )


if __name__ == "__main__":
    unittest.main()
