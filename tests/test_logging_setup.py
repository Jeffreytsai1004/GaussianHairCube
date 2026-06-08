"""Test logging setup module."""

import logging
import os
import tempfile
import unittest
from unittest.mock import patch

from src.config import logging_setup


def _reset_logging():
    """Close + remove every root handler so files are released (critical on Windows)."""
    logging_setup._configured = False
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:
            pass
        root.removeHandler(h)


def _make_tmp_appdata():
    """TemporaryDirectory that tolerates Windows leaving the .log open briefly."""
    # ignore_cleanup_errors is Python 3.10+
    return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


class TestLoggingSetup(unittest.TestCase):

    def setUp(self):
        _reset_logging()

    def tearDown(self):
        _reset_logging()

    def test_get_log_dir_under_appdata(self):
        with patch.dict(os.environ, {"APPDATA": "/fake/appdata"}):
            d = logging_setup.get_log_dir()
            self.assertIn("GaussianHairCube", str(d))
            self.assertIn("logs", str(d))

    def test_setup_creates_log_file(self):
        with _make_tmp_appdata() as tmpdir:
            with patch.dict(os.environ, {"APPDATA": tmpdir}):
                log_path = logging_setup.setup_logging(console=False)
                logger = logging.getLogger("test_module")
                logger.info("hello world")
                for h in logging.getLogger().handlers:
                    h.flush()
                self.assertTrue(log_path.exists())
                content = log_path.read_text(encoding="utf-8")
                self.assertIn("hello world", content)
                self.assertIn("test_module", content)
            _reset_logging()

    def test_setup_is_idempotent(self):
        """Calling setup_logging twice should not duplicate handlers."""
        with _make_tmp_appdata() as tmpdir:
            with patch.dict(os.environ, {"APPDATA": tmpdir}):
                logging_setup.setup_logging(console=False)
                n1 = len(logging.getLogger().handlers)
                logging_setup.setup_logging(console=False)
                n2 = len(logging.getLogger().handlers)
                self.assertEqual(n1, n2)
            _reset_logging()

    def test_logger_captures_exceptions(self):
        with _make_tmp_appdata() as tmpdir:
            with patch.dict(os.environ, {"APPDATA": tmpdir}):
                log_path = logging_setup.setup_logging(console=False)
                logger = logging.getLogger("err_test")
                try:
                    raise ValueError("boom!")
                except ValueError:
                    logger.exception("caught boom")
                for h in logging.getLogger().handlers:
                    h.flush()
                content = log_path.read_text(encoding="utf-8")
                self.assertIn("caught boom", content)
                self.assertIn("ValueError", content)
                self.assertIn("boom!", content)
            _reset_logging()


if __name__ == "__main__":
    unittest.main()
