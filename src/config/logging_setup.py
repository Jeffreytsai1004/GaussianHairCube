"""
Logging Setup
=============

Centralised logging configuration for the GaussianHairCube application.

Log file location (Windows):  %APPDATA%\\GaussianHairCube\\logs\\app.log
Rotation: 5 MB per file, 3 backups (15 MB total cap).

Usage in any module:

    import logging
    logger = logging.getLogger(__name__)
    logger.info("started")
    logger.exception("failed because…")
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


_LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def get_log_dir() -> Path:
    """Return the directory holding the rolling log files."""
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "GaussianHairCube" / "logs"


def get_log_file() -> Path:
    return get_log_dir() / "app.log"


def setup_logging(level: int = logging.INFO, console: bool = True) -> Path:
    """
    Configure the root logger once.  Subsequent calls are a no-op.

    Returns the log file path so callers can show it to the user.
    """
    global _configured
    if _configured:
        return get_log_file()

    log_dir = get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # capture everything; handlers filter

    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = RotatingFileHandler(
        get_log_file(),
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    if console:
        stream = logging.StreamHandler()
        stream.setLevel(logging.WARNING)   # console only shows WARN+
        stream.setFormatter(fmt)
        root.addHandler(stream)

    _configured = True
    logging.getLogger(__name__).info(
        "Logging initialised — writing to %s (level=%s)",
        get_log_file(), logging.getLevelName(level),
    )
    return get_log_file()
