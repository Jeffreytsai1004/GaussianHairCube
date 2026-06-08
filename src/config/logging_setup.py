"""
Logging Setup
=============

Centralised logging configuration for the GaussianHairCube application.

Log file location (Windows):  %APPDATA%\\GaussianHairCube\\logs\\app.log
Rotation: 5 MB per file, 3 backups (15 MB total cap).

In addition to the file handler, an in-memory ring buffer captures every
log record so that a Log Window opened later can replay recent history.

Usage in any module:

    import logging
    logger = logging.getLogger(__name__)
    logger.info("started")
    logger.exception("failed because…")
"""

import logging
import os
import threading
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, List, Tuple


_LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False
_BUFFER_SIZE = 2000

# Thread-safe ring buffer of (formatted_msg, level_name)
_log_buffer: deque = deque(maxlen=_BUFFER_SIZE)
_buffer_lock = threading.Lock()

# Listeners are called from the emitting thread; they must marshal to UI thread themselves
_listeners: List[Callable[[str, str], None]] = []


class _BufferHandler(logging.Handler):
    """Captures every log record into a bounded in-memory buffer and notifies listeners."""

    def emit(self, record):
        try:
            msg = self.format(record)
            with _buffer_lock:
                _log_buffer.append((msg, record.levelname))
            for cb in list(_listeners):
                try:
                    cb(msg, record.levelname)
                except Exception:
                    # Never let a listener kill the logger
                    pass
        except Exception:
            self.handleError(record)


def get_log_buffer() -> List[Tuple[str, str]]:
    """Snapshot of the (msg, level) ring buffer for newly-opened log windows."""
    with _buffer_lock:
        return list(_log_buffer)


def add_log_listener(callback: Callable[[str, str], None]):
    if callback not in _listeners:
        _listeners.append(callback)


def remove_log_listener(callback: Callable[[str, str], None]):
    try:
        _listeners.remove(callback)
    except ValueError:
        pass


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

    # Memory buffer — always captures DEBUG+ so the log window can show fine detail
    buf_handler = _BufferHandler()
    buf_handler.setLevel(logging.DEBUG)
    buf_handler.setFormatter(fmt)
    root.addHandler(buf_handler)

    if console:
        stream = logging.StreamHandler()
        stream.setLevel(logging.WARNING)
        stream.setFormatter(fmt)
        root.addHandler(stream)

    _configured = True
    logging.getLogger(__name__).info(
        "Logging initialised — writing to %s (level=%s)",
        get_log_file(), logging.getLevelName(level),
    )
    return get_log_file()
