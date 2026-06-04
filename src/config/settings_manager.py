"""
Settings Manager Module
=======================

Handles loading, saving, and defaulting of application settings.
"""

import os
import json
from pathlib import Path


DEFAULT_SETTINGS = {
    'num_iterations': 1000,
    'points_per_strand': 32,
    'num_strands': 10000,
    'theme': 'dark',
    'scale_factor': 1.0,
    'up_axis': 'Y',
    'include_color': True,
    'format_type': 'fbx',
    # HuggingFace endpoint — leave empty to use the default (huggingface.co).
    # Chinese users can set this to "https://hf-mirror.com" for faster downloads.
    'hf_endpoint': '',
}


def get_settings_path() -> Path:
    """Return path to the settings JSON file."""
    return Path(os.environ.get('APPDATA', Path.home())) / 'GaussianHairCube' / 'settings.json'


def load_settings() -> dict:
    """
    Load settings from disk, merging with defaults for any missing keys.

    Returns:
        dict: Settings dictionary with all keys present.
    """
    settings = DEFAULT_SETTINGS.copy()
    path = get_settings_path()
    try:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            # Merge: missing keys fall back to defaults
            settings.update(loaded)
    except json.JSONDecodeError:
        pass  # Return defaults on corrupt file
    return settings


def save_settings(data: dict):
    """
    Save settings to disk atomically.

    Args:
        data: Settings dictionary to persist.
    """
    path = get_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write via a temp file in the same directory
    tmp_path = path.with_suffix('.tmp')
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
