#!/usr/bin/env python3
"""
GaussianHairCube - Main Entry Point
=====================================

Hair Reconstruction with Strand-Aligned 3D Gaussians

A Windows desktop tool for reconstructing 3D hair from multiple photos
using Gaussian Splatting + multi-view reconstruction.

Based on the GaussianHaircut research from ETH Zurich AIT Lab.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.logging_setup import setup_logging
from src.ui.main_window import MainWindow


def main():
    """Application entry point."""
    # Configure logging first so every subsequent import can log
    log_path = setup_logging()

    # Apply HuggingFace mirror from settings before any network call
    try:
        from src.core.model_manager import apply_hf_mirror
        apply_hf_mirror()
    except Exception:
        import logging
        logging.getLogger(__name__).exception("apply_hf_mirror failed at startup")

    try:
        app = MainWindow()
        app.mainloop()
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("Fatal error in main loop")
        print(f"Error: {e}\nLog file: {log_path}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())