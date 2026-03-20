#!/usr/bin/env python3
"""
GaussianHairCube - Main Entry Point
===================================

Hair Reconstruction with Strand-Aligned 3D Gaussians

A professional Windows application for reconstructing 3D hair
from images or videos using Gaussian splatting technology.

Features:
- Single image or video input
- Automatic 3D Gaussian generation
- Hair strand curve extraction
- Interactive 3D preview
- Export to Maya (FBX) and Blender (GLB)

Based on the GaussianHaircut research from ETH Zurich.
"""

import sys
import os

# Ensure the src directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ui.main_window import MainWindow


def main():
    """Application entry point."""
    print("=" * 50)
    print("GaussianHairCube - Hair Reconstruction")
    print("=" * 50)
    print()
    print("Starting application...")
    print()
    
    try:
        # Create and run the main window
        app = MainWindow()
        app.mainloop()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())