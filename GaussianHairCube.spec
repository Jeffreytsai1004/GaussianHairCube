# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_all

datas = [('assets', 'assets')]
datas += collect_data_files('customtkinter')

# Collect tkinterdnd2 native binaries (drag-and-drop support)
try:
    datas += collect_data_files('tkinterdnd2')
except Exception:
    pass

hiddenimports = [
    # Image / UI
    'PIL', 'PIL.Image', 'PIL.ImageTk', 'PIL.ImageOps', 'PIL.ImageFilter',
    'customtkinter',
    'CTkMessagebox',
    'tkinterdnd2',
    # Numerics
    'numpy', 'numpy.core', 'numpy.linalg',
    'scipy', 'scipy.ndimage', 'scipy.spatial', 'scipy.spatial.ckdtree',
    'scipy.spatial.transform', 'scipy.linalg',
    'sklearn', 'sklearn.cluster', 'sklearn.neighbors',
    # Vision
    'cv2',
    # 3-D rendering
    'moderngl', 'pyrr', 'OpenGL', 'OpenGL.GL',
    'pyglet',
    # Geometry / mesh
    'trimesh', 'trimesh.transformations',
    # Export
    'pygltflib',
    # App modules (all must be explicit for PyInstaller)
    'src.config.settings_manager',
    'src.config.logging_setup',
    'src.core.model_manager',
    'src.core.multiview_reconstruction',
    'src.core.geometry_controller',
    'src.core.project_io',
    'src.core.batch_processor',
    'src.ui.model_download_dialog',
    'src.ui.batch_dialog',
    'src.ui.log_window',
    # HuggingFace (optional — only bundled if installed)
    'huggingface_hub', 'huggingface_hub.utils', 'safetensors',
]

# Add torch-related hidden imports only if torch is installed
try:
    import torch
    hiddenimports += ['torch', 'torch.nn', 'torch.nn.functional', 'torchvision']
except ImportError:
    pass

# Add transformers hidden imports only if installed
try:
    import transformers
    hiddenimports += [
        'transformers', 'transformers.models.segformer',
        'transformers.models.depth_anything',
        'accelerate',
    ]
except ImportError:
    pass

# Add pycolmap only if installed
try:
    import pycolmap
    hiddenimports += ['pycolmap']
except ImportError:
    pass


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'tkinter.test', 'unittest', 'pytest', 'IPython', 'jupyter', 'notebook'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GaussianHairCube',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets\\icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GaussianHairCube',
)
