# GaussianHairCube 💇

**Hair Reconstruction with Strand-Aligned 3D Gaussians**

A professional Windows application for reconstructing 3D hair from images or videos using Gaussian splatting technology. Based on the [GaussianHaircut](https://eth-ait.github.io/GaussianHaircut/) research from ETH Zurich.

![GaussianHairCube Screenshot](assets/screenshot.png)

## ✨ Features

- **Single Image Processing**: Generate 3D hair from a single photo
- **Video Processing**: Use multi-view video for enhanced reconstruction
- **3D Gaussian Splatting**: State-of-the-art hair representation
- **Interactive 3D Preview**: Real-time visualization with orbit controls
- **Geometry-Aware Editing**: Brush-based control for density, orientation, scale
- **Maya FBX Export**: NURBS curves compatible with Maya workflows
- **Blender GLB Export**: Full support for Blender 3D workflows
- **Professional UI**: Modern CustomTkinter interface

## 🚀 Quick Start

### Running from Source

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/GaussianHairCube.git
   cd GaussianHairCube
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or: source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

### Running the Executable

Download the latest release from the [Releases](releases) page and run `GaussianHairCube.exe`.

## 🛠️ Building from Source

### Prerequisites

- Python 3.9+
- pip
- PyInstaller (for building executable)

### Build Steps

1. **Install build dependencies**
   ```bash
   pip install pyinstaller
   ```

2. **Run the build script**
   ```bash
   python build.py
   ```

   Options:
   - `--onefile`: Create single executable (slower startup)
   - `--onedir`: Create directory distribution (default)
   - `--debug`: Build with debug console
   - `--clean`: Clean build directories first

3. **Find the output**
   - Directory build: `dist/GaussianHairCube/GaussianHairCube.exe`
   - Single file: `dist/GaussianHairCube.exe`

### Creating an Installer

1. Install [Inno Setup](https://jrsoftware.org/isinfo.php)
2. Generate the installer script:
   ```bash
   python build.py --installer
   ```
3. Open `installer.iss` in Inno Setup and compile

## 📖 Usage Guide

### Input

1. **Image Input**: 
   - Click "Upload Image" or drag & drop
   - Supported formats: PNG, JPG, JPEG, BMP, WEBP
   
2. **Video Input**:
   - Click "Upload Video" or drag & drop
   - Use the slider to extract frames
   - Supported formats: MP4, AVI, MOV, MKV, WEBM

### Processing

1. **Generate Gaussians**: Creates 3D Gaussian splats from input
2. **Extract Curves**: Converts Gaussians to hair strand curves
3. **Auto Process**: Runs both steps automatically

### Preview

- **Left Click + Drag**: Rotate view
- **Right Click + Drag**: Pan view
- **Scroll Wheel**: Zoom in/out
- **View Modes**: Toggle between Gaussian, Curves, and Combined views

### Export

1. Select export format (FBX for Maya, GLB for Blender)
2. Configure options (resolution, curve type)
3. Click "Export" and choose save location

## 🏗️ Architecture

```
GaussianHairCube/
├── main.py                 # Application entry point
├── build.py                # Build script
├── requirements.txt        # Python dependencies
├── GaussianHairCube.spec   # PyInstaller spec file
│
├── src/
│   ├── core/               # Core algorithms
│   │   ├── gaussian_generator.py   # 3D Gaussian generation
│   │   ├── hair_strands.py         # Strand extraction
│   │   └── geometry_controller.py  # Geometry editing
│   │
│   ├── rendering/          # 3D rendering
│   │   ├── viewer_3d.py            # 3D viewer component
│   │   └── gaussian_renderer.py    # Gaussian splatting renderer
│   │
│   ├── export/             # Export formats
│   │   ├── fbx_exporter.py         # Maya FBX export
│   │   └── glb_exporter.py         # Blender GLB export
│   │
│   └── ui/                 # User interface
│       ├── main_window.py          # Main application window
│       ├── input_panel.py          # Input controls
│       ├── viewer_widget.py        # 3D viewer widget
│       └── output_panel.py         # Export controls
│
└── assets/                 # Application assets
    └── icon.ico            # Application icon
```

## 🔬 Technical Details

### 3D Gaussian Representation

The application uses 3D Gaussian Splatting for hair representation:
- Each Gaussian has position, scale, rotation, opacity, and color
- Strand-aligned Gaussians follow hair flow direction
- Efficient rendering with depth-sorted alpha blending

### Curve Extraction

Hair curves are extracted using:
- K-means clustering for strand grouping
- Flow-field based path tracing
- NURBS fitting for smooth curves

### Export Formats

**FBX (Maya)**:
- ASCII FBX 7.5 format
- NURBS curves with control points
- Y-up coordinate system

**GLB (Blender)**:
- Binary glTF 2.0 format
- Mesh tube representation for curves
- Z-up coordinate system conversion

## 📋 Requirements

- **OS**: Windows 10/11 (64-bit)
- **Python**: 3.9 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional (software rendering available)

### Python Dependencies

```
customtkinter>=5.2.0
pillow>=10.0.0
numpy>=1.24.0
opencv-python>=4.8.0
scipy>=1.11.0
scikit-learn>=1.3.0
pygltflib>=1.16.0
pyinstaller>=6.0.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [GaussianHaircut](https://eth-ait.github.io/GaussianHaircut/) - ETH Zurich AIT Lab
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - INRIA
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) - Modern Python UI

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/GaussianHairCube/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/GaussianHairCube/discussions)