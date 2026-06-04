# GaussianHairCube

### 发丝三维重建工具 / 3D Hair Reconstruction Tool

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python) ![Platform](https://img.shields.io/badge/Platform-Windows-0078D4?logo=windows) ![Status](https://img.shields.io/badge/Status-WIP-orange) ![License](https://img.shields.io/badge/License-MIT-green)

> ⚠️ **开发中，功能尚未完整，请勿用于生产环境。/ Work in Progress — not production-ready.**

---

## 项目简介

GaussianHairCube 是一款面向影视和游戏美术师的 Windows 桌面工具，允许用户输入多张照片，通过 AI 自动重建真实感 3D 发丝并导出到 Maya 或 Blender 中使用。本工具以 3D Gaussian Splatting 技术为核心，参考 ETH Zurich AIT Lab 的 GaussianHaircut 研究成果实现。

---

## Introduction

GaussianHairCube is a Windows desktop tool aimed at film and game artists, enabling users to reconstruct photo-realistic 3D hair strands from multiple photographs and export them directly to Maya or Blender. The tool is built around 3D Gaussian Splatting, drawing on the GaussianHaircut research from ETH Zurich AIT Lab.

---

## 核心特性

1. **多图输入** — 最少 3 张照片，支持 PNG / JPG / BMP / TIFF / WEBP 格式。
2. **AI 发丝分割** — 集成 `jonathandinu/face-parsing`（基于 CelebAMask-HQ 数据集训练，约 85 MB），自动分离发丝区域。
3. **单目深度估计** — 集成 `depth-anything/Depth-Anything-V2-Small-hf`（约 99 MB），从单张图像预测稠密深度图。
4. **多视角三维重建** — 支持四级降级策略：pycolmap SfM → OpenCV 位姿链 → 合成占位符，确保在不同环境下均可运行。
5. **高斯点云生成与优化** — GPU 模式执行完整高斯优化；无 GPU 时自动切换至 CPU 模式，跳过优化步骤直接返回初始结果。
6. **发丝曲线提取** — 提供图遍历聚类与 RK4 流场积分两种发丝曲线提取方法。
7. **多格式导出** — 支持导出 FBX（供 Maya 使用）与 GLB（供 Blender 使用）。

---

## Features

1. **Multi-image Input** — Minimum 3 photos required; supports PNG / JPG / BMP / TIFF / WEBP formats.
2. **AI Hair Segmentation** — Integrates `jonathandinu/face-parsing` (trained on CelebAMask-HQ, ~85 MB) for automatic hair region extraction.
3. **Monocular Depth Estimation** — Integrates `depth-anything/Depth-Anything-V2-Small-hf` (~99 MB) to predict dense depth maps from single images.
4. **Multi-view 3D Reconstruction** — Four-level fallback strategy: pycolmap SfM → OpenCV pose chain → synthetic placeholder, ensuring operation across different environments.
5. **Gaussian Point Cloud Generation & Optimization** — Full Gaussian optimization in GPU mode; CPU mode skips optimization and returns initial results immediately.
6. **Hair Strand Curve Extraction** — Two extraction methods: graph-traversal clustering and RK4 flow-field integration.
7. **Multi-format Export** — Exports FBX (for Maya) and GLB (for Blender).

---

## 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10 / 11 64-bit（macOS / Linux 未经测试） |
| Python | 3.9 或更高版本 |
| 内存 | 最低 8 GB，推荐 16 GB |
| GPU | 可选，CUDA 11.8+（无 GPU 时跳过高斯优化步骤） |
| 网络 | 首次运行需从 HuggingFace 下载 AI 模型，合计约 184 MB |
| 磁盘空间 | 约 2 GB（含 Python 依赖与模型缓存） |

---

## System Requirements

| Item | Requirement |
|------|-------------|
| OS | Windows 10 / 11 64-bit (macOS / Linux untested) |
| Python | 3.9 or higher |
| RAM | Minimum 8 GB, 16 GB recommended |
| GPU | Optional, CUDA 11.8+ (Gaussian optimization is skipped without a GPU) |
| Network | ~184 MB AI model download from HuggingFace required on first run |
| Disk Space | ~2 GB (including Python dependencies and model cache) |

---

## 安装

**1. 获取代码**
```bash
git clone <仓库地址>
cd GaussianHairCube
```

**2. 创建虚拟环境**
```cmd
python -m venv venv
venv\Scripts\activate
```

**3. 安装依赖**
```bash
pip install -r requirements.txt
```

**4. 安装可选依赖（推荐，提升重建质量）**
```bash
pip install pycolmap
```
> `pycolmap` 提供完整 SfM 特征匹配与相机位姿估计能力。未安装时程序自动降级到 OpenCV 流程，重建质量会有所下降。

**5. 启动程序**
```bash
python main.py
```

**6. 首次运行说明**

首次启动会弹出模型下载对话框，程序需要从 HuggingFace 下载约 **184 MB** 的 AI 模型文件（face-parsing SegFormer + Depth-Anything-V2-Small）。下载完成后模型将缓存到本地，后续启动无需重新下载。请确保网络可正常访问 HuggingFace Hub，或预先配置镜像源。

---

## Installation

**1. Get the source code**
```bash
git clone <repository-url>
cd GaussianHairCube
```

**2. Create a virtual environment**
```cmd
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Install optional dependency (recommended — improves reconstruction quality)**
```bash
pip install pycolmap
```
> `pycolmap` enables full SfM feature matching and camera pose estimation. Without it, the pipeline automatically falls back to an OpenCV-based workflow with reduced reconstruction quality.

**5. Launch the application**
```bash
python main.py
```

**6. First-run note**

On the first launch, a model download dialog will appear. The application downloads approximately **184 MB** of AI model weights from HuggingFace (face-parsing SegFormer + Depth-Anything-V2-Small). Models are cached locally after the initial download; subsequent launches start immediately. Ensure your network can reach HuggingFace Hub, or configure a mirror endpoint in advance.

---

## 使用流程

### 第一步：输入图片

在主窗口左侧面板，点击 **Add Images** 按钮，或直接将图片文件拖放到面板区域。程序要求至少导入 **3 张**包含人物头部的照片。

- **推荐**：提供多角度图片（正面、侧面、斜 45° 等），角度越丰富，发丝重建的几何精度越高。
- 图片导入后会显示缩略图预览，可随时删除或补充。

### 第二步：处理

| 操作 | 说明 |
|---|---|
| **Auto Process** | 一键完成全流程：自动生成 3D 高斯点云并提取发丝曲线 |
| **Generate Gaussians** | 仅执行第一阶段：发丝分割 → 深度估计 → 多视图重建 → 高斯优化 |
| **Extract Curves** | 仅执行第二阶段：从已有高斯点云提取发丝曲线 |

### 第三步：导出

处理完成后，在主窗口右侧面板选择输出格式并导出：

- **Maya FBX**：导出为 `.fbx` 文件，包含发丝曲线，可直接导入 Autodesk Maya。
- **Blender GLB**：导出为 `.glb` 文件（glTF 2.0 二进制格式），可直接导入 Blender。

### 3D 预览交互

| 操作 | 效果 |
|---|---|
| 鼠标左键拖动 | 旋转视角 |
| 鼠标右键拖动 | 平移场景 |
| 滚轮滚动 | 缩放 |
| 视图模式切换 | 高斯点云 / 发丝曲线 / 混合显示 |

### 性能参考

- **GPU 模式**（NVIDIA CUDA）：约 **30–120 秒 / 组图片**，支持高斯优化，输出质量更高。
- **CPU 模式**：速度更快（跳过优化阶段），但输出几何细节较少。

---

## Usage

### Step 1: Import Images

In the left panel of the main window, click **Add Images** or drag and drop image files onto the panel. The application requires a minimum of **3 images** showing a person's head.

- **Recommended**: use images taken from multiple angles (front, side, 45° diagonal, etc.). Greater angular coverage leads to higher geometric accuracy.
- Imported images are shown as thumbnails and can be removed or supplemented at any time.

### Step 2: Process

| Action | Description |
|---|---|
| **Auto Process** | Full one-click pipeline: generates 3D Gaussians and extracts hair strand curves automatically |
| **Generate Gaussians** | First stage only: hair segmentation → depth estimation → multi-view reconstruction → Gaussian optimization |
| **Extract Curves** | Second stage only: extract strand curves from an existing Gaussian point cloud |

### Step 3: Export

After processing completes, select an output format in the right panel and export:

- **Maya FBX**: exports a `.fbx` file containing hair strand curves, ready for import into Autodesk Maya.
- **Blender GLB**: exports a `.glb` file (glTF 2.0 binary), ready for import into Blender.

### 3D Viewport Controls

| Input | Action |
|---|---|
| Left mouse button drag | Rotate view |
| Right mouse button drag | Pan scene |
| Scroll wheel | Zoom |
| View mode toggle | Gaussian point cloud / Hair curves / Combined |

### Performance Reference

- **GPU mode** (NVIDIA CUDA): approximately **30–120 seconds per image set**, with Gaussian optimization enabled for higher output quality.
- **CPU mode**: faster overall (optimization stage is skipped), but with less geometric detail in the output.

---

## 技术架构

### 数据流

```
输入图片（≥3张）
        │
        ▼
┌─────────────────────────────┐
│        InputPanel           │  多图上传 + 缩略图预览
└────────────┬────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────┐
│         GaussianGenerator.generate_from_images()         │
│                                                          │
│  1. _ensure_models_loaded()                              │
│       face-parsing SegFormer（懒加载）                   │
│       Depth-Anything-V2-Small（懒加载）                  │
│                                                          │
│  2. 每帧处理                                             │
│       _preprocess_image() → _generate_hair_mask()        │
│       → 深度图估计                                       │
│                                                          │
│  3. reconstruct_from_frames()  ← 4级降级 SfM            │
│       Tier 1: pycolmap 完整 SfM  （≥10帧, ≥500点）      │
│       Tier 2: pycolmap 部分 SfM  （≥5帧,  ≥200点）      │
│       Tier 3: OpenCV SIFT + 基础矩阵 + 三角化            │
│       Tier 4: 合成半球点云        （兜底，永不失败）      │
│                                                          │
│  4. _initialize_gaussians_from_points()  局部PCA初始化   │
│                                                          │
│  5. [GPU] _optimize_gaussians_single_view()              │
│       软光栅化 + Adam优化，1000次迭代，多视角轮询         │
│       [CPU] 跳过优化，直接返回初始点云                   │
└────────────┬─────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────┐
│           HairStrandsExtractor.extract()                 │
│                                                          │
│  方法A（Clustering）：主方向计算 → 方向图构建 → 图遍历   │
│  方法B（Flow Field）：RK4 流场积分                       │
│  后处理：平滑 + 重采样 + 长度过滤                        │
└────────────┬─────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────┐
│        OutputPanel          │  FBX（Maya）/ GLB（Blender）
└─────────────────────────────┘
```

### 核心模块

| 模块 | 文件 | 技术方案 | 状态 |
|---|---|---|---|
| 发丝分割 | `core/gaussian_generator.py` | face-parsing SegFormer，类别 ID 13 = hair | ✅ 已实现 |
| 深度估计 | `core/gaussian_generator.py` | Depth-Anything-V2-Small，单目深度估计 | ✅ 已实现 |
| 多视图重建 | `core/multiview_reconstruction.py` | 4 级降级 SfM，任意输入下均可返回点云 | ✅ 已实现 |
| 高斯优化 | `core/gaussian_generator.py` | 软光栅化 + Adam（简化版，非完整 3DGS），GPU 专属 | ⚠️ 简化版 |
| 发丝提取 | `core/hair_strands.py` | 主方向 PCA + 图遍历 / RK4 流场积分 | ✅ 已实现 |
| 模型缓存 | `core/model_manager.py` | HuggingFace Hub 下载，本地磁盘缓存，懒加载 | ✅ 已实现 |
| 3D 渲染 | `rendering/viewer_3d.py` | OpenGL / ModernGL，实时点云与曲线渲染 | ✅ 已实现 |
| FBX 导出 | `export/fbx_exporter.py` | 基于 trimesh，Maya 兼容 | ⚠️ 未验证 |
| GLB 导出 | `export/glb_exporter.py` | 基于 pygltflib，glTF 2.0 | ⚠️ 未验证 |
| 设置持久化 | `config/settings_manager.py` | JSON 文件读写，跨会话保存配置 | ✅ 已实现 |

### GPU / CPU 模式差异

- **GPU 模式**：完整运行全部5个阶段，包含高斯优化（1000 次迭代软光栅化）。发丝几何细节更丰富。
- **CPU 模式**：跳过高斯优化阶段，直接将 SfM 三角化点云传递给发丝提取器。处理速度更快，但几何精度相对降低。

---

## Technical Architecture

### Data Flow

*(See diagram above.)*

### Core Modules

| Module | File | Technology | Status |
|---|---|---|---|
| Hair segmentation | `core/gaussian_generator.py` | face-parsing SegFormer, class ID 13 = hair | ✅ Implemented |
| Depth estimation | `core/gaussian_generator.py` | Depth-Anything-V2-Small, monocular depth | ✅ Implemented |
| Multi-view reconstruction | `core/multiview_reconstruction.py` | 4-tier fallback SfM; always returns a point cloud | ✅ Implemented |
| Gaussian optimization | `core/gaussian_generator.py` | Soft rasterization + Adam (simplified, not full 3DGS); GPU only | ⚠️ Simplified |
| Hair strand extraction | `core/hair_strands.py` | PCA principal direction + graph traversal / RK4 flow field | ✅ Implemented |
| Model cache | `core/model_manager.py` | HuggingFace Hub download, local disk cache, lazy loading | ✅ Implemented |
| 3D rendering | `rendering/viewer_3d.py` | OpenGL / ModernGL, real-time point cloud and curve rendering | ✅ Implemented |
| FBX export | `export/fbx_exporter.py` | trimesh-based, Maya-compatible | ⚠️ Unverified |
| GLB export | `export/glb_exporter.py` | pygltflib-based, glTF 2.0 | ⚠️ Unverified |
| Settings persistence | `config/settings_manager.py` | JSON file read/write, cross-session config | ✅ Implemented |

### GPU vs. CPU Mode

- **GPU mode**: runs all five pipeline stages in full, including Gaussian optimization (1,000-iteration soft rasterization). Yields richer hair geometry detail.
- **CPU mode**: skips the Gaussian optimization stage and passes the raw SfM point cloud directly to the strand extractor. Faster, but with reduced geometric precision.

---

## 已知限制

> 当前版本为开发中（WIP）状态，以下限制已在代码层面确认，使用前请知悉。

| # | 限制 | 实际影响 |
|---|------|----------|
| 1 | **高斯优化为简化实现** — 使用 2D 软光栅化投影，并非真正的 3DGS 微分渲染器 | 发丝几何质量低于论文效果 |
| 2 | **CPU 模式下高斯优化被跳过** — 无 GPU 时直接返回初始化结果 | 纯 CPU 环境下输出质量明显低于 GPU 模式 |
| 3 | **多视图重建质量依赖 pycolmap** — 不可用时降级至合成占位符点云 | 若 pycolmap 未安装，三维重建精度大幅下降 |
| 4 | **FBX/GLB 导出未在 DCC 工具中验证** — 代码已实现，但未在 Maya/Blender 实际测试 | 导出文件可能出现导入错误或坐标轴偏差 |
| 5 | **仅 Windows 平台正式支持** — 打包与测试均仅在 Windows 上完成 | macOS / Linux 用户可能遭遇未知兼容性问题 |
| 6 | **无自动化测试覆盖** — 项目不包含任何单元测试或集成测试 | 代码稳定性未经系统验证 |
| 7 | **Geometry 编辑功能当前不可用** — 后端代码存在但尚无 UI 入口 | 用户无法通过界面对发丝几何进行手动调整 |

---

## Known Limitations

> This project is currently a work-in-progress (WIP). The following limitations have been confirmed at the code level.

| # | Limitation | Practical Impact |
|---|------------|-----------------|
| 1 | **Gaussian optimization is a simplified implementation** — uses 2D soft-rasterization, not a true 3DGS differentiable renderer | Hair geometry quality falls below paper-reported results |
| 2 | **Gaussian optimization is skipped in CPU mode** — returns the initialization result directly without a GPU | Output quality in CPU-only environments is significantly degraded |
| 3 | **Multi-view reconstruction quality depends on pycolmap** — degrades to a synthetic placeholder if unavailable | Without pycolmap, 3D reconstruction accuracy drops substantially |
| 4 | **FBX/GLB export is unverified in DCC tools** — code exists but has not been tested in Maya or Blender | Exported files may encounter import errors or axis misalignment |
| 5 | **Windows is the only officially supported platform** | macOS / Linux users may encounter unknown compatibility issues |
| 6 | **No automated test coverage** | Code stability is not systematically verified |
| 7 | **Geometry editing is currently unavailable** — backend code exists but has no UI entry point | Users cannot manually adjust hair geometry through the interface |

---

## 开发计划

### 近期

- [ ] 优化模型权重下载体验，添加进度条与断点续传支持
- [ ] 在 Maya 和 Blender 中实际验证 FBX/GLB 导出，修复兼容性问题
- [ ] 提升 UI 整体稳定性，覆盖边缘情况与异常处理

### 长期

- [ ] 集成真正的 3D Gaussian Splatting 优化器（如 [gsplat](https://github.com/nerfstudio-project/gsplat) 或 [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)），替换当前简化实现
- [ ] 实现 Geometry 编辑 UI，支持基于笔刷的发丝密度与朝向控制
- [ ] 扩展平台支持，适配 macOS 与 Linux
- [ ] 支持批量处理，允许一次性重建多组输入照片

---

## Roadmap

### Short-term

- [ ] Improve model weight download experience with progress indicators and resume-on-failure support
- [ ] Validate FBX/GLB export in Maya and Blender; fix discovered compatibility issues
- [ ] Improve overall UI stability, covering edge cases and exception handling

### Long-term

- [ ] Integrate a true 3D Gaussian Splatting optimizer (e.g. [gsplat](https://github.com/nerfstudio-project/gsplat) or [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)) to replace the current simplified implementation
- [ ] Implement a Geometry editing UI with brush-based controls for hair density and orientation
- [ ] Expand platform support to macOS and Linux
- [ ] Add batch processing support for reconstructing multiple input sets in a single run

---

## 致谢与许可

### 致谢

- **[GaussianHaircut](https://eth-ait.github.io/GaussianHaircut/)** — ETH Zurich AIT Lab。本项目整体技术框架的核心参考论文。
- **[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)** — INRIA。原始 3DGS 方法，为本项目的高斯表示提供理论基础。
- **[jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing)** — 用于发丝区域分割的预训练模型。
- **[depth-anything/Depth-Anything-V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)** — 用于单目深度估计的预训练模型。
- **[CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)** — Tom Schimansky，提供现代化 GUI 组件。

### 许可证

本项目基于 [MIT License](LICENSE) 开源发布。

---

## Acknowledgments & License

### Acknowledgments

- **[GaussianHaircut](https://eth-ait.github.io/GaussianHaircut/)** — ETH Zurich AIT Lab. The central technical reference for this project.
- **[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)** — INRIA. The original 3DGS method providing the theoretical foundation for the Gaussian representation.
- **[jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing)** — Pre-trained model used for hair region segmentation.
- **[depth-anything/Depth-Anything-V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)** — Pre-trained model used for monocular depth estimation.
- **[CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)** — Tom Schimansky. Modern GUI components powering the desktop interface.

### License

This project is released under the [MIT License](LICENSE).
