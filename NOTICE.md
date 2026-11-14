# Third-party components / 第三方组件

RUBBER STUDIO itself is MIT-licensed (see `LICENSE`). It bundles or uses the following components, each under its own
license. / RUBBER STUDIO 本身采用 MIT 许可证，并使用下列组件（各自遵循其许可证）。

| Component | Use | License |
|---|---|---|
| **PP-OCRv4 text detection model** (`ch_PP-OCRv4_det_infer.onnx`) — PaddleOCR / PaddlePaddle authors; ONNX build as distributed by RapidOCR | Finds text lines in images (bundled, ~4.7 MB) | Apache-2.0 |
| **LaMa – Large Mask Inpainting** (Suvorov et al.), ONNX export `Carve/LaMa-ONNX` | Optional neural fill for busy backgrounds. **Not bundled** – downloaded only after you confirm, or imported from a file you choose | Apache-2.0 |
| **Archivo Black** (Omnibus-Type) and **Space Mono** (Colophon Foundry) | UI typefaces, bundled in `assets/fonts` | SIL Open Font License 1.1 (texts next to the fonts) |
| **Qt for Python / PySide6-Essentials** (The Qt Company) | GUI toolkit | LGPL-3.0 – dynamically linked; the Qt libraries are separate files in the application folder and can be replaced |
| **OpenCV** (`opencv-python-headless`) | Image processing, inpainting | Apache-2.0 |
| **ONNX Runtime** (Microsoft) | Runs the ONNX models on the CPU | MIT |
| **NumPy** | Array maths | BSD-3-Clause (wheels also ship OpenBLAS, BSD-3-Clause) |
| **Pillow** | Image file I/O | HPND (MIT-CMU) |
| **Python** | Runtime | PSF License |

The Windows build is produced with [PyInstaller](https://pyinstaller.org) (GPL-2.0 with the bootloader exception, which
allows distributing the resulting program under any license).

Qt (LGPL-3.0) source code: <https://download.qt.io/official_releases/QtForPython/>.
