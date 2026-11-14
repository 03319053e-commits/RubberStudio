<div align="center">

<img src="assets/icon.png" width="112" alt="RUBBER STUDIO 图标">

# RUBBER STUDIO

**擦除 · 修复 · 还原**<br>
批量去除图片里的文字，另有修复、裁剪、加水印 —— 完全离线运行的 Windows 原生程序。

<!--badges-->
[![Build Windows app](https://github.com/03319053e-commits/RubberStudio/actions/workflows/build-windows.yml/badge.svg)](https://github.com/03319053e-commits/RubberStudio/actions/workflows/build-windows.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)
<!--/badges-->

[**⬇ 下载 Windows 版**](../../releases/latest) · [English](README.md)

<img src="docs/images/batch-zh.png" alt="RUBBER STUDIO —— 批量去文字与队列" width="900">

</div>

## 功能

- **批量去除文字** —— 把一堆图片或整个文件夹拖进窗口，点“开始”，每张图都会去掉文字后输出。队列实时显示每张图的状态；
  可随时停止、重新运行，也可以把任意一张打开到编辑器里对比“原图”。
- **真正的去字，而不是模糊** —— 两个本机文字检测器（PP-OCRv4 与 PP-OCRv5，支持中文、英文及混排，多尺度检测）找到文字，
  只重绘**笔画本身**，文字周围的画面保持不动。“彻底模式”下，每个笔画蒙版都会交给检测器自己复查——如果还能看到文字，
  就换下一个候选——所以细线、浅色、字距很大的文字也不会残留。
- **可选择的填补引擎** —— “自动 / 快速 / MI-GAN / LaMa”。纯色和渐变背景（横幅、字幕、说明文字、水印）由内置填充瞬间补好；
  有图案或照片般的背景则交给神经网络模型（确认后下载一次，之后离线使用）。详见[填补引擎](#填补引擎)。
- **默认安全** —— 绝不覆盖原图。结果默认写入每张图片所在文件夹下的 `RubberStudio_clean`（也可自选文件夹），可设置文件名
  后缀、输出格式（原格式 / PNG / JPG / WebP）及质量、是否保留子文件夹结构，以及文件已存在时的处理（改名 / 跳过 / 覆盖）。
  多张图片并行处理；没有文字的图片也可以一并复制过去。
- **一个区域，全部适用** —— 在示例图上框出一块位置（比如角落的水印条），所有图片都只清理同一相对位置。
- **单张图片工具** —— “去除文字”（整张图片或框选区域，带检测预览和“原图”对比开关）、“修复”画笔（修复、涂抹、
  重建颗粒）、“裁剪”，以及可调字体、颜色、大小、不透明度的文字“水印”。
- **隐私** —— 不上传任何内容。唯一的联网行为是可选的模型下载，并且只在你确认之后才会进行。
- **为 Windows 准备** —— 绿色免安装文件夹（无需管理员权限），全程支持中文/Unicode 路径，适配高分屏，界面有中文和英文。

<div align="center">
<img src="docs/images/erase-detect.png" alt="去除文字：检测预览" width="290">
<img src="docs/images/erase-result.png" alt="文字已去除；填补引擎的切换在左侧面板里" width="290">
<img src="docs/images/folded.png" alt="所有工具栏可折叠，让画面占满窗口" width="290">
</div>

## 下载（Windows 10 / 11，64 位）

1. 打开[**最新发布页**](../../releases/latest)，下载 `RubberStudio-<版本>-windows-x64.zip`。
2. 解压到任意位置（请先完整解压，不要在压缩包预览里直接运行）。
3. 运行 `RubberStudio.exe`。

程序没有代码签名，第一次运行时 Windows SmartScreen 可能提示“Windows 已保护你的电脑”：点**更多信息 → 仍要运行**即可。
每个发布版本都由本仓库公开的 [GitHub Actions 工作流](.github/workflows/build-windows.yml) 构建，并且在发布前会运行
打包后程序的自检。

如果在很新的 Windows 上无法启动，请安装
[Microsoft Visual C++ 运行库（x64）](https://aka.ms/vs/17/release/vc_redist.x64.exe)。

设置、日志和可选的模型都保存在 `%LOCALAPPDATA%\RubberStudio`；删除该文件夹即可恢复初始状态，删除程序文件夹即为卸载。

## 使用方法

**批量去文字** —— 选择“批量去文字”工具，然后：

1. 添加图片或文件夹（用按钮，或直接拖进窗口：多个文件或文件夹会直接进入队列，单个文件则在编辑器中打开）；
2. 在“输出”卡片里设置输出文件夹、格式和选项；
3. 点“**开始**”。每一帧依次变为“处理中 → 完成”（或“无文字”/“失败”）；点“**停止**”会干净地取消，不会留下写了一半的文件。

| 快捷键 | 作用 |
|---|---|
| `Ctrl+O` / `Ctrl+S` | 打开 / 保存 |
| `Ctrl+C` / `Ctrl+V` | 复制图片 / 粘贴剪贴板中的图片 |
| `Ctrl+Z` / `Ctrl+Y` | 撤销 / 重做 |
| `Ctrl+Enter` / `Esc` | 运行当前工具 / 关闭工具 |
| `0`、滚轮、`空格`+拖动 | 适应窗口、缩放、平移 |
| `F1` | 帮助 |

工具栏可以折叠（点边缘的黑色小标签），让图片占满整个窗口。

### 填补引擎

“填补引擎”开关（在“去除文字”和“批量去文字”里，紧挨着要用到它的按钮）决定如何补上文字留下的空白：

| 引擎 | 需要下载 | 速度¹ | 适合 |
|---|---|---|---|
| **自动**（默认） | – | 4–23 秒 | 平坦背景用内置填充，复杂背景用已安装的最佳模型（优先 LaMa，其次 MI-GAN） |
| **快速** | 无 | ≈ 4 秒 | 横幅、字幕、渐变、柔和的明暗 |
| **MI-GAN** | ≈ 28 MB | ≈ 7 秒 | 照片和简单纹理 |
| **LaMa** | ≈ 208 MB | ≈ 23 秒 | 复杂图案和纹理（丝绸、布料、枝叶） |

¹ 800 × 800 的图片、两行文字、2 核 CPU，含检测时间。

下方的“模型”卡片管理这些可选模型：点“**下载**”，确认后会从 Hugging Face（或镜像）获取模型，并用固定的 SHA-256 校验；或者点“**导入模型…**”选择你已有的 ONNX 文件。
选了但没安装的模型，会改用已安装的最佳模型，并在状态栏提示。模型保存在 `%LOCALAPPDATA%\RubberStudio\models`。

## 从源码运行

Python 3.10 及以上（CI 使用 3.12），Windows / Linux / macOS 均可：

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS：source .venv/bin/activate
pip install -r requirements.txt
python -m rubberstudio            # 也可以：python -m rubberstudio 图片.png 文件夹/
```

Windows 上双击 `run_windows.bat` 即可完成同样的事。常用参数：`--lang en|zh`、`--no-splash`、
`--selftest report.json`（检查内置模型、引擎、批量处理和界面）、
`--screenshot out.png --tool erase|batch|repair|crop|watermark`（无界面渲染窗口截图）。

## 自己打包 Windows 程序

要么交给 GitHub 来做（推送一个形如 `v1.1.0` 的标签，工作流会构建、测试，并把 zip 附到发布页），要么在装有 Python 3.12 的 Windows 电脑上运行：

```bat
build_windows.bat
```

它会创建虚拟环境、安装依赖并执行 `pyinstaller packaging\RubberStudio.spec`，产物为
`dist\RubberStudio\RubberStudio.exe` 和一个可直接分享的 zip。

测试（引擎与界面，无头运行）：`pip install -r requirements-dev.txt && python -m pytest`。

## 去字原理

1. **检测** —— 两个 DBNet 文字检测器（PP-OCRv4 与 PP-OCRv5-mobile，ONNX Runtime，CPU）在多个尺度上观察图片，
   并按行合并检测框；大图按重叠分块、以原始分辨率处理，小字不会因缩放而丢失。紧贴可靠文字的小符号
   （“©”、项目符号、句点）通过滞后阈值找回；模型完全没反应的符号，则直接检查文字行两端旁边的像素来补上
   （同墨色、字符大小、仅限平坦或平滑渐变的背景）。
2. **只取笔画** —— 在每个文字框内尝试多个候选墨色（颜色直方图的峰值、最亮和最暗的像素），每个候选生成一个笔画蒙版，
   并稍作扩展、闭合，避免残留的色点。候选按“把字形与周围分开的清晰程度”排序；“彻底模式”下第一个真正有效的获胜：
   把该蒙版涂掉后让检测器再看一遍，只有“再也检测不到文字”的蒙版才会被采用（否则使用整个带边距的文字框）。
3. **填充** —— 在原始分辨率的 512 × 512 窗口里带着充足的上下文补空白，边缘羽化融合，并把图片自带的感光颗粒放回去，
   修补处不会像被喷过一样平滑。

小技巧：文字很淡或有艺术效果时提高“灵敏度”，误检了非文字图形时降低它；字周围的细描边或阴影没去干净时提高“笔画扩展”；
背景是密集图案时选“LaMa”。

## 设计

界面遵循 1975 年《NASA 平面设计标准手册》的纪律：白色页面、黑色文字与线条、只用一种红色（`#DA291C`）来标记当前工具和主要操作、
Helvetica 风格字体（内置 Liberation Sans）、加宽字距的大写标签、每个分区上方的粗黑线，以及严格的直角矩形——没有渐变、阴影、
纹理和圆角。不使用任何 NASA 标识，也与 NASA 无关。所有颜色、字体和尺寸集中在
[`rubberstudio/theme.py`](rubberstudio/theme.py)。

## 目录结构

```
rubberstudio/
  core/        不依赖 Qt 的引擎：文字检测、笔画蒙版与复查、填补引擎（LaMa / MI-GAN / 快速）、批量运行器、图片读写
  widgets/     画布、工具页、队列、可折叠面板、扁平控件、矢量图标、对话框
  editor.py    编辑模型（工具、撤销/重做、异步擦除）     mainwindow.py   主窗口、拖放、快捷键
  theme.py     设计规范与样式表                           strings.py      中英文界面文字
assets/        字体（OFL）、图标、界面图片                packaging/      PyInstaller 规格、启动器、图标脚本
tests/         引擎、批量与界面测试                       .github/        Windows 构建工作流
```

## 合理使用

请只处理你拥有或获准编辑的图片。去除他人的水印、署名或授权声明可能侵犯其权利。

## 许可证

[MIT](LICENSE)。第三方组件（检测与填补模型、字体、Qt、OpenCV、ONNX Runtime 等）及其许可证见 [NOTICE.md](NOTICE.md)。
