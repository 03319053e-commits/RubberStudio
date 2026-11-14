## RUBBER STUDIO — Windows 10 / 11 (64-bit)

**Download `RubberStudio-<version>-windows-x64.zip` below → unzip it anywhere → double-click `RubberStudio.exe`.**
No installer and no Python needed. Everything runs offline on your PC.

### New in 1.1

- **Much stronger text removal** — two detectors (PP-OCRv4 + PP-OCRv5) at several scales, a stroke mask that is built from
  the real ink colour and *checked by the detector itself*, and a fill that works at native resolution with the picture's
  grain put back. Thin, soft and letter-spaced type no longer leaves stray letters or smeared patterns.
- **Choose the fill engine** — *Auto*, *Fast*, *MI-GAN* (~28 MB) or *LaMa* (~208 MB). The models are optional, downloaded
  only after you confirm and then used offline.
- **New look** — a black, white and one-red interface in the spirit of the 1975 NASA Graphics Standards Manual:
  Helvetica-style type, strict rectangles, no gradients or shadows.

### What it does

- **Batch text removal** — add images or whole folders (or drop them on the window), choose the output folder / name suffix /
  format / what to do with existing files, press START. Nothing is ever overwritten unless you pick *overwrite*.
- **Single-picture tools** — erase text, watermark removal, repair brush, crop, with undo / redo and a before / after view.
- English / 简体中文 interface.

> The program is not code-signed, so Windows SmartScreen may say *"Windows protected your PC"* the first time:
> click **More info → Run anyway**. The `.sha256` file lets you verify the download
> (`Get-FileHash .\RubberStudio-<version>-windows-x64.zip`).

## 中文说明

**下载下面的 `RubberStudio-<版本>-windows-x64.zip` → 解压到任意位置 → 双击 `RubberStudio.exe`。**
无需安装、无需 Python，全部在本机离线运行。

### 1.1 新变化

- **去字效果大幅增强**：两个检测器（PP-OCRv4 + PP-OCRv5）多尺度检测；笔画蒙版按真实墨色生成并由检测器自己复查；
  填补在原始分辨率下进行并还原图片的颗粒。细线、浅色、字距很大的文字不再残留笔画或把图案抹花。
- **可选择填补引擎**：“自动 / 快速 / MI-GAN（约 28 MB）/ LaMa（约 208 MB）”。模型可选，确认后才会下载，之后离线使用。
- **全新外观**：参考 1975 年 NASA 平面设计标准手册的黑、白、单一红色界面——Helvetica 风格字体、严格的直角矩形，无渐变与阴影。

### 功能

- **批量去除图片内文字**：添加图片或整个文件夹（也可直接拖进窗口），设置输出文件夹 / 文件名后缀 / 格式 / 同名文件处理方式，点“开始”。
  除非你选择“覆盖”，否则不会覆盖任何文件。
- **单张图片工具**：去字、去水印、修复笔刷、裁剪，支持撤销 / 重做与前后对比。
- 中 / 英文界面。

> 程序没有代码签名，首次运行时 Windows SmartScreen 可能提示“Windows 已保护你的电脑”：点**更多信息 → 仍要运行**即可。
> 可用 `.sha256` 文件校验下载是否完整。
