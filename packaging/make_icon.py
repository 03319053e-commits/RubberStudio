"""Regenerate assets/icon.png and assets/icon.ico (a white eraser on a red square, the mark of the interface).

    python packaging/make_icon.py

The original artwork the project started from stays in assets/icon-source.jpg."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image                                   # noqa: E402
from PySide6.QtCore import QBuffer, QIODevice, QRectF   # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter   # noqa: E402

from rubberstudio import theme as T                     # noqa: E402
from rubberstudio.widgets.icons import draw_icon        # noqa: E402

SIZES = (16, 24, 32, 48, 64, 128, 256)


def render(side: int) -> Image.Image:
    img = QImage(side, side, QImage.Format.Format_ARGB32)
    img.fill(QColor(T.RED))
    p = QPainter(img)
    pad = side * 0.2
    draw_icon(p, "erase", QRectF(pad, pad, side - 2 * pad, side - 2 * pad), QColor(T.PAPER), 2.0 + side / 48.0)
    p.end()
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA")


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication([])
    big = render(512)
    big.save(ROOT / "assets" / "icon.png")
    frames = [render(s) for s in SIZES]
    frames[-1].save(ROOT / "assets" / "icon.ico", format="ICO", sizes=[(s, s) for s in SIZES], append_images=frames[:-1])
    print("icon written")
    del app


if __name__ == "__main__":
    main()
