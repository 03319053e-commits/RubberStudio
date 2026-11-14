"""Design tokens — the single place to retune the look (colour-clash paper / film, pop-art halftone).

Everything visual reads from here, so matching the 风洞 palette exactly is a matter of editing these values.
"""
from __future__ import annotations

import sys
from functools import lru_cache

from PySide6.QtGui import QColor, QFont, QFontDatabase

from .core.paths import resource_path


def qc(hex_or_rgb, alpha: float = 1.0) -> QColor:
    c = QColor(hex_or_rgb) if isinstance(hex_or_rgb, str) else QColor(*hex_or_rgb)
    c.setAlphaF(alpha)
    return c


# --- paper & ink ------------------------------------------------------------------------------
PAPER = "#F4EEDF"        # warm cream, the page
PAPER_HI = "#FBF7EC"     # panels / fields
PAPER_LO = "#E7DDC6"     # the desk the picture lies on
INK = "#1C1A17"          # outlines, text, hard shadows
INK_SOFT = "#5A544A"     # secondary text
INK_FAINT = "#8C8575"    # disabled / hints

# --- clashing process inks --------------------------------------------------------------------
RED = "#F0452A"          # vermilion — also the eraser sleeve of the app icon
YELLOW = "#FFC52E"
BLUE = "#2E57E8"
GREEN = "#22B58B"
PINK = "#FF8DBA"

# every tool owns one colour
TOOL_COLORS = {
    "crop": YELLOW,
    "repair": GREEN,
    "watermark": PINK,
    "erase": RED,
    "batch": BLUE,
}

BORDER = 2.5             # ink outline width
SHADOW = 5               # hard offset shadow
MISREG = 2               # colour layer slips this many px against the ink (print mis-registration)
RADIUS = 11
DOT = 7.0                # halftone pitch (px)

TEXT_ON = {              # readable text colour on each ink
    PAPER: INK, PAPER_HI: INK, PAPER_LO: INK, YELLOW: INK, GREEN: INK, PINK: INK,
    RED: INK, BLUE: PAPER_HI, INK: PAPER_HI,
}


def mix(a: str, b: str, t: float) -> QColor:
    """Linear mix of two colours, t=0 -> a, t=1 -> b."""
    ca, cb = QColor(a), QColor(b)
    return QColor(round(ca.red() + (cb.red() - ca.red()) * t),
                  round(ca.green() + (cb.green() - ca.green()) * t),
                  round(ca.blue() + (cb.blue() - ca.blue()) * t))


# --- fonts ------------------------------------------------------------------------------------
DISPLAY_FAMILY = "Archivo Black"
MONO_FAMILY = "Space Mono"
_CJK_FALLBACK = ["Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
                 "Source Han Sans SC", "WenQuanYi Micro Hei", "SimHei"]
_UI_FAMILIES = ["Segoe UI", "Helvetica Neue", "Noto Sans", "DejaVu Sans", "Arial"] + _CJK_FALLBACK

_loaded = False


def load_fonts() -> None:
    global _loaded
    if _loaded:
        return
    for name in ("ArchivoBlack-Regular.ttf", "SpaceMono-Regular.ttf", "SpaceMono-Bold.ttf"):
        path = resource_path("assets", "fonts", name)
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    _loaded = True


def _font(families: list[str], px: float, weight: QFont.Weight, spacing: float = 0.0) -> QFont:
    f = QFont()
    f.setFamilies(families)
    f.setPixelSize(max(1, int(round(px))))
    f.setWeight(weight)
    f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    if spacing:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return f


@lru_cache(maxsize=64)
def font_display(px: float = 26) -> QFont:
    return _font([DISPLAY_FAMILY] + _CJK_FALLBACK, px, QFont.Weight.Normal)


@lru_cache(maxsize=64)
def font_mono(px: float = 11, bold: bool = True, spacing: float = 0.6) -> QFont:
    return _font([MONO_FAMILY] + _CJK_FALLBACK, px,
                 QFont.Weight.Bold if bold else QFont.Weight.Normal, spacing)


@lru_cache(maxsize=64)
def font_ui(px: float = 12, bold: bool = False) -> QFont:
    return _font(_UI_FAMILIES, px, QFont.Weight.DemiBold if bold else QFont.Weight.Normal)


def _arrow_url() -> str:
    return resource_path("assets", "ui", "arrow-down.png").as_posix()


def stylesheet() -> str:
    """QSS for the stock widgets (fields, scrollbars, menus, tooltips, dialogs)."""
    return f"""
    * {{ color: {INK}; }}
    QMainWindow, QDialog, QMessageBox {{ background: {PAPER}; }}
    QToolTip {{ background: {INK}; color: {PAPER_HI}; border: 2px solid {INK}; padding: 5px 8px; }}
    QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QFontComboBox {{
        background: {PAPER_HI}; border: 2px solid {INK}; border-radius: 7px;
        padding: 4px 8px; selection-background-color: {YELLOW}; selection-color: {INK};
        min-height: 20px;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ background: #FFFDF6; border-color: {BLUE}; }}
    QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{ color: {INK_FAINT}; border-color: {INK_FAINT}; }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox::down-arrow {{ image: url("{_arrow_url()}"); width: 14px; height: 10px; margin-right: 6px; }}
    QComboBox QAbstractItemView {{ background: {PAPER_HI}; border: 2px solid {INK}; selection-background-color: {YELLOW};
        selection-color: {INK}; outline: 0; padding: 2px; }}
    QSpinBox::up-button, QSpinBox::down-button {{ width: 16px; border: none; background: transparent; }}
    QScrollArea {{ background: transparent; border: none; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {INK}; border-radius: 3px; min-height: 36px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {INK}; border-radius: 3px; min-width: 36px; }}
    QMenu {{ background: {PAPER_HI}; border: 2px solid {INK}; padding: 4px; }}
    QMenu::item {{ padding: 6px 22px; border-radius: 4px; }}
    QMenu::item:selected {{ background: {YELLOW}; }}
    QPushButton {{ background: {PAPER_HI}; border: 2px solid {INK}; border-radius: 8px; padding: 6px 16px; font-weight: 600; }}
    QPushButton:hover {{ background: {YELLOW}; }}
    QPushButton:pressed {{ background: {RED}; }}
    QProgressBar {{ border: 2px solid {INK}; border-radius: 7px; background: {PAPER_HI}; text-align: center; height: 16px; }}
    QProgressBar::chunk {{ background: {GREEN}; border-radius: 4px; }}
    QListWidget {{ background: transparent; border: none; outline: 0; }}
    """


def apply_platform_tweaks() -> None:
    """Windows: make the taskbar use the app icon instead of python's."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RubberStudio.App")
        except Exception:
            pass
