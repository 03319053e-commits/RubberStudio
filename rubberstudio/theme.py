"""Design tokens — the single place to retune the look.

The look follows the NASA Graphics Standards Manual of 1975 (Danne & Blackburn): a white page, black type and rules,
exactly one red, a Helvetica-style grotesque, strict rectangles, bold tracked capitals for labels. No gradients, no
shadows, no texture, no rounded corners. Red means "this one": the active tool, the primary action, progress.
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


# --- black, white, one red ----------------------------------------------------------------------
PAPER = "#FFFFFF"        # the page
PAPER_HI = "#FFFFFF"     # panels / fields (the manual has no tinted boxes)
PAPER_LO = "#E8E8E8"     # the desk the picture lies on, hover tint
INK = "#000000"          # type, rules, outlines
INK_SOFT = "#4A4A4A"     # secondary text (8.9 : 1 on white)
INK_FAINT = "#8F8F8F"    # disabled text and rules only
RED = "#DA291C"          # white on it: 4.9 : 1 — AA for normal text
RED_DARK = "#A81E14"     # pressed

BORDER = 2.0             # control outline
RULE = 4.0               # the heavy rule above a panel title
HAIR = 1.0               # hairline

TEXT_ON = {              # readable text colour on each ink
    PAPER: INK, PAPER_LO: INK, INK: PAPER, RED: PAPER, RED_DARK: PAPER,
}


def mix(a: str, b: str, t: float) -> QColor:
    """Linear mix of two colours, t=0 -> a, t=1 -> b."""
    ca, cb = QColor(a), QColor(b)
    return QColor(round(ca.red() + (cb.red() - ca.red()) * t),
                  round(ca.green() + (cb.green() - ca.green()) * t),
                  round(ca.blue() + (cb.blue() - ca.blue()) * t))


# --- fonts: Liberation Sans (Arial-compatible, bundled) everywhere ---------------------------------
SANS_FAMILY = "Liberation Sans"
_CJK_FALLBACK = ["Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
                 "Source Han Sans SC", "WenQuanYi Micro Hei", "SimHei"]
_FAMILIES = [SANS_FAMILY, "Helvetica Neue", "Helvetica", "Arial"] + _CJK_FALLBACK

_loaded = False


def load_fonts() -> None:
    global _loaded
    if _loaded:
        return
    for name in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"):
        path = resource_path("assets", "fonts", name)
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    _loaded = True


def _font(px: float, bold: bool, spacing: float = 0.0) -> QFont:
    f = QFont()
    f.setFamilies(_FAMILIES)
    f.setPixelSize(max(1, int(round(px))))
    f.setWeight(QFont.Weight.Bold if bold else QFont.Weight.Normal)
    f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    if spacing:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return f


@lru_cache(maxsize=64)
def font_display(px: float = 26) -> QFont:
    """Big headlines: bold, a little tight."""
    return _font(px, True, -0.02 * px)


@lru_cache(maxsize=64)
def font_label(px: float = 11, bold: bool = True, spacing: float = 1.0) -> QFont:
    """Capitals with air between the letters: buttons, section labels, figures."""
    return _font(px, bold, spacing)


@lru_cache(maxsize=64)
def font_ui(px: float = 12, bold: bool = False) -> QFont:
    return _font(px, bold)


def _arrow_url() -> str:
    return resource_path("assets", "ui", "arrow-down.png").as_posix()


def stylesheet() -> str:
    """QSS for the stock widgets (fields, scrollbars, menus, tooltips, dialogs)."""
    return f"""
    * {{ color: {INK}; }}
    QMainWindow, QDialog, QMessageBox {{ background: {PAPER}; }}
    QToolTip {{ background: {INK}; color: {PAPER}; border: 0; padding: 5px 8px; }}
    QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QFontComboBox {{
        background: {PAPER_HI}; border: 2px solid {INK}; border-radius: 0;
        padding: 3px 8px; selection-background-color: {RED}; selection-color: {PAPER};
        min-height: 20px; placeholder-text-color: {INK_SOFT};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {RED}; }}
    QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{ color: {INK_FAINT}; border-color: {INK_FAINT}; }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox::down-arrow {{ image: url("{_arrow_url()}"); width: 14px; height: 10px; margin-right: 6px; }}
    QComboBox QAbstractItemView {{ background: {PAPER_HI}; border: 2px solid {INK}; selection-background-color: {INK};
        selection-color: {PAPER}; outline: 0; padding: 0; }}
    QSpinBox::up-button, QSpinBox::down-button {{ width: 16px; border: none; background: transparent; }}
    QScrollArea {{ background: transparent; border: none; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {INK}; border-radius: 0; min-height: 36px; margin: 0 3px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: {INK}; border-radius: 0; min-width: 36px; margin: 3px 0; }}
    QMenu {{ background: {PAPER_HI}; border: 2px solid {INK}; padding: 0; }}
    QMenu::item {{ padding: 6px 22px; }}
    QMenu::item:selected {{ background: {INK}; color: {PAPER}; }}
    QPushButton {{ background: {PAPER_HI}; border: 2px solid {INK}; border-radius: 0; padding: 6px 16px; font-weight: bold; }}
    QPushButton:hover {{ background: {PAPER_LO}; }}
    QPushButton:pressed {{ background: {INK}; color: {PAPER}; }}
    QProgressBar {{ border: 2px solid {INK}; border-radius: 0; background: {PAPER_HI}; text-align: center; height: 16px; }}
    QProgressBar::chunk {{ background: {RED}; }}
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
