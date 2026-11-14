"""Pop-art / colour-clash paper widgets: hard-shadow boxes, halftone dots, mis-registered colour."""
from __future__ import annotations

import math
from typing import Optional, Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QFontMetricsF, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QAbstractButton, QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from .. import theme as T
from ..i18n import Translatable, tr
from .icons import draw_icon
from .textures import halftone_pixmap, paper_tile, tint

# variant -> (fill, text, shadow)
VARIANTS = {
    "paper": (T.PAPER_HI, T.INK, T.INK),
    "ink": (T.INK, T.PAPER_HI, T.RED),
    "red": (T.RED, T.INK, T.INK),
    "yellow": (T.YELLOW, T.INK, T.INK),
    "green": (T.GREEN, T.INK, T.INK),
    "blue": (T.BLUE, T.PAPER_HI, T.INK),
    "pink": (T.PINK, T.INK, T.INK),
}


def rrect(rect: QRectF, r: float) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(rect, r, r)
    return path


def _luma(c: QColor) -> float:
    return 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()


def dot_color(fill: QColor) -> QColor:
    """Ben-Day dots: darker than light inks, lighter than dark inks."""
    if _luma(fill) > 150:
        c = T.mix(fill.name(), T.INK, 0.28)
        c.setAlphaF(0.55)
    elif _luma(fill) > 60:
        c = T.mix(fill.name(), "#FFFFFF", 0.42)
        c.setAlphaF(0.5)
    else:
        c = QColor(T.PAPER)
        c.setAlphaF(0.16)
    return c


def _q16(v: float) -> int:
    return (int(v) // 16 + 1) * 16


def paper_brush(base: str = T.PAPER, seed: int = 1) -> QBrush:
    c = QColor(base)
    return QBrush(paper_tile((c.red(), c.green(), c.blue()), seed))


def draw_dots(p: QPainter, rect: QRectF, color: QColor, *, ramp_angle: float = 45.0, lo: float = 0.0,
              hi: float = 0.9, spacing: float = T.DOT, speckle: float = 0.0) -> None:
    pm = halftone_pixmap(_q16(rect.width()), _q16(rect.height()), tint(color), spacing, 45.0,
                         ramp_angle, lo, hi, 0.62, speckle)
    p.drawPixmap(rect.topLeft(), pm)


def draw_pop_box(p: QPainter, rect: QRectF, *, fill: str = T.PAPER_HI, radius: float = T.RADIUS,
                 shadow: float = T.SHADOW, shadow_color: str = T.INK, border: float = T.BORDER,
                 misreg: float = 0.0, dots: bool = False, dots_hi: float = 0.9, dots_lo: float = 0.0,
                 ramp_angle: float = 45.0, speckle: float = 0.0, border_color: str = T.INK) -> QRectF:
    """Paint the signature box. ``rect`` is the outer edge of the ink outline; the shadow extends beyond it."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    if shadow > 0:
        p.fillPath(rrect(rect.translated(shadow, shadow), radius), QColor(shadow_color))
    inner = rect.adjusted(border / 2, border / 2, -border / 2, -border / 2)
    fc = QColor(fill)
    if misreg:
        p.fillPath(rrect(inner, radius), QColor(T.PAPER_HI))          # paper shows through the slip
    fbox = inner.translated(misreg, misreg) if misreg else inner
    fpath = rrect(fbox, radius)
    p.fillPath(fpath, fc)
    if dots:
        p.save()
        p.setClipPath(fpath)
        draw_dots(p, fbox, dot_color(fc), ramp_angle=ramp_angle, lo=dots_lo, hi=dots_hi, speckle=speckle)
        p.restore()
    pen = QPen(QColor(border_color), border)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(rrect(inner, radius))
    p.restore()
    return inner


# ====================================================================== buttons

class PopButton(QAbstractButton, Translatable):
    """Flat colour block with ink outline and a hard shadow; presses *into* the shadow."""

    def __init__(self, key: str = "", variant: str = "paper", icon: Optional[str] = None,
                 parent: Optional[QWidget] = None, *, text: Optional[str] = None, height: int = 36,
                 shadow: int = 4, px: float = 12, checked_variant: Optional[str] = None):
        super().__init__(parent)
        self.init_key(key)
        self._literal = text
        self.variant = variant
        self.checked_variant = checked_variant or variant
        self.icon_name = icon
        self._shadow = shadow
        self._px = px
        self._hover = False
        self.setFixedHeight(height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.retranslate()

    def retranslate(self) -> None:
        self.setText(self._literal if self._literal is not None else self.text_now())
        self.updateGeometry()
        self.update()

    def set_variant(self, variant: str) -> None:
        self.variant = variant
        self.update()

    def sizeHint(self) -> QSize:
        fm = QFontMetrics(T.font_mono(self._px, True))
        w = fm.horizontalAdvance(self.text()) + 30 + self._shadow + (24 if self.icon_name else 0)
        return QSize(max(w, 44), self.height())

    def minimumSizeHint(self) -> QSize:          # lets a row of buttons shrink; the label is elided when it no longer fits
        return QSize(40 + (24 if self.icon_name else 0), self.height())

    def enterEvent(self, e) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, e) -> None:
        self._hover = False
        self.update()

    def _look(self):
        v = self.checked_variant if self.isChecked() else self.variant
        if self.isCheckable() and not self.isChecked():
            v = self.variant
        return VARIANTS.get(v, VARIANTS["paper"])

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        enabled = self.isEnabled()
        fill, text_col, shadow_col = self._look()
        sh = self._shadow
        pressed = self.isDown() or (self.isCheckable() and self.isChecked())
        lift = 1 if (self._hover and enabled and not pressed) else 0
        if not enabled:
            fill, text_col, shadow_col = T.PAPER_LO, T.INK_FAINT, T.INK_FAINT
        base = QRectF(1, 1, self.width() - sh - 2, self.height() - sh - 2)
        if pressed:
            rect, shadow = base.translated(sh - 1, sh - 1), 0
        else:
            rect, shadow = base.translated(-lift, -lift), sh + lift
        draw_pop_box(p, rect, fill=fill, radius=9, shadow=shadow, shadow_color=shadow_col,
                     border=2.5 if enabled else 2.0, misreg=1.5 if enabled else 0, dots=enabled,
                     dots_hi=0.85, dots_lo=0.0, ramp_angle=60, speckle=0.003 if enabled else 0,
                     border_color=T.INK if enabled else T.INK_FAINT)
        self._paint_content(p, rect, QColor(text_col))

    def _paint_content(self, p: QPainter, rect: QRectF, col: QColor) -> None:
        p.setFont(T.font_mono(self._px, True))
        p.setPen(col)
        fm = p.fontMetrics()
        iw = 18 if self.icon_name else 0
        gap = 7 if self.icon_name and self.text() else 0
        text = fm.elidedText(self.text(), Qt.TextElideMode.ElideRight, max(8, int(rect.width() - 14 - iw - gap)))
        tw = fm.horizontalAdvance(text)
        total = tw + iw + gap
        x = rect.center().x() - total / 2
        if self.icon_name:
            draw_icon(p, self.icon_name, QRectF(x, rect.center().y() - iw / 2, iw, iw), col, 1.9)
            x += iw + gap
        if text:
            p.drawText(QRectF(x, rect.top(), tw + 2, rect.height()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       text)


class ToolButton(PopButton):
    """Wide tool selector: colour chip with icon + label; fills with the tool colour when active."""

    def __init__(self, key: str, color: str, icon: str, parent: Optional[QWidget] = None):
        super().__init__(key, "paper", icon, parent, height=46, shadow=4, px=12)
        self.color = color
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _look(self):
        if self.isChecked():
            return (self.color, T.INK, T.INK)
        return VARIANTS["paper"]

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        sh = self._shadow
        pressed = self.isChecked() or self.isDown()
        lift = 1 if (self._hover and not pressed and self.isEnabled()) else 0
        base = QRectF(1, 1, self.width() - sh - 2, self.height() - sh - 2)
        rect, shadow = (base.translated(sh - 1, sh - 1), 0) if pressed else (base.translated(-lift, -lift), sh + lift)
        fill = self.color if self.isChecked() else T.PAPER_HI
        draw_pop_box(p, rect, fill=fill, radius=10, shadow=shadow, border=2.5, misreg=1.5 if self.isChecked() else 0,
                     dots=self.isChecked(), dots_hi=0.9, ramp_angle=40, speckle=0.003)
        chip = QRectF(rect.left() + 8, rect.center().y() - 15, 30, 30)
        draw_pop_box(p, chip, fill=self.color if not self.isChecked() else T.PAPER_HI, radius=7, shadow=0,
                     border=2.2, misreg=1.2, dots=not self.isChecked(), dots_hi=0.7, ramp_angle=60)
        draw_icon(p, self.icon_name or "", chip.adjusted(5, 5, -5, -5), QColor(T.INK), 1.9)
        p.setFont(T.font_mono(self._px, True))
        p.setPen(QColor(T.INK))
        p.drawText(QRectF(chip.right() + 11, rect.top(), rect.width() - 56, rect.height()),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())


class PopCheck(QAbstractButton, Translatable):
    toggledBool = Signal(bool)

    def __init__(self, key: str = "", checked: bool = False, parent: Optional[QWidget] = None,
                 accent: str = T.BLUE, text: Optional[str] = None):
        super().__init__(parent)
        self.init_key(key)
        self._literal = text
        self.accent = accent
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.toggled.connect(self.toggledBool)
        self.retranslate()

    def retranslate(self) -> None:
        self.setText(self._literal if self._literal is not None else self.text_now())
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        fm = QFontMetrics(T.font_ui(12, True))
        return QSize(fm.horizontalAdvance(self.text()) + 34, 26)

    def minimumSizeHint(self) -> QSize:
        return QSize(60, 26)

    def hasHeightForWidth(self) -> bool:
        return False

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        on = self.isEnabled()
        box = QRectF(2, (self.height() - 20) / 2, 20, 20)
        draw_pop_box(p, box, fill=self.accent if self.isChecked() else T.PAPER_HI, radius=5, shadow=2 if on else 0,
                     border=2.2, dots=self.isChecked(), dots_hi=0.7,
                     border_color=T.INK if on else T.INK_FAINT)
        if self.isChecked():
            draw_icon(p, "check", box.adjusted(2, 2, -2, -2),
                      QColor(T.PAPER_HI if self.accent in (T.BLUE,) else T.INK), 2.6)
        f = T.font_ui(12, True)
        p.setFont(f)
        p.setPen(QColor(T.INK if on else T.INK_FAINT))
        label = QFontMetrics(f).elidedText(self.text(), Qt.TextElideMode.ElideRight, max(10, self.width() - 34))
        self.setToolTip(self.text() if label != self.text() else "")
        p.drawText(QRectF(32, 0, self.width() - 32, self.height()),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)


# ====================================================================== sliders / segments

class PopSlider(QWidget, Translatable):
    valueChanged = Signal(float)

    def __init__(self, key: str, minimum: float, maximum: float, value: float, *, step: float = 0.01,
                 decimals: int = 2, color: str = T.BLUE, parent: Optional[QWidget] = None,
                 suffix: str = ""):
        super().__init__(parent)
        self.init_key(key)
        self.min, self.max, self.step, self.decimals, self.color, self.suffix = minimum, maximum, step, decimals, color, suffix
        self._value = float(value)
        self._drag = False
        self.setFixedHeight(44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def retranslate(self) -> None:
        self.update()

    def value(self) -> float:
        return self._value

    def setValue(self, v: float, emit: bool = False) -> None:
        v = max(self.min, min(self.max, v))
        if self.step > 0:
            v = round((v - self.min) / self.step) * self.step + self.min
            v = max(self.min, min(self.max, v))
        if abs(v - self._value) > 1e-9:
            self._value = v
            self.update()
            if emit:
                self.valueChanged.emit(v)

    def setRange(self, lo: float, hi: float) -> None:
        self.min, self.max = lo, hi
        self.setValue(self._value)

    def _track(self) -> QRectF:
        return QRectF(11, 25, self.width() - 11 - 13, 10)

    def _pos(self, v: float) -> float:
        t = self._track()
        frac = (v - self.min) / max(self.max - self.min, 1e-9)
        return t.left() + t.width() * frac

    def _from_x(self, x: float) -> float:
        t = self._track()
        frac = (x - t.left()) / max(t.width(), 1)
        return self.min + max(0.0, min(1.0, frac)) * (self.max - self.min)

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = True
            self.setValue(self._from_x(e.position().x()), emit=True)

    def mouseMoveEvent(self, e) -> None:
        if self._drag:
            self.setValue(self._from_x(e.position().x()), emit=True)

    def mouseReleaseEvent(self, e) -> None:
        self._drag = False

    def wheelEvent(self, e) -> None:
        d = 1 if e.angleDelta().y() > 0 else -1
        self.setValue(self._value + d * max(self.step, (self.max - self.min) / 40.0), emit=True)
        e.accept()

    def keyPressEvent(self, e) -> None:
        k = e.key()
        if k in (Qt.Key.Key_Left, Qt.Key.Key_Down):
            self.setValue(self._value - max(self.step, (self.max - self.min) / 40.0), emit=True)
        elif k in (Qt.Key.Key_Right, Qt.Key.Key_Up):
            self.setValue(self._value + max(self.step, (self.max - self.min) / 40.0), emit=True)
        else:
            super().keyPressEvent(e)

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        on = self.isEnabled()
        p.setFont(T.font_mono(10.5, True))
        p.setPen(QColor(T.INK if on else T.INK_FAINT))
        p.drawText(QRectF(2, 0, self.width() - 60, 20), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   self.text_now())
        p.setFont(T.font_mono(10.5, False))
        txt = f"{self._value:.{self.decimals}f}{self.suffix}"
        p.drawText(QRectF(self.width() - 64, 0, 62, 20), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, txt)

        t = self._track()
        draw_pop_box(p, t, fill=T.PAPER_HI, radius=5, shadow=0, border=2.2,
                     border_color=T.INK if on else T.INK_FAINT)
        hx = self._pos(self._value)
        if hx > t.left() + 4:
            fillr = QRectF(t.left() + 1.4, t.top() + 1.4, hx - t.left() - 1.4, t.height() - 2.8)
            p.save()
            p.setClipPath(rrect(t.adjusted(1.4, 1.4, -1.4, -1.4), 3.6))
            p.fillRect(fillr, QColor(self.color if on else T.INK_FAINT))
            draw_dots(p, fillr, dot_color(QColor(self.color)), ramp_angle=0, lo=0.2, hi=0.9, spacing=5)
            p.restore()
        knob = QRectF(hx - 10, t.center().y() - 10, 20, 20)
        draw_pop_box(p, knob, fill=T.PAPER_HI, radius=10, shadow=2 if on else 0, border=2.4, misreg=0,
                     border_color=T.INK if on else T.INK_FAINT)
        if self.hasFocus():
            pen = QPen(QColor(T.BLUE), 1.6, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(QRectF(0.8, 0.8, self.width() - 1.6, self.height() - 1.6), 6, 6)


class PopSegmented(QWidget):
    """Equal-width segments; the active one is filled with ``color`` and dotted."""
    currentChanged = Signal(str)

    def __init__(self, items: Sequence[tuple[str, str]], current: str, color: str = T.GREEN,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.items = list(items)               # (id, translation key)
        self.color = color
        self._current = current
        self._hover = -1
        self.setFixedHeight(40)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def retranslate(self) -> None:
        self.update()

    def current(self) -> str:
        return self._current

    def setCurrent(self, ident: str, emit: bool = False) -> None:
        if ident != self._current:
            self._current = ident
            self.update()
            if emit:
                self.currentChanged.emit(ident)

    def _index_at(self, x: float) -> int:
        w = (self.width() - 6) / max(len(self.items), 1)
        return max(0, min(len(self.items) - 1, int(x // w)))

    def mouseMoveEvent(self, e) -> None:
        i = self._index_at(e.position().x())
        if i != self._hover:
            self._hover = i
            self.update()

    def leaveEvent(self, e) -> None:
        self._hover = -1
        self.update()

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton and self.items:
            self.setCurrent(self.items[self._index_at(e.position().x())][0], emit=True)

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        outer = QRectF(1, 1, self.width() - 5, self.height() - 5)
        draw_pop_box(p, outer, fill=T.PAPER_HI, radius=9, shadow=3, border=2.5)
        n = max(len(self.items), 1)
        seg_w = outer.width() / n
        p.setFont(T.font_mono(11, True))
        for i, (ident, key) in enumerate(self.items):
            r = QRectF(outer.left() + seg_w * i, outer.top(), seg_w, outer.height())
            if ident == self._current:
                inner = r.adjusted(3, 3, -3, -3)
                draw_pop_box(p, inner, fill=self.color, radius=6, shadow=0, border=2.0, misreg=1.2,
                             dots=True, dots_hi=0.9, ramp_angle=50, speckle=0.003)
            elif i == self._hover:
                p.fillPath(rrect(r.adjusted(4, 4, -4, -4), 5), QColor(T.PAPER_LO))
            if 0 < i < n and ident != self._current and self.items[i - 1][0] != self._current:
                p.setPen(QPen(QColor(T.INK), 1.4))
                p.drawLine(QPointF(r.left(), r.top() + 9), QPointF(r.left(), r.bottom() - 9))
            p.setPen(QColor(T.INK))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, tr(key))


# ====================================================================== labels / containers

class SectionLabel(QLabel, Translatable):
    def __init__(self, key: str = "", parent: Optional[QWidget] = None, *, text: Optional[str] = None):
        super().__init__(parent)
        self.init_key(key)
        self._literal = text
        self.setFont(T.font_mono(10.5, True, 1.0))
        self.setStyleSheet(f"color: {T.INK_SOFT}; background: transparent;")
        self.retranslate()

    def retranslate(self) -> None:
        self.setText(self._literal if self._literal is not None else self.text_now())


class HintLabel(QLabel, Translatable):
    def __init__(self, key: str = "", parent: Optional[QWidget] = None, *, text: Optional[str] = None,
                 px: float = 11.5, **fmt):
        super().__init__(parent)
        self.init_key(key, **fmt)
        self._literal = text
        self.setFont(T.font_ui(px))
        self.setWordWrap(True)
        self.setStyleSheet(f"color: {T.INK_SOFT}; background: transparent;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.retranslate()

    def retranslate(self) -> None:
        self.setText(self._literal if self._literal is not None else self.text_now())


class Chip(QLabel):
    """Small status pill."""

    def __init__(self, text: str = "", color: str = T.PAPER_HI, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.color = color
        self._text = text
        self.setFixedHeight(20)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_chip(self, text: str, color: str) -> None:
        self._text, self.color = text, color
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        fm = QFontMetrics(T.font_mono(9.5, True))
        return QSize(fm.horizontalAdvance(self._text) + 18, 20)

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        r = QRectF(1, 1, self.width() - 2, self.height() - 2)
        draw_pop_box(p, r, fill=self.color, radius=10, shadow=0, border=1.8, dots=True, dots_hi=0.8, ramp_angle=30)
        p.setFont(T.font_mono(9.5, True))
        p.setPen(QColor(T.TEXT_ON.get(self.color, T.INK)))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, self._text)


class PopPanel(QFrame, Translatable):
    """Paper card with a coloured, dotted title tab. Put content into ``self.body``."""

    def __init__(self, key: str = "", accent: str = T.YELLOW, parent: Optional[QWidget] = None, *,
                 text: Optional[str] = None, tab: bool = True):
        super().__init__(parent)
        self.init_key(key)
        self._literal = text
        self.accent = accent
        self._tab = tab
        self._title_h = 26 if tab else 0
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(12, self._title_h + 10, 12 + 4, 12 + 4)
        self.body.setSpacing(10)
        self.retranslate()

    def retranslate(self) -> None:
        self._title = self._literal if self._literal is not None else self.text_now()
        self.update()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        sh = 4
        outer = QRectF(1, 1, self.width() - sh - 2, self.height() - sh - 2)
        draw_pop_box(p, outer, fill=T.PAPER_HI, radius=10, shadow=sh, border=2.5)
        if self._tab:
            tab = QRectF(outer.left() + 2.6, outer.top() + 2.6, outer.width() - 5.2, self._title_h - 2)
            path = QPainterPath()
            path.addRoundedRect(tab, 7, 7)
            p.save()
            p.setClipPath(path)
            p.fillRect(tab.translated(1.5, 1.5), QColor(self.accent))
            draw_dots(p, tab, dot_color(QColor(self.accent)), ramp_angle=0, lo=0.1, hi=0.95, spacing=6, speckle=0.004)
            p.restore()
            p.setPen(QPen(QColor(T.INK), 2.2))
            p.drawLine(QPointF(tab.left(), tab.bottom()), QPointF(tab.right(), tab.bottom()))
            p.setFont(T.font_mono(10.5, True, 1.2))
            p.setPen(QColor(T.TEXT_ON.get(self.accent, T.INK)))
            p.drawText(tab.adjusted(10, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._title)


class PopProgress(QWidget):
    """Thick progress bar with a dotted fill."""

    def __init__(self, color: str = T.RED, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.color = color
        self._v = 0.0
        self._text = ""
        self.setFixedHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_progress(self, v: float, text: str = "") -> None:
        self._v, self._text = max(0.0, min(1.0, v)), text
        self.update()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(1.5, 1.5, self.width() - 3, self.height() - 3)
        draw_pop_box(p, r, fill=T.PAPER_HI, radius=8, shadow=0, border=2.4)
        if self._v > 0.001:
            inner = r.adjusted(2.4, 2.4, -2.4, -2.4)
            fr = QRectF(inner.left(), inner.top(), max(8.0, inner.width() * self._v), inner.height())
            p.save()
            p.setClipPath(rrect(inner, 5))
            p.fillRect(fr, QColor(self.color))
            draw_dots(p, fr, dot_color(QColor(self.color)), ramp_angle=0, lo=0.3, hi=1.0, spacing=5)
            p.setPen(QPen(QColor(T.INK), 2.0))
            p.drawLine(QPointF(fr.right(), fr.top()), QPointF(fr.right(), fr.bottom()))
            p.restore()
        if self._text:
            # ink text with a paper halo: readable on the dotted fill, on the empty track and across the divider line
            f = T.font_mono(10.5, True)
            fm = QFontMetricsF(f)
            x = r.center().x() - fm.horizontalAdvance(self._text) / 2.0
            y = r.center().y() + (fm.ascent() - fm.descent()) / 2.0
            path = QPainterPath()
            path.addText(QPointF(x, y), f, self._text)
            p.setPen(QPen(QColor(T.PAPER_HI), 3.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                          Qt.PenJoinStyle.RoundJoin))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(T.INK))
            p.drawPath(path)
