"""Flat widgets in the style of the 1975 NASA Graphics Standards Manual: white fields, black rules, red for the one
active thing. Rectangles only — no radius, no shadow, no texture."""
from __future__ import annotations

from typing import Optional, Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QFontMetricsF, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QAbstractButton, QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from .. import theme as T
from ..i18n import Translatable, tr
from .icons import draw_icon

# variant -> (fill, text, border, hover fill)
VARIANTS = {
    "paper": (T.PAPER_HI, T.INK, T.INK, T.PAPER_LO),
    "ink": (T.INK, T.PAPER, T.INK, "#2B2B2B"),
    "red": (T.RED, T.PAPER, T.RED, T.RED_DARK),
}
# what a pressed / checked button turns into
_PRESSED = {"paper": "ink", "ink": "red", "red": "ink"}


def draw_box(p: QPainter, rect: QRectF, *, fill: str = T.PAPER_HI, border: float = T.BORDER,
             border_color: str = T.INK) -> QRectF:
    """A plain rectangle; ``rect`` is the outer edge of the outline. Returns the area inside the outline."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    if border > 0:
        p.fillRect(rect, QColor(border_color))
    inner = rect.adjusted(border, border, -border, -border)
    p.fillRect(inner, QColor(fill))
    p.restore()
    return inner


def _focus_ring(p: QPainter, rect: QRectF, color: str, inset: float = 4.0) -> None:
    """Dashed keyboard-focus rectangle, ``inset`` px inside ``rect``."""
    pen = QPen(QColor(color), 1.0, Qt.PenStyle.DashLine)
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(rect.adjusted(inset, inset, -inset - 0.5, -inset - 0.5))
    p.restore()


# ====================================================================== buttons

class PopButton(QAbstractButton, Translatable):
    """Outlined rectangle with a bold capital label; inverts while pressed."""

    def __init__(self, key: str = "", variant: str = "paper", icon: Optional[str] = None,
                 parent: Optional[QWidget] = None, *, text: Optional[str] = None, height: int = 36,
                 px: float = 11.5, checked_variant: Optional[str] = None):
        super().__init__(parent)
        self.init_key(key)
        self._literal = text
        self.variant = variant
        self.checked_variant = checked_variant
        self.icon_name = icon
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
        fm = QFontMetrics(T.font_label(self._px, True))
        w = fm.horizontalAdvance(self.text()) + 30 + (24 if self.icon_name else 0)
        return QSize(max(w, 44), self.height())

    def minimumSizeHint(self) -> QSize:          # lets a row of buttons shrink; the label is elided when it no longer fits
        return QSize(40 + (24 if self.icon_name else 0), self.height())

    def enterEvent(self, e) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, e) -> None:
        self._hover = False
        self.update()

    def _look(self) -> tuple[str, str, str, int]:
        """(fill, text, border, border width) for the current state."""
        if not self.isEnabled():
            return T.PAPER_HI, T.INK_FAINT, T.INK_FAINT, 1
        name = self.variant
        pressed = self.isDown() or (self.isCheckable() and self.isChecked())
        if pressed:
            name = self.checked_variant or _PRESSED.get(name, "ink")
            fill, text, border, _ = VARIANTS[name]
        else:
            fill, text, border, hover = VARIANTS.get(name, VARIANTS["paper"])
            if self._hover:
                fill = hover
        return fill, text, border, 2

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        fill, text_col, border, bw = self._look()
        rect = QRectF(self.rect())
        draw_box(p, rect, fill=fill, border=bw, border_color=border)
        self._paint_content(p, rect, QColor(text_col))
        if self.hasFocus():
            _focus_ring(p, rect, T.PAPER if fill in (T.INK, T.RED, T.RED_DARK) else T.RED)

    def _paint_content(self, p: QPainter, rect: QRectF, col: QColor) -> None:
        p.setFont(T.font_label(self._px, True))
        p.setPen(col)
        fm = p.fontMetrics()
        iw = 18 if self.icon_name else 0
        gap = 8 if self.icon_name and self.text() else 0
        text = fm.elidedText(self.text(), Qt.TextElideMode.ElideRight, max(8, int(rect.width() - 14 - iw - gap)))
        tw = fm.horizontalAdvance(text)
        total = tw + iw + gap
        x = rect.center().x() - total / 2
        if self.icon_name:
            draw_icon(p, self.icon_name, QRectF(x, rect.center().y() - iw / 2, iw, iw), col, 1.9)
            x += iw + gap
        if text:
            p.drawText(QRectF(x, rect.top(), tw + 2, rect.height()),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)


class ToolButton(PopButton):
    """Wide tool selector: a black square with the icon, then the name; the active tool is a red bar."""

    def __init__(self, key: str, icon: str, parent: Optional[QWidget] = None):
        super().__init__(key, "paper", icon, parent, height=44, px=12)
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        on, enabled = self.isChecked(), self.isEnabled()
        rect = QRectF(self.rect())
        if not enabled:
            fill, edge, text_col, chip = T.PAPER_HI, T.INK_FAINT, T.INK_FAINT, T.INK_FAINT
        elif on:
            fill, edge, text_col, chip = T.RED, T.RED, T.PAPER, T.INK
        else:
            fill, edge, text_col, chip = (T.PAPER_LO if self._hover else T.PAPER_HI), T.INK, T.INK, T.INK
        draw_box(p, rect, fill=fill, border=2 if enabled else 1, border_color=edge)
        side = rect.height()
        p.fillRect(QRectF(rect.left(), rect.top(), side, side), QColor(chip))
        draw_icon(p, self.icon_name or "", QRectF(rect.left(), rect.top(), side, side).adjusted(11, 11, -11, -11),
                  QColor(T.PAPER), 1.9)
        p.setFont(T.font_label(self._px, True))
        p.setPen(QColor(text_col))
        p.drawText(QRectF(rect.left() + side + 12, rect.top(), rect.width() - side - 20, rect.height()),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())
        if self.hasFocus():
            _focus_ring(p, QRectF(rect.left() + side, rect.top(), rect.width() - side, rect.height()).adjusted(-3, 0, 0, 0),
                        T.PAPER if on else T.RED)


class PopCheck(QAbstractButton, Translatable):
    toggledBool = Signal(bool)

    def __init__(self, key: str = "", checked: bool = False, parent: Optional[QWidget] = None, *,
                 text: Optional[str] = None):
        super().__init__(parent)
        self.init_key(key)
        self._literal = text
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
        on = self.isEnabled()
        box = QRectF(1, (self.height() - 18) / 2, 18, 18)
        if self.isChecked():
            fill, edge = (T.RED, T.RED) if on else (T.INK_FAINT, T.INK_FAINT)
        else:
            fill, edge = T.PAPER_HI, (T.INK if on else T.INK_FAINT)
        draw_box(p, box, fill=fill, border=2 if on else 1, border_color=edge)
        if self.isChecked():
            draw_icon(p, "check", box.adjusted(3, 3, -3, -3), QColor(T.PAPER), 2.6)
        f = T.font_ui(12, True)
        p.setFont(f)
        p.setPen(QColor(T.INK if on else T.INK_FAINT))
        label = QFontMetrics(f).elidedText(self.text(), Qt.TextElideMode.ElideRight, max(10, self.width() - 32))
        self.setToolTip(self.text() if label != self.text() else "")
        text_rect = QRectF(30, 0, self.width() - 30, self.height())
        p.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
        if self.hasFocus():
            _focus_ring(p, QRectF(box).adjusted(-4, -4, 4.5, 4.5), T.RED)


# ====================================================================== sliders / segments

class PopSlider(QWidget, Translatable):
    valueChanged = Signal(float)

    def __init__(self, key: str, minimum: float, maximum: float, value: float, *, step: float = 0.01,
                 decimals: int = 2, parent: Optional[QWidget] = None, suffix: str = ""):
        super().__init__(parent)
        self.init_key(key)
        self.min, self.max, self.step, self.decimals, self.suffix = minimum, maximum, step, decimals, suffix
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
        return QRectF(7, 24, self.width() - 14, 14)

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
            self.update()

    def mouseMoveEvent(self, e) -> None:
        if self._drag:
            self.setValue(self._from_x(e.position().x()), emit=True)

    def mouseReleaseEvent(self, e) -> None:
        self._drag = False
        self.update()

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
        on = self.isEnabled()
        ink = QColor(T.INK if on else T.INK_FAINT)
        p.setFont(T.font_label(10.5, True, 1.0))
        p.setPen(ink)
        p.drawText(QRectF(0, 0, self.width() - 64, 20), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   self.text_now())
        p.setFont(T.font_ui(11.5, True))
        txt = f"{self._value:.{self.decimals}f}{self.suffix}"
        p.drawText(QRectF(self.width() - 64, 0, 64, 20), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, txt)

        t = self._track()
        cy = t.center().y()
        hx = self._pos(self._value)
        p.fillRect(QRectF(t.left(), cy - 1, t.width(), 2), ink)                         # the rule
        if hx > t.left():
            p.fillRect(QRectF(t.left(), cy - 2.5, hx - t.left(), 5), QColor(T.RED if on else T.INK_FAINT))
        knob = QRectF(round(hx) - 4, cy - 8, 8, 16)
        active = on and (self._drag or self.hasFocus())
        p.fillRect(knob, QColor(T.RED if active else ink))


class PopSegmented(QWidget):
    """Equal-width cells in one outline; the chosen one is black."""
    currentChanged = Signal(str)

    def __init__(self, items: Sequence[tuple[str, str]], current: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.items = list(items)               # (id, translation key)
        self._current = current
        self._hover = -1
        self.setFixedHeight(38)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def retranslate(self) -> None:
        self.update()

    def current(self) -> str:
        return self._current

    def keyPressEvent(self, e) -> None:
        step = {Qt.Key.Key_Left: -1, Qt.Key.Key_Up: -1, Qt.Key.Key_Right: 1, Qt.Key.Key_Down: 1}.get(e.key(), 0)
        if not step or not self.items:
            super().keyPressEvent(e)
            return
        ids = [i for i, _ in self.items]
        idx = ids.index(self._current) if self._current in ids else 0
        self.setCurrent(ids[max(0, min(len(ids) - 1, idx + step))], emit=True)

    def setCurrent(self, ident: str, emit: bool = False) -> None:
        if ident != self._current:
            self._current = ident
            self.update()
            if emit:
                self.currentChanged.emit(ident)

    def _index_at(self, x: float) -> int:
        w = self.width() / max(len(self.items), 1)
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
        outer = QRectF(self.rect())
        enabled = self.isEnabled()
        edge = T.INK if enabled else T.INK_FAINT
        draw_box(p, outer, fill=T.PAPER_HI, border=2, border_color=edge)
        n = max(len(self.items), 1)
        seg_w = outer.width() / n
        p.setFont(T.font_label(10.5, True, 0.8))
        for i, (ident, key) in enumerate(self.items):
            r = QRectF(outer.left() + seg_w * i, outer.top(), seg_w, outer.height())
            chosen = ident == self._current
            if chosen:
                p.fillRect(r.adjusted(2 if i == 0 else 0, 2, -2 if i == n - 1 else 0, -2), QColor(edge))
            elif i == self._hover and enabled:
                p.fillRect(r.adjusted(2 if i == 0 else 0, 2, -2 if i == n - 1 else 0, -2), QColor(T.PAPER_LO))
            if i > 0:
                p.fillRect(QRectF(round(r.left()) - 1, outer.top() + 2, 2, outer.height() - 4), QColor(edge))
            p.setPen(QColor(T.PAPER if chosen else (T.INK if enabled else T.INK_FAINT)))
            label = p.fontMetrics().elidedText(tr(key), Qt.TextElideMode.ElideRight, max(8, int(seg_w - 8)))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, label)
        if self.hasFocus():
            _focus_ring(p, outer, T.RED, inset=3.0)


# ====================================================================== labels / containers

class SectionLabel(QLabel, Translatable):
    def __init__(self, key: str = "", parent: Optional[QWidget] = None, *, text: Optional[str] = None):
        super().__init__(parent)
        self.init_key(key)
        self._literal = text
        self.setFont(T.font_label(10.5, True, 1.2))
        self.setStyleSheet(f"color: {T.INK}; background: transparent;")
        self.retranslate()

    def retranslate(self) -> None:
        self.setText(self._literal if self._literal is not None else self.text_now())


class HintLabel(QLabel, Translatable):
    """Small grey explanation text. ``provider`` (a callable returning the text) wins over ``key`` / ``text`` and is
    asked again on every language change."""

    def __init__(self, key: str = "", parent: Optional[QWidget] = None, *, text: Optional[str] = None,
                 px: float = 11.5, provider=None, **fmt):
        super().__init__(parent)
        self.init_key(key, **fmt)
        self._literal = text
        self._provider = provider
        self.setFont(T.font_ui(px))
        self.setWordWrap(True)
        self.setStyleSheet(f"color: {T.INK_SOFT}; background: transparent;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.retranslate()

    def retranslate(self) -> None:
        if self._provider is not None:
            self.setText(self._provider())
        else:
            self.setText(self._literal if self._literal is not None else self.text_now())


class Chip(QLabel):
    """Small status label. ``tone``: ``paper`` (outline), ``ink`` (black) or ``red``."""

    TONES = {"paper": (T.PAPER_HI, T.INK, T.INK), "ink": (T.INK, T.PAPER, T.INK), "red": (T.RED, T.PAPER, T.RED)}

    def __init__(self, text: str = "", tone: str = "paper", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tone = tone
        self._text = text
        self.setFixedHeight(20)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_chip(self, text: str, tone: str) -> None:
        self._text, self.tone = text, tone
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        fm = QFontMetrics(T.font_label(9.5, True, 0.8))
        return QSize(fm.horizontalAdvance(self._text) + 18, 20)

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        fill, text, edge = self.TONES.get(self.tone, self.TONES["paper"])
        r = QRectF(self.rect())
        draw_box(p, r, fill=fill, border=1.5, border_color=edge)
        p.setFont(T.font_label(9.5, True, 0.8))
        p.setPen(QColor(text))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, self._text)


class PopPanel(QFrame, Translatable):
    """A section: heavy black rule, a red marker, the title in bold capitals, a hairline. Put content into ``self.body``."""

    TITLE_H = 30

    def __init__(self, key: str = "", parent: Optional[QWidget] = None, *, text: Optional[str] = None):
        super().__init__(parent)
        self.init_key(key)
        self._literal = text
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(0, int(T.RULE) + self.TITLE_H + 12, 0, 16)
        self.body.setSpacing(10)
        self.retranslate()

    def retranslate(self) -> None:
        self._title = self._literal if self._literal is not None else self.text_now()
        self.update()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        w = float(self.width())
        p.fillRect(QRectF(0, 0, w, T.RULE), QColor(T.INK))
        top = T.RULE
        p.fillRect(QRectF(0, top + (self.TITLE_H - 9) / 2, 9, 9), QColor(T.RED))
        p.setFont(T.font_label(10.5, True, 1.4))
        p.setPen(QColor(T.INK))
        p.drawText(QRectF(18, top, w - 18, self.TITLE_H), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   self._title)
        p.fillRect(QRectF(0, top + self.TITLE_H, w, T.HAIR), QColor(T.INK))


class PopProgress(QWidget):
    """Progress bar: outlined track, red fill."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._v = 0.0
        self._text = ""
        self.setFixedHeight(24)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_progress(self, v: float, text: str = "") -> None:
        self._v, self._text = max(0.0, min(1.0, v)), text
        self.update()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        r = QRectF(self.rect())
        inner = draw_box(p, r, fill=T.PAPER_HI, border=2, border_color=T.INK)
        if self._v > 0.001:
            p.fillRect(QRectF(inner.left(), inner.top(), max(2.0, inner.width() * self._v), inner.height()), QColor(T.RED))
        if self._text:
            # black text with a white halo: readable on the red fill and on the empty track alike
            f = T.font_label(10.5, True, 0.6)
            fm = QFontMetricsF(f)
            x = r.center().x() - fm.horizontalAdvance(self._text) / 2.0
            y = r.center().y() + (fm.ascent() - fm.descent()) / 2.0
            path = QPainterPath()
            path.addText(QPointF(x, y), f, self._text)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setPen(QPen(QColor(T.PAPER_HI), 3.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(T.INK))
            p.drawPath(path)
