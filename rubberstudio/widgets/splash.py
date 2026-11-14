"""Opening animation: a paper title card is wiped away by a growing diagonal water-drop (port of the iOS splash)."""
from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QEasingCurve, QPointF, QRectF, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from .. import __version__
from .. import theme as T
from ..i18n import tr
from .pop import dot_color, draw_dots, draw_pop_box, paper_brush, rrect


def drop_path(rect: QRectF, progress: float) -> QPainterPath:
    """Round head slides along the diagonal growing as it goes; the tail tip stays outside the top-left corner."""
    t = max(0.0, min(1.0, progress))
    diag = math.hypot(rect.width(), rect.height())
    tail = QPointF(rect.left() - rect.width() * 0.08, rect.top() - rect.height() * 0.08)
    end = QPointF(rect.right() + rect.width() * 0.05, rect.bottom() + rect.height() * 0.05)
    c = QPointF(tail.x() + (end.x() - tail.x()) * t, tail.y() + (end.y() - tail.y()) * t)
    r = diag * (0.03 + 0.55 * t)
    d = math.hypot(c.x() - tail.x(), c.y() - tail.y())
    path = QPainterPath()
    if d <= r + 1:
        path.addEllipse(c, r, r)
        return path
    phi = math.atan2(tail.y() - c.y(), tail.x() - c.x())
    alpha = math.acos(min(1.0, r / d))
    start, stop = phi + alpha, phi + 2 * math.pi - alpha
    steps = 72
    pts = [QPointF(c.x() + math.cos(start + (stop - start) * i / steps) * r,
                   c.y() + math.sin(start + (stop - start) * i / steps) * r) for i in range(steps + 1)]
    pts.append(tail)
    path.moveTo(pts[0])
    for pt in pts[1:]:
        path.lineTo(pt)
    path.closeSubpath()
    return path


def paint_title_card(p: QPainter, rect: QRectF) -> None:
    """The poster: halftone sun, an eraser, the wordmark with its mis-registered colour plate."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.fillRect(rect, paper_brush(T.PAPER, 1))
    cx, cy = rect.center().x(), rect.center().y()
    s = min(rect.width() / 1000.0, rect.height() / 640.0)
    s = max(0.55, min(1.5, s))

    # big halftone disc behind everything
    R = 190 * s
    disc = QRectF(cx - R, cy - R * 1.12, 2 * R, 2 * R)
    p.save()
    clip = QPainterPath()
    clip.addEllipse(disc)
    p.setClipPath(clip)
    p.fillRect(disc, QColor(T.YELLOW))
    draw_dots(p, disc, dot_color(QColor(T.YELLOW)), ramp_angle=60, lo=0.1, hi=1.0, spacing=9 * s, speckle=0.004)
    p.restore()
    p.setPen(QPen(QColor(T.INK), 3.2 * s))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(disc)

    # eraser
    p.save()
    p.translate(cx + 70 * s, cy - 60 * s)
    p.rotate(-28)
    body = QRectF(-130 * s, -52 * s, 260 * s, 104 * s)
    draw_pop_box(p, body, fill=T.PAPER_HI, radius=26 * s, shadow=9 * s, shadow_color=T.INK, border=3.4 * s)
    sleeve = QRectF(-34 * s, body.top() + 2, 68 * s, body.height() - 4)
    p.fillRect(sleeve.translated(3 * s, 3 * s), QColor(T.RED))
    p.save()
    p.setClipRect(sleeve.translated(3 * s, 3 * s))
    draw_dots(p, sleeve.translated(3 * s, 3 * s), dot_color(QColor(T.RED)), ramp_angle=90, lo=0.2, hi=1.0, spacing=6 * s)
    p.restore()
    pen = QPen(QColor(T.INK), 3.4 * s)
    p.setPen(pen)
    p.drawLine(QPointF(sleeve.left(), body.top() + 1.7 * s), QPointF(sleeve.left(), body.bottom() - 1.7 * s))
    p.drawLine(QPointF(sleeve.right(), body.top() + 1.7 * s), QPointF(sleeve.right(), body.bottom() - 1.7 * s))
    p.restore()

    # pencil scribble being rubbed out
    pen = QPen(QColor(T.INK_SOFT), 3.0 * s)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    path = QPainterPath()
    x0, y0 = cx - 230 * s, cy + 90 * s
    path.moveTo(x0, y0)
    for i in range(1, 7):
        path.lineTo(x0 + i * 22 * s, y0 + (-26 if i % 2 else 8) * s)
    p.drawPath(path)

    # wordmark: colour plate slipped against the ink plate
    for word, dy, col in (("RUBBER", -2.0, T.RED), ("STUDIO", 0.0, T.BLUE)):
        f = T.font_display(86 * s)
        p.setFont(f)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(word)
        y = cy + (150 + (0 if word == "RUBBER" else 96)) * s
        x = cx - tw / 2
        p.setPen(QColor(col))
        p.drawText(QPointF(x + 7 * s, y + 7 * s), word)
        p.setPen(QColor(T.INK))
        p.drawText(QPointF(x, y), word)

    p.setFont(T.font_mono(14 * s + 1, True, 3.0))
    p.setPen(QColor(T.INK_SOFT))
    tag = tr("splash.tag")
    p.drawText(QRectF(rect.left(), cy + 290 * s, rect.width(), 30 * s), Qt.AlignmentFlag.AlignHCenter, tag)
    p.setFont(T.font_mono(11, False, 1.0))
    p.drawText(QRectF(rect.left(), rect.bottom() - 34, rect.width(), 22), Qt.AlignmentFlag.AlignHCenter,
               f"v{__version__}")
    p.restore()


class SplashOverlay(QWidget):
    finished = Signal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self._progress = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(1600)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim.valueChanged.connect(self._tick)
        self._anim.finished.connect(self._end)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._anim.start)
        self.setGeometry(parent.rect())
        self.raise_()
        self.show()

    def start(self, delay_ms: int = 900) -> None:
        self._timer.start(delay_ms)

    def skip(self) -> None:
        self._timer.stop()
        self._anim.stop()
        self._end()

    def mousePressEvent(self, e) -> None:
        self.skip()

    def keyPressEvent(self, e) -> None:
        self.skip()

    def _tick(self, v) -> None:
        self._progress = float(v)
        self.update()

    def _end(self) -> None:
        self.hide()
        self.finished.emit()
        self.deleteLater()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect())
        if self._progress > 0:
            mask = QPainterPath()
            mask.setFillRule(Qt.FillRule.OddEvenFill)
            mask.addRect(r)
            mask.addPath(drop_path(r, self._progress))
            p.setClipPath(mask)
        paint_title_card(p, r)
        if self._progress > 0:
            p.setClipping(False)
            pen = QPen(QColor(T.INK), 3.4)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(drop_path(r, self._progress))
