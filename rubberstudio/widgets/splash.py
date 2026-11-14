"""Opening animation: a black-white-red title card that a red edge wipes away from left to right."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEasingCurve, QRectF, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFontMetricsF, QPainter
from PySide6.QtWidgets import QWidget

from .. import __version__
from .. import theme as T
from ..i18n import tr
from .icons import draw_icon


def paint_title_card(p: QPainter, rect: QRectF) -> None:
    """The poster: a red square with the eraser, the wordmark in two lines, heavy rules above and below."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.fillRect(rect, QColor(T.PAPER))
    ink = QColor(T.INK)
    cx, cy = rect.center().x(), rect.center().y()
    s = max(0.55, min(1.6, min(rect.width() / 1000.0, rect.height() / 640.0)))
    m = 56 * s

    # heavy rules at the top and the bottom edge of the composition
    rule = max(4.0, 7 * s)
    p.fillRect(QRectF(rect.left() + m, rect.top() + m, rect.width() - 2 * m, rule), ink)
    bottom = rect.bottom() - m - 30 * s
    p.fillRect(QRectF(rect.left() + m, bottom, rect.width() - 2 * m, rule), ink)

    # red square + wordmark, centred as one block
    side = 220 * s
    word_px = 120 * s
    f = T.font_display(word_px)
    p.setFont(f)
    fm = QFontMetricsF(f)
    word_w = max(fm.horizontalAdvance("RUBBER"), fm.horizontalAdvance("STUDIO"))
    gap = 44 * s
    x0 = cx - (side + gap + word_w) / 2
    top = cy - side / 2
    mark = QRectF(x0, top, side, side)
    p.fillRect(mark, QColor(T.RED))
    draw_icon(p, "erase", mark.adjusted(side * 0.2, side * 0.2, -side * 0.2, -side * 0.2), QColor(T.PAPER), 2.0 * s + 1.5)
    cap = fm.capHeight()
    p.setPen(ink)
    p.drawText(x0 + side + gap, top + cap, "RUBBER")
    p.drawText(x0 + side + gap, top + side, "STUDIO")

    # caption row under the lower rule
    p.setFont(T.font_label(max(10.0, 12.5 * s), True, 3.0 * s))
    p.setPen(ink)
    row = QRectF(rect.left() + m, bottom + rule + 8 * s, rect.width() - 2 * m, 24 * s)
    p.drawText(row, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, tr("splash.tag"))
    p.setPen(QColor(T.INK_SOFT))
    p.drawText(row, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"V{__version__}")
    p.restore()


class SplashOverlay(QWidget):
    finished = Signal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self._progress = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(900)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
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
        r = QRectF(self.rect())
        if self._progress <= 0:
            paint_title_card(p, r)
            return
        edge = r.width() * self._progress
        p.setClipRect(QRectF(edge, 0, r.width() - edge, r.height()))
        paint_title_card(p, r)
        p.setClipping(False)
        p.fillRect(QRectF(edge, 0, 8, r.height()), QColor(T.RED))
