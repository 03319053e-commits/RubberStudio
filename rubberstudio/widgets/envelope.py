"""Fold / unfold: a panel folds toward its screen edge, leaving a black pull-tab."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEasingCurve, QPointF, QRectF, QSize, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF, QTransform
from PySide6.QtWidgets import QSizePolicy, QWidget

from .. import theme as T

TAB = 24          # thickness of the always-visible pull-tab strip


class EnvelopePanel(QWidget):
    """Wraps ``content``; ``edge`` is the screen edge the panel is hinged to (left/right/top/bottom)."""
    toggled = Signal(bool)          # True = open

    def __init__(self, content: QWidget, edge: str = "left", extent: int = 300,
                 parent: Optional[QWidget] = None, tooltip: str = ""):
        super().__init__(parent)
        assert edge in ("left", "right", "top", "bottom")
        self.content = content
        content.setParent(self)
        self.edge = edge
        self.extent = extent
        self._p = 1.0
        self._open = True
        self._snapshot: Optional[QPixmap] = None
        self._hover = False
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(420)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        self._anim.finished.connect(self._on_done)
        self.setMouseTracking(True)
        self.setToolTip(tooltip)
        self._apply_size()

    # --- geometry -----------------------------------------------------------------------------
    @property
    def vertical_axis(self) -> bool:       # folds along y (header / status bar)
        return self.edge in ("top", "bottom")

    def _total(self) -> int:
        return int(round(self.extent * self._p)) + TAB

    def _apply_size(self) -> None:
        total = self._total()
        if self.vertical_axis:
            self.setFixedHeight(total)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        else:
            self.setFixedWidth(total)
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._layout_content()
        self.update()

    def _content_rect(self) -> QRectF:
        """Where the fully-open content sits (inside this widget), ignoring the fold."""
        w, h, e = self.width(), self.height(), self.extent
        if self.edge == "left":
            return QRectF(0, 0, e, h)
        if self.edge == "right":
            return QRectF(w - e, 0, e, h)       # NOTE: pinned to the screen edge (outer side)
        if self.edge == "top":
            return QRectF(0, 0, w, e)
        return QRectF(0, h - e, w, e)

    def _layout_content(self) -> None:
        if self._p >= 0.999:
            r = self._content_rect().toRect()
            if self.edge == "right":
                r.moveLeft(TAB)
            elif self.edge == "bottom":
                r.moveTop(TAB)
            self.content.setGeometry(r)
            if self._snapshot is None:
                self.content.show()
        else:
            self.content.hide()

    def resizeEvent(self, e) -> None:
        self._layout_content()

    def sizeHint(self) -> QSize:
        return QSize(self._total(), self.extent) if not self.vertical_axis else QSize(self.extent, self._total())

    # --- state --------------------------------------------------------------------------------
    def is_open(self) -> bool:
        return self._open

    def set_open(self, open_: bool, animate: bool = True) -> None:
        if open_ == self._open and not self._anim.state() == QVariantAnimation.State.Running:
            return
        self._open = open_
        self._anim.stop()
        target = 1.0 if open_ else 0.0
        if not animate:
            self._snapshot = None
            self._p = target
            self._apply_size()
            self.toggled.emit(open_)
            return
        # snapshot of the fully laid-out content for the perspective fold
        full = self.extent
        if self.vertical_axis:
            self.content.setGeometry(0, 0, max(self.width(), 10), full)
        else:
            self.content.setGeometry(0, 0, full, max(self.height(), 10))
        self.content.show()
        self._snapshot = self.content.grab()
        self.content.hide()
        self._anim.setStartValue(self._p)
        self._anim.setEndValue(target)
        self._anim.start()
        self.toggled.emit(open_)

    def toggle(self) -> None:
        self.set_open(not self._open)

    def _on_anim(self, v) -> None:
        self._p = float(v)
        self._apply_size()

    def _on_done(self) -> None:
        self._snapshot = None
        self._apply_size()

    # --- tab ----------------------------------------------------------------------------------
    def _tab_rect(self) -> QRectF:
        w, h = self.width(), self.height()
        if self.edge == "left":
            return QRectF(w - TAB, 0, TAB, h)
        if self.edge == "right":
            return QRectF(0, 0, TAB, h)
        if self.edge == "top":
            return QRectF(0, h - TAB, w, TAB)
        return QRectF(0, 0, w, TAB)

    def _flap_rect(self) -> QRectF:
        t = self._tab_rect()
        if self.vertical_axis:
            return QRectF(t.center().x() - 36, t.top(), 72, TAB)
        return QRectF(t.left(), t.center().y() - 36, TAB, 72)

    def mouseMoveEvent(self, e) -> None:
        over = self._flap_rect().adjusted(-4, -4, 4, 4).contains(e.position())
        if over != self._hover:
            self._hover = over
            self.setCursor(Qt.CursorShape.PointingHandCursor if over else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, e) -> None:
        self._hover = False
        self.update()

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton and self._flap_rect().adjusted(-4, -4, 4, 4).contains(e.position()):
            self.toggle()
        else:
            e.ignore()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        if self._snapshot is not None and 0.001 < self._p < 0.999:
            self._paint_fold(p)
        # --- pull-tab: a black rectangle with a white chevron, red under the pointer
        r = self._flap_rect()
        p.fillRect(r, QColor(T.RED if self._hover else T.INK))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(T.PAPER), 2.4)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        p.setPen(pen)
        # chevron: points where the panel will go
        d = 4.5
        dirs = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}[self.edge]
        sgn = 1 if self._open else -1
        vx, vy = dirs[0] * sgn, dirs[1] * sgn
        c = r.center()
        if vx:
            tip = QPointF(c.x() + vx * d * 0.5, c.y())
            p.drawPolyline(QPolygonF([QPointF(tip.x() - vx * d, tip.y() - d), tip, QPointF(tip.x() - vx * d, tip.y() + d)]))
        else:
            tip = QPointF(c.x(), c.y() + vy * d * 0.5)
            p.drawPolyline(QPolygonF([QPointF(tip.x() - d, tip.y() - vy * d), tip, QPointF(tip.x() + d, tip.y() - vy * d)]))

    def _paint_fold(self, p: QPainter) -> None:
        snap = self._snapshot
        assert snap is not None
        prog = self._p
        sw, sh = float(snap.width()) / snap.devicePixelRatio(), float(snap.height()) / snap.devicePixelRatio()
        src = QPolygonF([QPointF(0, 0), QPointF(sw, 0), QPointF(sw, sh), QPointF(0, sh)])
        k = (1.0 - prog) * 0.085
        if self.edge in ("left", "right"):
            wv = sw * prog
            kk = sh * k
            if self.edge == "left":
                dst = QPolygonF([QPointF(0, 0), QPointF(wv, kk), QPointF(wv, sh - kk), QPointF(0, sh)])
            else:
                dst = QPolygonF([QPointF(TAB + sw, 0), QPointF(TAB + sw - wv, kk), QPointF(TAB + sw - wv, sh - kk), QPointF(TAB + sw, sh)])
        else:
            hv = sh * prog
            kk = sw * k
            if self.edge == "top":
                dst = QPolygonF([QPointF(0, 0), QPointF(sw, 0), QPointF(sw - kk, hv), QPointF(kk, hv)])
            else:
                dst = QPolygonF([QPointF(kk, TAB + sh - hv), QPointF(sw - kk, TAB + sh - hv), QPointF(sw, TAB + sh), QPointF(0, TAB + sh)])
        tr = QTransform()
        if not QTransform.quadToQuad(src, dst, tr):
            return
        p.save()
        p.setTransform(tr, True)
        p.drawPixmap(0, 0, snap)
        # the leaf darkens as it turns away
        shade = QColor(T.INK)
        shade.setAlphaF(0.30 * (1.0 - prog))
        p.fillRect(QRectF(0, 0, sw, sh), shade)
        p.restore()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(QPen(QColor(T.INK), 2.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolygon(dst)
