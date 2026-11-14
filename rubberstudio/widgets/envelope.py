"""Envelope-style fold / unfold: a panel folds toward its screen edge like a flap, leaving a pull-tab."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEasingCurve, QPointF, QRectF, QSize, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QTransform
from PySide6.QtWidgets import QSizePolicy, QWidget

from .. import theme as T
from .pop import dot_color, draw_dots

TAB = 24          # thickness of the always-visible pull-tab strip


class EnvelopePanel(QWidget):
    """Wraps ``content``; ``edge`` is the screen edge the panel is hinged to (left/right/top/bottom)."""
    toggled = Signal(bool)          # True = open

    def __init__(self, content: QWidget, edge: str = "left", extent: int = 300, color: str = T.YELLOW,
                 parent: Optional[QWidget] = None, tooltip: str = ""):
        super().__init__(parent)
        assert edge in ("left", "right", "top", "bottom")
        self.content = content
        content.setParent(self)
        self.edge = edge
        self.extent = extent
        self.color = color
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

    def _flap_path(self) -> tuple[QPainterPath, QPointF]:
        r = self._flap_rect()
        path = QPainterPath()
        pad = 3.0
        if self.edge == "left":      # base on the content edge (left of strip), apex toward the canvas
            a, b, c = QPointF(r.left(), r.top() + pad), QPointF(r.right() - 1, r.center().y()), QPointF(r.left(), r.bottom() - pad)
        elif self.edge == "right":
            a, b, c = QPointF(r.right(), r.top() + pad), QPointF(r.left() + 1, r.center().y()), QPointF(r.right(), r.bottom() - pad)
        elif self.edge == "top":
            a, b, c = QPointF(r.left() + pad, r.top()), QPointF(r.center().x(), r.bottom() - 1), QPointF(r.right() - pad, r.top())
        else:
            a, b, c = QPointF(r.left() + pad, r.bottom()), QPointF(r.center().x(), r.top() + 1), QPointF(r.right() - pad, r.bottom())
        path.moveTo(a)
        path.lineTo(b)
        path.lineTo(c)
        path.closeSubpath()
        return path, r.center()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        if self._snapshot is not None and 0.001 < self._p < 0.999:
            self._paint_fold(p)
        # --- pull-tab flap
        path, centre = self._flap_path()
        hover = self._hover
        off = 2.0 if not hover else 3.0
        p.fillPath(path.translated(off, off), QColor(T.INK))
        p.save()
        p.setClipPath(path)
        p.fillRect(path.boundingRect(), QColor(self.color if not hover else T.YELLOW))
        draw_dots(p, path.boundingRect(), dot_color(QColor(self.color)), ramp_angle=0 if not self.vertical_axis else 90,
                  lo=0.2, hi=1.0, spacing=5)
        p.restore()
        pen = QPen(QColor(T.INK), 2.4)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawPath(path)
        # chevron: points where the panel will go
        d = 4.0
        dirs = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}[self.edge]
        sgn = 1 if self._open else -1
        vx, vy = dirs[0] * sgn, dirs[1] * sgn
        c = centre
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
        # fold shading: the flap darkens as it turns away
        shade = QColor(T.INK)
        shade.setAlphaF(0.38 * (1.0 - prog))
        p.fillRect(QRectF(0, 0, sw, sh), shade)
        p.restore()
        pen = QPen(QColor(T.INK), 2.4)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolygon(dst)
