"""The picture desk: zoom / pan, crop + text frames, brush cursor, watermark, detection boxes, before / after."""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import QWidget

from .. import theme as T
from ..editor import EditorModel, Tool
from ..i18n import tr
from .icons import draw_icon
from .pop import PopButton, draw_box

HANDLE = 6.5            # half the side of a drag handle


def _checker() -> QBrush:
    pm = QPixmap(16, 16)
    pm.fill(QColor(T.PAPER_HI))
    p = QPainter(pm)
    p.fillRect(0, 0, 8, 8, QColor("#D6D6D6"))
    p.fillRect(8, 8, 8, 8, QColor("#D6D6D6"))
    p.end()
    return QBrush(pm)


class Canvas(QWidget):
    zoomChanged = Signal(float)
    openRequested = Signal()

    def __init__(self, model: EditorModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.model = model
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._img_rect = QRectF()
        self._space = False
        self._mode: Optional[str] = None          # pan | stroke | rect-move | rect-handle | rect-new | wm
        self._press = QPointF()
        self._pan0 = QPointF()
        self._rect0 = (0.0, 0.0, 0.0, 0.0)
        self._handle = -1
        self._mouse: Optional[QPointF] = None
        self._proxy_key = None
        self._proxy: Optional[QImage] = None
        self._proxy_buf: Optional[np.ndarray] = None
        self._last_size = (0, 0)
        self._checker = _checker()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 240)

        self.fit_btn = PopButton(variant="paper", icon="fit", text="", height=34)
        self.fit_btn.setFixedWidth(42)
        self.fit_btn.setParent(self)
        self.fit_btn.clicked.connect(self.fit)
        self.fit_btn.setToolTip(tr("canvas.fit"))
        self.cmp_btn = PopButton("canvas.compare", "ink", "compare", self, height=34, px=11)
        self.cmp_btn.setCheckable(True)
        self.cmp_btn.clicked.connect(self.model.toggle_compare)
        self.cmp_btn.hide()

        m = self.model
        m.imageChanged.connect(self._on_image)
        m.pixelsChanged.connect(self.update)
        m.overlayChanged.connect(self.update)
        m.toolChanged.connect(self._on_tool)
        m.compareChanged.connect(self._on_compare)
        m.busyChanged.connect(lambda _b: self.update())

    # ------------------------------------------------------------------ geometry

    def retranslate(self) -> None:
        self.fit_btn.setToolTip(tr("canvas.fit"))
        self.cmp_btn.retranslate()
        self.update()

    def _frame(self) -> QRectF:
        return QRectF(1, 1, self.width() - 2, self.height() - 2)

    def _fit_scale(self) -> float:
        iw, ih = self.model.size
        if iw <= 0 or ih <= 0:
            return 1.0
        avail = self._frame().adjusted(26, 26, -26, -26)
        return max(0.01, min(avail.width() / iw, avail.height() / ih, 4.0))

    def scale(self) -> float:
        return self._fit_scale() * self.zoom

    def _compute_rect(self) -> QRectF:
        iw, ih = self.model.size
        if iw <= 0:
            return QRectF()
        s = self.scale()
        f = self._frame()
        cx = f.center().x() + self.pan.x()
        cy = f.center().y() + self.pan.y()
        return QRectF(cx - iw * s / 2, cy - ih * s / 2, iw * s, ih * s)

    def to_image(self, pos: QPointF) -> QPointF:
        s = self.scale()
        r = self._img_rect
        return QPointF((pos.x() - r.left()) / s, (pos.y() - r.top()) / s)

    def fit(self) -> None:
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.zoomChanged.emit(self.scale())
        self.update()

    def resizeEvent(self, e) -> None:
        self.fit_btn.move(int(self.width() - 14 - self.fit_btn.width()), 14)
        self.cmp_btn.adjustSize()
        self.cmp_btn.move(int(self.width() - 14 - self.fit_btn.width() - 10 - self.cmp_btn.width()), 14)
        self.zoomChanged.emit(self.scale())

    # ------------------------------------------------------------------ model reactions

    def _on_image(self) -> None:
        size = self.model.size
        if size != self._last_size:
            self._last_size = size
            self.fit()
        self.update()

    def _on_tool(self, _t: str) -> None:
        self._mode = None
        self._update_cursor(self._mouse)
        self.update()

    def _on_compare(self) -> None:
        has = self.model.after is not None
        self.cmp_btn.setVisible(has)
        self.cmp_btn.setChecked(has and self.model.show_after)
        self.cmp_btn.adjustSize()
        self.resizeEvent(None)
        self.update()

    # ------------------------------------------------------------------ rectangles

    def _active_rect(self) -> Optional[tuple]:
        m = self.model
        if m.image is None:
            return None
        if m.tool == Tool.CROP:
            return m.crop_rect
        if m.tool == Tool.ERASE and not m.erase.whole_image:
            return m.erase_rect
        if m.tool == Tool.BATCH and m.batch_region_on:
            return m.batch_rect
        return None

    def _set_active_rect(self, rel: tuple) -> None:
        m = self.model
        if m.tool == Tool.CROP:
            m.crop_rect = rel
        elif m.tool == Tool.ERASE:
            m.erase_rect = rel
        elif m.tool == Tool.BATCH:
            m.batch_rect = rel
        m.overlayChanged.emit()

    def _rect_to_widget(self, rel: tuple) -> QRectF:
        r = self._img_rect
        return QRectF(r.left() + rel[0] * r.width(), r.top() + rel[1] * r.height(),
                      rel[2] * r.width(), rel[3] * r.height())

    def _handles(self, wr: QRectF) -> list[QPointF]:
        return [wr.topLeft(), QPointF(wr.center().x(), wr.top()), wr.topRight(),
                QPointF(wr.right(), wr.center().y()), wr.bottomRight(),
                QPointF(wr.center().x(), wr.bottom()), wr.bottomLeft(), QPointF(wr.left(), wr.center().y())]

    def _hit_handle(self, wr: QRectF, pos: QPointF) -> int:
        best, bd = -1, 13.0
        for i, h in enumerate(self._handles(wr)):
            d = math.hypot(h.x() - pos.x(), h.y() - pos.y())
            if d < bd:
                best, bd = i, d
        return best

    _HANDLE_CURSORS = [Qt.CursorShape.SizeFDiagCursor, Qt.CursorShape.SizeVerCursor, Qt.CursorShape.SizeBDiagCursor,
                       Qt.CursorShape.SizeHorCursor, Qt.CursorShape.SizeFDiagCursor, Qt.CursorShape.SizeVerCursor,
                       Qt.CursorShape.SizeBDiagCursor, Qt.CursorShape.SizeHorCursor]

    def _drag_handle(self, idx: int, pos: QPointF) -> None:
        r = self._img_rect
        if r.width() <= 0:
            return
        x0, y0, w0, h0 = self._rect0
        l, t, rr, b = x0, y0, x0 + w0, y0 + h0
        px = (pos.x() - r.left()) / r.width()
        py = (pos.y() - r.top()) / r.height()
        px, py = min(max(px, 0.0), 1.0), min(max(py, 0.0), 1.0)
        mn = 0.02
        if idx in (0, 6, 7):
            l = min(px, rr - mn)
        if idx in (2, 3, 4):
            rr = max(px, l + mn)
        if idx in (0, 1, 2):
            t = min(py, b - mn)
        if idx in (4, 5, 6):
            b = max(py, t + mn)
        self._set_active_rect((l, t, rr - l, b - t))

    # ------------------------------------------------------------------ input

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = True
            self._update_cursor(self._mouse)
        elif e.key() == Qt.Key.Key_0:
            self.fit()
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = False
            self._update_cursor(self._mouse)
        else:
            super().keyReleaseEvent(e)

    def _stroke(self, pos: QPointF) -> None:
        pt = self.to_image(pos)
        iw, ih = self.model.size
        if -40 <= pt.x() <= iw + 40 and -40 <= pt.y() <= ih + 40:
            self.model.stroke_to(min(max(pt.x(), 0), iw - 1), min(max(pt.y(), 0), ih - 1))

    def mousePressEvent(self, e) -> None:
        self.setFocus()
        pos = e.position()
        btn = e.button()
        m = self.model
        if m.image is None:
            if btn == Qt.MouseButton.LeftButton:
                self.openRequested.emit()
            return
        self._press = pos
        if btn in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton) or (
                btn == Qt.MouseButton.LeftButton and self._space):
            self._mode, self._pan0 = "pan", QPointF(self.pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if btn != Qt.MouseButton.LeftButton:
            return
        inside = self._img_rect.contains(pos)
        if m.tool == Tool.REPAIR and inside and not m.busy:
            m.begin_stroke()
            self._mode = "stroke"
            self._stroke(pos)
            return
        rel = self._active_rect()
        if rel is not None:
            wr = self._rect_to_widget(rel)
            h = self._hit_handle(wr, pos)
            if h >= 0:
                self._mode, self._handle, self._rect0 = "rect-handle", h, rel
                return
            if wr.contains(pos):
                self._mode, self._rect0 = "rect-move", rel
                return
            if inside:
                self._mode, self._rect0 = "rect-new", rel
                return
        if m.tool == Tool.WATERMARK and inside:
            wr = self._wm_widget_rect()
            if wr is not None and wr.adjusted(-8, -8, 8, 8).contains(pos):
                self._mode = "wm"
                return
        self._mode, self._pan0 = "pan", QPointF(self.pan)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e) -> None:
        pos = e.position()
        self._mouse = pos
        mode = self._mode
        m = self.model
        if mode == "pan":
            self.pan = self._pan0 + (pos - self._press)
        elif mode == "stroke":
            self._stroke(pos)
        elif mode == "rect-handle":
            self._drag_handle(self._handle, pos)
        elif mode == "rect-move":
            r = self._img_rect
            dx, dy = (pos.x() - self._press.x()) / r.width(), (pos.y() - self._press.y()) / r.height()
            x0, y0, w0, h0 = self._rect0
            self._set_active_rect((min(max(x0 + dx, 0.0), 1.0 - w0), min(max(y0 + dy, 0.0), 1.0 - h0), w0, h0))
        elif mode == "rect-new":
            r = self._img_rect
            ax = min(max((self._press.x() - r.left()) / r.width(), 0), 1)
            ay = min(max((self._press.y() - r.top()) / r.height(), 0), 1)
            bx = min(max((pos.x() - r.left()) / r.width(), 0), 1)
            by = min(max((pos.y() - r.top()) / r.height(), 0), 1)
            l, rr, t, b = min(ax, bx), max(ax, bx), min(ay, by), max(ay, by)
            if rr - l >= 0.01 and b - t >= 0.01:
                self._set_active_rect((l, t, rr - l, b - t))
        elif mode == "wm":
            r = self._img_rect
            m.wm.center = (min(max((pos.x() - r.left()) / r.width(), 0.0), 1.0),
                           min(max((pos.y() - r.top()) / r.height(), 0.0), 1.0))
            m.overlayChanged.emit()
        else:
            self._update_cursor(pos)
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        if self._mode == "stroke":
            self.model.end_stroke()
        self._mode = None
        self._update_cursor(e.position())
        self.update()

    def leaveEvent(self, e) -> None:
        self._mouse = None
        self.update()

    def wheelEvent(self, e) -> None:
        if self.model.image is None:
            return
        delta = e.angleDelta().y()
        if delta == 0:
            return
        factor = 1.18 ** (delta / 120.0)
        new_zoom = min(max(self.zoom * factor, 0.25), 40.0)
        if abs(new_zoom - self.zoom) < 1e-6:
            return
        pos = e.position()
        old_s, old_r = self.scale(), QRectF(self._img_rect)
        anchor = QPointF((pos.x() - old_r.left()) / old_s, (pos.y() - old_r.top()) / old_s)   # image px under cursor
        self.zoom = new_zoom
        new_s = self.scale()
        f = self._frame()
        iw, ih = self.model.size
        left = pos.x() - anchor.x() * new_s
        top = pos.y() - anchor.y() * new_s
        self.pan = QPointF(left + iw * new_s / 2 - f.center().x(), top + ih * new_s / 2 - f.center().y())
        if abs(self.zoom - 1.0) < 0.02:
            self.zoom, self.pan = 1.0, QPointF(0, 0)
        self.zoomChanged.emit(self.scale())
        self.update()
        e.accept()

    def mouseDoubleClickEvent(self, e) -> None:
        if self.model.image is not None and self.model.tool not in (Tool.REPAIR, Tool.CROP, Tool.ERASE):
            self.fit()

    def _update_cursor(self, pos: Optional[QPointF]) -> None:
        m = self.model
        if pos is None or m.image is None:
            self.setCursor(Qt.CursorShape.PointingHandCursor if m.image is None else Qt.CursorShape.ArrowCursor)
            return
        if self._space:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if m.tool == Tool.REPAIR and self._img_rect.contains(pos):
            self.setCursor(Qt.CursorShape.BlankCursor)
            return
        rel = self._active_rect()
        if rel is not None:
            wr = self._rect_to_widget(rel)
            h = self._hit_handle(wr, pos)
            if h >= 0:
                self.setCursor(self._HANDLE_CURSORS[h])
                return
            if wr.contains(pos):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                return
            if self._img_rect.contains(pos):
                self.setCursor(Qt.CursorShape.CrossCursor)
                return
        if m.tool == Tool.WATERMARK:
            wr = self._wm_widget_rect()
            if wr is not None and wr.adjusted(-8, -8, 8, 8).contains(pos):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                return
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    # ------------------------------------------------------------------ painting

    def _wm_widget_rect(self) -> Optional[QRectF]:
        m = self.model
        if m.image is None:
            return None
        r, s = m.watermark_rect(), self.scale()
        return QRectF(self._img_rect.left() + r.x() * s, self._img_rect.top() + r.y() * s, r.width() * s, r.height() * s)

    def _paint_desk(self, p: QPainter, f: QRectF) -> QPainterPath:
        """The grey desk the picture lies on, inside a black outline. Returns the area inside the outline."""
        inner = draw_box(p, f, fill=T.PAPER_LO, border=2)
        path = QPainterPath()
        path.addRect(inner)
        return path

    def _paint_empty(self, p: QPainter, f: QRectF) -> None:
        box = QRectF(0, 0, min(460, f.width() - 60), 270)
        box.moveCenter(f.center())
        pen = QPen(QColor(T.INK), 1.5, Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([4, 3])
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(box)
        ic = QRectF(0, 0, 76, 76)
        ic.moveCenter(QPointF(box.center().x(), box.top() + 88))
        p.fillRect(ic, QColor(T.RED))
        draw_icon(p, "photo", ic.adjusted(16, 16, -16, -16), QColor(T.PAPER), 2.2)
        p.setPen(QColor(T.INK))
        p.setFont(T.font_label(15, True, 1.6))
        p.drawText(QRectF(box.left(), box.top() + 146, box.width(), 30), Qt.AlignmentFlag.AlignHCenter, tr("canvas.empty.title"))
        p.setPen(QColor(T.INK_SOFT))
        p.setFont(T.font_ui(12))
        p.drawText(QRectF(box.left() + 20, box.top() + 184, box.width() - 40, 60),
                   Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap, tr("canvas.empty.sub"))

    def _proxy_for(self, arr: np.ndarray, k: int) -> QImage:
        key = (id(arr), self.model.version, k)
        if key != self._proxy_key:
            h, w = arr.shape[:2]
            small = cv2.resize(arr, (max(1, w >> k), max(1, h >> k)), interpolation=cv2.INTER_AREA)
            self._proxy_buf = np.ascontiguousarray(small)
            fmt = QImage.Format.Format_RGBA8888 if self.model.has_alpha else QImage.Format.Format_RGBX8888
            self._proxy = QImage(self._proxy_buf.data, self._proxy_buf.shape[1], self._proxy_buf.shape[0],
                                 self._proxy_buf.shape[1] * 4, fmt)
            self._proxy_key = key
        assert self._proxy is not None
        return self._proxy

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        f = self._frame()
        inner = self._paint_desk(p, f)
        m = self.model
        img = m.qimage()
        if img is None:
            self._img_rect = QRectF()
            self._paint_empty(p, f)
            return
        self._img_rect = self._compute_rect()
        ir = self._img_rect
        p.save()
        p.setClipPath(inner)
        scale = ir.width() / img.width()

        if m.has_alpha:                      # transparent pictures lie on a chequerboard
            p.fillRect(ir, self._checker)

        vis = ir.intersected(QRectF(f))
        if not vis.isEmpty():
            arr = m.after if (m.show_after and m.after is not None) else m.image
            src_img, f_ = img, 1.0
            if scale < 0.55 and self._mode != "stroke" and arr is not None:
                k = max(1, int(math.floor(-math.log2(scale))))
                src_img, f_ = self._proxy_for(arr, k), 1.0 / (1 << k)
            src = QRectF((vis.left() - ir.left()) / scale * f_, (vis.top() - ir.top()) / scale * f_,
                         vis.width() / scale * f_, vis.height() / scale * f_)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, scale < 2.0 and self._mode != "stroke")
            p.drawImage(vis, src_img, src)
        pen = QPen(QColor(T.INK), 2.0)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(ir.adjusted(-1, -1, 1, 1))

        self._paint_overlays(p, ir, scale)
        p.restore()

        # HUD: zoom readout
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        txt = f"{int(round(self.scale() * 100))}%   {img.width()}×{img.height()}"
        p.setFont(T.font_label(10.5, True, 0.8))
        tw = p.fontMetrics().horizontalAdvance(txt) + 22
        chip = QRectF(f.left() + 14, f.bottom() - 40, tw, 26)
        draw_box(p, chip, fill=T.PAPER_HI, border=2)
        p.setPen(QColor(T.INK))
        p.drawText(chip, Qt.AlignmentFlag.AlignCenter, txt)
        if m.busy:
            self._paint_busy(p, f)

    def _paint_busy(self, p: QPainter, f: QRectF) -> None:
        w = 170
        chip = QRectF(f.center().x() - w / 2, f.top() + 16, w, 34)
        draw_box(p, chip, fill=T.RED, border=0)
        p.setFont(T.font_label(11, True, 1.4))
        p.setPen(QColor(T.PAPER))
        p.drawText(chip, Qt.AlignmentFlag.AlignCenter, tr("busy"))

    def _paint_overlays(self, p: QPainter, ir: QRectF, scale: float) -> None:
        m = self.model
        # detection boxes
        if m.boxes:
            fillc = QColor(T.RED)
            fillc.setAlphaF(0.16)
            for b in m.boxes:
                poly = QPolygonF([QPointF(ir.left() + x * scale, ir.top() + y * scale) for x, y in b.poly])
                p.setBrush(fillc)
                halo = QPen(QColor(T.PAPER_HI), 4.2)             # light halo so the box shows on red or dark picture areas
                halo.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                p.setPen(halo)
                p.drawPolygon(poly)
                p.setBrush(Qt.BrushStyle.NoBrush)
                inner = QPen(QColor(T.RED), 2.0)
                inner.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                p.setPen(inner)
                p.drawPolygon(poly)
        rel = self._active_rect()
        if rel is not None:
            wr = self._rect_to_widget(rel)
            dim = QColor(T.INK)
            dim.setAlphaF(0.42)
            path = QPainterPath()
            path.addRect(ir)
            hole = QPainterPath()
            hole.addRect(wr)
            p.fillPath(path.subtracted(hole), dim)
            p.setBrush(Qt.BrushStyle.NoBrush)
            if m.tool == Tool.CROP:
                guide = QColor(T.PAPER_HI)
                guide.setAlphaF(0.55)
                p.setPen(QPen(guide, 1.2))
                for i in (1, 2):
                    p.drawLine(QPointF(wr.left() + wr.width() * i / 3, wr.top()), QPointF(wr.left() + wr.width() * i / 3, wr.bottom()))
                    p.drawLine(QPointF(wr.left(), wr.top() + wr.height() * i / 3), QPointF(wr.right(), wr.top() + wr.height() * i / 3))
            p.setPen(QPen(QColor(T.PAPER_HI), 3.2))
            p.drawRect(wr)
            dash = QPen(QColor(T.INK), 2.2, Qt.PenStyle.CustomDashLine)
            dash.setDashPattern([3.5, 3.5])
            p.setPen(dash)
            p.drawRect(wr)
            for hpt in self._handles(wr):
                hb = QRectF(hpt.x() - HANDLE, hpt.y() - HANDLE, HANDLE * 2, HANDLE * 2)
                draw_box(p, hb, fill=T.RED, border=2, border_color=T.PAPER_HI)
            # size label
            iw, ih = m.size
            lab = f"{int(round(rel[2] * iw))} × {int(round(rel[3] * ih))}"
            p.setFont(T.font_label(10, True, 0.6))
            tw = p.fontMetrics().horizontalAdvance(lab) + 16
            lr = QRectF(wr.left(), wr.top() - 28, tw, 22)
            if lr.top() < ir.top():
                lr.moveTop(wr.top() + 6)
                lr.moveLeft(wr.left() + 6)
            draw_box(p, lr, fill=T.RED, border=0)
            p.setPen(QColor(T.PAPER))
            p.drawText(lr, Qt.AlignmentFlag.AlignCenter, lab)
        # watermark preview
        if m.tool == Tool.WATERMARK and m.wm.text.strip():
            wr = self._wm_widget_rect()
            if wr is not None:
                f = m.watermark_font()
                f.setPixelSize(max(1, int(round(f.pixelSize() * scale))))
                col = QColor(255, 255, 255) if m.wm.color == "white" else QColor(0, 0, 0)
                col.setAlphaF(m.wm.opacity)
                p.setFont(f)
                p.setPen(col)
                text = m.wm.text.upper() if m.wm.text.isascii() else m.wm.text
                p.drawText(wr, Qt.AlignmentFlag.AlignCenter, text)
                dash = QPen(QColor(T.RED), 2.0, Qt.PenStyle.CustomDashLine)
                dash.setDashPattern([3.5, 3.5])
                p.setPen(dash)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(wr.adjusted(-4, -4, 4, 4))
        # brush cursor
        if m.tool == Tool.REPAIR and self._mouse is not None and ir.contains(self._mouse) and self._mode in (None, "stroke"):
            r = m.brush_radius() * scale
            c = self._mouse
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(T.PAPER_HI), 3.6))
            p.drawEllipse(c, r, r)
            p.setPen(QPen(QColor(T.INK), 1.8))
            p.drawEllipse(c, r, r)
            hard = QPen(QColor(T.RED), 1.4, Qt.PenStyle.DashLine)
            p.setPen(hard)
            ri = r * (0.1 + 0.85 * m.brush.hardness)
            p.drawEllipse(c, ri, ri)
            p.setPen(QPen(QColor(T.INK), 1.6))
            p.drawLine(QPointF(c.x() - 4, c.y()), QPointF(c.x() + 4, c.y()))
            p.drawLine(QPointF(c.x(), c.y() - 4), QPointF(c.x(), c.y() + 4))
