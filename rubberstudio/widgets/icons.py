"""Procedural line icons (24×24 grid) — crisp at any DPI, no font or image dependency."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QBrush, QPolygonF


def _poly(*pts) -> QPolygonF:
    return QPolygonF([QPointF(x, y) for x, y in pts])


def draw_icon(p: QPainter, name: str, rect: QRectF, color: QColor, stroke: float = 2.0) -> None:
    """Draw icon ``name`` into ``rect`` (square recommended)."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    side = min(rect.width(), rect.height())
    p.translate(rect.center().x() - side / 2, rect.center().y() - side / 2)
    p.scale(side / 24.0, side / 24.0)
    pen = QPen(color, stroke * 24.0 / max(side, 1) * 0.5 + 0.9)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    fill = QBrush(color)

    def line(*pts):
        p.drawPolyline(_poly(*pts))

    if name == "sidebar":
        p.drawRoundedRect(QRectF(3, 4.5, 18, 15), 3, 3)
        p.drawLine(QPointF(9.5, 4.5), QPointF(9.5, 19.5))
    elif name == "photo":
        p.drawRoundedRect(QRectF(3, 4, 18, 16), 3, 3)
        p.drawEllipse(QPointF(9, 9.5), 1.8, 1.8)
        line((4, 18), (10, 12.5), (14, 16), (17, 13), (20, 17))
    elif name == "export":
        line((12, 4), (12, 15))
        line((7.5, 10.5), (12, 15), (16.5, 10.5))
        line((4.5, 17), (4.5, 20), (19.5, 20), (19.5, 17))
    elif name == "info":
        p.drawEllipse(QPointF(12, 12), 8.5, 8.5)
        p.drawLine(QPointF(12, 11), QPointF(12, 16.5))
        p.setBrush(fill)
        p.drawEllipse(QPointF(12, 7.7), 0.9, 0.9)
    elif name == "crop":
        line((7, 2.5), (7, 17), (21.5, 17))
        line((2.5, 7), (17, 7), (17, 21.5))
    elif name == "repair":                      # bandage
        p.save()
        p.translate(12, 12)
        p.rotate(-45)
        p.drawRoundedRect(QRectF(-9.5, -4.2, 19, 8.4), 4.2, 4.2)
        p.drawLine(QPointF(-3, -4.2), QPointF(-3, 4.2))
        p.drawLine(QPointF(3, -4.2), QPointF(3, 4.2))
        p.setBrush(fill)
        for dx in (-0.4, 0.8):
            for dy in (-1.4, 1.4):
                p.drawEllipse(QPointF(dx, dy), 0.55, 0.55)
        p.restore()
    elif name == "watermark":                   # drop
        path = QPainterPath()
        path.moveTo(12, 3.2)
        path.cubicTo(12, 3.2, 5.2, 10.4, 5.2, 14.6)
        path.cubicTo(5.2, 18.4, 8.2, 20.8, 12, 20.8)
        path.cubicTo(15.8, 20.8, 18.8, 18.4, 18.8, 14.6)
        path.cubicTo(18.8, 10.4, 12, 3.2, 12, 3.2)
        p.drawPath(path)
        line((9, 15.2), (10.6, 17.2), (13.4, 17.4))
    elif name == "erase":                       # eraser, like the app icon
        p.save()
        p.translate(12, 12)
        p.rotate(-40)
        p.drawRoundedRect(QRectF(-9.5, -5, 19, 10), 3.4, 3.4)
        p.drawLine(QPointF(-1.5, -5), QPointF(-1.5, 5))
        p.restore()
        p.drawLine(QPointF(4, 21), QPointF(20, 21))
    elif name == "batch":                       # film frames
        p.drawRoundedRect(QRectF(7, 3, 14, 13), 2.4, 2.4)
        line((3, 8), (3, 20.5), (16, 20.5))
        p.drawLine(QPointF(10.5, 7.5), QPointF(17.5, 7.5))
        p.drawLine(QPointF(10.5, 11.5), QPointF(15, 11.5))
    elif name in ("undo", "redo"):
        if name == "redo":
            p.translate(24, 0)
            p.scale(-1, 1)
        line((9, 6), (4.5, 10.5), (9, 15))
        path = QPainterPath()
        path.moveTo(5, 10.5)
        path.lineTo(14, 10.5)
        path.cubicTo(18.5, 10.5, 20.5, 13, 20.5, 15.6)
        path.cubicTo(20.5, 18.4, 18.4, 20.5, 15.4, 20.5)
        path.lineTo(10, 20.5)
        p.drawPath(path)
    elif name == "reset":
        p.drawArc(QRectF(4, 4, 16, 16), 60 * 16, 270 * 16)
        line((18, 3.5), (18.6, 7.8), (14.2, 8.2))
    elif name == "folder":
        path = QPainterPath()
        path.moveTo(3, 7)
        path.lineTo(3, 18.5)
        path.cubicTo(3, 19.5, 3.6, 20, 4.6, 20)
        path.lineTo(19.4, 20)
        path.cubicTo(20.4, 20, 21, 19.5, 21, 18.5)
        path.lineTo(21, 9.5)
        path.cubicTo(21, 8.6, 20.4, 8, 19.4, 8)
        path.lineTo(11.5, 8)
        path.lineTo(9.5, 5.5)
        path.lineTo(4.6, 5.5)
        path.cubicTo(3.6, 5.5, 3, 6, 3, 7)
        p.drawPath(path)
    elif name == "play":
        p.setBrush(fill)
        p.drawPolygon(_poly((7.5, 4.5), (19.5, 12), (7.5, 19.5)))
    elif name == "stop":
        p.setBrush(fill)
        p.drawRoundedRect(QRectF(6, 6, 12, 12), 2, 2)
    elif name == "plus":
        p.drawLine(QPointF(12, 5), QPointF(12, 19))
        p.drawLine(QPointF(5, 12), QPointF(19, 12))
    elif name == "trash":
        line((4.5, 7), (19.5, 7))
        line((9.5, 7), (9.5, 4.5), (14.5, 4.5), (14.5, 7))
        line((6.5, 7), (7.5, 20), (16.5, 20), (17.5, 7))
        p.drawLine(QPointF(10.2, 11), QPointF(10.2, 16))
        p.drawLine(QPointF(13.8, 11), QPointF(13.8, 16))
    elif name == "eye":
        path = QPainterPath()
        path.moveTo(2.5, 12)
        path.cubicTo(6, 5.5, 18, 5.5, 21.5, 12)
        path.cubicTo(18, 18.5, 6, 18.5, 2.5, 12)
        p.drawPath(path)
        p.drawEllipse(QPointF(12, 12), 3.2, 3.2)
    elif name == "download":
        line((12, 4), (12, 14.5))
        line((7.5, 10.5), (12, 15), (16.5, 10.5))
        p.drawLine(QPointF(5, 20), QPointF(19, 20))
    elif name == "check":
        line((5, 12.8), (10, 17.6), (19.2, 7))
    elif name == "close":
        p.drawLine(QPointF(6, 6), QPointF(18, 18))
        p.drawLine(QPointF(18, 6), QPointF(6, 18))
    elif name == "globe":
        p.drawEllipse(QPointF(12, 12), 8.6, 8.6)
        p.drawEllipse(QRectF(8, 3.4, 8, 17.2))
        p.drawLine(QPointF(3.4, 12), QPointF(20.6, 12))
    elif name == "fit":
        line((4, 9), (4, 4), (9, 4))
        line((15, 4), (20, 4), (20, 9))
        line((20, 15), (20, 20), (15, 20))
        line((9, 20), (4, 20), (4, 15))
    elif name == "compare":
        p.drawRoundedRect(QRectF(3.5, 5, 17, 14), 2.5, 2.5)
        p.drawLine(QPointF(12, 5), QPointF(12, 19))
        p.setBrush(fill)
        p.drawRect(QRectF(5.5, 8, 4.5, 8))
    else:                                       # unknown -> small square, never crash painting
        p.drawRect(QRectF(6, 6, 12, 12))
    p.restore()
