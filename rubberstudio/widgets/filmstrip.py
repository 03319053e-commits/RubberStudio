"""The right-hand "film" — the batch queue drawn as strips of film with sprocket holes."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QModelIndex, QPoint, QPointF, QRectF, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFontMetrics, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QListView, QMenu, QSizePolicy, QStackedLayout,
                               QStyle, QStyledItemDelegate, QVBoxLayout, QWidget)

from .. import theme as T
from ..controllers import BatchController, BatchModel, ItemRole, ThumbRole
from ..core.batch import BatchItem, Status
from ..i18n import tr
from .icons import draw_icon
from .panels import PaperPane
from .pop import Chip, PopButton, SectionLabel, draw_dots, draw_pop_box, dot_color

ROW_H = 94

STATUS_LOOK = {                       # status -> (translation key, chip colour, shadow colour)
    Status.QUEUED: ("q.st.queued", T.PAPER_HI, T.INK_FAINT),
    Status.RUNNING: ("q.st.running", T.YELLOW, T.YELLOW),
    Status.DONE: ("q.st.done", T.GREEN, T.GREEN),
    Status.NO_TEXT: ("q.st.no_text", T.PINK, T.PINK),
    Status.FAILED: ("q.st.failed", T.RED, T.RED),
    Status.CANCELED: ("q.st.canceled", T.PAPER_LO, T.INK_FAINT),
}


class FilmDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index) -> QSize:
        return QSize(option.rect.width(), ROW_H)

    def paint(self, p: QPainter, option, index: QModelIndex) -> None:
        item: Optional[BatchItem] = index.data(ItemRole)
        if item is None:
            return
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        key, chip_col, shadow_col = STATUS_LOOK[item.status]
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        r = QRectF(option.rect).adjusted(5, 4, -9, -8)
        if hover and not selected:
            r.translate(-1, -1)
        strip = draw_pop_box(p, r, fill=T.INK, radius=8, shadow=4 if not selected else 5, shadow_color=shadow_col,
                             border=2.5, border_color=T.INK)
        # sprocket holes
        hole = QColor(T.PAPER)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(hole)
        y = strip.top() + 8
        while y + 10 < strip.bottom() - 3:
            p.drawRoundedRect(QRectF(strip.left() + 6, y, 8, 10), 2, 2)
            p.drawRoundedRect(QRectF(strip.right() - 14, y, 8, 10), 2, 2)
            y += 15
        frame = QRectF(strip.left() + 20, strip.top() + 6, strip.width() - 40, strip.height() - 12)
        draw_pop_box(p, frame, fill=T.PAPER_HI, radius=4, shadow=0, border=1.6, border_color=T.PAPER_LO)

        # thumbnail
        th = QRectF(frame.left() + 5, frame.top() + 5, 76, frame.height() - 10)
        p.fillRect(th, QColor(T.PAPER_LO))
        img = index.data(ThumbRole)
        if img is not None and not img.isNull():
            s = min(th.width() / img.width(), th.height() / img.height())
            w, h = img.width() * s, img.height() * s
            tr_ = QRectF(th.center().x() - w / 2, th.center().y() - h / 2, w, h)
            p.drawImage(tr_, img)
            p.setPen(QPen(QColor(T.INK), 1.6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(tr_)
        else:
            draw_icon(p, "photo", th.adjusted(20, 14, -20, -14), QColor(T.INK_FAINT), 1.6)
            p.setPen(QPen(QColor(T.INK), 1.6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(th)

        # text
        tx = th.right() + 9
        tw = frame.right() - 6 - tx
        p.setFont(T.font_mono(10.5, True, 0.2))
        p.setPen(QColor(T.INK))
        name = QFontMetrics(p.font()).elidedText(item.src.name, Qt.TextElideMode.ElideMiddle, int(tw))
        p.drawText(QRectF(tx, frame.top() + 5, tw, 18), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, name)

        label = tr(key)
        pf = T.font_mono(9.5, True)
        p.setFont(pf)
        cw = QFontMetrics(pf).horizontalAdvance(label) + 16
        chip = QRectF(tx, frame.top() + 28, cw, 19)
        draw_pop_box(p, chip, fill=chip_col, radius=9.5, shadow=0, border=1.8, dots=item.status != Status.QUEUED,
                     dots_hi=0.8, ramp_angle=30)
        p.setPen(QColor(T.TEXT_ON.get(chip_col, T.INK)))
        p.drawText(chip, Qt.AlignmentFlag.AlignCenter, label)

        detail = ""
        if item.status == Status.DONE:
            detail = tr("q.detail", n=item.boxes, ai=item.neural, fast=item.fast)
        elif item.status in (Status.FAILED, Status.NO_TEXT, Status.CANCELED) and item.message:
            detail = item.message
        elif item.status == Status.RUNNING:
            detail = tr("q.working")
        if detail:
            f = T.font_ui(10)
            p.setFont(f)
            p.setPen(QColor(T.INK_SOFT))
            p.drawText(QRectF(tx, frame.top() + 50, tw, 16), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       QFontMetrics(f).elidedText(detail, Qt.TextElideMode.ElideRight, int(tw)))
        if selected:
            p.setPen(QPen(QColor(T.YELLOW), 3.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r.adjusted(-2.5, -2.5, 2.5, 2.5), 10, 10)
        p.restore()


class _Empty(QWidget):
    """Dashed "drop here" box shown while the film is empty."""

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        box = QRectF(6, 6, self.width() - 16, min(self.height() - 16, 230))
        pen = QPen(QColor(T.INK_SOFT), 2.4, Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([4, 4])
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(box, 14, 14)
        ic = QRectF(0, 0, 64, 64)
        ic.moveCenter(QPointF(box.center().x(), box.center().y() - 36))
        draw_pop_box(p, ic, fill=T.BLUE, radius=14, shadow=4, border=2.8, misreg=2, dots=True, dots_hi=0.9)
        draw_icon(p, "batch", ic.adjusted(13, 13, -13, -13), QColor(T.PAPER_HI), 2.2)
        p.setPen(QColor(T.INK))
        p.setFont(T.font_mono(12, True, 1.0))
        p.drawText(QRectF(box.left(), box.center().y() + 8, box.width(), 24), Qt.AlignmentFlag.AlignHCenter, tr("q.empty.title"))
        p.setPen(QColor(T.INK_SOFT))
        p.setFont(T.font_ui(11.5))
        p.drawText(QRectF(box.left() + 16, box.center().y() + 36, box.width() - 32, 70),
                   Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap, tr("q.empty.sub"))


class FilmView(QListView):
    removeRequested = Signal(list)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            rows = sorted({i.row() for i in self.selectedIndexes()})
            if rows:
                self.removeRequested.emit(rows)
                return
        super().keyPressEvent(e)


class QueuePanel(PaperPane):
    addFilesRequested = Signal()
    addFolderRequested = Signal()
    openRequested = Signal(str)

    def __init__(self, model: BatchModel, controller: BatchController, parent: Optional[QWidget] = None):
        super().__init__("left", parent, seed=4)
        self.model, self.controller = model, controller
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 14, 12)
        lay.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(SectionLabel("q.title"))
        head.addStretch(1)
        self.count = Chip("0", T.PAPER_HI)
        head.addWidget(self.count)
        lay.addLayout(head)

        self.view = FilmView()
        self.view.setModel(model)
        self.view.setItemDelegate(FilmDelegate(self.view))
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setFrameShape(QListView.Shape.NoFrame)
        self.view.setMouseTracking(True)
        self.view.setUniformItemSizes(True)
        self.view.viewport().setAutoFillBackground(False)
        self.view.setStyleSheet("QListView { background: transparent; }")
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._menu)
        self.view.doubleClicked.connect(self._open_index)
        self.view.removeRequested.connect(self._remove)
        self.empty = _Empty()
        self.stack_host = QWidget()
        self.stack = QStackedLayout(self.stack_host)
        self.stack.setContentsMargins(0, 0, 0, 0)
        self.stack.addWidget(self.empty)
        self.stack.addWidget(self.view)
        lay.addWidget(self.stack_host, 1)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_add = PopButton("q.add", "paper", "plus", height=36, px=10.5, shadow=3)
        self.btn_folder = PopButton("q.folder", "paper", "folder", height=36, px=10.5, shadow=3)
        self.btn_clear = PopButton("q.clear", "paper", "trash", height=36, px=10.5, shadow=3)
        for b in (self.btn_add, self.btn_folder, self.btn_clear):
            row.addWidget(b, 1)
        lay.addLayout(row)
        self.btn_add.clicked.connect(self.addFilesRequested)
        self.btn_folder.clicked.connect(self.addFolderRequested)
        self.btn_clear.clicked.connect(self._clear)

        model.countsChanged.connect(self.refresh)
        model.modelReset.connect(self.refresh)
        controller.started.connect(self.refresh)
        controller.finished.connect(lambda _s: self.refresh())
        self.refresh()

    def refresh(self) -> None:
        c = self.model.counts()
        total = c["total"]
        running = self.controller.running
        if running:
            done = c["done"] + c["no_text"] + c["failed"] + c["canceled"]
            self.count.set_chip(f"{done}/{total}", T.YELLOW)
        elif total:
            self.count.set_chip(tr("q.files", n=total), T.PAPER_HI if not c["done"] else T.GREEN)
        else:
            self.count.set_chip("0", T.PAPER_HI)
        self.stack.setCurrentWidget(self.view if total else self.empty)
        self.btn_clear.setEnabled(total > 0 and not running)
        self.btn_add.setEnabled(not running)
        self.btn_folder.setEnabled(not running)

    def retranslate(self) -> None:
        self.refresh()
        self.view.viewport().update()
        self.empty.update()

    # --- interaction -----------------------------------------------------------------------------
    def _open_index(self, ix: QModelIndex) -> None:
        it = ix.data(ItemRole)
        if it is not None:
            self.openRequested.emit(str(it.src))

    def _remove(self, rows: list) -> None:
        if not self.controller.running:
            self.model.remove_rows(rows)

    def _clear(self) -> None:
        if not self.controller.running:
            self.model.clear()

    def _menu(self, pos: QPoint) -> None:
        ix = self.view.indexAt(pos)
        if not ix.isValid():
            return
        it: BatchItem = ix.data(ItemRole)
        menu = QMenu(self)
        a_open = menu.addAction(tr("q.open"))
        a_show = menu.addAction(tr("q.show"))
        a_out = menu.addAction(tr("q.show_out")) if it.out_path else None
        menu.addSeparator()
        a_rm = menu.addAction(tr("q.remove"))
        a_rm.setEnabled(not self.controller.running)
        act = menu.exec(self.view.viewport().mapToGlobal(pos))
        if act is None:
            return
        if act is a_open:
            self.openRequested.emit(str(it.src))
        elif act is a_show:
            reveal(it.src)
        elif a_out is not None and act is a_out and it.out_path:
            reveal(it.out_path)
        elif act is a_rm:
            self._remove([r.row() for r in self.view.selectedIndexes()] or [ix.row()])


def reveal(path: Path) -> None:
    """Show a file in the system file manager (Explorer selects it on Windows)."""
    path = Path(path)
    try:
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))
    except Exception:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))
