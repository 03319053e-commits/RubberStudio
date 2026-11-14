"""The right-hand queue — the batch list drawn as plain rows: status bar, thumbnail, name, status, detail."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QModelIndex, QPoint, QRectF, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFontMetrics, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QListView, QMenu, QStackedLayout, QStyle,
                               QStyledItemDelegate, QVBoxLayout, QWidget)

from .. import theme as T
from ..controllers import BatchController, BatchModel, ItemRole, ThumbRole
from ..core.batch import BatchItem, Status
from ..i18n import tr
from .icons import draw_icon
from .panels import PaperPane
from .pop import Chip, PopButton, SectionLabel, draw_box

ROW_H = 94

STATUS_LOOK = {                       # status -> (translation key, chip tone)
    Status.QUEUED: ("q.st.queued", "paper"),
    Status.RUNNING: ("q.st.running", "red"),
    Status.DONE: ("q.st.done", "ink"),
    Status.NO_TEXT: ("q.st.no_text", "paper"),
    Status.FAILED: ("q.st.failed", "red"),
    Status.CANCELED: ("q.st.canceled", "paper"),
}
_BAR = {"red": T.RED, "ink": T.INK, "paper": T.INK_FAINT}       # colour of the status bar at the row's left edge


class FilmDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index) -> QSize:
        return QSize(option.rect.width(), ROW_H)

    def paint(self, p: QPainter, option, index: QModelIndex) -> None:
        item: Optional[BatchItem] = index.data(ItemRole)
        if item is None:
            return
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        key, tone = STATUS_LOOK[item.status]
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        r = QRectF(option.rect).adjusted(2, 3, -8, -3)
        inner = draw_box(p, r, fill=T.PAPER_LO if hover and not selected else T.PAPER_HI, border=3 if selected else 2,
                         border_color=T.RED if selected else T.INK)
        p.fillRect(QRectF(inner.left(), inner.top(), 6, inner.height()), QColor(_BAR[tone]))

        # thumbnail
        th = QRectF(inner.left() + 14, inner.top() + 8, 68, inner.height() - 16)
        p.fillRect(th, QColor(T.PAPER_LO))
        img = index.data(ThumbRole)
        if img is not None and not img.isNull():
            s = min(th.width() / img.width(), th.height() / img.height())
            w, h = img.width() * s, img.height() * s
            tr_ = QRectF(th.center().x() - w / 2, th.center().y() - h / 2, w, h)
            p.drawImage(tr_, img)
            p.setPen(QPen(QColor(T.INK), 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(tr_.adjusted(0.5, 0.5, -0.5, -0.5))
        else:
            draw_icon(p, "photo", th.adjusted(18, 14, -18, -14), QColor(T.INK_FAINT), 1.6)

        # text
        tx = th.right() + 12
        tw = inner.right() - 8 - tx
        f = T.font_ui(11.5, True)
        p.setFont(f)
        p.setPen(QColor(T.INK))
        name = QFontMetrics(f).elidedText(item.src.name, Qt.TextElideMode.ElideMiddle, int(tw))
        p.drawText(QRectF(tx, inner.top() + 7, tw, 18), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, name)

        label = tr(key)
        pf = T.font_label(9, True, 0.8)
        cw = QFontMetrics(pf).horizontalAdvance(label) + 16
        chip = QRectF(tx, inner.top() + 30, min(cw, tw), 18)
        fill, text, edge = Chip.TONES[tone]
        draw_box(p, chip, fill=fill, border=1.5, border_color=edge)
        p.setFont(pf)
        p.setPen(QColor(text))
        p.drawText(chip, Qt.AlignmentFlag.AlignCenter, label)

        detail = ""
        if item.status == Status.DONE:
            detail = tr("q.detail", n=item.boxes, ai=item.neural, fast=item.fast)
        elif item.status in (Status.FAILED, Status.NO_TEXT, Status.CANCELED) and item.message:
            detail = item.message
        elif item.status == Status.RUNNING:
            detail = tr("q.working")
        if detail:
            f = T.font_ui(10.5)
            p.setFont(f)
            p.setPen(QColor(T.INK_SOFT))
            p.drawText(QRectF(tx, inner.top() + 52, tw, 16), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       QFontMetrics(f).elidedText(detail, Qt.TextElideMode.ElideRight, int(tw)))
        p.restore()


class _Empty(QWidget):
    """Dashed "drop here" box shown while the queue is empty."""

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        box = QRectF(6, 6, self.width() - 16, min(self.height() - 16, 230))
        pen = QPen(QColor(T.INK), 1.5, Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([4, 3])
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(box)
        ic = QRectF(0, 0, 56, 56)
        ic.moveCenter(box.center())
        ic.moveTop(box.top() + 28)
        p.fillRect(ic, QColor(T.INK))
        draw_icon(p, "batch", ic.adjusted(12, 12, -12, -12), QColor(T.PAPER), 2.0)
        p.setPen(QColor(T.INK))
        p.setFont(T.font_label(11.5, True, 1.2))
        p.drawText(QRectF(box.left(), ic.bottom() + 14, box.width(), 22), Qt.AlignmentFlag.AlignHCenter, tr("q.empty.title"))
        p.setPen(QColor(T.INK_SOFT))
        p.setFont(T.font_ui(11.5))
        p.drawText(QRectF(box.left() + 16, ic.bottom() + 40, box.width() - 32, 70),
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
        super().__init__("left", parent)
        self.model, self.controller = model, controller
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 14, 12)
        lay.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(SectionLabel("q.title"))
        head.addStretch(1)
        self.count = Chip("0", "paper")
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
        self.btn_add = PopButton("q.add", "paper", "plus", height=36, px=10.5)
        self.btn_folder = PopButton("q.folder", "paper", "folder", height=36, px=10.5)
        self.btn_clear = PopButton("q.clear", "paper", "trash", height=36, px=10.5)
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
            self.count.set_chip(f"{done}/{total}", "red")
        elif total:
            self.count.set_chip(tr("q.files", n=total), "paper" if not c["done"] else "ink")
        else:
            self.count.set_chip("0", "paper")
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
