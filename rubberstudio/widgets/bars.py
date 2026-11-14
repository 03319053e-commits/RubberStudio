"""Header (mark, wordmark, file actions) and status bar — both fold away inside envelope panels."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter
from PySide6.QtWidgets import QHBoxLayout, QWidget

from .. import __version__, theme as T
from ..i18n import language, tr
from .icons import draw_icon
from .panels import PaperPane
from .pop import PopButton

HEADER_H = 62
STATUS_H = 32


class HeaderBar(PaperPane):
    openClicked = Signal()
    pasteClicked = Signal()
    saveClicked = Signal()
    langClicked = Signal()
    helpClicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("bottom", parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(300, 8, 18, 12)
        lay.setSpacing(10)
        lay.addStretch(1)
        self.btn_open = PopButton("hdr.open", "red", "photo", height=38)
        self.btn_paste = PopButton("hdr.paste", "paper", "plus", height=38)
        self.btn_save = PopButton("hdr.save", "ink", "export", height=38)
        self.btn_lang = PopButton("", "paper", "globe", text=self._lang_label(), height=38)
        self.btn_help = PopButton("", "paper", "info", text="", height=38)
        self.btn_help.setFixedWidth(46)
        for b in (self.btn_open, self.btn_paste, self.btn_save, self.btn_lang, self.btn_help):
            lay.addWidget(b)
        self.btn_open.clicked.connect(self.openClicked)
        self.btn_paste.clicked.connect(self.pasteClicked)
        self.btn_save.clicked.connect(self.saveClicked)
        self.btn_lang.clicked.connect(self.langClicked)
        self.btn_help.clicked.connect(self.helpClicked)
        self.retranslate()

    @staticmethod
    def _lang_label() -> str:
        return "EN" if language() == "zh" else "中文"

    def retranslate(self) -> None:
        self.btn_lang._literal = self._lang_label()
        self.btn_lang.retranslate()
        self.btn_help.setToolTip(tr("hdr.help"))
        self.btn_lang.setToolTip(tr("hdr.lang"))
        self.btn_paste.setToolTip(tr("hdr.paste.tip"))
        self.update()

    def paintEvent(self, e) -> None:
        super().paintEvent(e)
        p = QPainter(self)
        mark = QRectF(18, 10, 40, 40)                  # a red square with the eraser: the whole logo
        p.fillRect(mark, QColor(T.RED))
        draw_icon(p, "erase", mark.adjusted(8, 8, -8, -8), QColor(T.PAPER), 2.2)
        f = T.font_display(25)
        p.setFont(f)
        p.setPen(QColor(T.INK))
        p.drawText(72, 35, "RUBBER STUDIO")
        p.setFont(T.font_label(9.5, True, 2.2))
        p.setPen(QColor(T.INK_SOFT))
        p.drawText(73, 51, tr("splash.tag"))


class StatusBar(PaperPane):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("top", parent, thick=2)
        self._text = ""
        self._busy = False

    def set_text(self, text: str) -> None:
        self._text = text
        self.update()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.update()

    def paintEvent(self, e) -> None:
        super().paintEvent(e)
        p = QPainter(self)
        h = self.height()
        p.fillRect(QRectF(18, h / 2 - 5 + 1, 10, 10), QColor(T.RED if self._busy else T.INK))
        p.setPen(QColor(T.INK))
        p.setFont(T.font_ui(12, True))
        right = f"V{__version__}"
        rf = T.font_label(10, True, 1.2)
        rw = QFontMetrics(rf).horizontalAdvance(right)
        avail = self.width() - 46 - rw - 24
        text = QFontMetrics(p.font()).elidedText(self._text, Qt.TextElideMode.ElideRight, max(40, avail))
        p.drawText(QRectF(40, 1, avail + 4, h - 1), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
        p.setFont(rf)
        p.setPen(QColor(T.INK_SOFT))
        p.drawText(QRectF(self.width() - rw - 20, 1, rw + 4, h - 1), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, right)
