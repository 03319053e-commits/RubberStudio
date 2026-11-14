"""Header (logo + file actions) and status bar — both fold away inside envelope panels."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QWidget

from .. import __version__, theme as T
from ..i18n import language, tr
from .icons import draw_icon
from .panels import PaperPane
from .pop import PopButton, draw_pop_box

HEADER_H = 62
STATUS_H = 32


class HeaderBar(PaperPane):
    openClicked = Signal()
    pasteClicked = Signal()
    saveClicked = Signal()
    langClicked = Signal()
    helpClicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("bottom", parent, seed=3)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(300, 8, 18, 12)
        lay.setSpacing(12)
        lay.addStretch(1)
        self.btn_open = PopButton("hdr.open", "yellow", "photo", height=38)
        self.btn_paste = PopButton("hdr.paste", "paper", "plus", height=38)
        self.btn_save = PopButton("hdr.save", "green", "export", height=38)
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

    def paintEvent(self, e) -> None:
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        mark = QRectF(18, 9, 42, 42)
        draw_pop_box(p, mark, fill=T.YELLOW, radius=10, shadow=3, border=2.6, misreg=2, dots=True, dots_hi=0.95,
                     ramp_angle=40)
        draw_icon(p, "erase", mark.adjusted(8, 8, -8, -8), QColor(T.INK), 2.3)
        f = T.font_display(25)
        p.setFont(f)
        fm = QFontMetrics(f)
        base_y = 41
        p.setPen(QColor(T.RED))                       # colour plate, slipped against the black plate
        p.drawText(73 + 2, base_y + 2, "RUBBER")
        p.setPen(QColor(T.INK))
        p.drawText(73, base_y, "RUBBER")
        x = 73 + fm.horizontalAdvance("RUBBER") + 10
        f2 = T.font_mono(12, True, 3.2)
        p.setFont(f2)
        p.setPen(QColor(T.INK))
        p.drawText(x, base_y - 1, "STUDIO")
        w = QFontMetrics(f2).horizontalAdvance("STUDIO")
        p.fillRect(QRectF(x, base_y + 4, w + 1, 4), QColor(T.BLUE))


class StatusBar(PaperPane):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("top", parent, seed=3)
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
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        h = self.height()
        dot = QRectF(18, h / 2 - 6, 12, 12)
        draw_pop_box(p, dot, fill=T.YELLOW if self._busy else T.GREEN, radius=3, shadow=0, border=2.0, misreg=1.0)
        p.setPen(QColor(T.INK))
        p.setFont(T.font_mono(11.5, True, 0.3))
        right = f"v{__version__}"
        rw = QFontMetrics(T.font_mono(10.5, False)).horizontalAdvance(right)
        avail = self.width() - 46 - rw - 24
        text = QFontMetrics(p.font()).elidedText(self._text, Qt.TextElideMode.ElideRight, max(40, avail))
        p.drawText(QRectF(40, 0, avail + 4, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
        p.setFont(T.font_mono(10.5, False))
        p.setPen(QColor(T.INK_SOFT))
        p.drawText(QRectF(self.width() - rw - 20, 0, rw + 4, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, right)
