"""Message dialogs (confirm / info / error) and the help sheet — white page, black banner, one red button."""
from __future__ import annotations

from typing import Optional, Sequence

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFontDatabase, QPainter
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from .. import theme as T
from ..i18n import tr
from .icons import draw_icon
from .pop import PopButton

INFO = T.INK            # banner colour of an ordinary message
ALERT = T.RED           # banner colour of an error

# (translation key, button variant)
OK = (("dlg.ok", "red"),)
OK_CANCEL = (("dlg.cancel", "paper"), ("dlg.ok", "red"))


class _Banner(QWidget):
    """Solid title strip: white bold capitals and an icon on black (red for errors)."""

    def __init__(self, title: str, accent: str, icon: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.title, self.accent, self.icon = title, accent, icon
        self.setFixedHeight(54)

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        r = QRectF(self.rect())
        p.fillRect(r, QColor(T.RED if self.accent == T.RED else T.INK))
        draw_icon(p, self.icon, QRectF(18, (r.height() - 22) / 2, 22, 22), QColor(T.PAPER), 2.0)
        p.setFont(T.font_label(13.5, True, 1.6))
        p.setPen(QColor(T.PAPER))
        label = p.fontMetrics().elidedText(self.title, Qt.TextElideMode.ElideRight, max(40, int(r.width() - 66)))
        p.drawText(QRectF(54, 0, r.width() - 66, r.height()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)


def _page(w: QWidget) -> None:
    p = QPainter(w)
    p.fillRect(w.rect(), QColor(T.PAPER))


class PopDialog(QDialog):
    def __init__(self, parent: Optional[QWidget], title: str, body: str, *,
                 buttons: Sequence[tuple[str, str]] = OK, accent: str = INFO, icon: str = "info",
                 detail: str = "", width: int = 460):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(width)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(_Banner(title, accent, icon, self))
        inner = QVBoxLayout()
        inner.setContentsMargins(22, 20, 22, 18)
        inner.setSpacing(14)
        outer.addLayout(inner)
        lab = QLabel(body)
        lab.setWordWrap(True)
        lab.setFont(T.font_ui(13))
        lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        inner.addWidget(lab)
        if detail:
            d = QLabel(detail)
            d.setWordWrap(True)
            f = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
            f.setPixelSize(11)
            d.setFont(f)
            d.setStyleSheet(f"color:{T.INK_SOFT}; background:{T.PAPER_LO}; border:0; padding:9px;")
            d.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            inner.addWidget(d)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addStretch(1)
        for i, (key, variant) in enumerate(buttons):
            b = PopButton(key, variant, parent=self, height=38)
            b.clicked.connect(lambda _=False, idx=i: self.done(idx + 1))
            row.addWidget(b)
        row.setContentsMargins(0, 8, 0, 0)
        inner.addLayout(row)

    def paintEvent(self, e) -> None:
        _page(self)

    @staticmethod
    def choose(parent: Optional[QWidget], title: str, body: str, buttons: Sequence[tuple[str, str]] = OK,
               accent: str = INFO, icon: str = "info", detail: str = "") -> int:
        """Run the dialog; returns the 0-based index of the clicked button, ``-1`` when it was just closed."""
        dlg = PopDialog(parent, title, body, buttons=buttons, accent=accent, icon=icon, detail=detail)
        return dlg.exec() - 1


def confirm(parent: Optional[QWidget], title: str, body: str, ok_key: str = "dlg.ok", *, accent: str = INFO,
            icon: str = "info", detail: str = "") -> bool:
    return PopDialog.choose(parent, title, body, (("dlg.cancel", "paper"), (ok_key, "red")), accent, icon, detail) == 1


def notify(parent: Optional[QWidget], title: str, body: str, *, accent: str = INFO, icon: str = "info",
           detail: str = "") -> None:
    PopDialog.choose(parent, title, body, OK, accent, icon, detail)


def error(parent: Optional[QWidget], body: str, detail: str = "") -> None:
    notify(parent, tr("dlg.error"), body, accent=ALERT, icon="close", detail=detail)


class HelpDialog(QDialog):
    """Scrolling how-to sheet."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(tr("help.title"))
        self.resize(560, 640)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(_Banner(tr("help.title"), INFO, "info", self))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        holder = QWidget()
        holder.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(22, 10, 22, 16)
        lay.setSpacing(6)
        for key in ("crop", "repair", "watermark", "erase", "batch", "keys"):
            h = QLabel(tr(f"help.{key}.h"))
            h.setFont(T.font_label(11.5, True, 1.4))
            h.setStyleSheet(f"color:{T.INK}; background:transparent; border-top:3px solid {T.INK}; padding-top:6px;")
            lay.addSpacing(10)
            lay.addWidget(h)
            body = QLabel(tr(f"help.{key}.b"))
            body.setWordWrap(True)
            body.setFont(T.font_ui(12.5))
            body.setTextFormat(Qt.TextFormat.PlainText)
            lay.addWidget(body)
        lay.addStretch(1)
        scroll.setWidget(holder)
        outer.addWidget(scroll, 1)
        row = QHBoxLayout()
        row.setContentsMargins(22, 8, 22, 16)
        row.addStretch(1)
        ok = PopButton("dlg.ok", "red", parent=self, height=38)
        ok.clicked.connect(self.accept)
        row.addWidget(ok)
        outer.addLayout(row)

    def paintEvent(self, e) -> None:
        _page(self)
