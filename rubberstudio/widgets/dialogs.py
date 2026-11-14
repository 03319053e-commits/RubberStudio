"""Paper-styled message dialogs (confirm / info / error) and the help sheet."""
from __future__ import annotations

from typing import Optional, Sequence

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from .. import theme as T
from ..i18n import tr
from .icons import draw_icon
from .pop import PopButton, dot_color, draw_dots, draw_pop_box, paper_brush

# (translation key, button variant)
OK = (("dlg.ok", "yellow"),)
OK_CANCEL = (("dlg.cancel", "paper"), ("dlg.ok", "yellow"))


class _Banner(QWidget):
    """Coloured, dotted title strip with an icon."""

    def __init__(self, title: str, accent: str, icon: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.title, self.accent, self.icon = title, accent, icon
        self.setFixedHeight(58)

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(0, 0, self.width(), self.height())
        p.fillRect(r.translated(2, 2), QColor(self.accent))
        draw_dots(p, r, dot_color(QColor(self.accent)), ramp_angle=20, lo=0.1, hi=1.0, spacing=6, speckle=0.004)
        p.setPen(QPen(QColor(T.INK), 3.0))
        p.drawLine(0, int(r.bottom()) - 1, self.width(), int(r.bottom()) - 1)
        chip = QRectF(16, 11, 36, 36)
        draw_pop_box(p, chip, fill=T.PAPER_HI, radius=9, shadow=3, border=2.4, misreg=1.2)
        draw_icon(p, self.icon, chip.adjusted(7, 7, -7, -7), QColor(T.INK), 2.0)
        p.setFont(T.font_mono(14, True, 1.2))
        p.setPen(QColor(T.TEXT_ON.get(self.accent, T.INK)))
        p.drawText(QRectF(64, 0, self.width() - 80, self.height() - 3), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   self.title)


class PopDialog(QDialog):
    def __init__(self, parent: Optional[QWidget], title: str, body: str, *,
                 buttons: Sequence[tuple[str, str]] = OK, accent: str = T.YELLOW, icon: str = "info",
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
        inner.setContentsMargins(22, 18, 22, 18)
        inner.setSpacing(12)
        outer.addLayout(inner)
        lab = QLabel(body)
        lab.setWordWrap(True)
        lab.setFont(T.font_ui(13))
        lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        inner.addWidget(lab)
        if detail:
            d = QLabel(detail)
            d.setWordWrap(True)
            d.setFont(T.font_mono(10.5, False, 0.2))
            d.setStyleSheet(f"color:{T.INK_SOFT}; background:{T.PAPER_HI}; border:2px solid {T.INK}; border-radius:7px; padding:8px;")
            d.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            inner.addWidget(d)
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addStretch(1)
        for i, (key, variant) in enumerate(buttons):
            b = PopButton(key, variant, parent=self, height=38, shadow=4)
            b.clicked.connect(lambda _=False, idx=i: self.done(idx + 1))
            row.addWidget(b)
            if i == len(buttons) - 1:
                b.setDefault(True) if hasattr(b, "setDefault") else None
        row.setContentsMargins(0, 8, 4, 4)
        inner.addLayout(row)

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), paper_brush(T.PAPER, 3))

    @staticmethod
    def choose(parent: Optional[QWidget], title: str, body: str, buttons: Sequence[tuple[str, str]] = OK,
               accent: str = T.YELLOW, icon: str = "info", detail: str = "") -> int:
        """Run the dialog; returns the 0-based index of the clicked button, ``-1`` when it was just closed."""
        dlg = PopDialog(parent, title, body, buttons=buttons, accent=accent, icon=icon, detail=detail)
        return dlg.exec() - 1


def confirm(parent: Optional[QWidget], title: str, body: str, ok_key: str = "dlg.ok", *, accent: str = T.YELLOW,
            icon: str = "info", detail: str = "") -> bool:
    return PopDialog.choose(parent, title, body, (("dlg.cancel", "paper"), (ok_key, "yellow")), accent, icon, detail) == 1


def notify(parent: Optional[QWidget], title: str, body: str, *, accent: str = T.YELLOW, icon: str = "info",
           detail: str = "") -> None:
    PopDialog.choose(parent, title, body, OK, accent, icon, detail)


def error(parent: Optional[QWidget], body: str, detail: str = "") -> None:
    notify(parent, tr("dlg.error"), body, accent=T.RED, icon="close", detail=detail)


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
        outer.addWidget(_Banner(tr("help.title"), T.BLUE, "info", self))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        holder = QWidget()
        holder.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(22, 16, 22, 16)
        lay.setSpacing(6)
        for key, color in (("crop", T.YELLOW), ("repair", T.GREEN), ("watermark", T.PINK), ("erase", T.RED), ("batch", T.BLUE),
                           ("keys", T.INK)):
            h = QLabel(tr(f"help.{key}.h"))
            h.setFont(T.font_mono(12, True, 1.0))
            h.setStyleSheet(f"color:{T.INK}; background:{color if color != T.INK else T.PAPER_LO}; "
                            f"border:2px solid {T.INK}; border-radius:6px; padding:3px 9px;"
                            + (f" color:{T.PAPER_HI}; background:{T.INK};" if color == T.BLUE else ""))
            h.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            lay.addSpacing(8)
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
        ok = PopButton("dlg.ok", "yellow", parent=self, height=38)
        ok.clicked.connect(self.accept)
        row.addWidget(ok)
        outer.addLayout(row)

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), paper_brush(T.PAPER, 3))
