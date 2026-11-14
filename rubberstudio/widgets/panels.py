"""The left drawer: tool selector, the contextual tool panels, the models card and the history row."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QRectF, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter
from PySide6.QtWidgets import (QFileDialog, QFontComboBox, QHBoxLayout, QLabel, QLineEdit, QScrollArea, QVBoxLayout,
                               QWidget)

from .. import settings as S
from .. import theme as T
from ..controllers import BatchController, ModelController
from ..core.neural import MODELS
from ..editor import ENGINES, BrushSettings, EditorModel, RepairTool, Tool
from ..i18n import tr
from . import dialogs
from .pop import (Chip, HintLabel, PopButton, PopCheck, PopPanel, PopProgress, PopSegmented, PopSlider, SectionLabel,
                  ToolButton)


# ====================================================================== base pieces

class PaperPane(QWidget):
    """White page with a black rule on the side that faces the canvas."""

    def __init__(self, rule: str = "right", parent: Optional[QWidget] = None, thick: float = T.RULE):
        super().__init__(parent)
        self.rule = rule
        self.thick = int(thick)

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(T.PAPER))
        ink = QColor(T.INK)
        w, h, t = self.width(), self.height(), self.thick
        if self.rule == "right":
            p.fillRect(w - t, 0, t, h, ink)
        elif self.rule == "left":
            p.fillRect(0, 0, t, h, ink)
        elif self.rule == "bottom":
            p.fillRect(0, h - t, w, t, ink)
        elif self.rule == "top":
            p.fillRect(0, 0, w, t, ink)


class Rule(QWidget):
    """A plain black line of a fixed thickness."""

    def __init__(self, thickness: int = 2, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(int(thickness))

    def paintEvent(self, e) -> None:
        QPainter(self).fillRect(self.rect(), QColor(T.INK))


def _transparent_scroll(scroll: QScrollArea) -> None:
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.viewport().setAutoFillBackground(False)
    scroll.setStyleSheet("QScrollArea { background: transparent; border: none; } "
                         "QScrollArea > QWidget > QWidget { background: transparent; }")


def _row(*widgets: QWidget, spacing: int = 10) -> QHBoxLayout:
    r = QHBoxLayout()
    r.setSpacing(spacing)
    r.setContentsMargins(0, 0, 0, 0)
    for w in widgets:
        r.addWidget(w, 1)
    return r


# ====================================================================== detection options + fill engine

class DetectOptions(QWidget):
    """Sensitivity, stroke-grow and thorough-mode controls shared by the single-image eraser and the batch tool."""

    def __init__(self, editor: EditorModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.editor = editor
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.sens = PopSlider("er.sens", 0.0, 1.0, editor.erase.sensitivity, step=0.01)
        self.grow = PopSlider("er.grow", 0.0, 0.4, editor.erase.grow, step=0.01)
        self.thorough = PopCheck("er.thorough", editor.erase.thorough)
        lay.addWidget(self.sens)
        lay.addWidget(self.grow)
        lay.addSpacing(4)
        lay.addWidget(self.thorough)
        self.retranslate()
        self.sens.valueChanged.connect(self._sens)
        self.grow.valueChanged.connect(self._grow)
        self.thorough.toggledBool.connect(self._thorough)
        editor.prefsChanged.connect(self.refresh)

    def _sens(self, v: float) -> None:
        self.editor.erase.sensitivity = v
        S.put("erase/sensitivity", v)
        self.editor.prefsChanged.emit()

    def _grow(self, v: float) -> None:
        self.editor.erase.grow = v
        S.put("erase/grow", v)
        self.editor.prefsChanged.emit()

    def _thorough(self, on: bool) -> None:
        self.editor.erase.thorough = on
        S.put("erase/thorough", on)
        self.editor.prefsChanged.emit()

    def retranslate(self) -> None:
        self.thorough.setToolTip(tr("er.thorough.hint"))

    def refresh(self) -> None:
        self.sens.setValue(self.editor.erase.sensitivity)
        self.grow.setValue(self.editor.erase.grow)
        self.thorough.blockSignals(True)
        self.thorough.setChecked(self.editor.erase.thorough)
        self.thorough.blockSignals(False)


class ModelRow(QWidget):
    """One downloadable fill model: name and size, a status chip and the DOWNLOAD / CANCEL / REMOVE buttons."""

    def __init__(self, model: str, models: ModelController, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.model, self.models = model, models
        self.spec = MODELS[model]
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(8)
        self.name = QLabel(f"{self.spec.label.upper()}  ·  {self.spec.size_mb} MB")
        self.name.setFont(T.font_label(11, True, 1.0))
        self.chip = Chip("", "paper")
        top.addWidget(self.name, 1)
        top.addWidget(self.chip)
        lay.addLayout(top)
        self.bar = PopProgress()
        lay.addWidget(self.bar)
        self.btn_dl = PopButton("ai.download", "paper", "download", height=34, px=10.5)
        self.btn_cancel = PopButton("ai.cancel", "paper", "close", height=34, px=10.5)
        self.btn_rm = PopButton("ai.remove", "paper", "trash", height=34, px=10.5)
        row = QHBoxLayout()
        row.setSpacing(8)
        for b in (self.btn_dl, self.btn_cancel, self.btn_rm):
            row.addWidget(b, 1)
        lay.addLayout(row)
        self.btn_dl.clicked.connect(self._download)
        self.btn_cancel.clicked.connect(models.cancel)
        self.btn_rm.clicked.connect(self._remove)

    # --- actions ---------------------------------------------------------------------------------
    def _download(self) -> None:
        host = self.spec.urls[0].split("/")[2] if self.spec.urls else ""
        title = tr("ai.confirm.title", name=self.spec.label.upper())
        body = tr("ai.confirm.body", name=self.spec.label, mb=self.spec.size_mb, host=host)
        if dialogs.confirm(self.window(), title, body, "ai.confirm.ok", icon="download"):
            self.models.download(self.model)

    def _remove(self) -> None:
        title = tr("ai.remove.title", name=self.spec.label.upper())
        if dialogs.confirm(self.window(), title, tr("ai.remove.body", name=self.spec.label), "ai.remove",
                           accent=dialogs.ALERT, icon="trash"):
            self.models.remove(self.model)

    # --- state -----------------------------------------------------------------------------------
    def refresh(self) -> None:
        st = self.models.state(self.model)
        busy_elsewhere = bool(self.models.busy) and st not in ("downloading",)
        if st == "ready":
            self.chip.set_chip(tr("ai.state.ready", mb=int(round(self.models.size_mb(self.model)))), "ink")
        elif st == "downloading":
            self.chip.set_chip(tr("ai.state.downloading", pct=int(round(self.models.progress * 100))), "red")
        elif st == "importing":
            self.chip.set_chip(tr("ai.state.importing"), "red")
        else:
            self.chip.set_chip(tr("ai.state.missing"), "paper")
        self.bar.setVisible(st == "downloading")
        if st == "downloading":
            self.bar.set_progress(self.models.progress, self.models.host)
        self.btn_dl.setVisible(st == "missing")
        self.btn_dl.setEnabled(not busy_elsewhere)
        self.btn_cancel.setVisible(st == "downloading")
        self.btn_rm.setVisible(st == "ready")
        self.btn_rm.setEnabled(not busy_elsewhere)


class EngineChoice(QWidget):
    """The fill-engine switch (Auto / Fast / MI-GAN / LaMa) with a short explanation. It sits inside the erase card and
    the batch detection card, right above the buttons that use it."""

    def __init__(self, editor: EditorModel, models: ModelController, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.editor, self.models = editor, models
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(SectionLabel("eng.title"))
        self.seg = PopSegmented([(e, f"eng.{e}") for e in ENGINES],
                                editor.erase.engine if editor.erase.engine in ENGINES else "auto")
        lay.addWidget(self.seg)
        self.hint = HintLabel("", provider=self._hint_text, px=11)
        lay.addWidget(self.hint)
        self.seg.currentChanged.connect(self._engine)
        models.changed.connect(self.refresh)
        editor.prefsChanged.connect(self.refresh)

    def _engine(self, ident: str) -> None:
        self.editor.erase.engine = ident
        S.put("erase/engine", ident)
        self.editor.prefsChanged.emit()

    def _hint_text(self) -> str:
        eng = self.editor.erase.engine
        if eng in MODELS and self.models.state(eng) == "missing":
            return tr("eng.missing", name=MODELS[eng].label)
        return tr(f"eng.hint.{eng if eng in ENGINES else 'auto'}")

    def refresh(self) -> None:
        eng = self.editor.erase.engine
        if eng in ENGINES:
            self.seg.setCurrent(eng)
        self.hint.retranslate()


class ModelsPanel(PopPanel):
    """The optional fill models: download, import or remove each one."""

    _ready = False              # PopPanel.__init__ retranslates before our widgets exist

    def __init__(self, models: ModelController, parent: Optional[QWidget] = None):
        super().__init__("eng.models", parent)
        self.models = models
        self.body.addWidget(HintLabel("eng.models.hint", px=11))
        self.rows = {m: ModelRow(m, models) for m in ("migan", "lama")}
        for row in self.rows.values():
            self.body.addWidget(row)
        self.btn_imp = PopButton("ai.import", "paper", "folder", height=34, px=10.5)
        self.body.addWidget(self.btn_imp)
        self.btn_imp.clicked.connect(self._import)
        models.changed.connect(self.refresh)
        self._ready = True
        self.refresh()

    def _import(self) -> None:
        start = S.get_str("ai/import_dir", "")
        path, _ = QFileDialog.getOpenFileName(self.window(), tr("ai.import.title"), start, tr("ai.import.filter"))
        if path:
            S.put("ai/import_dir", str(Path(path).parent))
            self.models.import_file(path)

    def refresh(self) -> None:
        if not self._ready:
            return
        for row in self.rows.values():
            row.refresh()
        self.btn_imp.setEnabled(not self.models.busy)

    def retranslate(self) -> None:
        super().retranslate()
        self.refresh()


# ====================================================================== tool pages

class CropPage(PopPanel):
    def __init__(self, editor: EditorModel, parent: Optional[QWidget] = None):
        super().__init__("tool.crop", parent)
        self.editor = editor
        self.body.addWidget(HintLabel("crop.hint"))
        full = PopButton("crop.full", "paper", "fit", height=38, px=11)
        apply_ = PopButton("crop.apply", "red", "check", height=38, px=11)
        self.body.addLayout(_row(full, apply_))
        full.clicked.connect(self._full)
        apply_.clicked.connect(editor.confirm_crop)

    def _full(self) -> None:
        self.editor.crop_rect = (0.0, 0.0, 1.0, 1.0)
        self.editor.overlayChanged.emit()


class RepairPage(PopPanel):
    def __init__(self, editor: EditorModel, parent: Optional[QWidget] = None):
        super().__init__("tool.repair", parent)
        self.editor = editor
        b = editor.brush
        self.seg = PopSegmented([(RepairTool.HEAL.value, "rep.tool.heal"), (RepairTool.SMUDGE.value, "rep.tool.smudge"),
                                 (RepairTool.TEXTURE.value, "rep.tool.texture")], b.tool.value)
        self.body.addWidget(self.seg)
        self.hint = HintLabel("rep.hint.heal")
        self.body.addWidget(self.hint)
        self.size_sl = PopSlider("rep.size", 4, 120, b.size, step=1, decimals=0)
        self.hard = PopSlider("rep.hardness", 0.0, 1.0, b.hardness, step=0.01)
        self.flow = PopSlider("rep.flow", 0.05, 1.0, b.flow, step=0.01)
        self.strength = PopSlider("rep.strength", 0.0, 1.0, b.strength, step=0.01)
        self.smear = PopSlider("rep.smear", 0.1, 1.0, b.smear, step=0.01)
        self.block = PopSlider("rep.block", 2, 40, b.block, step=1, decimals=0)
        for w in (self.size_sl, self.hard, self.flow, self.strength, self.smear, self.block):
            self.body.addWidget(w)
        self.seg.currentChanged.connect(self._tool)
        self.size_sl.valueChanged.connect(lambda v: self._set("size", v))
        self.hard.valueChanged.connect(lambda v: self._set("hardness", v))
        self.flow.valueChanged.connect(lambda v: self._set("flow", v))
        self.strength.valueChanged.connect(lambda v: self._set("strength", v))
        self.smear.valueChanged.connect(lambda v: self._set("smear", v))
        self.block.valueChanged.connect(lambda v: self._set("block", v))
        self._show_for(b.tool)

    def _set(self, name: str, v: float) -> None:
        setattr(self.editor.brush, name, v)
        S.put(f"brush/{name}", v)
        self.editor.overlayChanged.emit()

    def _tool(self, ident: str) -> None:
        t = RepairTool(ident)
        self.editor.brush.tool = t
        S.put("brush/tool", ident)
        self._show_for(t)

    def _show_for(self, t: RepairTool) -> None:
        self.hint.set_key(f"rep.hint.{t.value}")
        self.strength.setVisible(t != RepairTool.SMUDGE)
        self.smear.setVisible(t == RepairTool.SMUDGE)
        self.block.setVisible(t == RepairTool.TEXTURE)
        self.hard.setVisible(True)


class WatermarkPage(PopPanel):
    def __init__(self, editor: EditorModel, parent: Optional[QWidget] = None):
        super().__init__("tool.watermark", parent)
        self.editor = editor
        wm = editor.wm
        self.body.addWidget(SectionLabel("wm.text"))
        self.edit = QLineEdit(wm.text)
        self.edit.setMaxLength(80)
        self.edit.setPlaceholderText(tr("wm.placeholder"))
        self.body.addWidget(self.edit)
        self.body.addWidget(SectionLabel("wm.font"))
        self.font_box = QFontComboBox()
        self.font_box.setFontFilters(QFontComboBox.FontFilter.ScalableFonts)
        self.font_box.setSizeAdjustPolicy(QFontComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.font_box.setMinimumContentsLength(8)
        self.font_box.setCurrentFont(QFont(wm.font))
        self.body.addWidget(self.font_box)
        self.body.addWidget(SectionLabel("wm.color"))
        self.color = PopSegmented([("white", "wm.white"), ("black", "wm.black")], wm.color)
        self.body.addWidget(self.color)
        self.size_sl = PopSlider("wm.size", 0.3, 3.0, wm.size, step=0.05)
        self.opacity = PopSlider("wm.opacity", 0.05, 1.0, wm.opacity, step=0.01)
        self.body.addWidget(self.size_sl)
        self.body.addWidget(self.opacity)
        self.body.addWidget(HintLabel("wm.hint"))
        self.apply = PopButton("wm.apply", "red", "check", height=38, px=11)
        self.body.addWidget(self.apply)
        self.edit.textChanged.connect(self._text)
        self.font_box.currentFontChanged.connect(self._font)
        self.color.currentChanged.connect(self._color)
        self.size_sl.valueChanged.connect(lambda v: self._set("size", v))
        self.opacity.valueChanged.connect(lambda v: self._set("opacity", v))
        self.apply.clicked.connect(editor.apply_watermark)

    def _text(self, t: str) -> None:
        self.editor.wm.text = t
        S.put("wm/text", t)
        self.editor.overlayChanged.emit()

    def _font(self, f: QFont) -> None:
        self.editor.wm.font = f.family()
        S.put("wm/font", f.family())
        self.editor.overlayChanged.emit()

    def _color(self, ident: str) -> None:
        self.editor.wm.color = ident
        S.put("wm/color", ident)
        self.editor.overlayChanged.emit()

    def _set(self, name: str, v: float) -> None:
        setattr(self.editor.wm, name, v)
        S.put(f"wm/{name}", v)
        self.editor.overlayChanged.emit()


class ErasePage(QWidget):
    """ERASE TEXT card (with the fill-engine switch) + the models card below it."""

    def __init__(self, editor: EditorModel, models: ModelController, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.editor = editor
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        card = PopPanel("tool.erase")
        self.scope = PopSegmented([("region", "er.region"), ("whole", "er.whole")],
                                  "whole" if editor.erase.whole_image else "region")
        card.body.addWidget(self.scope)
        self.hint = HintLabel("er.hint.region")
        card.body.addWidget(self.hint)
        # the two actions stay at the top, so they are in reach however short the window is
        self.btn_detect = PopButton("er.detect", "paper", "eye", height=38, px=11)
        self.btn_run = PopButton("er.run", "red", "erase", height=38, px=11)
        card.body.addLayout(_row(self.btn_detect, self.btn_run))
        self.opts = DetectOptions(editor)
        card.body.addWidget(self.opts)
        card.body.addWidget(EngineChoice(editor, models))
        lay.addWidget(card)
        lay.addWidget(ModelsPanel(models))
        self.scope.currentChanged.connect(self._scope)
        self.btn_detect.clicked.connect(editor.detect_text)
        self.btn_run.clicked.connect(editor.erase_text)
        editor.busyChanged.connect(self._busy)
        self._scope(self.scope.current())

    def _scope(self, ident: str) -> None:
        self.editor.erase.whole_image = ident == "whole"
        S.put("erase/whole", self.editor.erase.whole_image)
        self.hint.set_key("er.hint.whole" if self.editor.erase.whole_image else "er.hint.region")
        self.editor.boxes = []
        self.editor.overlayChanged.emit()

    def _busy(self, busy: bool) -> None:
        self.btn_detect.setEnabled(not busy)
        self.btn_run.setEnabled(not busy)


class BatchPage(QWidget):
    """Batch run card, output card, detection card (with the fill-engine switch) and the models card."""
    addFilesRequested = Signal()
    addFolderRequested = Signal()

    def __init__(self, editor: EditorModel, models: ModelController, batch: BatchController,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.editor, self.batch = editor, batch
        s = batch.settings
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        # --- run card -------------------------------------------------------------------------
        run = PopPanel("tool.batch")
        run.body.addWidget(HintLabel("bt.hint"))
        add = PopButton("bt.add", "paper", "plus", height=36, px=11)
        folder = PopButton("bt.folder", "paper", "folder", height=36, px=11)
        run.body.addLayout(_row(add, folder))
        add.clicked.connect(self.addFilesRequested)
        folder.clicked.connect(self.addFolderRequested)
        self.btn_start = PopButton("bt.start", "red", "play", height=42, px=12)
        self.btn_stop = PopButton("bt.stop", "ink", "stop", height=42, px=12)
        run.body.addLayout(_row(self.btn_start, self.btn_stop))
        self.bar = PopProgress()
        run.body.addWidget(self.bar)
        self.summary = HintLabel("", text="")
        run.body.addWidget(self.summary)
        self.btn_open = PopButton("bt.open_out", "paper", "folder", height=36, px=11)
        run.body.addWidget(self.btn_open)
        lay.addWidget(run)

        # --- output card ----------------------------------------------------------------------
        out = PopPanel("bt.out")
        out.body.addWidget(SectionLabel("bt.out.dir"))
        self.dir_edit = QLineEdit(S.get_str("batch/out_dir", ""))
        self.dir_edit.setPlaceholderText(tr("bt.out.same"))
        browse = PopButton("bt.browse", "paper", "folder", height=34, px=11)
        out.body.addWidget(self.dir_edit)
        out.body.addWidget(browse)
        out.body.addWidget(SectionLabel("bt.suffix"))
        self.suffix = QLineEdit(s.suffix)
        self.suffix.setMaxLength(40)
        out.body.addWidget(self.suffix)
        out.body.addWidget(SectionLabel("bt.format"))
        self.fmt = PopSegmented([("same", "bt.fmt.same"), ("png", "bt.fmt.png"), ("jpg", "bt.fmt.jpg"), ("webp", "bt.fmt.webp")],
                                s.fmt)
        out.body.addWidget(self.fmt)
        self.quality = PopSlider("bt.quality", 50, 100, s.quality, step=1, decimals=0)
        out.body.addWidget(self.quality)
        out.body.addWidget(SectionLabel("bt.exists"))
        self.exists = PopSegmented([("rename", "bt.ex.rename"), ("skip", "bt.ex.skip"), ("overwrite", "bt.ex.overwrite")],
                                   s.existing)
        out.body.addWidget(self.exists)
        self.keep = PopCheck("bt.keep", s.keep_structure)
        self.unchanged = PopCheck("bt.unchanged", s.save_unchanged)
        out.body.addWidget(self.keep)
        out.body.addWidget(self.unchanged)
        lay.addWidget(out)

        # --- detection card -------------------------------------------------------------------
        det = PopPanel("bt.detect")
        self.region = PopCheck("bt.region", editor.batch_region_on)
        det.body.addWidget(self.region)
        self.region_hint = HintLabel("bt.region.hint")
        det.body.addWidget(self.region_hint)
        det.body.addWidget(DetectOptions(editor))
        det.body.addWidget(EngineChoice(editor, models))
        max_workers = max(1, min(4, (os.cpu_count() or 2) // 2))
        self.workers = PopSlider("bt.workers", 1, max(2, max_workers), min(s.workers, max_workers), step=1, decimals=0)
        self.workers.setVisible(max_workers > 1)
        det.body.addWidget(self.workers)
        lay.addWidget(det)
        lay.addWidget(ModelsPanel(models))

        # --- wiring ---------------------------------------------------------------------------
        browse.clicked.connect(self._browse)
        self.dir_edit.textChanged.connect(self._dir)
        self.suffix.textChanged.connect(self._suffix)
        self.fmt.currentChanged.connect(self._fmt)
        self.quality.valueChanged.connect(lambda v: self._set("quality", int(v)))
        self.exists.currentChanged.connect(lambda v: self._set("existing", v))
        self.keep.toggledBool.connect(lambda v: self._set("keep_structure", v))
        self.unchanged.toggledBool.connect(lambda v: self._set("save_unchanged", v))
        self.region.toggledBool.connect(self._region)
        self.workers.valueChanged.connect(lambda v: self._set("workers", int(v)))
        self.btn_open.clicked.connect(self.open_output)
        self._fmt(s.fmt)
        self.set_running(False)
        self.btn_open.hide()
        self.summary.hide()
        self.bar.hide()

    # --- settings --------------------------------------------------------------------------------
    def _set(self, name: str, v) -> None:
        setattr(self.batch.settings, name, v)
        S.put(f"batch/{name}", v)

    def _dir(self, text: str) -> None:
        text = text.strip()
        self.batch.settings.out_dir = Path(text) if text else None
        S.put("batch/out_dir", text)

    def _suffix(self, text: str) -> None:
        self.batch.settings.suffix = text
        S.put("batch/suffix", text)

    def _fmt(self, ident: str) -> None:
        self._set("fmt", ident)
        self.quality.setVisible(ident in ("jpg", "webp", "same"))

    def _region(self, on: bool) -> None:
        self.editor.batch_region_on = on
        S.put("batch/region_on", on)
        self.editor.overlayChanged.emit()

    def _browse(self) -> None:
        start = self.dir_edit.text().strip() or S.get_str("batch/last_dir", "")
        d = QFileDialog.getExistingDirectory(self.window(), tr("bt.browse.title"), start)
        if d:
            self.dir_edit.setText(d)

    def open_output(self) -> None:
        d = self.batch.summary.out_dir if self.batch.summary else None
        if d and Path(d).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    # --- run state -------------------------------------------------------------------------------
    def set_running(self, running: bool, pending: int = 0) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_start.setVisible(not running)
        self.btn_stop.setVisible(running)
        if running:
            self.bar.show()
            self.btn_open.hide()

    def set_progress(self, done: int, total: int) -> None:
        self.bar.show()
        self.bar.set_progress(done / total if total else 0.0, f"{done} / {total}")

    def set_summary(self, text: str, out_dir: Optional[Path]) -> None:
        self.summary.setText(text)
        self.summary.show()
        self.btn_open.setVisible(bool(out_dir))

    def set_start_key(self, key: str) -> None:
        self.btn_start.set_key(key)


# ====================================================================== the drawer itself

class HistoryRow(QWidget):
    def __init__(self, editor: EditorModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.editor = editor
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.undo = PopButton("hist.undo", "paper", "undo", height=36, px=10.5)
        self.redo = PopButton("hist.redo", "paper", "redo", height=36, px=10.5)
        self.reset = PopButton("hist.reset", "paper", "reset", height=36, px=10.5)
        for b in (self.undo, self.redo, self.reset):
            lay.addWidget(b, 1)
        self.undo.clicked.connect(editor.undo)
        self.redo.clicked.connect(editor.redo)
        self.reset.clicked.connect(editor.reset)
        editor.historyChanged.connect(self.refresh)
        editor.imageChanged.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        e = self.editor
        self.undo.setEnabled(e.can_undo())
        self.redo.setEnabled(e.can_redo())
        self.reset.setEnabled(e.image is not None and e.can_undo())


class ToolDrawer(PaperPane):
    """TOOLSET buttons, the active tool's panel (scrolls when the window is short) and undo / redo / reset."""
    toolPicked = Signal(str)

    def __init__(self, editor: EditorModel, models: ModelController, batch: BatchController,
                 parent: Optional[QWidget] = None):
        super().__init__("right", parent)
        self.editor = editor
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 18, 10)
        lay.setSpacing(6)
        lay.addWidget(SectionLabel("tool.set"))
        self.buttons: dict[Tool, ToolButton] = {}
        for tool, key, icon in ((Tool.CROP, "tool.crop", "crop"), (Tool.REPAIR, "tool.repair", "repair"),
                                (Tool.WATERMARK, "tool.watermark", "watermark"), (Tool.ERASE, "tool.erase", "erase"),
                                (Tool.BATCH, "tool.batch", "batch")):
            b = ToolButton(key, icon)
            b.clicked.connect(lambda _=False, t=tool: self._clicked(t))
            self.buttons[tool] = b
            lay.addWidget(b)

        self.pages: dict[Tool, QWidget] = {
            Tool.CROP: CropPage(editor),
            Tool.REPAIR: RepairPage(editor),
            Tool.WATERMARK: WatermarkPage(editor),
            Tool.ERASE: ErasePage(editor, models),
            Tool.BATCH: BatchPage(editor, models, batch),
        }
        holder = QWidget()
        holder.setAutoFillBackground(False)
        hl = QVBoxLayout(holder)
        hl.setContentsMargins(0, 8, 6, 8)
        hl.setSpacing(0)
        for page in self.pages.values():
            hl.addWidget(page)
            page.hide()
        hl.addStretch(1)
        self.scroll = QScrollArea()
        _transparent_scroll(self.scroll)
        self.scroll.setWidget(holder)
        lay.addWidget(self.scroll, 1)
        lay.addWidget(Rule(2))          # a card that is cut off by the scroll area must not read as part of the buttons below
        self.history = HistoryRow(editor)
        lay.addWidget(self.history)

        editor.toolChanged.connect(self._tool_changed)
        editor.imageChanged.connect(self._enable)
        self._enable()

    @property
    def batch_page(self) -> BatchPage:
        return self.pages[Tool.BATCH]       # type: ignore[return-value]

    def _clicked(self, tool: Tool) -> None:
        new = Tool.NONE if self.editor.tool == tool else tool
        self.editor.set_tool(new)

    def _tool_changed(self, _name: str) -> None:
        t = self.editor.tool
        for tool, b in self.buttons.items():
            b.setChecked(tool == t)
        for tool, page in self.pages.items():
            page.setVisible(tool == t)
        self.scroll.verticalScrollBar().setValue(0)
        self.toolPicked.emit(t.value)

    def _enable(self) -> None:
        has = self.editor.image is not None
        for tool, b in self.buttons.items():
            b.setEnabled(has or tool == Tool.BATCH)
        if not has and self.editor.tool not in (Tool.NONE, Tool.BATCH):
            self.editor.set_tool(Tool.NONE)
