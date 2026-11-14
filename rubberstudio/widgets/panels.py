"""The left drawer: tool selector, the contextual tool panels, the AI-model card and the history row."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QRectF, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter, QPen
from PySide6.QtWidgets import (QFileDialog, QFontComboBox, QHBoxLayout, QLabel, QLineEdit, QScrollArea, QSizePolicy,
                               QVBoxLayout, QWidget)

from .. import settings as S
from .. import theme as T
from ..controllers import BatchController, ModelController
from ..editor import BrushSettings, EditorModel, RepairTool, Tool
from ..i18n import tr
from . import dialogs
from .pop import (Chip, HintLabel, PopButton, PopCheck, PopPanel, PopProgress, PopSegmented, PopSlider, SectionLabel,
                  ToolButton, paper_brush)


MODEL_MB = 208      # size of the default LaMa ONNX file, shown in the download confirmation

# ====================================================================== base pieces

class PaperPane(QWidget):
    """Paper background with an ink rule on the side that faces the canvas."""

    def __init__(self, rule: str = "right", parent: Optional[QWidget] = None, seed: int = 2):
        super().__init__(parent)
        self.rule = rule
        self.seed = seed

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), paper_brush(T.PAPER, self.seed))
        pen = QPen(QColor(T.INK), 3.0)
        p.setPen(pen)
        w, h = self.width(), self.height()
        if self.rule == "right":
            p.drawLine(w - 2, 0, w - 2, h)
        elif self.rule == "left":
            p.drawLine(1, 0, 1, h)
        elif self.rule == "bottom":
            p.drawLine(0, h - 2, w, h - 2)
        elif self.rule == "top":
            p.drawLine(0, 1, w, 1)


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


# ====================================================================== detection options + AI model

class DetectOptions(QWidget):
    """Sensitivity and stroke-grow sliders shared by the single-image eraser and the batch tool."""

    def __init__(self, editor: EditorModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.editor = editor
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.sens = PopSlider("er.sens", 0.0, 1.0, editor.erase.sensitivity, step=0.01, color=T.RED)
        self.grow = PopSlider("er.grow", 0.0, 0.4, editor.erase.grow, step=0.01, color=T.RED)
        lay.addWidget(self.sens)
        lay.addWidget(self.grow)
        self.sens.valueChanged.connect(self._sens)
        self.grow.valueChanged.connect(self._grow)
        editor.prefsChanged.connect(self.refresh)

    def _sens(self, v: float) -> None:
        self.editor.erase.sensitivity = v
        S.put("erase/sensitivity", v)
        self.editor.prefsChanged.emit()

    def _grow(self, v: float) -> None:
        self.editor.erase.grow = v
        S.put("erase/grow", v)
        self.editor.prefsChanged.emit()

    def refresh(self) -> None:
        self.sens.setValue(self.editor.erase.sensitivity)
        self.grow.setValue(self.editor.erase.grow)


class AiModelPanel(PopPanel):
    """Optional neural inpainting model: download (after a confirmation), import, remove, use."""

    def __init__(self, editor: EditorModel, models: ModelController, parent: Optional[QWidget] = None):
        super().__init__("ai.title", T.BLUE, parent)
        self.editor, self.models = editor, models
        top = QHBoxLayout()
        top.setSpacing(8)
        self.chip = Chip("", T.PAPER_HI)
        top.addWidget(self.chip)
        top.addStretch(1)
        self.body.addLayout(top)
        self.hint = HintLabel("ai.hint")
        self.body.addWidget(self.hint)
        self.bar = PopProgress(T.BLUE)
        self.body.addWidget(self.bar)
        self.btn_dl = PopButton("ai.download", "blue", "download", height=36, px=11)
        self.btn_cancel = PopButton("ai.cancel", "paper", "close", height=36, px=11)
        self.btn_imp = PopButton("ai.import", "paper", "folder", height=36, px=11)
        self.btn_rm = PopButton("ai.remove", "paper", "trash", height=36, px=11)
        self.body.addLayout(_row(self.btn_dl, self.btn_cancel))
        self.body.addLayout(_row(self.btn_imp, self.btn_rm))
        self.use = PopCheck("ai.use", editor.erase.use_ai, accent=T.BLUE)
        self.always = PopCheck("ai.always", editor.erase.always_ai, accent=T.BLUE)
        self.body.addWidget(self.use)
        self.body.addWidget(self.always)

        self.btn_dl.clicked.connect(self._download)
        self.btn_cancel.clicked.connect(models.cancel)
        self.btn_imp.clicked.connect(self._import)
        self.btn_rm.clicked.connect(self._remove)
        self.use.toggledBool.connect(self._use)
        self.always.toggledBool.connect(self._always)
        models.changed.connect(self.refresh)
        editor.prefsChanged.connect(self.refresh)
        self._ready = True
        self.refresh()

    # --- actions ---------------------------------------------------------------------------------
    def _download(self) -> None:
        body = tr("ai.confirm.body", mb=MODEL_MB, host="huggingface.co")
        if dialogs.confirm(self.window(), tr("ai.confirm.title"), body, "ai.confirm.ok", accent=T.BLUE, icon="download"):
            self.models.download()

    def _import(self) -> None:
        start = S.get_str("ai/import_dir", "")
        path, _ = QFileDialog.getOpenFileName(self.window(), tr("ai.import.title"), start, tr("ai.import.filter"))
        if path:
            S.put("ai/import_dir", str(Path(path).parent))
            self.models.import_file(path)

    def _remove(self) -> None:
        if dialogs.confirm(self.window(), tr("ai.remove.title"), tr("ai.remove.body"), "ai.remove", accent=T.RED, icon="trash"):
            self.models.remove()
            self.editor.erase.always_ai = False
            self.editor.prefsChanged.emit()

    def _use(self, on: bool) -> None:
        self.editor.erase.use_ai = on
        if not on:
            self.editor.erase.always_ai = False
        S.put("erase/use_ai", on)
        S.put("erase/always_ai", self.editor.erase.always_ai)
        self.editor.prefsChanged.emit()

    def _always(self, on: bool) -> None:
        self.editor.erase.always_ai = on
        if on:
            self.editor.erase.use_ai = True
        S.put("erase/use_ai", self.editor.erase.use_ai)
        S.put("erase/always_ai", on)
        self.editor.prefsChanged.emit()

    # --- state -----------------------------------------------------------------------------------
    def refresh(self) -> None:
        st = self.models.state
        ready = st == "ready"
        busy = st in ("downloading", "importing")
        if st == "ready":
            self.chip.set_chip(tr("ai.state.ready", mb=int(round(self.models.size_mb()))), T.GREEN)
        elif st == "downloading":
            pct = int(round(self.models.progress * 100))
            self.chip.set_chip(tr("ai.state.downloading", pct=pct), T.YELLOW)
        elif st == "importing":
            self.chip.set_chip(tr("ai.state.importing"), T.YELLOW)
        else:
            self.chip.set_chip(tr("ai.state.missing"), T.PAPER_HI)
        self.bar.setVisible(st == "downloading")
        if st == "downloading":
            self.bar.set_progress(self.models.progress, self.models.host)
        self.btn_dl.setVisible(not ready and not busy)
        self.btn_cancel.setVisible(st == "downloading")
        self.btn_imp.setVisible(not ready and not busy)
        self.btn_rm.setVisible(ready)
        for cb, val, enabled in ((self.use, self.editor.erase.use_ai, ready),
                                 (self.always, self.editor.erase.always_ai, ready and self.editor.erase.use_ai)):
            cb.blockSignals(True)
            cb.setChecked(bool(val and enabled))
            cb.blockSignals(False)
            cb.setEnabled(enabled)

    def retranslate(self) -> None:
        super().retranslate()
        if getattr(self, "_ready", False):            # PopPanel.__init__ calls this before our attributes exist
            self.refresh()


# ====================================================================== tool pages

class CropPage(PopPanel):
    def __init__(self, editor: EditorModel, parent: Optional[QWidget] = None):
        super().__init__("tool.crop", T.YELLOW, parent)
        self.editor = editor
        self.body.addWidget(HintLabel("crop.hint"))
        full = PopButton("crop.full", "paper", "fit", height=38, px=11)
        apply_ = PopButton("crop.apply", "yellow", "check", height=38, px=11)
        self.body.addLayout(_row(full, apply_))
        full.clicked.connect(self._full)
        apply_.clicked.connect(editor.confirm_crop)

    def _full(self) -> None:
        self.editor.crop_rect = (0.0, 0.0, 1.0, 1.0)
        self.editor.overlayChanged.emit()


class RepairPage(PopPanel):
    def __init__(self, editor: EditorModel, parent: Optional[QWidget] = None):
        super().__init__("tool.repair", T.GREEN, parent)
        self.editor = editor
        b = editor.brush
        self.seg = PopSegmented([(RepairTool.HEAL.value, "rep.tool.heal"), (RepairTool.SMUDGE.value, "rep.tool.smudge"),
                                 (RepairTool.TEXTURE.value, "rep.tool.texture")], b.tool.value, T.GREEN)
        self.body.addWidget(self.seg)
        self.hint = HintLabel("rep.hint.heal")
        self.body.addWidget(self.hint)
        self.size_sl = PopSlider("rep.size", 4, 120, b.size, step=1, decimals=0, color=T.GREEN)
        self.hard = PopSlider("rep.hardness", 0.0, 1.0, b.hardness, step=0.01, color=T.GREEN)
        self.flow = PopSlider("rep.flow", 0.05, 1.0, b.flow, step=0.01, color=T.GREEN)
        self.strength = PopSlider("rep.strength", 0.0, 1.0, b.strength, step=0.01, color=T.GREEN)
        self.smear = PopSlider("rep.smear", 0.1, 1.0, b.smear, step=0.01, color=T.GREEN)
        self.block = PopSlider("rep.block", 2, 40, b.block, step=1, decimals=0, color=T.GREEN)
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
        super().__init__("tool.watermark", T.PINK, parent)
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
        self.color = PopSegmented([("white", "wm.white"), ("black", "wm.black")], wm.color, T.PINK)
        self.body.addWidget(self.color)
        self.size_sl = PopSlider("wm.size", 0.3, 3.0, wm.size, step=0.05, color=T.PINK)
        self.opacity = PopSlider("wm.opacity", 0.05, 1.0, wm.opacity, step=0.01, color=T.PINK)
        self.body.addWidget(self.size_sl)
        self.body.addWidget(self.opacity)
        self.body.addWidget(HintLabel("wm.hint"))
        self.apply = PopButton("wm.apply", "pink", "check", height=38, px=11)
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
    """ERASE TEXT card + the AI model card below it."""

    def __init__(self, editor: EditorModel, models: ModelController, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.editor = editor
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        card = PopPanel("tool.erase", T.RED)
        self.scope = PopSegmented([("region", "er.region"), ("whole", "er.whole")],
                                  "whole" if editor.erase.whole_image else "region", T.RED)
        card.body.addWidget(self.scope)
        self.hint = HintLabel("er.hint.region")
        card.body.addWidget(self.hint)
        self.opts = DetectOptions(editor)
        card.body.addWidget(self.opts)
        self.btn_detect = PopButton("er.detect", "paper", "eye", height=38, px=11)
        self.btn_run = PopButton("er.run", "red", "erase", height=38, px=11)
        card.body.addLayout(_row(self.btn_detect, self.btn_run))
        lay.addWidget(card)
        lay.addWidget(AiModelPanel(editor, models))
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
    """Batch settings card, run controls and the AI model card."""
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
        run = PopPanel("tool.batch", T.BLUE)
        run.body.addWidget(HintLabel("bt.hint"))
        add = PopButton("bt.add", "paper", "plus", height=36, px=11)
        folder = PopButton("bt.folder", "paper", "folder", height=36, px=11)
        run.body.addLayout(_row(add, folder))
        add.clicked.connect(self.addFilesRequested)
        folder.clicked.connect(self.addFolderRequested)
        self.btn_start = PopButton("bt.start", "blue", "play", height=42, px=12)
        self.btn_stop = PopButton("bt.stop", "red", "stop", height=42, px=12)
        run.body.addLayout(_row(self.btn_start, self.btn_stop))
        self.bar = PopProgress(T.BLUE)
        run.body.addWidget(self.bar)
        self.summary = HintLabel("", text="")
        run.body.addWidget(self.summary)
        self.btn_open = PopButton("bt.open_out", "paper", "folder", height=36, px=11)
        run.body.addWidget(self.btn_open)
        lay.addWidget(run)

        # --- output card ----------------------------------------------------------------------
        out = PopPanel("bt.out", T.YELLOW)
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
                                s.fmt, T.YELLOW)
        out.body.addWidget(self.fmt)
        self.quality = PopSlider("bt.quality", 50, 100, s.quality, step=1, decimals=0, color=T.YELLOW)
        out.body.addWidget(self.quality)
        out.body.addWidget(SectionLabel("bt.exists"))
        self.exists = PopSegmented([("rename", "bt.ex.rename"), ("skip", "bt.ex.skip"), ("overwrite", "bt.ex.overwrite")],
                                   s.existing, T.YELLOW)
        out.body.addWidget(self.exists)
        self.keep = PopCheck("bt.keep", s.keep_structure, accent=T.YELLOW)
        self.unchanged = PopCheck("bt.unchanged", s.save_unchanged, accent=T.YELLOW)
        out.body.addWidget(self.keep)
        out.body.addWidget(self.unchanged)
        lay.addWidget(out)

        # --- detection card -------------------------------------------------------------------
        det = PopPanel("bt.detect", T.RED)
        self.region = PopCheck("bt.region", editor.batch_region_on, accent=T.RED)
        det.body.addWidget(self.region)
        self.region_hint = HintLabel("bt.region.hint")
        det.body.addWidget(self.region_hint)
        det.body.addWidget(DetectOptions(editor))
        max_workers = max(1, min(4, (os.cpu_count() or 2) // 2))
        self.workers = PopSlider("bt.workers", 1, max(2, max_workers), min(s.workers, max_workers), step=1, decimals=0,
                                 color=T.RED)
        self.workers.setVisible(max_workers > 1)
        det.body.addWidget(self.workers)
        lay.addWidget(det)
        lay.addWidget(AiModelPanel(editor, models))

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
        self.undo = PopButton("hist.undo", "paper", "undo", height=36, px=10.5, shadow=3)
        self.redo = PopButton("hist.redo", "paper", "redo", height=36, px=10.5, shadow=3)
        self.reset = PopButton("hist.reset", "paper", "reset", height=36, px=10.5, shadow=3)
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
        lay.setContentsMargins(14, 14, 18, 12)
        lay.setSpacing(8)
        lay.addWidget(SectionLabel("tool.set"))
        self.buttons: dict[Tool, ToolButton] = {}
        for tool, key, icon in ((Tool.CROP, "tool.crop", "crop"), (Tool.REPAIR, "tool.repair", "repair"),
                                (Tool.WATERMARK, "tool.watermark", "watermark"), (Tool.ERASE, "tool.erase", "erase"),
                                (Tool.BATCH, "tool.batch", "batch")):
            b = ToolButton(key, T.TOOL_COLORS[tool.value], icon)
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
