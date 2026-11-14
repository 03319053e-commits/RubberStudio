"""The application window: header, left tool drawer, canvas, right film queue and status bar."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QAction, QCloseEvent, QColor, QDragEnterEvent, QDropEvent, QKeySequence, QPainter
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QMainWindow, QVBoxLayout, QWidget)

from . import APP_NAME, i18n
from . import settings as S
from . import theme as T
from .controllers import BatchController, BatchModel, ModelController
from .core.imageio import READ_EXTS, iter_images, is_supported
from .core.neural import MODELS, ModelManager
from .editor import ENGINES, EditorModel, Tool
from .i18n import tr
from .widgets import dialogs
from .widgets.bars import HEADER_H, STATUS_H, HeaderBar, StatusBar
from .widgets.canvas import Canvas
from .widgets.envelope import EnvelopePanel
from .widgets.filmstrip import QueuePanel
from .widgets.panels import ToolDrawer
from .widgets.splash import SplashOverlay

log = logging.getLogger("rubberstudio")

LEFT_W = 332
RIGHT_W = 332


class _Central(QWidget):
    def paintEvent(self, e) -> None:
        QPainter(self).fillRect(self.rect(), QColor(T.PAPER))


def load_prefs(editor: EditorModel, batch: BatchController) -> None:
    """Restore the panel settings of the last session."""
    e, b, wm, s = editor.erase, editor.brush, editor.wm, batch.settings
    engine = S.get_str("erase/engine", "")
    if engine not in ENGINES:                       # settings of version 1.0: two switches instead of one choice
        engine = "fast" if not S.get_bool("erase/use_ai", True) else ("lama" if S.get_bool("erase/always_ai", False) else "auto")
    e.engine = engine
    e.thorough = S.get_bool("erase/thorough", e.thorough)
    e.sensitivity = min(max(S.get_float("erase/sensitivity", e.sensitivity), 0.0), 1.0)
    e.grow = min(max(S.get_float("erase/grow", e.grow), 0.0), 0.4)
    e.whole_image = S.get_bool("erase/whole", e.whole_image)
    for name in ("size", "hardness", "flow", "strength", "smear", "block"):
        setattr(b, name, S.get_float(f"brush/{name}", getattr(b, name)))
    try:
        from .editor import RepairTool
        b.tool = RepairTool(S.get_str("brush/tool", b.tool.value))
    except ValueError:
        pass
    wm.text = S.get_str("wm/text", wm.text)
    wm.font = S.get_str("wm/font", wm.font)
    wm.color = S.get_str("wm/color", wm.color) if S.get_str("wm/color", wm.color) in ("white", "black") else wm.color
    wm.size = S.get_float("wm/size", wm.size)
    wm.opacity = S.get_float("wm/opacity", wm.opacity)
    out = S.get_str("batch/out_dir", "")
    s.out_dir = Path(out) if out else None
    s.suffix = S.get_str("batch/suffix", s.suffix)
    s.fmt = S.get_str("batch/fmt", s.fmt) if S.get_str("batch/fmt", s.fmt) in ("same", "png", "jpg", "webp") else s.fmt
    s.quality = S.get_int("batch/quality", s.quality)
    s.existing = S.get_str("batch/existing", s.existing) if S.get_str("batch/existing", s.existing) in ("rename", "skip", "overwrite") else s.existing
    s.keep_structure = S.get_bool("batch/keep_structure", s.keep_structure)
    s.save_unchanged = S.get_bool("batch/save_unchanged", s.save_unchanged)
    s.workers = max(1, S.get_int("batch/workers", s.workers))
    editor.batch_region_on = S.get_bool("batch/region_on", editor.batch_region_on)


class MainWindow(QMainWindow):
    def __init__(self, models: ModelManager, *, splash: bool = True, restore: bool = True):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setAcceptDrops(True)
        self.setMinimumSize(1020, 660)
        self.resize(1360, 860)
        self._splash: Optional[SplashOverlay] = None
        self._right_auto = False

        # --- state -------------------------------------------------------------------------------
        self.editor = EditorModel(models)
        self.model_ctl = ModelController(models)
        self.batch_model = BatchModel()
        self.batch_ctl = BatchController(self.batch_model, self.model_ctl)
        load_prefs(self.editor, self.batch_ctl)

        # --- widgets -----------------------------------------------------------------------------
        central = _Central()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = HeaderBar()
        self.header_env = EnvelopePanel(self.header, "top", HEADER_H, tooltip=tr("fold.header"))
        root.addWidget(self.header_env)

        mid = QHBoxLayout()
        mid.setContentsMargins(0, 0, 0, 0)
        mid.setSpacing(0)
        self.drawer = ToolDrawer(self.editor, self.model_ctl, self.batch_ctl)
        self.left_env = EnvelopePanel(self.drawer, "left", LEFT_W, tooltip=tr("fold.left"))
        mid.addWidget(self.left_env)
        self.canvas = Canvas(self.editor)
        holder = QWidget()
        hl = QVBoxLayout(holder)
        hl.setContentsMargins(14, 10, 10, 10)
        hl.addWidget(self.canvas)
        mid.addWidget(holder, 1)
        self.queue = QueuePanel(self.batch_model, self.batch_ctl)
        self.right_env = EnvelopePanel(self.queue, "right", RIGHT_W, tooltip=tr("fold.right"))
        mid.addWidget(self.right_env)
        root.addLayout(mid, 1)

        self.statusbar = StatusBar()
        self.status_env = EnvelopePanel(self.statusbar, "bottom", STATUS_H, tooltip=tr("fold.status"))
        root.addWidget(self.status_env)
        self.right_env.set_open(False, animate=False)

        self._build_actions()
        self._wire()
        self._refresh_status()
        self._refresh_title()
        if restore:
            self._restore_state()
        if splash:
            self._splash = SplashOverlay(central)
            self._splash.finished.connect(self._splash_done)
            self._splash.start()
            self._splash.setFocus()

    # ------------------------------------------------------------------ construction

    def _build_actions(self) -> None:
        def act(name: str, keys, slot) -> QAction:
            a = QAction(self)
            a.setShortcuts([QKeySequence(k) for k in (keys if isinstance(keys, (list, tuple)) else [keys])])
            a.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            a.triggered.connect(slot)
            self.addAction(a)
            setattr(self, f"act_{name}", a)
            return a

        act("open", "Ctrl+O", self.open_files)
        act("save", "Ctrl+S", self.save_image)
        act("copy", "Ctrl+C", self.copy_image)
        act("paste", "Ctrl+V", self.paste_image)
        act("undo", "Ctrl+Z", self.editor.undo)
        act("redo", ["Ctrl+Y", "Ctrl+Shift+Z"], self.editor.redo)
        act("help", "F1", self.show_help)
        act("run", ["Ctrl+Return", "Ctrl+Enter"], self.run_current)
        act("escape", "Esc", lambda: self.editor.set_tool(Tool.NONE))

    def _wire(self) -> None:
        e, h = self.editor, self.header
        h.openClicked.connect(self.open_files)
        h.pasteClicked.connect(self.paste_image)
        h.saveClicked.connect(self.save_image)
        h.langClicked.connect(self.toggle_language)
        h.helpClicked.connect(self.show_help)
        self.canvas.openRequested.connect(self.open_files)
        e.statusChanged.connect(self._refresh_status)
        e.busyChanged.connect(self.statusbar.set_busy)
        e.imageChanged.connect(self._refresh_title)
        e.toolChanged.connect(self._tool_changed)
        i18n.bus.changed.connect(self._retranslate)

        page = self.drawer.batch_page
        page.addFilesRequested.connect(self.add_batch_files)
        page.addFolderRequested.connect(self.add_batch_folder)
        page.btn_start.clicked.connect(self.start_batch)
        page.btn_stop.clicked.connect(self.batch_ctl.stop)
        self.queue.addFilesRequested.connect(self.add_batch_files)
        self.queue.addFolderRequested.connect(self.add_batch_folder)
        self.queue.openRequested.connect(self._open_from_queue)
        self.batch_model.countsChanged.connect(self._queue_counts)
        self.batch_ctl.started.connect(lambda: page.set_running(True))
        self.batch_ctl.progress.connect(page.set_progress)
        self.batch_ctl.finished.connect(self._batch_finished)
        self.model_ctl.failed.connect(lambda msg: dialogs.error(self, tr("ai.failed"), msg))
        self.model_ctl.changed.connect(self._model_changed)
        self._last_model_states = {m: self.model_ctl.state(m) for m in MODELS}
        self._queue_counts()

    # ------------------------------------------------------------------ language / chrome

    def _retranslate(self, _lang: str = "") -> None:
        S.put("ui/lang", i18n.language())
        for w in self.findChildren(QWidget):
            fn = getattr(w, "retranslate", None)
            if callable(fn):
                try:
                    fn()
                except TypeError:
                    pass
        self.header_env.setToolTip(tr("fold.header"))
        self.left_env.setToolTip(tr("fold.left"))
        self.right_env.setToolTip(tr("fold.right"))
        self.status_env.setToolTip(tr("fold.status"))
        self._refresh_status()
        self._refresh_title()
        self._queue_counts()

    def toggle_language(self) -> None:
        i18n.set_language("en" if i18n.language() == "zh" else "zh")

    def _refresh_status(self) -> None:
        self.statusbar.set_text(self.editor.status_text())

    def _refresh_title(self) -> None:
        p = self.editor.path
        self.setWindowTitle(f"{p.name} — {APP_NAME}" if p else APP_NAME)

    def _splash_done(self) -> None:
        self._splash = None
        self.canvas.setFocus()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self._splash is not None:
            self._splash.setGeometry(self.centralWidget().rect())

    # ------------------------------------------------------------------ tools / panels

    def _tool_changed(self, name: str) -> None:
        if name == Tool.BATCH.value:
            if not self.right_env.is_open():
                self.right_env.set_open(True)
                self._right_auto = True
        elif self._right_auto and not self.batch_ctl.running:
            self.right_env.set_open(False)
            self._right_auto = False
        self.canvas.setFocus()

    def run_current(self) -> None:
        t = self.editor.tool
        if t == Tool.CROP:
            self.editor.confirm_crop()
        elif t == Tool.WATERMARK:
            self.editor.apply_watermark()
        elif t == Tool.ERASE:
            self.editor.erase_text()
        elif t == Tool.BATCH:
            self.start_batch()

    # ------------------------------------------------------------------ file actions

    def _images_filter(self) -> str:
        exts = " ".join(f"*{e}" for e in READ_EXTS)
        return f"{tr('file.images')} ({exts});;{tr('file.all')} (*)"

    def _last_dir(self) -> str:
        d = S.get_str("ui/last_dir", "")
        return d if d and Path(d).is_dir() else str(Path.home())

    def _remember_dir(self, path: str) -> None:
        S.put("ui/last_dir", str(Path(path).parent if Path(path).is_file() else path))

    def open_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, tr("file.open"), self._last_dir(), self._images_filter())
        if paths:
            self._remember_dir(paths[0])
            self.open_paths(paths)

    def open_paths(self, paths: Sequence[str]) -> None:
        """Open one image for editing; several images / folders go to the batch queue."""
        paths = [str(p) for p in paths]
        single = len(paths) == 1 and Path(paths[0]).is_file() and is_supported(paths[0])
        if single:
            if not self.editor.busy and not self.editor.load_path(paths[0]):
                dialogs.error(self, self.editor.status_text())
            return
        added = self.batch_model.add_paths(paths)
        if added:
            self.editor.set_tool(Tool.BATCH)
            self.editor.set_status("st.queued", n=added)
            if self.editor.image is None:                    # show the first one so a region can be framed on it
                first = next(iter(iter_images([Path(p) for p in paths])), None)
                if first is not None:
                    self.editor.load_path(first)
                    self.editor.set_status("st.queued", n=added)
        else:
            self.editor.set_status("st.no_images")

    def _open_from_queue(self, path: str) -> None:
        if not self.editor.busy:
            self.editor.load_path(path)

    def add_batch_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, tr("file.add"), self._last_dir(), self._images_filter())
        if paths:
            self._remember_dir(paths[0])
            self._add_to_batch(paths)

    def add_batch_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, tr("file.folder"), self._last_dir())
        if d:
            S.put("ui/last_dir", d)
            self._add_to_batch([d])

    def _add_to_batch(self, paths: Sequence[str]) -> None:
        n = self.batch_model.add_paths(paths)
        if not n:
            self.editor.set_status("st.no_images")
            return
        if self.editor.image is None:                         # show the first one so a region can be framed on it
            self.editor.load_path(self.batch_model.items[0].src)
        self.editor.set_status("st.queued", n=n)

    def paste_image(self) -> None:
        if not self.editor.paste_from_clipboard(QApplication.clipboard()):
            self.editor.set_status("st.clip_empty")

    def copy_image(self) -> None:
        img = self.editor.qimage(self.editor.image)
        if img is not None:
            QApplication.clipboard().setImage(img.copy())
            self.editor.set_status("st.copied")

    def save_image(self) -> None:
        e = self.editor
        if e.image is None:
            return
        stem = (e.path.stem if e.path else "image") + "_edit"
        start = str((Path(S.get_str("ui/save_dir", "")) if S.get_str("ui/save_dir", "") else
                     (e.path.parent if e.path else Path(self._last_dir()))) / f"{stem}.png")
        filters = "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp);;BMP (*.bmp);;TIFF (*.tif *.tiff)"
        path, chosen = QFileDialog.getSaveFileName(self, tr("file.save"), start, filters)
        if not path:
            return
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".jpg" if "JPEG" in chosen else ".webp" if "WebP" in chosen else ".bmp" if "BMP" in chosen
                              else ".tif" if "TIFF" in chosen else ".png")
        try:
            e.export(p, quality=95)
            S.put("ui/save_dir", str(p.parent))
        except Exception as exc:
            log.exception("save failed")
            dialogs.error(self, tr("file.save_failed"), str(exc))

    # ------------------------------------------------------------------ batch

    def start_batch(self) -> None:
        if self.batch_ctl.running:
            return
        if not self.batch_model.items:
            dialogs.notify(self, tr("bt.title"), tr("bt.nofiles"), icon="batch")
            return
        e = self.editor
        region = e.batch_rect if e.batch_region_on else None
        page = self.drawer.batch_page
        self.batch_ctl.settings.workers = int(page.workers.value()) if page.workers.isVisible() else 1
        page.set_start_key("bt.start")
        if self.batch_ctl.start(e.erase.options(), region):
            e.set_status("st.batch_running")

    def _queue_counts(self) -> None:
        c = self.batch_model.counts()
        page = self.drawer.batch_page
        again = c["total"] > 0 and not self.batch_model.pending()
        page.set_start_key("bt.again" if again else "bt.start")
        page.btn_start.setEnabled(c["total"] > 0 and not self.batch_ctl.running)

    def _batch_finished(self, summary) -> None:
        page = self.drawer.batch_page
        page.set_running(False)
        text = tr("bt.summary", done=summary.done, nothing=summary.no_text, failed=summary.failed,
                  canceled=summary.canceled, secs=f"{summary.seconds:.1f}")
        page.set_summary(text, summary.out_dir)
        self.editor.set_status("st.batch_done", done=summary.done, nothing=summary.no_text, failed=summary.failed)
        self._queue_counts()
        self.queue.refresh()

    def _model_changed(self) -> None:
        now = {m: self.model_ctl.state(m) for m in MODELS}
        for m, st in now.items():
            if st == "ready" and self._last_model_states.get(m) in ("downloading", "importing"):
                self.editor.set_status("st.model_ready", name=MODELS[m].label)
        self._last_model_states = now

    # ------------------------------------------------------------------ drag & drop

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        if e.mimeData().hasUrls() and any(u.isLocalFile() for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent) -> None:
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            e.acceptProposedAction()
            self.open_paths(paths)

    # ------------------------------------------------------------------ misc

    def show_help(self) -> None:
        dialogs.HelpDialog(self).exec()

    def _restore_state(self) -> None:
        geo = S.store().value("ui/geometry")
        if isinstance(geo, (QByteArray, bytes)) and geo:
            self.restoreGeometry(geo)
        self.header_env.set_open(S.get_bool("ui/header_open", True), animate=False)
        self.left_env.set_open(S.get_bool("ui/left_open", True), animate=False)
        self.status_env.set_open(S.get_bool("ui/status_open", True), animate=False)

    def closeEvent(self, e: QCloseEvent) -> None:
        if self.batch_ctl.running:
            if not dialogs.confirm(self, tr("bt.title"), tr("bt.quit_running"), "bt.stop", accent=dialogs.ALERT, icon="stop"):
                e.ignore()
                return
            self.batch_ctl.stop()
        S.put("ui/geometry", self.saveGeometry())
        S.put("ui/header_open", self.header_env.is_open())
        S.put("ui/left_open", self.left_env.is_open())
        S.put("ui/status_open", self.status_env.is_open())
        S.store().sync()
        super().closeEvent(e)
