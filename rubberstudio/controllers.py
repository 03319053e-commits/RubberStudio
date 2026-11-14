"""Long-running jobs the panels share: the AI model file and the batch queue."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional, Sequence

from PIL import Image, ImageOps
from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, Qt, Signal
from PySide6.QtGui import QImage

from .core.batch import BatchItem, BatchRunner, BatchSettings, BatchSummary, Status, make_items
from .core.eraser import EraseOptions, RelRect
from .core.imageio import iter_images, is_supported
from .core.neural import ModelError, ModelManager
from .dispatch import Dispatcher

THUMB = 168


# ====================================================================== AI model file

class ModelController(QObject):
    """Wraps :class:`ModelManager` so that download / import / validation never block the GUI.

    One file operation runs at a time (``busy``); the state of each model is asked by name (``lama`` | ``migan``)."""
    changed = Signal()                  # state or progress changed
    failed = Signal(str)

    def __init__(self, manager: ModelManager, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.manager = manager
        self.busy = ""                  # "" | "download" | "import"
        self.busy_model = ""            # which model the running operation is about ("" while importing: slot unknown)
        self.progress = 0.0
        self.host = ""
        self._cancel = threading.Event()
        self._disp = Dispatcher()

    def state(self, model: str = "lama") -> str:
        """``ready`` | ``missing`` | ``downloading`` | ``importing``."""
        if self.busy == "import":
            return "importing"
        if self.busy == "download" and self.busy_model == model:
            return "downloading"
        return "ready" if self.manager.is_ready(model) else "missing"

    def size_mb(self, model: str = "lama") -> float:
        return self.manager.status(model).size_mb

    def _run(self, kind: str, model: str, work: Callable[[], None]) -> None:
        if self.busy:
            return
        self.busy, self.busy_model, self.progress, self.host = kind, model, 0.0, ""
        self._cancel.clear()
        self.changed.emit()

        def job() -> None:
            err = ""
            try:
                work()
            except ModelError as exc:
                err = "" if str(exc).endswith("canceled") else str(exc)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"

            def done() -> None:
                self.busy = self.busy_model = ""
                self.changed.emit()
                if err:
                    self.failed.emit(err)

            self._disp.post(done)

        threading.Thread(target=job, daemon=True, name=f"model-{kind}").start()

    def download(self, model: str = "lama", urls: Optional[Sequence[str]] = None) -> None:
        def prog(v: float, host: str) -> None:
            def ui() -> None:
                self.progress, self.host = v, host
                self.changed.emit()
            self._disp.post(ui)

        self._run("download", model, lambda: self.manager.download(urls, prog, self._cancel, model=model))

    def import_file(self, path: str, model: Optional[str] = None) -> None:
        self._run("import", model or "", lambda: self.manager.import_file(Path(path), model))

    def cancel(self) -> None:
        self._cancel.set()

    def remove(self, model: str = "lama") -> None:
        if self.busy:
            return
        self.manager.remove(model)
        self.changed.emit()


# ====================================================================== batch queue

ItemRole = Qt.ItemDataRole.UserRole + 1
ThumbRole = Qt.ItemDataRole.UserRole + 2
StatusRole = Qt.ItemDataRole.UserRole + 3

TERMINAL = (Status.DONE, Status.NO_TEXT, Status.FAILED, Status.CANCELED)


class BatchModel(QAbstractListModel):
    """The film strip's data: one row per source image, with lazily loaded thumbnails."""

    countsChanged = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.items: list[BatchItem] = []
        self._thumbs: dict[str, Optional[QImage]] = {}
        self._asked: set[str] = set()
        self._index: dict[int, int] = {}
        self._disp = Dispatcher()
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="thumb")

    # --- QAbstractListModel --------------------------------------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.items)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self.items)):
            return None
        it = self.items[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return it.src.name
        if role == Qt.ItemDataRole.ToolTipRole:
            return str(it.src) + (f"\n{it.message}" if it.message else "")
        if role == ItemRole:
            return it
        if role == StatusRole:
            return it.status.value
        if role == ThumbRole:
            return self._thumb(it)
        return None

    # --- content -------------------------------------------------------------------------------
    def _reindex(self) -> None:
        self._index = {id(it): i for i, it in enumerate(self.items)}

    def known(self) -> set[str]:
        return {self._key(it.src) for it in self.items}

    @staticmethod
    def _key(p: Path) -> str:
        import os
        return os.path.normcase(os.path.abspath(p))

    def add_paths(self, paths: Sequence, *, recursive: bool = True) -> int:
        """Add image files and/or folders. Returns the number of images that were new."""
        have = self.known()
        new: list[BatchItem] = []
        for raw in paths:
            p = Path(raw)
            if p.is_dir():
                for f in iter_images([p], recursive=recursive):
                    k = self._key(f)
                    if k not in have:
                        have.add(k)
                        new.append(BatchItem(f, p))
            elif p.is_file() and is_supported(p):
                k = self._key(p)
                if k not in have:
                    have.add(k)
                    new.append(BatchItem(p, None))
        if new:
            self.beginInsertRows(QModelIndex(), len(self.items), len(self.items) + len(new) - 1)
            self.items.extend(new)
            self.endInsertRows()
            self._reindex()
            self.countsChanged.emit()
        return len(new)

    def remove_rows(self, rows: Sequence[int]) -> None:
        for r in sorted(set(rows), reverse=True):
            if 0 <= r < len(self.items):
                self.beginRemoveRows(QModelIndex(), r, r)
                it = self.items.pop(r)
                self.endRemoveRows()
                self._thumbs.pop(self._key(it.src), None)
                self._asked.discard(self._key(it.src))
        self._reindex()
        self.countsChanged.emit()

    def clear(self) -> None:
        self.beginResetModel()
        self.items = []
        self._thumbs.clear()
        self._asked.clear()
        self.endResetModel()
        self._reindex()
        self.countsChanged.emit()

    def requeue_all(self) -> None:
        for it in self.items:
            it.status, it.message, it.out_path = Status.QUEUED, "", None
            it.boxes = it.neural = it.fast = 0
        if self.items:
            self.dataChanged.emit(self.index(0), self.index(len(self.items) - 1))
        self.countsChanged.emit()

    def touched(self, item: BatchItem) -> None:
        row = self._index.get(id(item))
        if row is not None:
            ix = self.index(row)
            self.dataChanged.emit(ix, ix)
        self.countsChanged.emit()

    def counts(self) -> dict[str, int]:
        c = {s.value: 0 for s in Status}
        for it in self.items:
            c[it.status.value] += 1
        c["total"] = len(self.items)
        return c

    def pending(self) -> list[BatchItem]:
        return [it for it in self.items if it.status in (Status.QUEUED, Status.FAILED, Status.CANCELED)]

    # --- thumbnails ----------------------------------------------------------------------------
    def _thumb(self, it: BatchItem) -> Optional[QImage]:
        k = self._key(it.src)
        if k in self._thumbs:
            return self._thumbs[k]
        if k not in self._asked:
            self._asked.add(k)
            self._pool.submit(self._load_thumb, it.src, k)
        return None

    def _load_thumb(self, path: Path, key: str) -> None:
        img: Optional[QImage] = None
        try:
            with Image.open(path) as im:
                try:
                    im.draft("RGB", (THUMB * 2, THUMB * 2))       # fast JPEG down-scaling
                except Exception:
                    pass
                im = ImageOps.exif_transpose(im)
                im.thumbnail((THUMB * 2, THUMB * 2), Image.Resampling.BILINEAR)
                im = im.convert("RGBA")
                w, h = im.size
                data = im.tobytes("raw", "RGBA")
                img = QImage(data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        except Exception:
            img = None

        def put() -> None:
            self._thumbs[key] = img if img is not None else QImage()
            row = next((i for i, it in enumerate(self.items) if self._key(it.src) == key), None)
            if row is not None:
                ix = self.index(row)
                self.dataChanged.emit(ix, ix, [ThumbRole])

        self._disp.post(put)


class BatchController(QObject):
    """Runs the queue on a worker thread and reports progress to the GUI."""
    started = Signal()
    progress = Signal(int, int)             # finished, total (of this run)
    finished = Signal(object)               # BatchSummary

    def __init__(self, model: BatchModel, models: ModelController, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.model = model
        self.models = models
        self.settings = BatchSettings()
        self.running = False
        self.summary: Optional[BatchSummary] = None
        self._runner: Optional[BatchRunner] = None
        self._disp = Dispatcher()

    def start(self, erase: EraseOptions, region: Optional[RelRect]) -> bool:
        if self.running:
            return False
        if not self.model.pending():
            self.model.requeue_all()
        pending = self.model.pending()
        if not pending:
            return False
        s = self.settings
        s.erase = erase
        s.region = region
        mgr = self.models.manager
        run_items = list(pending)
        for it in run_items:
            it.status, it.message = Status.QUEUED, ""
        self.model.dataChanged.emit(self.model.index(0), self.model.index(len(self.model.items) - 1))
        self.running = True
        self.summary = None
        total = len(run_items)

        def on_update(_i: int, item: BatchItem) -> None:
            done = sum(1 for it in run_items if it.status in TERMINAL)
            self._disp.post(lambda: (self.model.touched(item), self.progress.emit(done, total)))

        def job() -> None:
            neural, _note = mgr.resolve(erase.engine)
            runner = BatchRunner(run_items, s, neural=neural)
            self._runner = runner
            try:
                summary = runner.run(on_update)
            except Exception as exc:                 # the runner guards each file; this is the last line of defence
                summary = BatchSummary(total=total, failed=total)
                summary.out_dir = None
                for it in run_items:
                    if it.status not in TERMINAL:
                        it.status, it.message = Status.FAILED, str(exc)

            def end() -> None:
                self.running = False
                self.summary = summary
                self._runner = None
                self.model.dataChanged.emit(self.model.index(0), self.model.index(max(0, len(self.model.items) - 1)))
                self.model.countsChanged.emit()
                self.finished.emit(summary)

            self._disp.post(end)

        self.started.emit()
        self.progress.emit(0, total)
        threading.Thread(target=job, daemon=True, name="batch").start()
        return True

    def stop(self) -> None:
        r = self._runner
        if r is not None:
            r.cancel.set()
