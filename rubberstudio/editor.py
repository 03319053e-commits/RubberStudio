"""EditorModel — the document: image, history, tool state and the operations (port of the iOS EditorModel)."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Signal, Slot
from PySide6.QtGui import QClipboard, QColor, QFont, QFontMetricsF, QImage, QPainter

from .core import repair
from .core.eraser import EraseOptions, RelRect, TextEraser, rel_to_px
from .core.imageio import ImageMeta, load_image, save_image
from .core.neural import ModelManager
from .core.textdet import TextBox, TextDetector
from .dispatch import Dispatcher
from .i18n import tr


def engine_label(engine: str) -> str:
    """Name of a fill engine as shown to the user (the fast fill is the only one that is translated)."""
    return {"lama": "LaMa", "migan": "MI-GAN", "auto": "AUTO"}.get(engine) or tr("eng.fast")


class Tool(str, Enum):
    NONE = "none"
    CROP = "crop"
    REPAIR = "repair"
    WATERMARK = "watermark"
    ERASE = "erase"
    BATCH = "batch"


class RepairTool(str, Enum):
    HEAL = "heal"
    SMUDGE = "smudge"
    TEXTURE = "texture"


@dataclass
class BrushSettings:
    size: float = 28.0
    hardness: float = 0.5
    flow: float = 0.8
    strength: float = 1.0
    smear: float = 1.0
    block: float = 8.0
    tool: RepairTool = RepairTool.HEAL


@dataclass
class WatermarkSettings:
    text: str = "RUBBER STUDIO"
    opacity: float = 0.5
    size: float = 1.0                 # 1.0 = 5 % of the image width (the iOS default)
    font: str = "Times New Roman"
    color: str = "white"              # white | black
    center: tuple = (0.78, 0.9)       # relative position of the label's centre


ENGINES = ("auto", "fast", "migan", "lama")       # fill engines in the order the selector shows them


@dataclass
class ErasePrefs:
    whole_image: bool = False         # False: only the framed region (like the iOS app)
    sensitivity: float = 0.5          # 0..1, higher finds more (and fainter) text
    grow: float = 0.12
    engine: str = "auto"              # auto | fast | migan | lama — see ModelManager.resolve
    thorough: bool = True             # multi-scale detection + a check that the text is really gone (a bit slower)

    def options(self) -> EraseOptions:
        eng = self.engine if self.engine in ENGINES else "auto"
        return EraseOptions(box_thresh=round(0.8 - 0.6 * float(self.sensitivity), 3), grow=float(self.grow),
                            thorough=bool(self.thorough), verify=bool(self.thorough), engine=eng,
                            use_neural=eng != "fast", always_neural=eng in ("migan", "lama"))


class EditorModel(QObject):
    imageChanged = Signal()           # a different pixel buffer (load / undo / crop / apply …)
    pixelsChanged = Signal()          # same buffer, edited in place (brush)
    toolChanged = Signal(str)
    statusChanged = Signal()
    busyChanged = Signal(bool)
    historyChanged = Signal()
    overlayChanged = Signal()         # rectangles / detection boxes / watermark moved
    compareChanged = Signal()
    prefsChanged = Signal()           # shared erase options changed (keeps the two option boxes in sync)

    MAX_UNDO = 30
    UNDO_BUDGET = 700 * 1024 * 1024

    def __init__(self, models: ModelManager, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.models = models
        self.image: Optional[np.ndarray] = None
        self.original: Optional[np.ndarray] = None
        self.meta = ImageMeta()
        self.path: Optional[Path] = None
        self.version = 0
        self.has_alpha = False
        self.tool = Tool.NONE
        self.brush = BrushSettings()
        self.wm = WatermarkSettings()
        self.erase = ErasePrefs()
        self.crop_rect: RelRect = (0.08, 0.08, 0.84, 0.84)
        self.erase_rect: RelRect = (0.3, 0.3, 0.4, 0.2)
        self.batch_rect: RelRect = (0.05, 0.78, 0.9, 0.18)
        self.batch_region_on = False                 # BATCH tool: framed region instead of the whole image
        self.boxes: list[TextBox] = []
        self.after: Optional[np.ndarray] = None        # processed version for before / after
        self.show_after = False
        self.busy = False
        self._status = ("st.ready", {})
        self._undo: list[np.ndarray] = []
        self._redo: list[np.ndarray] = []
        self._stroke_last: Optional[tuple[float, float]] = None
        self._stroking = False
        self._rng = np.random.default_rng()
        self._qimg: Optional[QImage] = None
        self._qsrc: Optional[np.ndarray] = None
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._disp = Dispatcher()
        self._detector: Optional[TextDetector] = None
        self._detector_lock = threading.Lock()

    # ------------------------------------------------------------------ basics

    @property
    def size(self) -> tuple[int, int]:
        if self.image is None:
            return (0, 0)
        return (self.image.shape[1], self.image.shape[0])

    def status_text(self) -> str:
        from .i18n import tr
        key, fmt = self._status
        return tr(key, **fmt)

    def set_status(self, key: str, **fmt) -> None:
        self._status = (key, fmt)
        self.statusChanged.emit()

    def set_busy(self, busy: bool) -> None:
        if busy != self.busy:
            self.busy = busy
            self.busyChanged.emit(busy)

    def set_tool(self, tool: Tool) -> None:
        if tool != self.tool:
            self.tool = tool
            self.boxes = []
            self.toolChanged.emit(tool.value)
            self.overlayChanged.emit()

    def qimage(self, source: Optional[np.ndarray] = None) -> Optional[QImage]:
        """Zero-copy QImage over the pixel buffer (kept alive here)."""
        arr = source if source is not None else (self.after if (self.show_after and self.after is not None) else self.image)
        if arr is None:
            return None
        if self._qsrc is not arr or self._qimg is None or self._qimg.width() != arr.shape[1]:
            fmt = QImage.Format.Format_RGBA8888 if self.has_alpha else QImage.Format.Format_RGBX8888
            self._qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1] * 4, fmt)
            self._qsrc = arr
        return self._qimg

    def _bump(self) -> None:
        self.version += 1

    # ------------------------------------------------------------------ loading

    def load_path(self, path) -> bool:
        from .core.imageio import ImageIOError
        try:
            arr, meta = load_image(path)
        except ImageIOError as exc:
            self.set_status("st.import_failed", why=str(exc))
            return False
        self.load_array(arr, meta, Path(path))
        self.set_status("st.imported")
        return True

    def load_array(self, arr: np.ndarray, meta: Optional[ImageMeta] = None, path: Optional[Path] = None,
                   keep_history: bool = False) -> None:
        self.image = np.ascontiguousarray(arr)
        self.original = self.image.copy()
        self.meta = meta or ImageMeta()
        self.path = path
        self.has_alpha = bool((self.image[..., 3] < 255).any())
        self.boxes = []
        self.after = None
        self.show_after = False
        if not keep_history:
            self._undo.clear()
            self._redo.clear()
        self._qimg = None
        self._bump()
        self.imageChanged.emit()
        self.historyChanged.emit()
        self.compareChanged.emit()

    def paste_from_clipboard(self, clip: QClipboard) -> bool:
        qimg = clip.image()
        if qimg.isNull():
            return False
        qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = qimg.width(), qimg.height()
        ptr = qimg.constBits()
        arr = np.frombuffer(ptr, np.uint8, count=qimg.bytesPerLine() * h).reshape(h, qimg.bytesPerLine())[:, :w * 4]
        self.load_array(arr.reshape(h, w, 4).copy(), ImageMeta(), None)
        self.set_status("st.pasted")
        return True

    def set_compare(self, after: Optional[np.ndarray]) -> None:
        self.after = after
        self.show_after = after is not None
        self._qimg = None
        self._bump()
        self.compareChanged.emit()
        self.imageChanged.emit()

    def toggle_compare(self) -> None:
        if self.after is not None:
            self.show_after = not self.show_after
            self._qimg = None
            self._bump()
            self.imageChanged.emit()
            self.compareChanged.emit()

    # ------------------------------------------------------------------ history

    def _bytes(self) -> int:
        return sum(a.nbytes for a in self._undo) + sum(a.nbytes for a in self._redo)

    def clear_compare(self) -> None:
        if self.after is not None:
            self.after = None
            self.show_after = False
            self._qimg = None
            self.compareChanged.emit()

    def checkpoint(self) -> None:
        if self.image is None:
            return
        self.clear_compare()
        self._undo.append(self.image.copy())
        while len(self._undo) > self.MAX_UNDO or (self._bytes() > self.UNDO_BUDGET and len(self._undo) > 1):
            self._undo.pop(0)
        self._redo.clear()
        self.historyChanged.emit()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def _swap_in(self, arr: np.ndarray) -> None:
        self.clear_compare()
        self.image = arr
        self.has_alpha = bool((arr[..., 3] < 255).any())
        self._qimg = None
        self.boxes = []
        self._bump()
        self.imageChanged.emit()
        self.historyChanged.emit()

    def undo(self) -> None:
        if not self._undo or self.image is None:
            return
        self._redo.append(self.image)
        self._swap_in(self._undo.pop())
        self.set_status("st.undo")

    def redo(self) -> None:
        if not self._redo or self.image is None:
            return
        self._undo.append(self.image)
        self._swap_in(self._redo.pop())
        self.set_status("st.redo")

    def reset(self) -> None:
        if self.original is None or self.image is None:
            return
        self.checkpoint()
        self._swap_in(self.original.copy())
        self.set_status("st.reset")

    # ------------------------------------------------------------------ crop

    def confirm_crop(self) -> bool:
        if self.image is None:
            return False
        h, w = self.image.shape[:2]
        x0, y0, x1, y1 = rel_to_px(self.crop_rect, w, h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            self.set_status("st.crop_failed")
            return False
        self.checkpoint()
        self._swap_in(np.ascontiguousarray(self.image[y0:y1, x0:x1]))
        self.crop_rect = (0.08, 0.08, 0.84, 0.84)
        self.set_tool(Tool.NONE)
        self.set_status("st.cropped")
        return True

    # ------------------------------------------------------------------ repair brush

    def brush_radius(self) -> float:
        if self.image is None:
            return 8.0
        h, w = self.image.shape[:2]
        return repair.brush_radius_px(w, h, self.brush.size)

    def begin_stroke(self) -> None:
        if self.image is None:
            return
        if not self._stroking:
            self.checkpoint()
            self._stroking = True
        self._stroke_last = None

    def end_stroke(self) -> None:
        self._stroking = False
        self._stroke_last = None
        self._bump()
        self.pixelsChanged.emit()

    def stroke_to(self, x: float, y: float) -> None:
        """Brush stroke towards image-pixel position (x, y)."""
        if self.image is None:
            return
        b, r = self.brush, self.brush_radius()
        if b.tool == RepairTool.SMUDGE:
            if self._stroke_last is None:
                self._stroke_last = (x, y)
                return
            if repair.smudge(self.image, self._stroke_last, (x, y), radius=r, hardness=b.hardness,
                             flow=b.flow, smear=b.smear):
                self._stroke_last = (x, y)
                self.set_status("st.smudged")
        else:
            last = self._stroke_last
            if last is None:
                pts = [(x, y)]
            else:
                spacing = max(r * 0.5, 4.0)
                dist = float(np.hypot(x - last[0], y - last[1]))
                n = min(max(1, int(dist / spacing)), 24)
                pts = [(last[0] + (x - last[0]) * i / n, last[1] + (y - last[1]) * i / n) for i in range(1, n + 1)]
            for px, py in pts:
                if b.tool == RepairTool.HEAL:
                    repair.heal(self.image, px, py, radius=r, hardness=b.hardness, flow=b.flow, strength=b.strength)
                else:
                    repair.texture(self.image, px, py, radius=r, hardness=b.hardness, flow=b.flow,
                                   strength=b.strength, block=b.block, rng=self._rng)
            self._stroke_last = (x, y)
            self.set_status("st.repaired" if b.tool == RepairTool.HEAL else "st.textured")
        self._bump()
        self.pixelsChanged.emit()

    # ------------------------------------------------------------------ watermark

    def watermark_font(self) -> QFont:
        w = self.size[0] or 1000
        f = QFont(self.wm.font)
        f.setPixelSize(max(8, int(w * 0.05 * self.wm.size)))
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, f.pixelSize() * 0.04)
        return f

    def watermark_rect(self) -> QRectF:
        """Label rectangle in image pixels (centred on ``wm.center``, kept inside the picture)."""
        w, h = self.size
        fm = QFontMetricsF(self.watermark_font())
        text = self.wm.text.upper() if self.wm.text.isascii() else self.wm.text
        tw = fm.horizontalAdvance(text) + fm.height() * 0.1
        th = fm.height()
        cx, cy = self.wm.center[0] * w, self.wm.center[1] * h
        x = min(max(0.0, cx - tw / 2), max(0.0, w - tw))
        y = min(max(0.0, cy - th / 2), max(0.0, h - th))
        return QRectF(x, y, tw, th)

    def apply_watermark(self) -> None:
        if self.image is None or not self.wm.text.strip():
            return
        self.checkpoint()
        arr = self.image
        qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1] * 4, QImage.Format.Format_RGBA8888)
        p = QPainter(qimg)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setFont(self.watermark_font())
        col = QColor(255, 255, 255) if self.wm.color == "white" else QColor(0, 0, 0)
        col.setAlphaF(float(self.wm.opacity))
        p.setPen(col)
        text = self.wm.text.upper() if self.wm.text.isascii() else self.wm.text
        p.drawText(self.watermark_rect(), int(0x0084), text)         # AlignCenter
        p.end()
        self._bump()
        self.pixelsChanged.emit()
        self.set_status("st.watermarked")

    # ------------------------------------------------------------------ text erasing (async)

    def detector(self) -> TextDetector:
        with self._detector_lock:
            if self._detector is None:
                self._detector = TextDetector()
            return self._detector

    def _region_px(self) -> Optional[tuple[int, int, int, int]]:
        if self.image is None or self.erase.whole_image:
            return None
        h, w = self.image.shape[:2]
        return rel_to_px(self.erase_rect, w, h)

    def _run_async(self, work: Callable[[], object], done: Callable[[object], None]) -> None:
        self.set_busy(True)

        def job() -> None:
            try:
                res: object = work()
            except Exception as exc:        # surface any failure on the GUI thread
                res = exc
            self._disp.post(lambda: self._finish(done, res))

        self._pool.submit(job)

    def _finish(self, done: Callable[[object], None], res: object) -> None:
        self.set_busy(False)
        done(res)

    def detect_text(self) -> None:
        if self.image is None or self.busy:
            return
        arr, region, opts = self.image, self._region_px(), self.erase.options()
        self.set_status("st.detecting")

        def work():
            return TextEraser(self.detector()).detect(arr, region=region, opts=opts)

        def done(res) -> None:
            if isinstance(res, Exception):
                self.set_status("st.erase_failed", why=str(res))
                return
            self.boxes = list(res)
            self.overlayChanged.emit()
            if self.boxes:
                self.set_status("st.found_text", n=len(self.boxes))
            else:
                self.set_status("st.no_text")

        self._run_async(work, done)

    def erase_text(self) -> None:
        if self.image is None or self.busy:
            return
        arr, region, opts = self.image, self._region_px(), self.erase.options()
        neural, _ = self.models.resolve(self.erase.engine)
        wanted = self.erase.engine
        self.set_status("st.detecting")
        src_version = self.version

        def work():
            eraser = TextEraser(self.detector(), neural)

            def prog(done_n: int, total: int) -> None:
                self._disp.post(lambda: self.set_status("st.erasing", done=done_n, total=total))

            return eraser.erase(arr, region=region, opts=opts, progress=prog)

        def done(res) -> None:
            if isinstance(res, Exception):
                self.set_status("st.erase_failed", why=str(res))
                return
            if not res.found:
                self.set_status("st.no_text")
                return
            if self.version != src_version or self.image is None:
                self.set_status("st.erase_stale")
                return
            before = self.image
            self.checkpoint()
            self._swap_in(np.ascontiguousarray(res.image))
            self.after = before.copy() if before is not None else None      # BEFORE view on the canvas
            self.show_after = False
            self.compareChanged.emit()
            used = getattr(neural, "kind", "fast")
            note = res.note
            if not note and wanted in ("migan", "lama") and used != wanted:
                note = tr("st.engine_fallback", want=engine_label(wanted), got=engine_label(used))
            if note:
                self.set_status("st.erased_note", ai=res.neural_count, fast=res.fast_count, note=note)
            else:
                self.set_status("st.erased", ai=res.neural_count, fast=res.fast_count)

        self._run_async(work, done)

    # ------------------------------------------------------------------ export

    def export(self, path, fmt: Optional[str] = None, quality: int = 95) -> Path:
        if self.image is None:
            raise ValueError("no image")
        out = save_image(path, self.image, fmt=fmt, quality=quality, meta=self.meta)
        self.set_status("st.saved", name=Path(out).name)
        return out
