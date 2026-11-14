"""Batch text removal: many images, one set of options, a thread pool, cancel + live progress."""
from __future__ import annotations

import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence

from .eraser import EraseOptions, RelRect, TextEraser, rel_to_px
from .imageio import FORMATS, ImageIOError, ImageMeta, key_for_path, load_image, save_image, unique_path
from .neural import NeuralInpainter
from .textdet import TextDetector


class Status(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    NO_TEXT = "no_text"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class BatchItem:
    src: Path
    root: Optional[Path] = None          # folder the file was added from (to mirror sub-folders)
    status: Status = Status.QUEUED
    out_path: Optional[Path] = None
    boxes: int = 0
    neural: int = 0
    fast: int = 0
    seconds: float = 0.0
    message: str = ""


@dataclass
class BatchSettings:
    out_dir: Optional[Path] = None       # None -> "<image folder>/RubberStudio_clean"
    suffix: str = "_clean"
    fmt: str = "same"                    # same | png | jpg | webp
    quality: int = 95
    existing: str = "rename"             # rename | skip | overwrite   (a source file is never overwritten)
    keep_structure: bool = True
    save_unchanged: bool = True          # also write images in which no text was found
    region: Optional[RelRect] = None     # same relative region for every image; None = whole image
    erase: EraseOptions = field(default_factory=EraseOptions)
    workers: int = 2


@dataclass
class BatchSummary:
    total: int = 0
    done: int = 0
    no_text: int = 0
    failed: int = 0
    canceled: int = 0
    seconds: float = 0.0
    out_dir: Optional[Path] = None


def make_items(paths: Sequence[Path], root: Optional[Path] = None) -> list[BatchItem]:
    return [BatchItem(Path(p), root) for p in paths]


def plan_output(item: BatchItem, s: BatchSettings) -> Path:
    """Where the result for ``item`` will be written (before any rename-on-clash)."""
    src = item.src
    key = s.fmt if s.fmt in FORMATS else key_for_path(src)
    ext = src.suffix if s.fmt == "same" else FORMATS[key][1]
    name = f"{src.stem}{s.suffix}{ext}"
    if s.out_dir is None:
        return src.parent / "RubberStudio_clean" / name
    base = Path(s.out_dir)
    if s.keep_structure and item.root is not None:
        try:
            rel = src.parent.relative_to(item.root)
            return base / rel / name
        except ValueError:
            pass
    return base / name


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def resolve_output(item: BatchItem, s: BatchSettings) -> Optional[Path]:
    """Final output path honoring the clash policy; ``None`` means "skip this file"."""
    out = plan_output(item, s)
    if _same_file(out, item.src):                 # never clobber the source
        return unique_path(out)
    if out.exists():
        if s.existing == "skip":
            return None
        if s.existing == "rename":
            return unique_path(out)
    return out


class BatchRunner:
    """Runs a list of :class:`BatchItem` through the eraser. Thread-safe progress via ``on_update``."""

    def __init__(self, items: Sequence[BatchItem], settings: BatchSettings,
                 neural: Optional[NeuralInpainter] = None,
                 detector: Optional[TextDetector] = None):
        self.items = list(items)
        self.settings = settings
        self.neural = neural
        workers = max(1, int(settings.workers))
        cpu = os.cpu_count() or 2
        self.detector = detector or TextDetector(intra_threads=max(1, cpu // workers))
        self.cancel = threading.Event()
        self._local = threading.local()

    def _eraser(self) -> TextEraser:
        e = getattr(self._local, "eraser", None)
        if e is None:
            e = TextEraser(self.detector, self.neural)
            self._local.eraser = e
        return e

    def run(self, on_update: Optional[Callable[[int, BatchItem], None]] = None) -> BatchSummary:
        t0 = time.time()
        s = self.settings
        summary = BatchSummary(total=len(self.items))

        def notify(i: int) -> None:
            if on_update is not None:
                try:
                    on_update(i, self.items[i])
                except Exception:
                    pass

        def work(i: int) -> None:
            item = self.items[i]
            if self.cancel.is_set():
                item.status = Status.CANCELED
                notify(i)
                return
            item.status = Status.RUNNING
            notify(i)
            start = time.time()
            try:
                self._process(item)
            except ImageIOError as exc:
                item.status, item.message = Status.FAILED, str(exc)
            except Exception as exc:     # one bad file must never stop the batch
                item.status, item.message = Status.FAILED, f"{type(exc).__name__}: {exc}"
            item.seconds = time.time() - start
            notify(i)

        with ThreadPoolExecutor(max_workers=max(1, int(s.workers))) as pool:
            for _ in pool.map(work, range(len(self.items))):
                pass

        for it in self.items:
            if it.status == Status.DONE:
                summary.done += 1
            elif it.status == Status.NO_TEXT:
                summary.no_text += 1
            elif it.status == Status.FAILED:
                summary.failed += 1
            elif it.status == Status.CANCELED:
                summary.canceled += 1
        summary.seconds = time.time() - t0
        first = next((it.out_path for it in self.items if it.out_path), None)
        summary.out_dir = first.parent if first else None
        return summary

    # ------------------------------------------------------------------

    def _process(self, item: BatchItem) -> None:
        s = self.settings
        rgba, meta = load_image(item.src)
        h, w = rgba.shape[:2]
        region = rel_to_px(s.region, w, h) if s.region else None
        eraser = self._eraser()
        res = eraser.erase(rgba, region=region, opts=s.erase, cancel=self.cancel.is_set)
        if self.cancel.is_set():            # stopped mid-way: write nothing for this file
            item.status = Status.CANCELED
            return
        item.boxes, item.neural, item.fast = len(res.boxes), res.neural_count, res.fast_count
        if res.note:
            item.message = res.note

        key = s.fmt if s.fmt in FORMATS else None
        if not res.found:
            item.status = Status.NO_TEXT
            if not s.save_unchanged:
                return
            out = resolve_output(item, s)
            if out is None:
                item.message = "exists, skipped"
                return
            out.parent.mkdir(parents=True, exist_ok=True)
            if key is None or key == key_for_path(item.src):
                shutil.copy2(item.src, out)          # byte-for-byte copy, no re-compression
            else:
                save_image(out, rgba, fmt=key, quality=s.quality, meta=meta)
            item.out_path = out
            return

        out = resolve_output(item, s)
        if out is None:
            item.status, item.message = Status.NO_TEXT, "exists, skipped"
            return
        save_image(out, res.image, fmt=key, quality=s.quality, meta=meta)
        item.out_path = out
        item.status = Status.DONE
