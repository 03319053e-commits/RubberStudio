"""Optional neural inpainting (LaMa, ONNX) and the manager that downloads / imports / removes it.

The model is *not* bundled: it is ~200 MB. Nothing is ever downloaded without an explicit
user action. Without the model the app uses the fast OpenCV fill, which needs no download.
"""
from __future__ import annotations

import os
import shutil
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from .ortutil import open_session
from .paths import models_dir

MODEL_FILE = "lama.onnx"
MODEL_LABEL = "LaMa (Carve/LaMa-ONNX, fp32, ≈200 MB)"
DEFAULT_URLS: tuple[str, ...] = (
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
    "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
)
MIN_MODEL_BYTES = 20 * 1024 * 1024


class ModelError(Exception):
    pass


# ----------------------------------------------------------------------- inference

class NeuralInpainter:
    """LaMa-style inpainting network: ``image`` (1,3,512,512, 0‥1) + ``mask`` (1,1,512,512, 0/1)."""

    SIZE = 512

    def __init__(self, model_path: Path, intra_threads: int = 0):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if intra_threads > 0:
            so.intra_op_num_threads = intra_threads
        self._session = open_session(model_path, so)
        names = [i.name for i in self._session.get_inputs()]
        if len(names) != 2:
            raise ModelError("unexpected model inputs: " + ", ".join(names))
        lowered = [n.lower() for n in names]
        self._img_name = names[lowered.index("image")] if "image" in lowered else names[0]
        self._mask_name = names[lowered.index("mask")] if "mask" in lowered else names[1]
        self._lock = threading.Lock()   # one heavy inference at a time keeps RAM in check

    def inpaint(self, rgb: np.ndarray, mask: np.ndarray,
                bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Fill every ``mask`` pixel inside a window around ``bbox`` (``rgb``: H×W×3 uint8, edited in place).

        Returns the window ``(x0, y0, x1, y1)`` — all masked pixels inside it have been filled."""
        S = self.SIZE
        H, W = rgb.shape[:2]
        x0, y0, x1, y1 = bbox
        side = max(x1 - x0, y1 - y0, 8)
        need = side * 1.5 + 32                      # window: mask area + context
        scale = min(1.0, S / need)
        win = int(round(S / scale))
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        wx0 = int(round(cx - win / 2.0))
        wy0 = int(round(cy - win / 2.0))
        wx0 = min(max(wx0, 0), max(W - win, 0))
        wy0 = min(max(wy0, 0), max(H - win, 0))
        wx1, wy1 = min(wx0 + win, W), min(wy0 + win, H)

        crop = np.ascontiguousarray(rgb[wy0:wy1, wx0:wx1])
        mcrop = np.ascontiguousarray(mask[wy0:wy1, wx0:wx1])
        ch, cw = crop.shape[:2]
        th, tw = max(1, int(round(ch * scale))), max(1, int(round(cw * scale)))
        if (th, tw) != (ch, cw):
            crop_s = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
            mask_s = cv2.resize(mcrop, (tw, th), interpolation=cv2.INTER_AREA)
        else:
            crop_s, mask_s = crop, mcrop
        mask_s = ((mask_s > 127).astype(np.uint8)) * 255
        th, tw = crop_s.shape[:2]
        th, tw = min(th, S), min(tw, S)
        crop_s, mask_s = crop_s[:th, :tw], mask_s[:th, :tw]

        canvas = cv2.copyMakeBorder(crop_s, 0, S - th, 0, S - tw, cv2.BORDER_REFLECT_101)
        mcanvas = cv2.copyMakeBorder(mask_s, 0, S - th, 0, S - tw, cv2.BORDER_CONSTANT, value=0)

        x = np.ascontiguousarray((canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        m = np.ascontiguousarray((mcanvas > 127).astype(np.float32)[None, None])
        with self._lock:
            out = self._session.run(None, {self._img_name: x, self._mask_name: m})[0]
        out = out[0].transpose(1, 2, 0)
        if float(out.max()) <= 1.5:                  # some exports emit 0‥1, others 0‥255
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)[:th, :tw]
        if (th, tw) != (ch, cw):
            out = cv2.resize(out, (cw, ch), interpolation=cv2.INTER_CUBIC)

        hard = (mcrop > 127)
        soft = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 1.2)
        alpha = np.maximum(soft, hard.astype(np.float32))[..., None]
        blended = crop.astype(np.float32) * (1.0 - alpha) + out.astype(np.float32) * alpha
        rgb[wy0:wy1, wx0:wx1] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
        return wx0, wy0, wx1, wy1


# --------------------------------------------------------------------- management

@dataclass
class ModelStatus:
    state: str = "missing"            # missing | ready | downloading | error
    progress: float = 0.0
    message: str = ""
    size_mb: float = 0.0


class ModelManager:
    """Owns the on-disk model file; validates, downloads, imports and removes it."""

    def __init__(self, directory: Optional[Path] = None):
        self.dir = Path(directory) if directory else models_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._inpainter: Optional[NeuralInpainter] = None
        self._lock = threading.Lock()
        self.min_bytes = MIN_MODEL_BYTES

    @property
    def path(self) -> Path:
        return self.dir / MODEL_FILE

    def is_ready(self) -> bool:
        return self.path.is_file() and self.path.stat().st_size >= self.min_bytes

    def status(self) -> ModelStatus:
        if self.is_ready():
            return ModelStatus("ready", 1.0, str(self.path), self.path.stat().st_size / 1048576.0)
        return ModelStatus("missing")

    def inpainter(self, intra_threads: int = 0) -> Optional[NeuralInpainter]:
        """Loaded model, or ``None`` when it is not installed / cannot be loaded."""
        with self._lock:
            if self._inpainter is None and self.is_ready():
                try:
                    self._inpainter = NeuralInpainter(self.path, intra_threads)
                except Exception:
                    return None
            return self._inpainter

    def remove(self) -> None:
        with self._lock:
            self._inpainter = None
        for p in (self.path, self.path.with_name(self.path.name + ".part")):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def validate(path: Path) -> None:
        """Raise :class:`ModelError` unless ``path`` is a usable image+mask inpainting model."""
        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            sess = open_session(path, so)
        except Exception as exc:
            raise ModelError(f"not a valid ONNX model: {exc}") from exc
        ins = sess.get_inputs()
        if len(ins) != 2:
            raise ModelError("expected an inpainting model with two inputs (image, mask)")
        shapes = [tuple(i.shape) for i in ins]
        if not any(len(s) == 4 and s[1] == 3 for s in shapes) or not any(len(s) == 4 and s[1] == 1 for s in shapes):
            raise ModelError("expected inputs image (1×3×H×W) and mask (1×1×H×W)")

    def import_file(self, src: Path) -> None:
        src = Path(src)
        if not src.is_file():
            raise ModelError("file not found")
        self.validate(src)
        tmp = self.path.with_name(self.path.name + ".part")
        shutil.copyfile(src, tmp)
        os.replace(tmp, self.path)
        with self._lock:
            self._inpainter = None

    def download(self, urls: Sequence[str] = DEFAULT_URLS,
                 progress: Optional[Callable[[float, str], None]] = None,
                 cancel: Optional[threading.Event] = None) -> None:
        """Download from the first URL that works. Only called after the user confirmed."""
        errors: list[str] = []
        tmp = self.path.with_name(self.path.name + ".part")
        for url in urls:
            if not url.lower().startswith("https://"):
                errors.append(f"{url}: only https:// is allowed")
                continue
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "RubberStudio"})
                with urllib.request.urlopen(req, timeout=30) as resp, open(tmp, "wb") as fh:
                    total = int(resp.headers.get("Content-Length") or 0)
                    done = 0
                    host = url.split("/")[2]
                    while True:
                        if cancel is not None and cancel.is_set():
                            raise ModelError("canceled")
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(done / total if total else 0.0, host)
                if tmp.stat().st_size < self.min_bytes:
                    raise ModelError("downloaded file is too small")
                self.validate(tmp)
                os.replace(tmp, self.path)
                with self._lock:
                    self._inpainter = None
                return
            except ModelError as exc:
                errors.append(f"{url}: {exc}")
                if str(exc) == "canceled":
                    break
            except Exception as exc:
                errors.append(f"{url}: {exc}")
            finally:
                try:
                    tmp.unlink()          # gone already after a successful os.replace
                except OSError:
                    pass
        raise ModelError("; ".join(errors) or "download failed")
