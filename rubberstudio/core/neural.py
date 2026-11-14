"""Optional neural inpainting (ONNX) and the manager that downloads / imports / removes the models.

Two engines are supported and they are *not* bundled (nothing is ever downloaded without an explicit user action):

* **LaMa** (~208 MB) — the best quality, especially on busy or patterned backgrounds.
* **MI-GAN** (~28 MB) — a small mobile network, several times faster, good on photos and simple textures.

Without any model the app uses the fast OpenCV fill, which needs no download. Whatever engine runs, it works on 512 × 512
windows cut from the picture at *native resolution* (large areas are tiled, only extreme sizes are scaled down), so
nothing gets blurred by a down- and up-scale round trip.
"""
from __future__ import annotations

import hashlib
import math
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
from .texture import match_grain

Rect = tuple[int, int, int, int]


class ModelError(Exception):
    pass


# --------------------------------------------------------------------------- catalogue

@dataclass(frozen=True)
class ModelSpec:
    id: str                         # "lama" | "migan"
    file: str                       # file name inside the models folder
    label: str                      # short name shown in the UI
    size_mb: int                    # approximate download size
    urls: tuple[str, ...]
    min_bytes: int                  # smaller than this = broken / truncated download
    sha256: str = ""                # pinned checksum of the official file ("" = not checked)
    license: str = "Apache-2.0"


LAMA = ModelSpec(
    id="lama", file="lama.onnx", label="LaMa", size_mb=208,
    urls=("https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
          "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"),
    min_bytes=20 * 1024 * 1024,
    sha256="1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
)
MIGAN = ModelSpec(
    id="migan", file="migan.onnx", label="MI-GAN", size_mb=28,
    urls=("https://huggingface.co/andraniksargsyan/migan/resolve/main/migan.onnx",
          "https://hf-mirror.com/andraniksargsyan/migan/resolve/main/migan.onnx"),
    min_bytes=10 * 1024 * 1024,
    sha256="593eba0b7e04730f1b61c0a3cbca68d97d8d6a7ff5c6a44a7b9d7fcd880fc5ae",
    license="MIT",
)
MODELS: dict[str, ModelSpec] = {LAMA.id: LAMA, MIGAN.id: MIGAN}
PREFERENCE = ("lama", "migan")          # which installed engine "auto" picks first

# kept for older code / tests
MODEL_FILE = LAMA.file
MODEL_LABEL = "LaMa (Carve/LaMa-ONNX, fp32, ≈200 MB)"
DEFAULT_URLS: tuple[str, ...] = LAMA.urls
MIN_MODEL_BYTES = LAMA.min_bytes


# ----------------------------------------------------------------------- inference

class InpaintEngine:
    """Base class: fills masked pixels window by window. Subclasses implement :meth:`_infer` for one 512 × 512 window."""

    SIZE = 512                      # the networks take exactly this window
    CTX = 96                        # context (px) kept around the part of a window that is actually committed
    MAX_TILES = 9                   # more windows than this -> the area is scaled down instead
    kind = "base"

    def __init__(self, model_path: Path, intra_threads: int = 0):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if intra_threads > 0:
            so.intra_op_num_threads = intra_threads
        self._session = open_session(model_path, so)
        inputs = self._session.get_inputs()
        if len(inputs) != 2:
            raise ModelError("unexpected model inputs: " + ", ".join(i.name for i in inputs))
        names = [i.name for i in inputs]
        lowered = [n.lower() for n in names]
        self._img_name = names[lowered.index("image")] if "image" in lowered else names[0]
        self._mask_name = names[lowered.index("mask")] if "mask" in lowered else names[1]
        self._lock = threading.Lock()   # one heavy inference at a time keeps RAM in check
        self._rng = np.random.default_rng(20241114)

    # -- to be provided by subclasses ------------------------------------------------------
    def _infer(self, canvas: np.ndarray, hole: np.ndarray) -> np.ndarray:
        """``canvas`` (512,512,3 uint8), ``hole`` (512,512 uint8, 255 = missing) -> filled (512,512,3 uint8)."""
        raise NotImplementedError

    # -- public ----------------------------------------------------------------------------
    def inpaint(self, rgb: np.ndarray, mask: np.ndarray, bbox: Rect) -> Rect:
        """Fill the ``mask`` (uint8, 255 = fill) pixels inside ``bbox`` of ``rgb`` (H×W×3 uint8, edited in place).

        Mask pixels *outside* ``bbox`` are treated as holes for context but are left for their own call.
        Returns ``bbox`` (clipped): every masked pixel inside it has been filled."""
        H, W = rgb.shape[:2]
        x0, y0, x1, y1 = max(0, bbox[0]), max(0, bbox[1]), min(W, bbox[2]), min(H, bbox[3])
        if x1 <= x0 or y1 <= y0:
            return (x0, y0, max(x0, x1), max(y0, y1))
        core = self.SIZE - 2 * self.CTX
        s = 1.0
        while math.ceil((x1 - x0) * s / core) * math.ceil((y1 - y0) * s / core) > self.MAX_TILES:
            s *= 0.85
        if s < 1.0:
            self._inpaint_scaled(rgb, mask, (x0, y0, x1, y1), s)
        else:
            self._tile_loop(rgb, mask, (x0, y0, x1, y1))
        return (x0, y0, x1, y1)

    # -- internals -------------------------------------------------------------------------
    def _tile_loop(self, work: np.ndarray, mask: np.ndarray, bbox: Rect) -> None:
        """Native-resolution tiling of ``bbox``; ``work`` is edited in place, ``mask`` is only read."""
        H, W = work.shape[:2]
        S = self.SIZE
        x0, y0, x1, y1 = bbox
        core = S - 2 * self.CTX
        nx, ny = max(1, math.ceil((x1 - x0) / core)), max(1, math.ceil((y1 - y0) / core))
        xs = np.linspace(x0, x1, nx + 1).round().astype(int)
        ys = np.linspace(y0, y1, ny + 1).round().astype(int)
        # working copy of the hole mask around the area; committed pixels are cleared so later windows see them filled
        pad = S
        mx0, my0 = max(0, x0 - pad), max(0, y0 - pad)
        mx1, my1 = min(W, x1 + pad), min(H, y1 + pad)
        hole_all = np.ascontiguousarray(mask[my0:my1, mx0:mx1] > 127)
        before = np.ascontiguousarray(work[y0:y1, x0:x1]).copy()
        mask_before = np.ascontiguousarray(mask[y0:y1, x0:x1]).copy()

        for j in range(ny):
            for i in range(nx):
                cx0, cx1, cy0, cy1 = int(xs[i]), int(xs[i + 1]), int(ys[j]), int(ys[j + 1])
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                if not hole_all[cy0 - my0:cy1 - my0, cx0 - mx0:cx1 - mx0].any():
                    continue
                wx0 = int(np.clip((cx0 + cx1) // 2 - S // 2, 0, max(W - S, 0)))
                wy0 = int(np.clip((cy0 + cy1) // 2 - S // 2, 0, max(H - S, 0)))
                wx1, wy1 = min(wx0 + S, W), min(wy0 + S, H)
                crop = np.ascontiguousarray(work[wy0:wy1, wx0:wx1])
                # the window may reach beyond the mask copy only when the image is tiny; then there is simply no hole
                hole = np.zeros(crop.shape[:2], np.uint8)
                ax0, ay0 = max(wx0, mx0), max(wy0, my0)
                ax1, ay1 = min(wx1, mx1), min(wy1, my1)
                hole[ay0 - wy0:ay1 - wy0, ax0 - wx0:ax1 - wx0] = hole_all[ay0 - my0:ay1 - my0, ax0 - mx0:ax1 - mx0] * 255
                ch, cw = crop.shape[:2]
                canvas = cv2.copyMakeBorder(crop, 0, S - ch, 0, S - cw, cv2.BORDER_REFLECT_101) if (ch < S or cw < S) else crop
                hcanvas = cv2.copyMakeBorder(hole, 0, S - ch, 0, S - cw, cv2.BORDER_CONSTANT, value=0) if (ch < S or cw < S) else hole
                with self._lock:
                    out = self._infer(canvas, hcanvas)
                out = out[:ch, :cw]
                commit = np.zeros((ch, cw), np.float32)
                commit[cy0 - wy0:cy1 - wy0, cx0 - wx0:cx1 - wx0] = 1.0
                hard = (hole > 127).astype(np.float32) * commit
                alpha = (np.maximum(cv2.GaussianBlur(hard, (0, 0), 1.0), hard) * commit)[..., None]
                blended = crop.astype(np.float32) * (1.0 - alpha) + out.astype(np.float32) * alpha
                work[wy0:wy1, wx0:wx1] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
                hole_all[cy0 - my0:cy1 - my0, cx0 - mx0:cx1 - mx0] = False

        # a network's output is smoother than a real photo: put back the grain that is missing
        area = np.ascontiguousarray(work[y0:y1, x0:x1])
        matched = match_grain(before, area, mask_before, self._rng)
        if matched is not area:
            work[y0:y1, x0:x1] = matched

    def _inpaint_scaled(self, rgb: np.ndarray, mask: np.ndarray, bbox: Rect, s: float) -> None:
        """Very large areas: work on a reduced copy (still tiled), then bring the result back at full size."""
        H, W = rgb.shape[:2]
        x0, y0, x1, y1 = bbox
        margin = int(math.ceil((self.CTX + 16) / s))
        X0, Y0, X1, Y1 = max(0, x0 - margin), max(0, y0 - margin), min(W, x1 + margin), min(H, y1 + margin)
        crop = np.ascontiguousarray(rgb[Y0:Y1, X0:X1])
        mcrop = np.ascontiguousarray(mask[Y0:Y1, X0:X1])
        ch, cw = crop.shape[:2]
        sw, sh = max(8, int(round(cw * s))), max(8, int(round(ch * s)))
        small = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
        msmall = ((cv2.resize(mcrop, (sw, sh), interpolation=cv2.INTER_AREA) > 40).astype(np.uint8)) * 255
        sb = (int((x0 - X0) * s), int((y0 - Y0) * s), min(sw, int(math.ceil((x1 - X0) * s))), min(sh, int(math.ceil((y1 - Y0) * s))))
        work = small.copy()
        self._tile_loop(work, msmall, sb)
        up = cv2.resize(work, (cw, ch), interpolation=cv2.INTER_CUBIC)
        sel = np.zeros((ch, cw), np.float32)
        sel[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = (mcrop[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] > 127)
        alpha = np.maximum(cv2.GaussianBlur(sel, (0, 0), 1.2), sel)[..., None]
        out = crop.astype(np.float32) * (1.0 - alpha) + up.astype(np.float32) * alpha
        rgb[Y0:Y1, X0:X1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)


class LamaEngine(InpaintEngine):
    """LaMa-style network: ``image`` (1,3,512,512, 0‥1) + ``mask`` (1,1,512,512, 1 = hole)."""
    kind = "lama"

    def _infer(self, canvas: np.ndarray, hole: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray((canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        m = np.ascontiguousarray((hole > 127).astype(np.float32)[None, None])
        out = self._session.run(None, {self._img_name: x, self._mask_name: m})[0][0].transpose(1, 2, 0)
        if float(out.max()) <= 1.5:                  # some exports emit 0‥1, others 0‥255
            out = out * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)


class MiganEngine(InpaintEngine):
    """MI-GAN pipeline: ``image`` uint8 (1,3,H,W) + ``mask`` uint8 (1,1,H,W, 255 = keep, 0 = hole) -> uint8 image."""
    kind = "migan"

    def _infer(self, canvas: np.ndarray, hole: np.ndarray) -> np.ndarray:
        keep = np.where(hole > 127, 0, 255).astype(np.uint8)
        feeds = {self._img_name: np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]),
                 self._mask_name: np.ascontiguousarray(keep[None, None])}
        out = self._session.run(None, feeds)[0][0].transpose(1, 2, 0)
        return np.clip(out, 0, 255).astype(np.uint8)


def make_engine(model_path: Path, intra_threads: int = 0) -> InpaintEngine:
    """The right engine class for the model file (integer image input = MI-GAN pipeline, float = LaMa style)."""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = open_session(model_path, so)
    uint8_in = any("uint8" in i.type for i in sess.get_inputs())
    del sess
    return (MiganEngine if uint8_in else LamaEngine)(model_path, intra_threads)


NeuralInpainter = LamaEngine            # name used by older code and tests


# --------------------------------------------------------------------- management

@dataclass
class ModelStatus:
    state: str = "missing"            # missing | ready | downloading | error
    progress: float = 0.0
    message: str = ""
    size_mb: float = 0.0


class ModelManager:
    """Owns the on-disk model files; validates, downloads, imports and removes them."""

    def __init__(self, directory: Optional[Path] = None):
        self.dir = Path(directory) if directory else models_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._engines: dict[str, InpaintEngine] = {}
        self._lock = threading.Lock()
        self.min_bytes = MIN_MODEL_BYTES          # lowered by tests that use a tiny dummy model
        self.verify_checksum = True

    # -- files -----------------------------------------------------------------------------
    def spec(self, model: str = "lama") -> ModelSpec:
        try:
            return MODELS[model]
        except KeyError:
            raise ModelError(f"unknown model: {model}") from None

    def path_for(self, model: str = "lama") -> Path:
        return self.dir / self.spec(model).file

    @property
    def path(self) -> Path:                     # LaMa, as before
        return self.path_for("lama")

    def _min_bytes(self, model: str) -> int:
        return min(self.min_bytes, self.spec(model).min_bytes)

    def is_ready(self, model: str = "lama") -> bool:
        p = self.path_for(model)
        return p.is_file() and p.stat().st_size >= self._min_bytes(model)

    def status(self, model: str = "lama") -> ModelStatus:
        if self.is_ready(model):
            p = self.path_for(model)
            return ModelStatus("ready", 1.0, str(p), p.stat().st_size / 1048576.0)
        return ModelStatus("missing")

    def installed(self) -> list[str]:
        return [m for m in PREFERENCE if self.is_ready(m)]

    # -- engines ---------------------------------------------------------------------------
    def engine(self, model: str, intra_threads: int = 0) -> Optional[InpaintEngine]:
        """Loaded engine, or ``None`` when that model is not installed / cannot be loaded."""
        with self._lock:
            eng = self._engines.get(model)
            if eng is None and self.is_ready(model):
                try:
                    eng = make_engine(self.path_for(model), intra_threads)
                except Exception:
                    return None
                self._engines[model] = eng
            return eng

    def inpainter(self, intra_threads: int = 0) -> Optional[InpaintEngine]:
        """The best installed engine (LaMa, else MI-GAN), or ``None``."""
        for m in PREFERENCE:
            eng = self.engine(m, intra_threads)
            if eng is not None:
                return eng
        return None

    def resolve(self, choice: str = "auto", intra_threads: int = 0) -> tuple[Optional[InpaintEngine], str]:
        """Engine for a user choice (``auto`` | ``fast`` | ``lama`` | ``migan``) and a note when it had to differ.

        ``fast`` -> no neural engine. A named model that is not installed falls back to the best one that is."""
        choice = (choice or "auto").lower()
        if choice == "fast":
            return None, ""
        if choice in MODELS:
            eng = self.engine(choice, intra_threads)
            if eng is not None:
                return eng, ""
            fb = self.inpainter(intra_threads)
            return fb, f"{MODELS[choice].label} is not installed - used {'the other model' if fb else 'the fast fill'}"
        return self.inpainter(intra_threads), ""

    # -- changes ---------------------------------------------------------------------------
    def remove(self, model: str = "lama") -> None:
        with self._lock:
            self._engines.pop(model, None)
        p = self.path_for(model)
        for q in (p, p.with_name(p.name + ".part")):
            try:
                q.unlink()
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

    def import_file(self, src: Path, model: Optional[str] = None) -> str:
        """Install a model file the user picked. ``model`` = which slot; by default it is guessed from the file
        (integer inputs = MI-GAN, float = LaMa). Returns the slot used."""
        src = Path(src)
        if not src.is_file():
            raise ModelError("file not found")
        self.validate(src)
        if model is None:
            try:
                so = ort.SessionOptions()
                so.log_severity_level = 3
                uint8_in = any("uint8" in i.type for i in open_session(src, so).get_inputs())
            except Exception:
                uint8_in = False
            model = "migan" if uint8_in else "lama"
        dest = self.path_for(model)
        tmp = dest.with_name(dest.name + ".part")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
        with self._lock:
            self._engines.pop(model, None)
        return model

    def download(self, urls: Optional[Sequence[str]] = None,
                 progress: Optional[Callable[[float, str], None]] = None,
                 cancel: Optional[threading.Event] = None, model: str = "lama") -> None:
        """Download from the first URL that works. Only called after the user confirmed."""
        spec = self.spec(model)
        urls = tuple(urls) if urls is not None else spec.urls
        dest = self.path_for(model)
        min_bytes = self._min_bytes(model)
        errors: list[str] = []
        tmp = dest.with_name(dest.name + ".part")
        for url in urls:
            if not url.lower().startswith("https://"):
                errors.append(f"{url}: only https:// is allowed")
                continue
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "RubberStudio"})
                digest = hashlib.sha256()
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
                        digest.update(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(done / total if total else 0.0, host)
                if tmp.stat().st_size < min_bytes:
                    raise ModelError("downloaded file is too small")
                if self.verify_checksum and spec.sha256 and url in spec.urls and digest.hexdigest() != spec.sha256:
                    raise ModelError("checksum differs from the official file")
                self.validate(tmp)
                os.replace(tmp, dest)
                with self._lock:
                    self._engines.pop(model, None)
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
