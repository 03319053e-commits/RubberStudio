"""Scene-text detector: PP-OCRv4 + PP-OCRv5 DBNet (ONNX) with a small, dependency-light post-processor.

Works for Latin, CJK and mixed text. Two detectors that were trained differently are run at several scales and their
boxes are merged, because each one misses things the other finds (very large or widely letter-spaced display text,
tiny captions, stylised fonts). Large / very tall images ("long screenshots") are detected tile by tile at native
resolution so small text is not lost to down-scaling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from .ortutil import open_session
from .paths import resource_path

MODEL_NAME = "ch_PP-OCRv4_det_infer.onnx"
MODEL_NAMES = (MODEL_NAME, "ch_PP-OCRv5_mobile_det.onnx")


@dataclass
class TextBox:
    """A detected text line / block: a rotated rectangle in image pixel coordinates."""
    poly: np.ndarray          # (4, 2) float32
    score: float = 1.0

    def xyxy(self) -> tuple[float, float, float, float]:
        return (float(self.poly[:, 0].min()), float(self.poly[:, 1].min()),
                float(self.poly[:, 0].max()), float(self.poly[:, 1].max()))

    @property
    def height(self) -> float:
        """Length of the short side of the rotated rectangle (≈ text height)."""
        rect = cv2.minAreaRect(self.poly.astype(np.float32))
        return float(min(rect[1]))

    def area(self) -> float:
        return float(cv2.contourArea(self.poly.astype(np.float32)))

    def shifted(self, dx: float, dy: float) -> "TextBox":
        return TextBox(self.poly + np.array([dx, dy], dtype=np.float32), self.score)

    def scaled(self, sx: float, sy: float) -> "TextBox":
        return TextBox(self.poly * np.array([sx, sy], dtype=np.float32), self.score)


def default_model_path() -> Path:
    return resource_path("rubberstudio", "core", "models", MODEL_NAME)


def default_model_paths() -> list[Path]:
    """Every bundled detector that is present (the PP-OCRv4 model is mandatory, the others are a bonus)."""
    return [p for p in (resource_path("rubberstudio", "core", "models", n) for n in MODEL_NAMES) if p.is_file()]


def _round32(v: float) -> int:
    return max(32, int(round(v / 32.0)) * 32)


class TextDetector:
    """DBNet text detector(s). One instance is safe to share between threads."""

    def __init__(self, model_path: Optional[Path] = None, intra_threads: int = 0,
                 extra_models: Optional[Sequence[Path]] = None):
        if model_path is not None:
            paths = [Path(model_path)] + [Path(p) for p in (extra_models or [])]
        else:
            paths = default_model_paths() + [Path(p) for p in (extra_models or [])]
            if not paths:
                paths = [default_model_path()]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"text detection model not found: {path}")
        self._sessions: list[tuple[ort.InferenceSession, str]] = []
        for path in paths:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if intra_threads > 0:
                so.intra_op_num_threads = intra_threads
            sess = open_session(path, so)
            self._sessions.append((sess, sess.get_inputs()[0].name))
        # tuning (PP-OCRv4 defaults, as used by RapidOCR)
        self.limit_min_side = 736
        self.max_side = 2048
        self.tile = 1536
        self.tile_overlap = 192
        self.thresh = 0.3
        self.unclip_ratio = 1.6

    @property
    def model_count(self) -> int:
        return len(self._sessions)

    # ------------------------------------------------------------------ public

    def detect(self, rgb: np.ndarray, *, box_thresh: float = 0.5,
               region: Optional[tuple[int, int, int, int]] = None,
               cancel: Optional[Callable[[], bool]] = None, thorough: bool = True) -> list[TextBox]:
        """Find text in ``rgb`` (H×W×3/4 uint8). ``region`` = (x0, y0, x1, y1) pixel rectangle to search.

        ``thorough`` adds a coarse pass (huge display lettering) and, for small pictures, a fine pass (tiny captions)
        on top of the normal one, for every loaded model."""
        h, w = rgb.shape[:2]
        ox = oy = 0
        if region is not None:
            x0, y0, x1, y1 = region
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(w, int(x1)), min(h, int(y1))
            if x1 - x0 < 8 or y1 - y0 < 8:
                return []
            rgb = rgb[y0:y1, x0:x1]
            ox, oy = x0, y0
            h, w = rgb.shape[:2]
        img = np.ascontiguousarray(rgb[..., :3])

        boxes: list[TextBox] = []
        for (tx0, ty0, tx1, ty1) in self._tiles(w, h):
            tile = img[ty0:ty1, tx0:tx1]
            for scale in self._scales(ty1 - ty0, tx1 - tx0, thorough):
                for sess, name in self._sessions:
                    if cancel is not None and cancel():
                        break
                    for b in self._detect_single(tile, box_thresh, sess, name, scale):
                        boxes.append(b.shifted(tx0 + ox, ty0 + oy))
        boxes = _merge_overlaps(_hysteresis(_suppress_duplicates(boxes), box_thresh))
        boxes.sort(key=lambda b: (round(b.xyxy()[1] / 12.0), b.xyxy()[0]))
        return boxes

    def text_score(self, rgb: np.ndarray, poly: np.ndarray, text_h: float) -> float:
        """How strongly the detectors still see text inside ``poly`` (polygon in the pixel coordinates of ``rgb``).

        This is the "did it work?" question for an erase mask: the crop is shown to every model at the size the text
        is normally read at, and the best score of any box whose centre lies inside ``poly`` is returned (``0.0`` when
        nothing is found). Text that is really gone scores 0; a mask that missed the strokes scores like the original."""
        h, w = rgb.shape[:2]
        if h < 8 or w < 8:
            return 0.0
        img = np.ascontiguousarray(rgb[..., :3])
        inside = np.zeros((h, w), np.uint8)
        cv2.fillPoly(inside, [np.round(np.asarray(poly, np.float32)).astype(np.int32)], 255)
        base = float(np.clip(32.0 / max(float(text_h), 4.0), 0.6, 3.0))
        best = 0.0
        for sess, name in self._sessions:
            for scale in (base, base * 1.5):
                scale = min(scale, self.max_side / float(max(h, w)))
                for b in self._detect_single(img, 0.3, sess, name, scale):
                    cx, cy = b.poly.mean(axis=0)
                    ix, iy = int(cx), int(cy)
                    if 0 <= ix < w and 0 <= iy < h and inside[iy, ix]:
                        best = max(best, b.score)
        return best

    # ----------------------------------------------------------------- internal

    def _tiles(self, w: int, h: int) -> list[tuple[int, int, int, int]]:
        if max(w, h) <= self.max_side:
            return [(0, 0, w, h)]
        xs = _spans(w, self.tile, self.tile_overlap)
        ys = _spans(h, self.tile, self.tile_overlap)
        return [(x0, y0, x1, y1) for (y0, y1) in ys for (x0, x1) in xs]

    def _scales(self, th: int, tw: int, thorough: bool) -> list[float]:
        """Input scales for one tile: the normal one, plus a coarse (big text) and a fine (small text) one."""
        base = 1.0
        if min(th, tw) < self.limit_min_side:
            base = self.limit_min_side / float(min(th, tw))
        if max(th, tw) * base > self.max_side:
            base = self.max_side / float(max(th, tw))
        scales = [base]
        if thorough:
            coarse = min(base * 0.65, 640.0 / float(max(th, tw)))
            if coarse * min(th, tw) >= 96 and coarse < base * 0.92:
                scales.append(coarse)
            if max(th, tw) * base * 1.6 <= 1800:
                scales.append(base * 1.6)
        return scales

    def _detect_single(self, tile: np.ndarray, box_thresh: float, sess: ort.InferenceSession, name: str,
                       scale: float) -> list[TextBox]:
        th, tw = tile.shape[:2]
        if th < 8 or tw < 8:
            return []
        rh, rw = _round32(th * scale), _round32(tw * scale)
        interp = cv2.INTER_AREA if (rh < th or rw < tw) else cv2.INTER_LINEAR
        resized = cv2.resize(tile, (rw, rh), interpolation=interp)

        x = resized.astype(np.float32) * (1.0 / 255.0)
        x = (x - 0.5) / 0.5
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        prob = sess.run(None, {name: x})[0][0, 0]   # InferenceSession.run is thread-safe
        return self._postprocess(prob, tw, th, weak_threshold(box_thresh))

    def _postprocess(self, prob: np.ndarray, orig_w: int, orig_h: int, min_score: float) -> list[TextBox]:
        """Boxes scoring at least ``min_score`` (callers pass the *weak* threshold; see ``_hysteresis``)."""
        ph, pw = prob.shape
        seg = (prob > self.thresh).astype(np.uint8) * 255
        seg = cv2.dilate(seg, np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out: list[TextBox] = []
        sx, sy = orig_w / float(pw), orig_h / float(ph)
        for c in contours[:1000]:
            if len(c) < 4:
                continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), ang = rect
            if min(rw, rh) < 3:
                continue
            pts = cv2.boxPoints(rect)
            score = _box_score(prob, pts)
            if score < min_score:
                continue
            d = (rw * rh) * self.unclip_ratio / (2.0 * (rw + rh))
            grown = ((cx, cy), (rw + 2 * d, rh + 2 * d), ang)
            if min(grown[1]) < 5:
                continue
            poly = cv2.boxPoints(grown).astype(np.float32)
            poly[:, 0] = np.clip(poly[:, 0] * sx, 0, orig_w - 1)
            poly[:, 1] = np.clip(poly[:, 1] * sy, 0, orig_h - 1)
            out.append(TextBox(poly, float(score)))
        return out


def weak_threshold(box_thresh: float) -> float:
    """Score above which a box is kept *if* it sits right next to a confident one."""
    return max(0.15, 0.4 * float(box_thresh))


def _hysteresis(boxes: Sequence[TextBox], box_thresh: float) -> list[TextBox]:
    """Canny-style box selection.

    Tiny glyphs next to real text — "©", bullets, dots, a lone punctuation mark — score low on their own and would be
    left behind as a speck. A low-scoring box is therefore kept when it lies on the same line as a confident box
    (centre inside that box's line band, horizontal gap at most one text height). Isolated low scorers stay dropped.
    A kept fragment is widened to the whole line height: the detector only marks the dense core of such a glyph."""
    strong = [b for b in boxes if b.score >= box_thresh]
    pending = [b for b in boxes if b.score < box_thresh]
    kept = list(strong)
    if not strong or not pending:
        return kept
    for _ in range(2):                       # a second pass lets "© •" chains hang on one another
        taken: dict[int, TextBox] = {}
        for w in pending:
            wx0, wy0, wx1, wy1 = w.xyxy()
            wcy = 0.5 * (wy0 + wy1)
            for s in kept:
                sx0, sy0, sx1, sy1 = s.xyxy()
                sh = sy1 - sy0
                gap = max(sx0 - wx1, wx0 - sx1, 0.0)
                small = (wy1 - wy0) <= 1.0 * sh and (wx1 - wx0) <= 2.5 * sh       # a fragment, not a line or blob
                if small and sy0 - 0.25 * sh <= wcy <= sy1 + 0.25 * sh and gap <= 1.0 * sh:
                    taken[id(w)] = _widen_fragment(w, s)
                    break
        if not taken:
            break
        kept += list(taken.values())
        pending = [w for w in pending if id(w) not in taken]
    return kept


def _widen_fragment(frag: TextBox, anchor: TextBox) -> TextBox:
    x0, y0, x1, y1 = frag.xyxy()
    ax0, ay0, ax1, ay1 = anchor.xyxy()
    g = 0.3 * (ay1 - ay0)
    ang = cv2.minAreaRect(anchor.poly.astype(np.float32))[2] % 90.0
    if ang < 6.0 or ang > 84.0:              # horizontal line: take the full line band
        y0, y1 = min(y0 - g, ay0), max(y1 + g, ay1)
    else:                                    # slanted text: just grow the fragment
        y0, y1 = y0 - g, y1 + g
    x0, x1 = x0 - g, x1 + g
    poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    return TextBox(poly, frag.score)


def _spans(length: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    if length <= tile:
        return [(0, length)]
    step = tile - overlap
    starts = list(range(0, length - tile, step)) + [length - tile]
    return [(s, s + tile) for s in starts]


def _box_score(prob: np.ndarray, pts: np.ndarray) -> float:
    h, w = prob.shape
    xmin = int(np.clip(np.floor(pts[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(pts[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(pts[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(pts[:, 1].max()), 0, h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
    shifted = pts.copy()
    shifted[:, 0] -= xmin
    shifted[:, 1] -= ymin
    cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)
    return float(cv2.mean(prob[ymin:ymax + 1, xmin:xmax + 1], mask)[0])


def _suppress_duplicates(boxes: Sequence[TextBox]) -> list[TextBox]:
    """Drop boxes that are mostly covered by a better one (the same text found by another pass, model or tile).

    Best score first, so a confident small box is never swallowed by a bigger but doubtful one."""
    if len(boxes) < 2:
        return list(boxes)
    order = sorted(range(len(boxes)), key=lambda i: (-boxes[i].score, -boxes[i].area()))
    kept: list[TextBox] = []
    for i in order:
        b = boxes[i]
        bx0, by0, bx1, by1 = b.xyxy()
        barea = max((bx1 - bx0) * (by1 - by0), 1.0)
        dup = False
        for k in kept:
            kx0, ky0, kx1, ky1 = k.xyxy()
            iw = min(bx1, kx1) - max(bx0, kx0)
            ih = min(by1, ky1) - max(by0, ky0)
            if iw > 0 and ih > 0 and (iw * ih) / barea > 0.8:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


def _axis_aligned(b: TextBox) -> bool:
    ang = cv2.minAreaRect(b.poly.astype(np.float32))[2] % 90.0
    return ang < 4.0 or ang > 86.0


def _merge_overlaps(boxes: Sequence[TextBox]) -> list[TextBox]:
    """One box per text line: overlapping axis-aligned boxes (the same line seen by different passes / models, each
    with slightly different reach and height) become one box — the widest horizontal reach of the group, with the
    typical (median) vertical extent. Rotated boxes are left alone."""
    n = len(boxes)
    if n < 2:
        return list(boxes)
    rects = [b.xyxy() for b in boxes]
    flat = [_axis_aligned(b) for b in boxes]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        if not flat[i]:
            continue
        for j in range(i + 1, n):
            if not flat[j]:
                continue
            ax0, ay0, ax1, ay1 = rects[i]
            bx0, by0, bx1, by1 = rects[j]
            iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
            if iw <= 0 or ih <= 0:
                continue
            small = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
            if iw * ih >= 0.5 * small and ih >= 0.6 * min(ay1 - ay0, by1 - by0):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out: list[TextBox] = []
    for members in groups.values():
        if len(members) == 1:
            out.append(boxes[members[0]])
            continue
        xs0 = min(rects[i][0] for i in members)
        xs1 = max(rects[i][2] for i in members)
        ys0 = float(np.median([rects[i][1] for i in members]))
        ys1 = float(np.median([rects[i][3] for i in members]))
        poly = np.array([[xs0, ys0], [xs1, ys0], [xs1, ys1], [xs0, ys1]], np.float32)
        out.append(TextBox(poly, max(boxes[i].score for i in members)))
    return out
