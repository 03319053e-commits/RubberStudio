"""Text eraser: detected boxes -> stroke mask -> fill (fast OpenCV, or neural when it pays off).

Pipeline (same idea as the iOS app): find text boxes, build a mask of just the glyph strokes
inside each box, fill the mask from its surroundings, and leave every other pixel untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from .neural import NeuralInpainter
from .textdet import TextBox, TextDetector

Rect = tuple[int, int, int, int]          # x0, y0, x1, y1 (pixels)
RelRect = tuple[float, float, float, float]   # x, y, w, h (0‥1)


@dataclass
class EraseOptions:
    box_thresh: float = 0.5          # detector confidence threshold
    pad: float = 0.15                # grow each box by this × text height before looking for strokes
    grow: float = 0.12               # dilate the stroke mask by this × text height (covers anti-alias halos)
    whole_box: bool = False          # fill entire boxes instead of just glyph strokes
    side_glyphs: bool = True         # also take small symbols (©, •, @ …) hugging the end of a detected line
    fast_radius: int = 3             # OpenCV inpaint radius
    use_neural: bool = True          # allow the neural model (when installed) on complex backgrounds
    always_neural: bool = False      # force the neural model for every box (slower)
    complexity_texture: float = 7.0  # background "fine detail" above this       -> complex
    complexity_edge: float = 10.0    # strong-edge pixels in the ring beside text -> complex


@dataclass
class EraseResult:
    image: np.ndarray
    boxes: list = field(default_factory=list)
    neural_count: int = 0
    fast_count: int = 0
    note: Optional[str] = None

    @property
    def found(self) -> bool:
        return bool(self.boxes)


def rel_to_px(rel: RelRect, width: int, height: int) -> Rect:
    x, y, w, h = rel
    x0 = int(round(max(0.0, x) * width))
    y0 = int(round(max(0.0, y) * height))
    x1 = int(round(min(1.0, x + w) * width))
    y1 = int(round(min(1.0, y + h) * height))
    return x0, y0, x1, y1


def _odd(v: int) -> int:
    return v if v % 2 == 1 else v + 1


def build_stroke_mask(rgb: np.ndarray, boxes: Sequence[TextBox], opts: EraseOptions) -> np.ndarray:
    """Return a full-size uint8 mask (0/255) of the glyph strokes inside ``boxes``."""
    H, W = rgb.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    for box in boxes:
        rect = cv2.minAreaRect(box.poly.astype(np.float32))
        (cx, cy), (rw, rh), ang = rect
        text_h = max(min(rw, rh), 4.0)
        pad = opts.pad * text_h
        grown = ((cx, cy), (rw + 2 * pad, rh + 2 * pad), ang)
        poly = cv2.boxPoints(grown)
        g = int(np.clip(round(opts.grow * text_h), 1, 10))

        x0 = int(max(0, np.floor(poly[:, 0].min()) - g - 2))
        y0 = int(max(0, np.floor(poly[:, 1].min()) - g - 2))
        x1 = int(min(W, np.ceil(poly[:, 0].max()) + g + 3))
        y1 = int(min(H, np.ceil(poly[:, 1].max()) + g + 3))
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue

        local = poly - np.array([x0, y0], dtype=np.float32)
        region = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(region, [np.round(local).astype(np.int32)], 255)

        strokes = None
        if not opts.whole_box:
            strokes = _strokes_in_region(np.ascontiguousarray(rgb[y0:y1, x0:x1]), region, text_h, pad)
        if strokes is None:
            strokes = region.copy()
        else:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
            strokes = cv2.dilate(strokes, k)
            strokes = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            # a speck of glyph colour left inside a thick letter would be *propagated* by the fill — close such holes
            strokes = _fill_small_holes(strokes, (0.5 * text_h) ** 2)
            # never leak past the (slightly grown) box
            limit = cv2.dilate(region, k)
            strokes = cv2.bitwise_and(strokes, limit)
        mask[y0:y1, x0:x1] |= strokes
    return mask


def _fill_small_holes(mask: np.ndarray, max_area: float) -> np.ndarray:
    """Fill enclosed holes of at most ``max_area`` pixels (counters of letters, specks inside thick strokes)."""
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return mask
    for i, c in enumerate(cnts):
        if hier[0][i][3] >= 0 and cv2.contourArea(c) <= max_area:
            cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    return mask


def _band_background(roi: np.ndarray, region: np.ndarray, pad: float) -> Optional[tuple[np.ndarray, float]]:
    """Background model for flat or smoothly shaded surroundings.

    The padding band (outer ``pad`` px of the grown box) lies outside the detected text, so it is almost pure
    background. A plane per colour channel is fitted to it (robustly, dropping the worst 15 %) and evaluated over the
    whole ROI. Unlike a local median this does not break down when glyphs cover more than half of a window (heavy
    display fonts). Returns ``(background image, residual sigma)`` or ``None`` when the band is too small / busy."""
    k = max(3, int(round(pad)) * 2 + 1)
    inner = cv2.erode(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ys, xs = np.nonzero((region > 0) & (inner == 0))
    if xs.size < 60:
        return None
    if xs.size > 4000:
        sel = np.random.default_rng(0).choice(xs.size, 4000, replace=False)
        ys, xs = ys[sel], xs[sel]
    h, w = roi.shape[:2]
    sx, sy = max(w, 1), max(h, 1)
    A = np.stack([xs / sx, ys / sy, np.ones_like(xs, dtype=np.float64)], 1)
    Y = roi[ys, xs].astype(np.float64)
    keep = np.ones(len(xs), bool)
    coef = None
    for _ in range(2):
        coef, *_r = np.linalg.lstsq(A[keep], Y[keep], rcond=None)
        res = np.abs(A @ coef - Y).max(axis=1)
        keep = res <= np.quantile(res, 0.85)
        if keep.sum() < 30:
            return None
    res = np.abs(A[keep] @ coef - Y[keep])
    sigma = float(1.4826 * np.median(res.max(axis=1)) + 1e-3)
    gx, gy = np.meshgrid(np.arange(w) / sx, np.arange(h) / sy)
    grid = np.stack([gx, gy, np.ones_like(gx)], -1).reshape(-1, 3)
    bg = np.clip(grid @ coef, 0, 255).reshape(h, w, 3)
    return bg.astype(np.float32), sigma


def _strokes_from_diff(diff: np.ndarray, inside: np.ndarray, floor: float) -> Optional[np.ndarray]:
    vals = diff[inside]
    if vals.size < 16:
        return None
    otsu, _ = cv2.threshold(np.clip(vals, 0, 255).astype(np.uint8).reshape(-1, 1), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(float(otsu), floor)
    strokes = ((diff > thr) & inside).astype(np.uint8) * 255
    ratio = float(strokes.sum() / 255.0) / float(inside.sum())
    if ratio < 0.015 or ratio > 0.65:
        return None            # not convincing
    return strokes


def _strokes_in_region(roi: np.ndarray, region: np.ndarray, text_h: float, pad: float = 0.0) -> Optional[np.ndarray]:
    """Glyph strokes = pixels that differ from the background.

    1. Flat / smooth surroundings: distance to a plane fitted on the padding band (robust to dense, heavy glyphs).
    2. Anything else: distance to a local median of a window ≥ the text height."""
    inside = region > 0
    if pad >= 2.0:
        model = _band_background(roi, region, pad)
        if model is not None and model[1] <= 7.0:
            bg, sigma = model
            diff = np.abs(roi.astype(np.float32) - bg).max(axis=2)
            strokes = _strokes_from_diff(diff, inside, floor=max(8.0, 5.0 + 4.0 * sigma))
            if strokes is not None:
                return strokes
    k = _odd(int(np.clip(text_h * 0.9, 11, 121)))
    k = min(k, _odd(max(3, min(roi.shape[:2]) // 1)))
    if k < 3:
        return None
    bg = cv2.medianBlur(roi, k if k <= 255 else 255)
    diff = cv2.absdiff(roi, bg).max(axis=2).astype(np.float32)
    vals = diff[inside]
    if vals.size < 16:
        return None
    return _strokes_from_diff(diff, inside, floor=max(float(np.median(vals)) * 2.5 + 3.0, 8.0))


# ------------------------------------------------- symbols the detector has no response for

_SIDE_REACH = 2.0      # look up to this many line heights beside a text line ...
_SIDE_GAP = 1.0        # ... for symbols starting within this many line heights of the line end (or of the previous one)
_SIDE_MAX = 4          # at most this many symbols per line end
_SIDE_MAX_SIGMA = 5.0  # background must be this smooth (residual of a plane fitted around the line, in grey levels)


def _is_horizontal_line(box: TextBox) -> bool:
    x0, y0, x1, y1 = box.xyxy()
    if (x1 - x0) < 1.2 * (y1 - y0):
        return False
    ang = cv2.minAreaRect(box.poly.astype(np.float32))[2] % 90.0
    return ang < 5.0 or ang > 85.0


def glyphs_beside_lines(rgb: np.ndarray, boxes: Sequence[TextBox], box_thresh: float,
                        bounds: Optional[Rect] = None) -> list[TextBox]:
    """Small symbols (©, ®, •, @ …) right beside a text line that the detector gave no response for at all.

    The network ignores some isolated tiny glyphs — and whether it does depends on pixel-level details such as how the
    font was rasterised — so they would be the speck left behind next to an otherwise clean line. Pixels beside the end
    of a confident, horizontal line are therefore checked directly: a compact blob of the *same ink colour*, glyph-sized
    (clearly smaller than the line), sitting on the line and close to it, counts as part of the text. Anything bigger
    (icons, objects), another colour, off the line or far away is left alone — and so is anything on a busy background."""
    H, W = rgb.shape[:2]
    bx0, by0, bx1, by1 = bounds if bounds is not None else (0, 0, W, H)
    bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(W, bx1), min(H, by1)
    rects = [list(b.xyxy()) for b in boxes]
    found: list[TextBox] = []
    for b in boxes:
        if b.score < box_thresh or not _is_horizontal_line(b):
            continue
        x0, y0, x1, y1 = b.xyxy()
        bh, bw = y1 - y0, x1 - x0
        if bh < 10 or bw > 40 * bh:
            continue
        reach = int(round(_SIDE_REACH * bh))
        band_pad = int(round(0.1 * bh))
        rx0, rx1 = max(bx0, int(np.floor(x0)) - reach), min(bx1, int(np.ceil(x1)) + reach)
        ry0, ry1 = max(by0, int(np.floor(y0)) - band_pad), min(by1, int(np.ceil(y1)) + band_pad)
        if rx1 - rx0 < 8 or ry1 - ry0 < 8:
            continue
        m = 3
        ox0, oy0 = max(0, rx0 - m), max(0, ry0 - m)
        ox1, oy1 = min(W, rx1 + m), min(H, ry1 + m)
        roi = np.ascontiguousarray(rgb[oy0:oy1, ox0:ox1, :3])
        region = np.zeros(roi.shape[:2], np.uint8)
        region[ry0 - oy0:ry1 - oy0, rx0 - ox0:rx1 - ox0] = 255
        # Only on flat or smoothly shaded surroundings, where "differs from the background" really means "is ink".
        # Photos / busy textures are skipped: there every speck would look like a symbol.
        model = _band_background(roi, region, 0.15 * bh)
        if model is None or model[1] > _SIDE_MAX_SIGMA:
            continue
        bg, sigma = model
        diff = np.abs(roi.astype(np.float32) - bg).max(axis=2)
        strokes = _strokes_from_diff(diff, region > 0, floor=max(8.0, 5.0 + 4.0 * sigma))
        if strokes is None:
            continue

        # ink colour of the line itself and the background right around it
        lx0, lx1 = int(np.floor(x0)) - ox0, int(np.ceil(x1)) - ox0
        own = np.zeros_like(strokes)
        own[:, max(0, lx0):max(0, lx1)] = strokes[:, max(0, lx0):max(0, lx1)]
        if int((own > 0).sum()) < 20:
            continue
        ink = np.median(roi[own > 0], axis=0)
        near = cv2.dilate(strokes, np.ones((5, 5), np.uint8)) == 0
        near &= region > 0
        if int(near.sum()) < 30:
            continue
        contrast = float(np.abs(ink - np.median(roi[near], axis=0)).max())
        if contrast < 24.0:
            continue

        wing = strokes.copy()
        wing[:, max(0, lx0):max(0, lx1)] = 0            # the line itself is handled by its own box
        for ex0, ey0, ex1, ey1 in rects:                  # ... and so is every other box
            if ex1 > ox0 and ex0 < ox1 and ey1 > oy0 and ey0 < oy1:
                wing[max(0, int(ey0) - oy0):max(0, int(np.ceil(ey1)) - oy0),
                     max(0, int(ex0) - ox0):max(0, int(np.ceil(ex1)) - ox0)] = 0
        if not wing.any():
            continue

        k = max(3, int(round(0.12 * bh)) | 1)
        n, lab, _stats, _c = cv2.connectedComponentsWithStats(
            cv2.dilate(wing, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), connectivity=8)
        cands = []
        for j in range(1, n):
            ys, xs = np.nonzero((lab == j) & (wing > 0))
            if xs.size < 6:
                continue
            ix0, ix1 = int(xs.min()), int(xs.max()) + 1
            iy0, iy1 = int(ys.min()), int(ys.max()) + 1
            w_, h_ = ix1 - ix0, iy1 - iy0
            if not (0.12 * bh <= h_ <= 0.8 * bh and 0.08 * bh <= w_ <= 1.2 * bh):
                continue                                  # not glyph-sized (specks, icons, objects)
            top, bot = oy0 + iy0, oy0 + iy1
            if min(bot, y1) - max(top, y0) < 0.6 * h_:
                continue                                  # not sitting on the line
            if xs.size / float(w_ * h_) > 0.8 and min(w_, h_) > 0.5 * bh:
                continue                                  # a solid blob, not a glyph
            if float(np.abs(np.median(roi[ys, xs], axis=0) - ink).max()) > 0.5 * contrast:
                continue                                  # a different colour than the text
            cands.append((ox0 + ix0, top, ox0 + ix1, bot))

        for side in (-1, 1):
            edge = x0 if side < 0 else x1
            taken = 0
            pool = [c for c in cands if (c[2] <= x0 + 1 if side < 0 else c[0] >= x1 - 1)]
            while pool and taken < _SIDE_MAX:
                nxt = min(pool, key=(lambda c: edge - c[2]) if side < 0 else (lambda c: c[0] - edge))
                pool.remove(nxt)
                gap = (edge - nxt[2]) if side < 0 else (nxt[0] - edge)
                if gap > _SIDE_GAP * bh:
                    break
                g = 0.25 * bh
                fx0, fx1 = max(bx0, nxt[0] - g), min(bx1 - 1, nxt[2] + g)
                fy0, fy1 = max(by0, min(nxt[1] - g, y0)), min(by1 - 1, max(nxt[3] + g, y1))
                poly = np.array([[fx0, fy0], [fx1, fy0], [fx1, fy1], [fx0, fy1]], np.float32)
                found.append(TextBox(poly, b.score))
                rects.append([fx0, fy0, fx1, fy1])
                edge = nxt[0] if side < 0 else nxt[2]
                taken += 1
    return found


# ------------------------------------------------------------------------ clustering

def _rect_union(a: Rect, b: Rect) -> Rect:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _touching(a: Rect, b: Rect, gap: int) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def cluster_rects(rects: Sequence[Rect], max_span: int, gap: int = 6) -> list[Rect]:
    """Greedy grouping of nearby rectangles; groups stay ≤ ``max_span`` unless they touch."""
    clusters: list[Rect] = []
    for r in sorted(rects, key=lambda q: (q[1], q[0])):
        merged = False
        for i, c in enumerate(clusters):
            u = _rect_union(c, r)
            if _touching(c, r, gap) or (u[2] - u[0] <= max_span and u[3] - u[1] <= max_span):
                clusters[i] = u
                merged = True
                break
        if not merged:
            clusters.append(r)
    # touching clusters may have grown into each other
    changed = True
    while changed and len(clusters) > 1:
        changed = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if _touching(clusters[i], clusters[j], 0):
                    clusters[i] = _rect_union(clusters[i], clusters[j])
                    del clusters[j]
                    changed = True
                    break
            if changed:
                break
    return clusters


def _mask_bbox(mask: np.ndarray, area: Rect) -> Optional[Rect]:
    x0, y0, x1, y1 = area
    sub = mask[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub.any(axis=1))[0], np.nonzero(sub.any(axis=0))[0]
    if ys.size == 0 or xs.size == 0:
        return None
    return x0 + int(xs[0]), y0 + int(ys[0]), x0 + int(xs[-1]) + 1, y0 + int(ys[-1]) + 1


# ------------------------------------------------------------------------- fillers

def background_complexity(rgb: np.ndarray, mask: np.ndarray, bbox: Rect) -> tuple[float, float]:
    """(fine texture energy, number of strong-edge pixels) in the ring of background around the masked pixels.

    Both are colour-aware (max over channels) so a red/blue boundary with equal brightness still counts."""
    H, W = mask.shape
    m = 14
    x0, y0 = max(0, bbox[0] - m), max(0, bbox[1] - m)
    x1, y1 = min(W, bbox[2] + m), min(H, bbox[3] + m)
    cm = mask[y0:y1, x0:x1]
    f = np.ascontiguousarray(rgb[y0:y1, x0:x1]).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner = cv2.dilate(cm, kernel, iterations=1)
    outer = cv2.dilate(cm, kernel, iterations=4)
    ring = (outer > 0) & (inner == 0)
    if int(ring.sum()) < 30:
        return 0.0, 0.0
    hf = np.abs(f - cv2.GaussianBlur(f, (0, 0), 1.5)).max(axis=2)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy).max(axis=2)
    return float(hf[ring].mean()), float((grad[ring] > 150.0).sum())


def _ring_noise_sigma(roi: np.ndarray, mask: np.ndarray) -> float:
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ring = (cv2.dilate(mask, kernel, iterations=4) > 0) & (cv2.dilate(mask, kernel, iterations=1) == 0)
    if int(ring.sum()) < 30:
        return 0.0
    hf = (gray - cv2.GaussianBlur(gray, (0, 0), 1.0))[ring]
    mad = float(np.median(np.abs(hf - np.median(hf))))
    return 1.4826 * mad


def fast_fill(rgb: np.ndarray, mask: np.ndarray, bbox: Rect, radius: int,
              rng: Optional[np.random.Generator] = None) -> None:
    """Telea inpainting on the masked pixels of ``rgb`` (edited in place) + grain matched to the surroundings."""
    H, W = mask.shape
    margin = radius * 4 + 18
    x0, y0 = max(0, bbox[0] - margin), max(0, bbox[1] - margin)
    x1, y1 = min(W, bbox[2] + margin), min(H, bbox[3] + margin)
    roi = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    m = np.ascontiguousarray(mask[y0:y1, x0:x1])
    filled = cv2.inpaint(roi, m, float(radius), cv2.INPAINT_TELEA)
    sigma = _ring_noise_sigma(roi, m)
    sel = m > 0
    if sigma > 0.7:
        rng = rng or np.random.default_rng(1234)
        h, w = m.shape
        lum = rng.normal(0.0, sigma, (h, w, 1)).astype(np.float32)
        chroma = rng.normal(0.0, sigma * 0.35, (h, w, 3)).astype(np.float32)
        noise = cv2.GaussianBlur(lum + chroma, (0, 0), 0.55).reshape(h, w, 3) * 1.25
        vals = filled.astype(np.float32) + noise
        filled = np.where(sel[..., None], np.clip(vals, 0, 255), filled).astype(np.uint8)
    view = rgb[y0:y1, x0:x1]
    view[sel] = filled[sel]


# ----------------------------------------------------------------------- the eraser

class TextEraser:
    def __init__(self, detector: Optional[TextDetector] = None,
                 neural: Optional[NeuralInpainter] = None):
        self.detector = detector
        self.neural = neural
        self._rng = np.random.default_rng(20241114)

    def ensure_detector(self) -> TextDetector:
        if self.detector is None:
            self.detector = TextDetector()
        return self.detector

    def detect(self, rgba: np.ndarray, *, region: Optional[Rect] = None,
               opts: Optional[EraseOptions] = None,
               cancel: Optional[Callable[[], bool]] = None) -> list[TextBox]:
        opts = opts or EraseOptions()
        boxes = self.ensure_detector().detect(rgba, box_thresh=opts.box_thresh, region=region, cancel=cancel)
        if opts.side_glyphs and boxes and not (cancel is not None and cancel()):
            extra = glyphs_beside_lines(rgba, boxes, opts.box_thresh, region)
            if extra:
                boxes = sorted(list(boxes) + extra, key=lambda b: (round(b.xyxy()[1] / 12.0), b.xyxy()[0]))
        return boxes

    def erase(self, rgba: np.ndarray, *, boxes: Optional[Sequence[TextBox]] = None,
              region: Optional[Rect] = None, opts: Optional[EraseOptions] = None,
              progress: Optional[Callable[[int, int], None]] = None,
              cancel: Optional[Callable[[], bool]] = None) -> EraseResult:
        """Erase text from ``rgba`` and return a new image (input is left untouched)."""
        opts = opts or EraseOptions()
        if boxes is None:
            boxes = self.detect(rgba, region=region, opts=opts, cancel=cancel)
        boxes = list(boxes)
        if not boxes:
            return EraseResult(rgba.copy(), [], 0, 0, None)

        out = rgba.copy()
        rgb = out[..., :3]
        H, W = rgb.shape[:2]
        work = np.ascontiguousarray(rgb)
        mask = build_stroke_mask(work, boxes, opts)

        use_neural = self.neural is not None and (opts.use_neural or opts.always_neural)
        rects: list[Rect] = []
        for b in boxes:
            x0, y0, x1, y1 = b.xyxy()
            pad = int(opts.pad * b.height) + 12
            rects.append((max(0, int(x0) - pad), max(0, int(y0) - pad),
                          min(W, int(x1) + pad), min(H, int(y1) + pad)))
        clusters = cluster_rects(rects, max_span=380 if use_neural else 160, gap=4)

        neural_count = fast_count = 0
        note = None
        total = len(clusters)
        # the network gets a slightly bigger mask so it has room to hide anti-alias halos
        grown_all = (cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                     if use_neural else None)
        for i, area in enumerate(clusters):
            if cancel is not None and cancel():
                note = "canceled"
                break
            bbox = _mask_bbox(mask, area)
            if bbox is not None:
                go_neural = False
                if use_neural:
                    if opts.always_neural:
                        go_neural = True
                    else:
                        tex, edge = background_complexity(work, mask, bbox)
                        go_neural = tex > opts.complexity_texture or edge >= opts.complexity_edge
                if go_neural:
                    try:
                        wx0, wy0, wx1, wy1 = self.neural.inpaint(work, grown_all, bbox)   # type: ignore[union-attr]
                        # everything masked inside that window is done now
                        mask[wy0:wy1, wx0:wx1] = 0
                        grown_all[wy0:wy1, wx0:wx1] = 0
                        neural_count += 1
                    except Exception as exc:       # model hiccup -> fall back, never lose the job
                        fast_fill(work, mask, bbox, opts.fast_radius, self._rng)
                        fast_count += 1
                        note = f"AI model failed, used fast fill ({exc})"
                else:
                    fast_fill(work, mask, bbox, opts.fast_radius, self._rng)
                    fast_count += 1
            if progress is not None:
                progress(i + 1, total)

        out[..., :3] = work
        return EraseResult(out, boxes, neural_count, fast_count, note)
