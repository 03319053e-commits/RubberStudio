"""Repair brushes (HEAL, SMUDGE, TEXTURE) — ports of the iOS app's brushes, edited in place on RGBA arrays."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import cv2
import numpy as np


def brush_radius_px(width: int, height: int, size: float) -> float:
    """Brush radius in *image* pixels. ``size`` is the slider diameter; it scales with image resolution
    so the brush covers roughly the same share of the picture whatever its pixel size."""
    return max(4.0, 0.5 * float(size) * max(width, height) / 1400.0)


@lru_cache(maxsize=64)
def _soft_disc(r: int, hardness_pct: int) -> np.ndarray:
    hardness = hardness_pct / 100.0
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.sqrt(xx * xx + yy * yy) / max(r, 1)
    inner = 0.1 + 0.85 * hardness
    a = np.clip((1.0 - d) / max(1.0 - inner, 1e-3), 0.0, 1.0)
    return (a * a * (3.0 - 2.0 * a)).astype(np.float32)


def _disc(radius: float, hardness: float) -> np.ndarray:
    return _soft_disc(max(2, int(round(radius))), int(round(float(np.clip(hardness, 0, 1)) * 100)))


def _clip_box(cx: float, cy: float, r: int, w: int, h: int):
    x0, y0 = int(round(cx)) - r, int(round(cy)) - r
    x1, y1 = x0 + 2 * r + 1, y0 + 2 * r + 1
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if cx1 - cx0 < 2 or cy1 - cy0 < 2:
        return None
    return x0, y0, cx0, cy0, cx1, cy1


def _blend(img: np.ndarray, x0: int, y0: int, cx0: int, cy0: int, cx1: int, cy1: int,
           patch: np.ndarray, disc: np.ndarray, alpha: float) -> None:
    """img[ROI] = lerp(img[ROI], patch, disc*alpha); ``patch`` covers the clipped ROI."""
    a = (disc[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] * float(alpha))[..., None]
    roi = img[cy0:cy1, cx0:cx1, :3].astype(np.float32)
    mixed = roi * (1.0 - a) + patch.astype(np.float32) * a
    img[cy0:cy1, cx0:cx1, :3] = np.clip(mixed + 0.5, 0, 255).astype(np.uint8)


def _context(img: np.ndarray, cx0: int, cy0: int, cx1: int, cy1: int, margin: int):
    """RGB crop with ``margin`` px of context (clamped at the image border) + the offset of the ROI inside it."""
    h, w = img.shape[:2]
    ex0, ey0 = max(0, cx0 - margin), max(0, cy0 - margin)
    ex1, ey1 = min(w, cx1 + margin), min(h, cy1 + margin)
    big = np.ascontiguousarray(img[ey0:ey1, ex0:ex1, :3])
    return big, cx0 - ex0, cy0 - ey0


def heal(img: np.ndarray, cx: float, cy: float, *, radius: float, hardness: float,
         flow: float, strength: float) -> None:
    """Soften blemishes by blending with a blurred copy of the surroundings."""
    h, w = img.shape[:2]
    r = max(2, int(round(radius)))
    box = _clip_box(cx, cy, r, w, h)
    if box is None:
        return
    x0, y0, cx0, cy0, cx1, cy1 = box
    sigma = max(0.8, radius * 0.2 * float(strength))
    margin = int(3 * sigma) + 2
    big, ox, oy = _context(img, cx0, cy0, cx1, cy1, margin)
    blurred = cv2.GaussianBlur(big, (0, 0), sigma)[oy:oy + (cy1 - cy0), ox:ox + (cx1 - cx0)]
    _blend(img, x0, y0, cx0, cy0, cx1, cy1, blurred, _disc(r, hardness), flow)


def smudge(img: np.ndarray, last: tuple[float, float], cur: tuple[float, float], *, radius: float,
           hardness: float, flow: float, smear: float) -> bool:
    """Drag pixels from ``last`` to ``cur``. Returns False when the move was too small to act on."""
    h, w = img.shape[:2]
    r = max(2, int(round(radius)))
    dx, dy = cur[0] - last[0], cur[1] - last[1]
    if np.hypot(dx, dy) < max(2.0, r * 0.12):
        return False
    idx, idy = int(round(dx)), int(round(dy))
    # destination ROI around ``cur``; source ROI is the same box shifted back by the move
    box = _clip_box(cur[0], cur[1], r, w, h)
    if box is None:
        return False
    x0, y0, cx0, cy0, cx1, cy1 = box
    sx0, sy0, sx1, sy1 = cx0 - idx, cy0 - idy, cx1 - idx, cy1 - idy
    # clip the source to the image and shrink the destination to match
    cl = max(0, -sx0), max(0, -sy0), max(0, sx1 - w), max(0, sy1 - h)
    cx0, cy0, cx1, cy1 = cx0 + cl[0], cy0 + cl[1], cx1 - cl[2], cy1 - cl[3]
    sx0, sy0, sx1, sy1 = cx0 - idx, cy0 - idy, cx1 - idx, cy1 - idy
    if cx1 - cx0 < 2 or cy1 - cy0 < 2:
        return False
    patch = np.ascontiguousarray(img[sy0:sy1, sx0:sx1, :3])
    alpha = min(1.0, float(flow) * (0.2 + 0.16 * float(smear)))
    _blend(img, x0, y0, cx0, cy0, cx1, cy1, patch, _disc(r, hardness), alpha)
    return True


def _soft_light(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    d = np.where(b <= 0.25, ((16.0 * b - 12.0) * b + 4.0) * b, np.sqrt(b))
    return np.where(s <= 0.5, b - (1.0 - 2.0 * s) * b * (1.0 - b), b + (2.0 * s - 1.0) * (d - b))


def texture(img: np.ndarray, cx: float, cy: float, *, radius: float, hardness: float, flow: float,
            strength: float, block: float, rng: Optional[np.random.Generator] = None) -> None:
    """Rebuild fine grain after a mosaic: blur to the block size, unsharp-mask, add soft-light grain."""
    h, w = img.shape[:2]
    r = max(2, int(round(radius)))
    box = _clip_box(cx, cy, r, w, h)
    if box is None:
        return
    x0, y0, cx0, cy0, cx1, cy1 = box
    block = max(1.0, float(block))
    sig_b, sig_u = max(0.6, block * 0.55), max(0.6, block * 0.9)
    margin = int(3 * max(sig_b, sig_u)) + 2
    big, ox, oy = _context(img, cx0, cy0, cx1, cy1, margin)
    bf = cv2.GaussianBlur(big, (0, 0), sig_b).astype(np.float32)
    sharp = bf + (bf - cv2.GaussianBlur(bf, (0, 0), sig_u)) * (0.6 * float(strength))
    sharp = np.clip(sharp, 0, 255)[oy:oy + (cy1 - cy0), ox:ox + (cx1 - cx0)] / 255.0
    rng = rng or np.random.default_rng()
    k = float(min(1.0, 0.1 + 0.25 * float(strength)))
    noise = 0.5 + (rng.random(sharp.shape[:2], dtype=np.float32) - 0.5) * k
    out = _soft_light(sharp, noise[..., None]) * 255.0
    _blend(img, x0, y0, cx0, cy0, cx1, cy1, np.clip(out, 0, 255), _disc(r, hardness), flow)
