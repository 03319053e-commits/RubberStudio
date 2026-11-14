"""Procedural paper, halftone (Ben-Day) dots and print noise — generated with numpy, cached as pixmaps."""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PySide6.QtGui import QColor, QImage, QPixmap

from ..theme import DOT


def qimage_from_rgba(arr: np.ndarray, premultiply: bool = True) -> QImage:
    """Detached QImage from an H×W×4 uint8 array (straight alpha in, premultiplied out)."""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    if premultiply:
        a = arr[..., 3:4].astype(np.float32) / 255.0
        pm = arr.copy()
        pm[..., :3] = (arr[..., :3].astype(np.float32) * a + 0.5).astype(np.uint8)
        img = QImage(pm.data, w, h, w * 4, QImage.Format.Format_RGBA8888_Premultiplied)
        return img.copy()
    return QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()


def _wrap_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    pad = int(sigma * 3) + 1
    big = np.pad(a, pad, mode="wrap")
    big = cv2.GaussianBlur(big, (0, 0), sigma)
    return big[pad:-pad, pad:-pad]


@lru_cache(maxsize=8)
def paper_tile(base_rgb: tuple, seed: int = 1, size: int = 384, strength: float = 1.0) -> QPixmap:
    """Seamlessly tiling paper: mottling, grain, fibres and the odd coloured fleck of recycled stock."""
    rng = np.random.default_rng(seed)
    S = size
    fine = rng.normal(0, 1, (S, S)).astype(np.float32)
    mid = _wrap_blur(rng.normal(0, 1, (S, S)).astype(np.float32), 1.3) * 2.2
    low = _wrap_blur(rng.normal(0, 1, (S, S)).astype(np.float32), 22.0) * 11.0
    lum = (fine * 0.011 + mid * 0.012 + low * 0.016) * strength
    img = np.array(base_rgb, np.float32)[None, None, :] * (1.0 + lum[..., None])

    # fibres, drawn four times so they wrap around the tile edge
    fib = np.zeros((S, S), np.float32)
    for _ in range(int(S * S / 2600)):
        x, y = rng.uniform(0, S, 2)
        ang = rng.uniform(0, np.pi)
        ln = rng.uniform(5, 16)
        dx, dy = np.cos(ang) * ln, np.sin(ang) * ln
        v = rng.choice([-1.0, 1.0]) * rng.uniform(0.02, 0.05)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                cv2.line(fib, (int(x + ox), int(y + oy)), (int(x + dx + ox), int(y + dy + oy)), v, 1, cv2.LINE_AA)
    img *= (1.0 + fib * strength)[..., None]

    # flecks (red / blue / grey), as in the app icon's recycled paper
    colours = [(222, 98, 66), (70, 96, 190), (96, 92, 86)]
    for _ in range(int(S * S / 5200)):
        x, y = int(rng.integers(0, S)), int(rng.integers(0, S))
        col = colours[int(rng.integers(0, 3))]
        a = rng.uniform(0.25, 0.6) * strength
        r = 1 if rng.random() < 0.8 else 2
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                px, py = x + ox, y + oy
                if -r <= px < S + r and -r <= py < S + r:
                    x0, x1, y0, y1 = max(px, 0), min(px + r, S), max(py, 0), min(py + r, S)
                    if x1 > x0 and y1 > y0:
                        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + np.array(col, np.float32) * a
    rgba = np.dstack([np.clip(img, 0, 255).astype(np.uint8), np.full((S, S), 255, np.uint8)])
    return QPixmap.fromImage(qimage_from_rgba(rgba, premultiply=False))


def halftone_alpha(w: int, h: int, spacing: float = DOT, angle: float = 45.0,
                   ramp_angle: float = 45.0, lo: float = 0.0, hi: float = 1.0,
                   rmax: float = 0.62) -> np.ndarray:
    """Alpha mask (uint8) of round dots on a rotated grid; dot area follows a linear tone ramp."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    th = np.deg2rad(angle)
    c, s = np.cos(th), np.sin(th)
    u = (xx * c + yy * s) / spacing
    v = (-xx * s + yy * c) / spacing
    cu, cv_ = np.round(u), np.round(v)
    du, dv = u - cu, v - cv_
    dist = np.sqrt(du * du + dv * dv)
    # tone sampled at the dot centre so every dot stays a clean circle
    cx = (cu * c - cv_ * s) * spacing
    cy = (cu * s + cv_ * c) * spacing
    ra = np.deg2rad(ramp_angle)
    t = 0.5 + ((cx / max(w, 1) - 0.5) * np.cos(ra) + (cy / max(h, 1) - 0.5) * np.sin(ra)) / (abs(np.cos(ra)) + abs(np.sin(ra)))
    tone = np.clip(lo + (hi - lo) * np.clip(t, 0, 1), 0, 1)
    radius = rmax * np.sqrt(tone)
    aa = 1.0 / spacing
    alpha = np.clip((radius - dist) / aa + 0.5, 0, 1)
    return (alpha * 255).astype(np.uint8)


@lru_cache(maxsize=96)
def halftone_pixmap(w: int, h: int, rgba: tuple, spacing: float = DOT, angle: float = 45.0,
                    ramp_angle: float = 45.0, lo: float = 0.0, hi: float = 1.0,
                    rmax: float = 0.62, speckle: float = 0.0, seed: int = 7) -> QPixmap:
    """Tinted dot screen as a pixmap (alpha = dots). ``speckle`` adds print-noise flecks."""
    w, h = max(1, int(w)), max(1, int(h))
    a = halftone_alpha(w, h, spacing, angle, ramp_angle, lo, hi, rmax).astype(np.float32) / 255.0
    if speckle > 0:
        rng = np.random.default_rng(seed)
        fl = (rng.random((h, w)) < speckle).astype(np.uint8)
        fl = cv2.dilate(fl, np.ones((2, 2), np.uint8)).astype(np.float32)
        a = np.clip(a + fl * 0.55, 0, 1)
    r, g, b, al = rgba
    out = np.zeros((h, w, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = r, g, b
    out[..., 3] = (a * al * 255).astype(np.uint8)
    return QPixmap.fromImage(qimage_from_rgba(out))


def tint(c: QColor) -> tuple:
    return (c.red(), c.green(), c.blue(), c.alphaF())
