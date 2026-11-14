"""Film grain / sensor noise helpers: a filled patch must have the same fine texture as the picture around it,
otherwise it shows up as a smooth "plastic" sticker even when its colours are right."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

_RING_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def noise_sigma(gray: np.ndarray, where: np.ndarray) -> float:
    """Robust (MAD) standard deviation of the fine detail of ``gray`` (float32) over the boolean mask ``where``."""
    if int(where.sum()) < 30:
        return 0.0
    hf = (gray - cv2.GaussianBlur(gray, (0, 0), 1.0))[where]
    mad = float(np.median(np.abs(hf - np.median(hf))))
    return 1.4826 * mad


def ring_noise_sigma(roi: np.ndarray, mask: np.ndarray) -> float:
    """Noise level of the untouched pixels in a ring around ``mask`` (uint8 0/255)."""
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ring = (cv2.dilate(mask, _RING_KERNEL, iterations=4) > 0) & (cv2.dilate(mask, _RING_KERNEL, iterations=1) == 0)
    return noise_sigma(gray, ring)


_GRAIN_GAIN = 0.52       # what :func:`noise_sigma` reads for grain made with a unit ``sigma`` (it is soft, so it reads low)


def grain(shape: tuple[int, int], sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Luma-dominant grain (h, w, 3), slightly softened like real sensor noise.

    ``sigma`` is the noise level as :func:`noise_sigma` measures it, so grain made for a ring that measures σ reads σ
    again once it is in the picture."""
    h, w = shape
    sigma = sigma / _GRAIN_GAIN
    lum = rng.normal(0.0, sigma, (h, w, 1)).astype(np.float32)
    chroma = rng.normal(0.0, sigma * 0.35, (h, w, 3)).astype(np.float32)
    return cv2.GaussianBlur(lum + chroma, (0, 0), 0.55).reshape(h, w, 3) * 1.25


def match_grain(before: np.ndarray, filled: np.ndarray, mask: np.ndarray,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Add grain to the ``mask`` pixels of ``filled`` so they are as noisy as the ring around the hole in ``before``.

    Only the missing amount is added (a fill that is already textured is left alone). Returns a new array."""
    sel = mask > 0
    if not sel.any():
        return filled
    sigma_ring = ring_noise_sigma(before, mask)
    if sigma_ring < 0.7:
        return filled
    gray = cv2.cvtColor(filled, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma_fill = noise_sigma(gray, sel)
    extra = float(np.sqrt(max(sigma_ring * sigma_ring - sigma_fill * sigma_fill, 0.0)))
    if extra < 0.6:
        return filled
    rng = rng or np.random.default_rng(1234)
    vals = filled.astype(np.float32) + grain(mask.shape, extra, rng)
    return np.where(sel[..., None], np.clip(vals, 0, 255), filled).astype(np.uint8)
