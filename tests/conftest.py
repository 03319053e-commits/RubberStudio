import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


def font(size: int):
    for cand in _FONT_CANDIDATES:
        if Path(cand).exists():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default(size=size)


def gradient_bg(w=900, h=600, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.stack([200 + 40 * xx / w, 210 - 30 * yy / h, 230 - 60 * xx / w], -1).astype(np.float32)
    return np.clip(bg + rng.normal(0, 2.0, bg.shape), 0, 255).astype(np.uint8)


def with_text(bg: np.ndarray, items=None) -> np.ndarray:
    im = Image.fromarray(bg)
    d = ImageDraw.Draw(im)
    for xy, text, size, fill in items or [
        ((60, 60), "SUMMER SALE 50% OFF", 56, (255, 255, 255)),
        ((60, 200), "www.example-shop.com", 34, (20, 20, 20)),
        ((520, 500), "@watermark_user", 30, (255, 255, 255)),
    ]:
        d.text(xy, text, font=font(size), fill=fill)
    return np.array(im)


@pytest.fixture(scope="session")
def detector():
    from rubberstudio.core.textdet import TextDetector
    return TextDetector()


@pytest.fixture()
def clean_and_text():
    clean = gradient_bg()
    return clean, with_text(clean)
