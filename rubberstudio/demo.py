"""A synthetic sample picture with text on it — used by the self-test, the screenshots and the tests."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core.paths import resource_path

DEMO_TEXTS = ("SUMMER SALE 50% OFF", "www.example-shop.com", "© 2026 Sample Studio")


def _font(name: str, size: int, basic_layout: bool = True) -> ImageFont.FreeTypeFont:
    """Bundled sample-picture font (the interface itself uses Liberation Sans). The basic layout engine is forced: with Raqm (present in Pillow on some systems, absent on
    others, e.g. plain Windows) the glyph advances differ by a fraction of a pixel and the picture would not be
    pixel-identical from one machine to the next — which the self-test, the screenshots and the tests rely on."""
    path = resource_path("assets", "fonts", "demo", name)
    try:
        if basic_layout:
            return ImageFont.truetype(str(path), size, layout_engine=ImageFont.Layout.BASIC)
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default(size=size)       # type: ignore[return-value]


def make_demo_image(w: int = 1280, h: int = 800, seed: int = 7, basic_layout: bool = True) -> np.ndarray:
    """RGBA uint8 picture: sky gradient, sun, hills and three different kinds of text."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = yy / h
    sky = np.stack([255 * (0.36 + 0.62 * t), 255 * (0.58 + 0.28 * t), 255 * (0.92 - 0.30 * t)], -1)
    img = sky.copy()
    # sun
    d = np.hypot(xx - w * 0.74, yy - h * 0.3)
    sun = np.clip((h * 0.115 - d) / 3.0, 0, 1)[..., None]
    img = img * (1 - sun) + np.array([255, 224, 120], np.float32) * sun
    glow = np.exp(-(d / (h * 0.28)) ** 2)[..., None] * 0.35
    img = img * (1 - glow) + np.array([255, 238, 190], np.float32) * glow
    # hills
    for k, (base, amp, col) in enumerate(((0.62, 0.10, (72, 150, 110)), (0.72, 0.08, (44, 118, 92)), (0.84, 0.06, (28, 88, 74)))):
        ridge = h * (base + amp * np.sin(xx / w * (3.1 + k) * np.pi + k * 1.7) * 0.5 + amp * 0.35 * np.sin(xx / w * 11 + k))
        m = np.clip((yy - ridge) / 2.0, 0, 1)[..., None]
        img = img * (1 - m) + np.array(col, np.float32) * m
    img += rng.normal(0, 3.2, (h, w, 1)).astype(np.float32)               # film grain
    base = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

    over = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    # 1: big banner text on a colour stripe
    big = _font("ArchivoBlack-Regular.ttf", int(h * 0.088), basic_layout)
    tw = d.textlength(DEMO_TEXTS[0], font=big)
    x0, y0 = int(w * 0.06), int(h * 0.1)
    d.rectangle((x0 - 18, y0 - 10, x0 + tw + 18, y0 + int(h * 0.088) + 22), fill=(240, 69, 42, 235))
    d.text((x0, y0), DEMO_TEXTS[0], font=big, fill=(255, 250, 235, 255))
    # 2: small dark caption on bright sky
    cap = _font("SpaceMono-Bold.ttf", int(h * 0.034), basic_layout)
    d.text((int(w * 0.07), int(h * 0.27)), DEMO_TEXTS[1], font=cap, fill=(28, 26, 23, 255))
    # 3: semi-transparent white "watermark" low on the picture
    wm = _font("SpaceMono-Bold.ttf", int(h * 0.03), basic_layout)
    tw3 = d.textlength(DEMO_TEXTS[2], font=wm)
    d.text((w - tw3 - int(w * 0.04), int(h * 0.92)), DEMO_TEXTS[2], font=wm, fill=(255, 255, 255, 170))
    out = Image.alpha_composite(base, over)
    return np.ascontiguousarray(np.asarray(out, dtype=np.uint8))
