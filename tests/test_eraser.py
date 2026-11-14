import numpy as np
import pytest

from conftest import gradient_bg, with_text
from rubberstudio.core.eraser import EraseOptions, TextEraser, glyphs_beside_lines, rel_to_px, build_stroke_mask
from rubberstudio.core.textdet import TextDetector


def _demo_font(name: str) -> str:
    """Heavy display type and a monospace face (bundled for the sample picture) used to draw test pictures."""
    from rubberstudio.core.paths import resource_path
    return str(resource_path("assets", "fonts", "demo", name))


def rgba(a):
    return np.dstack([a, np.full(a.shape[:2], 255, np.uint8)])


def test_detects_and_erases_text(detector, clean_and_text):
    clean, texty = clean_and_text
    boxes = detector.detect(texty)
    assert len(boxes) >= 3
    res = TextEraser(detector).erase(rgba(texty), boxes=boxes)
    out = res.image[..., :3].astype(int)
    text_px = np.abs(texty.astype(int) - clean.astype(int)).sum(axis=2) > 0
    err = np.abs(out - clean.astype(int)).mean(axis=2)
    assert err[text_px].mean() < 6.0, "text should be filled with background"
    assert res.fast_count >= 1 and res.neural_count == 0


def test_pixels_far_from_text_are_untouched(detector, clean_and_text):
    _, texty = clean_and_text
    src = rgba(texty)
    res = TextEraser(detector).erase(src)
    mask = build_stroke_mask(texty, res.boxes, EraseOptions())
    import cv2
    near = cv2.dilate(mask, np.ones((41, 41), np.uint8)) > 0
    assert (res.image[~near] == src[~near]).all()
    assert (res.image[..., 3] == 255).all()
    assert res.image is not src and (src[..., :3] == rgba(texty)[..., :3]).all()


def test_no_text_returns_unchanged(detector):
    flat = gradient_bg()
    res = TextEraser(detector).erase(rgba(flat))
    assert not res.found and (res.image == rgba(flat)).all()


def test_region_limits_erasing(detector, clean_and_text):
    _, texty = clean_and_text
    h, w = texty.shape[:2]
    # only the top band (the SUMMER SALE line)
    region = rel_to_px((0.0, 0.0, 1.0, 0.25), w, h)
    res = TextEraser(detector).erase(rgba(texty), region=region)
    out = res.image[..., :3]
    top = np.abs(out[: int(h * 0.25)].astype(int) - texty[: int(h * 0.25)].astype(int)).sum()
    assert top > 0
    # the watermark at the bottom must survive untouched
    assert (out[480:540, 500:800] == texty[480:540, 500:800]).all()


def test_alpha_channel_is_preserved(detector, clean_and_text):
    _, texty = clean_and_text
    a = rgba(texty)
    a[:50, :, 3] = 90
    res = TextEraser(detector).erase(a)
    assert (res.image[..., 3] == a[..., 3]).all()


def test_tall_long_image_is_tiled(detector):
    parts = [with_text(gradient_bg(790, 700, s)) for s in range(5)]
    tall = np.concatenate(parts, 0)             # 790 x 3500
    boxes = detector.detect(tall)
    ys = sorted({int(b.xyxy()[1] // 700) for b in boxes})
    assert ys == [0, 1, 2, 3, 4], "text in every segment of a long image must be found"


# ---------------------------------------------------------------- regressions: heavy type, tiny neighbours

@pytest.mark.parametrize("basic_layout", [True, False], ids=["basic-layout", "raqm-layout"])
def test_demo_scene_is_fully_cleaned(detector, basic_layout):
    """Heavy display type on a flat banner, a caption on the sky and a '©' hugging a watermark: nothing may stay.

    Pillow lays text out differently with and without Raqm (Windows builds usually lack it); with the basic layout the
    detector has *no* response at all for the "©", so it must be picked up beside the line instead."""
    from PIL import features
    from rubberstudio.demo import make_demo_image
    if not basic_layout and not features.check("raqm"):
        pytest.skip("this Pillow build has no Raqm layout engine")
    demo = make_demo_image(basic_layout=basic_layout)
    res = TextEraser(detector).erase(demo.copy())
    out = res.image[..., :3].astype(int)
    # banner interior is banner-red again (the glyphs were cream; a leftover speck would be spread by the fill)
    y0, y1, x0, x1 = 80, 165, 70, 1015
    band = out[y0:y1, x0:x1]
    assert band[..., 0].min() >= 215 and band[..., 1].max() <= 95 and band[..., 2].max() <= 80, \
        (band.reshape(-1, 3).min(0), band.reshape(-1, 3).max(0))
    # watermark with its "©": dark hills only
    wm = out[735:775, 925:1240]
    assert wm.max() <= 110, wm.reshape(-1, 3).max(0)
    # caption on the sky: no strokes left (deviation from a wide median is just grain)
    import cv2
    cap = res.image[215:262, 70:440, :3]
    dev = np.abs(cap.astype(int) - cv2.medianBlur(cap, 41).astype(int)).max(-1)
    assert dev.max() <= 30, dev.max()
    # the sun is nowhere near any text
    diff = np.abs(out - demo[..., :3].astype(int)).max(-1)
    assert diff[300:330, 930:960].max() == 0


def test_thick_display_type_is_covered_without_holes():
    """Very dense glyphs (text covers > 50 % of a median window) must be masked completely."""
    from PIL import Image, ImageDraw, ImageFont
    from rubberstudio.core.textdet import TextBox
    w, h = 900, 260
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.stack([60 + 30 * xx / w, 120 - 40 * yy / h, 200 - 50 * xx / w], -1).astype(np.uint8)
    glyphs = Image.new("L", (w, h), 0)
    ImageDraw.Draw(glyphs).text((40, 40), "MMMMMMMM", font=ImageFont.truetype(
        _demo_font("ArchivoBlack-Regular.ttf"), 150), fill=255)
    gm = np.array(glyphs) > 200
    ys, xs = np.nonzero(gm)
    pad = 12
    poly = np.array([[xs.min() - pad, ys.min() - pad], [xs.max() + pad, ys.min() - pad],
                     [xs.max() + pad, ys.max() + pad], [xs.min() - pad, ys.max() + pad]], np.float32)
    rgb = bg.copy()
    rgb[gm] = (250, 238, 215)
    mask = build_stroke_mask(rgb, [TextBox(poly, 0.9)], EraseOptions())
    covered = (mask[gm] > 0).mean()
    assert covered >= 0.9995, covered


def test_fill_small_holes_only_fills_small_ones():
    from rubberstudio.core.eraser import _fill_small_holes
    m = np.zeros((120, 120), np.uint8)
    m[10:110, 10:110] = 255
    m[30:34, 30:34] = 0           # 16 px speck
    m[60:100, 60:100] = 0         # 1600 px counter
    out = _fill_small_holes(m.copy(), 100.0)
    assert out[31, 31] == 255
    assert out[80, 80] == 0


def test_low_scoring_fragment_next_to_text_is_kept_and_widened():
    from rubberstudio.core.textdet import TextBox, _hysteresis

    def box(x0, y0, x1, y1, score):
        return TextBox(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32), score)

    strong = box(100, 100, 400, 130, 0.8)
    near = box(84, 110, 92, 118, 0.25)        # a "©" in front of the line
    far = box(700, 110, 708, 118, 0.25)       # same size, nowhere near
    above = box(150, 20, 158, 28, 0.25)       # near in x, but on another line
    kept = _hysteresis([strong, near, far, above], 0.5)
    assert len(kept) == 2
    frag = [b for b in kept if b is not strong][0]
    x0, y0, x1, y1 = frag.xyxy()
    assert x0 < 84 and x1 > 92               # widened sideways
    assert y0 <= 100 and y1 >= 130           # and to the full line height
    assert _hysteresis([near, far], 0.5) == []      # without a confident box nothing is kept


def test_model_path_with_non_ascii_characters(tmp_path, clean_and_text):
    """A Chinese folder / user name must not stop the detector from loading (Windows code-page trap)."""
    import shutil
    from rubberstudio.core.paths import resource_path
    from rubberstudio.core.textdet import MODEL_NAME, TextDetector
    folder = tmp_path / "中文 用户" / "模型"
    folder.mkdir(parents=True)
    target = folder / ("检测-" + MODEL_NAME)
    shutil.copyfile(resource_path("rubberstudio", "core", "models", MODEL_NAME), target)
    _, texty = clean_and_text
    assert len(TextDetector(target).detect(texty)) >= 3


# ------------------------------------------------------ symbols the detector does not see (©, •, @ ...)

def test_side_glyph_step_can_be_switched_off():
    """With the single PP-OCRv4 model on its normal scale (the detector that has no response for the '©') the symbol
    next to the watermark is only found by the side-glyph step; it must be switchable."""
    from rubberstudio.core.textdet import default_model_path
    from rubberstudio.demo import make_demo_image
    demo = make_demo_image()
    single = TextDetector(default_model_path())
    on = TextEraser(single).detect(demo, opts=EraseOptions(thorough=False))
    off = TextEraser(single).detect(demo, opts=EraseOptions(thorough=False, side_glyphs=False))
    assert len(on) == len(off) + 1, "the '©' in front of the watermark is only found by the side-glyph step"


def _spacemono(size):
    from PIL import ImageFont
    return ImageFont.truetype(_demo_font("SpaceMono-Bold.ttf"), size, layout_engine=ImageFont.Layout.BASIC)


def _line_scene(*, where="left", gap=12, ink=(255, 255, 255), symbol_ink=None, symbol="©", symbol_size=32,
                dy=0, noise=2.0, bg=(40, 110, 90), draw_symbol=True, shape=None, shape_size=44):
    """A text line on a smooth background with one stray symbol (or ``shape``: "ring" / "square") beside it
    -> (rgb, line box, symbol bbox)."""
    from PIL import Image, ImageDraw
    from rubberstudio.core.textdet import TextBox
    w, h = 760, 260
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([bg[0] + 14 * xx / w, bg[1] + 10 * yy / h, bg[2] - 12 * xx / w], -1)
    base = base + np.random.default_rng(3).normal(0, noise, base.shape)
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    text, base_y, lx = "Sample Studio 2026", 150, 250
    d.text((lx, base_y), text, font=_spacemono(32), fill=ink, anchor="ls")
    l, t, r, b = d.textbbox((lx, base_y), text, font=_spacemono(32), anchor="ls")
    pad = 8
    line = TextBox(np.array([[l - pad, t - pad], [r + pad, t - pad], [r + pad, b + pad], [l - pad, b + pad]],
                            np.float32), 0.9)
    if shape:                                    # an object, not a glyph: centred on the line, drawn with primitives
        cy = (t + b) // 2 + dy
        half = shape_size // 2
        sx0 = (l - pad - gap - shape_size) if where == "left" else (r + pad + gap)
        sym = (sx0, cy - half, sx0 + shape_size, cy + half)
        if draw_symbol and shape == "ring":
            d.ellipse(sym, outline=symbol_ink or ink, width=4)
        elif draw_symbol:
            d.rectangle(sym, fill=symbol_ink or ink)
        return np.asarray(im), line, sym
    sf = _spacemono(symbol_size)
    sl, _, sr, _ = d.textbbox((0, base_y), symbol, font=sf, anchor="ls")
    sx = (l - pad - gap - sr) if where == "left" else (r + pad + gap - sl)
    sym = d.textbbox((sx, base_y + dy), symbol, font=sf, anchor="ls")
    if draw_symbol:
        d.text((sx, base_y + dy), symbol, font=sf, fill=symbol_ink or ink, anchor="ls")
    return np.asarray(im), line, tuple(int(v) for v in sym)


@pytest.mark.parametrize("where, symbol", [("left", "©"), ("right", "®"), ("left", "•"), ("right", "@")])
def test_symbol_beside_a_line_is_taken_and_erased(where, symbol):
    rgb, line, sym = _line_scene(where=where, symbol=symbol)
    found = glyphs_beside_lines(rgb, [line], 0.5)
    assert len(found) == 1, found
    x0, y0, x1, y1 = found[0].xyxy()
    assert x0 <= sym[0] and x1 >= sym[2] and y0 <= sym[1] and y1 >= sym[3], ((x0, y0, x1, y1), sym)
    out = TextEraser().erase(np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]),
                             boxes=[line] + found).image[..., :3]
    assert out[sym[1]:sym[3], sym[0]:sym[2]].max() <= 150, "the symbol must be gone"


def test_symbol_chain_hangs_on_the_line():
    """"© •" — the second symbol is within reach of the first one, not of the line."""
    from PIL import Image, ImageDraw
    rgb, line, sym = _line_scene(where="left", gap=10, symbol="•")
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    f = _spacemono(32)
    x = sym[0] - 12 - (sym[2] - sym[0])
    d.text((x, 150), "•", font=f, fill=(255, 255, 255), anchor="ls")
    found = glyphs_beside_lines(np.asarray(im), [line], 0.5)
    assert len(found) == 2


@pytest.mark.parametrize("case, kwargs", [
    ("too far away (inside the search window, beyond the allowed gap)", dict(gap=62)),
    ("far outside the search window", dict(gap=140)),
    ("another colour", dict(symbol_ink=(255, 60, 60))),
    ("giant glyph", dict(symbol_size=96)),
    ("ring bigger than a glyph, fully on the line", dict(shape="ring", shape_size=44)),
    ("solid square, glyph-sized but not glyph-shaped", dict(shape="square", shape_size=30)),
    ("not on the line", dict(dy=-95)),
    ("half above the line", dict(dy=-30)),
])
def test_things_that_are_not_part_of_the_line_are_left_alone(case, kwargs):
    rgb, line, _sym = _line_scene(**kwargs)
    assert glyphs_beside_lines(rgb, [line], 0.5) == [], case


def test_side_glyph_guards():
    rgb, line, sym = _line_scene()
    from rubberstudio.core.textdet import TextBox
    assert len(glyphs_beside_lines(rgb, [line], 0.5)) == 1
    weak = TextBox(line.poly, 0.3)                                   # only confident lines are anchors
    assert glyphs_beside_lines(rgb, [weak], 0.5) == []
    h, w = rgb.shape[:2]
    assert glyphs_beside_lines(rgb, [line], 0.5, bounds=(sym[2] + 4, 0, w, h)) == []        # outside the search area
    cover = TextBox(np.array([[sym[0] - 4, sym[1] - 4], [sym[2] + 4, sym[1] - 4], [sym[2] + 4, sym[3] + 4],
                              [sym[0] - 4, sym[3] + 4]], np.float32), 0.9)
    assert glyphs_beside_lines(rgb, [line, cover], 0.5) == []        # another box already owns it
    plain, line2, _ = _line_scene(draw_symbol=False)
    assert glyphs_beside_lines(plain, [line2], 0.5) == []            # nothing there, nothing invented
    faint, line3, _ = _line_scene(ink=(60, 125, 105))
    assert glyphs_beside_lines(faint, [line3], 0.5) == []            # a line barely above the background: no analysis


@pytest.mark.parametrize("ink, noise", [((255, 255, 255), 14.0), ((25, 25, 25), 14.0), ((255, 255, 255), 30.0)])
def test_noisy_background_does_not_grow_phantom_symbols(ink, noise):
    rgb, line, _sym = _line_scene(ink=ink, noise=noise, draw_symbol=False, bg=(120, 130, 110))
    assert glyphs_beside_lines(rgb, [line], 0.5) == []
