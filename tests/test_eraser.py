import numpy as np

from conftest import gradient_bg, with_text
from rubberstudio.core.eraser import EraseOptions, TextEraser, rel_to_px, build_stroke_mask


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

def test_demo_scene_is_fully_cleaned(detector):
    """Heavy display type on a flat banner, a caption on the sky and a '©' hugging a watermark: nothing may stay."""
    from rubberstudio.demo import make_demo_image
    demo = make_demo_image()
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
    from rubberstudio.core.paths import resource_path
    from rubberstudio.core.textdet import TextBox
    w, h = 900, 260
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.stack([60 + 30 * xx / w, 120 - 40 * yy / h, 200 - 50 * xx / w], -1).astype(np.uint8)
    glyphs = Image.new("L", (w, h), 0)
    ImageDraw.Draw(glyphs).text((40, 40), "MMMMMMMM", font=ImageFont.truetype(
        str(resource_path("assets", "fonts", "ArchivoBlack-Regular.ttf")), 150), fill=255)
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
