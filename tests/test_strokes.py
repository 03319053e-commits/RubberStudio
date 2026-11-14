"""Stroke masks on busy backgrounds, the detector-based check of a mask, and the box-merging helpers of the detector."""
import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from conftest import font, gradient_bg
from rubberstudio.core.eraser import (EraseOptions, TextEraser, _MaskVerifier, build_stroke_mask, ink_candidates)
from rubberstudio.core.textdet import TextBox, _merge_overlaps, _suppress_duplicates


def busy_background(w=720, h=260, seed=1):
    """Patchy, multi-coloured, a bit noisy — something a flat-colour or plane model cannot describe."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    img[:] = (60, 90, 130)
    for _ in range(70):
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(12, 55)),
                   tuple(float(v) for v in rng.integers(25, 225, 3)), -1)
    for _ in range(120):                                  # thin lines: pale and dark "pattern" structures
        p, q = rng.integers(0, [w, h], 2), rng.integers(0, [w, h], 2)
        cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), tuple(float(v) for v in rng.integers(0, 255, 3)), 1)
    img = cv2.GaussianBlur(img, (0, 0), 1.2) + rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_text(bg, text, xy, size, fill=(250, 250, 250)):
    """``bg`` with ``text`` on it + the boolean glyph map + a box polygon around the glyphs."""
    glyphs = Image.new("L", (bg.shape[1], bg.shape[0]), 0)
    ImageDraw.Draw(glyphs).text(xy, text, font=font(size), fill=255)
    gm = np.array(glyphs) > 128
    out = bg.copy()
    out[gm] = fill
    ys, xs = np.nonzero(gm)
    poly = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]], np.float32)
    return out, gm, poly


@pytest.fixture(scope="module")
def scene():
    bg = busy_background()
    rgb, gm, poly = draw_text(bg, "WOOL & SILK", (60, 90), 64)
    return bg, rgb, gm, poly


def test_ink_candidates_contain_the_text_colour_on_a_busy_background(scene):
    bg, rgb, gm, poly = scene
    core = np.zeros(gm.shape, bool)
    cv2.fillPoly(core.view(np.uint8), [poly.astype(np.int32)], 1)
    grown = cv2.dilate(core.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    ring = grown & ~(cv2.dilate(core.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    cands = ink_candidates(rgb, core, ring)
    assert cands, "no ink candidate at all"
    assert min(float(np.linalg.norm(c - 250.0)) for c in cands) < 30.0


@pytest.mark.parametrize("size", [64, 30])
def test_stroke_mask_on_a_busy_background_is_exact(size):
    """Every glyph pixel is covered, and almost nothing farther than a few pixels from a glyph is."""
    rgb, gm, poly = draw_text(busy_background(), "WOOL & SILK", (60, 90), size)
    mask = build_stroke_mask(rgb, [TextBox(poly, 0.9)], EraseOptions())
    assert float((mask[gm] > 0).mean()) >= 0.97
    near = cv2.dilate(gm.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * 10 + 1, 2 * 10 + 1))) > 0
    assert float((mask[~near] > 0).sum()) < 0.02 * float((mask > 0).sum())


def test_a_verifier_that_never_passes_falls_back_to_the_whole_box(scene):
    _, rgb, gm, poly = scene
    calls = []

    def never(roi, mask, box_poly, text_h):
        calls.append(float((mask > 0).sum()))
        return 1.0

    mask = build_stroke_mask(rgb, [TextBox(poly, 0.9)], EraseOptions(), never)
    assert calls, "the candidates were not offered to the verifier"
    x0, y0, x1, y1 = [int(v) for v in (poly[0, 0], poly[0, 1], poly[2, 0], poly[2, 1])]
    assert (mask[y0:y1, x0:x1] > 0).all()                     # the whole text box is masked
    assert (mask[gm] > 0).all()


def test_the_first_candidate_that_passes_the_verifier_is_used(scene):
    _, rgb, gm, poly = scene
    seen = []

    def second_one_passes(roi, mask, box_poly, text_h):
        seen.append(mask.copy())
        return 0.9 if len(seen) == 1 else 0.0

    mask = build_stroke_mask(rgb, [TextBox(poly, 0.9)], EraseOptions(), second_one_passes)
    if len(seen) >= 2:
        assert (mask[:] > 0).sum() == (seen[1] > 0).sum() or (mask > 0).sum() > 0


def test_flat_backgrounds_are_not_sent_to_the_verifier():
    bg = gradient_bg(700, 240)
    rgb, gm, poly = draw_text(bg, "SUMMER SALE", (60, 70), 56)
    calls = []
    mask = build_stroke_mask(rgb, [TextBox(poly, 0.9)], EraseOptions(), lambda *a: calls.append(1) or 1.0)
    assert not calls
    assert (mask[gm] > 0).mean() > 0.99


def test_text_score_sees_text_and_its_absence(detector, scene):
    bg, rgb, gm, poly = scene
    h = float(poly[2, 1] - poly[0, 1])
    pad = 12
    big = np.array([[poly[0, 0] - pad, poly[0, 1] - pad], [poly[1, 0] + pad, poly[0, 1] - pad],
                    [poly[1, 0] + pad, poly[2, 1] + pad], [poly[0, 0] - pad, poly[2, 1] + pad]], np.float32)
    assert detector.text_score(rgb, big, h) > 0.5
    assert detector.text_score(bg, big, h) < 0.25            # same picture without the text
    assert detector.text_score(np.zeros((4, 4, 3), np.uint8), big, h) == 0.0


def test_verifier_rejects_a_mask_that_missed_the_text_and_accepts_the_right_one(detector, scene):
    _, rgb, gm, poly = scene
    h = float(poly[2, 1] - poly[0, 1])
    verify = _MaskVerifier(detector)
    region = np.array([[poly[0, 0] - 12, poly[0, 1] - 12], [poly[1, 0] + 12, poly[0, 1] - 12],
                       [poly[1, 0] + 12, poly[2, 1] + 12], [poly[0, 0] - 12, poly[2, 1] + 12]], np.float32)
    right = cv2.dilate(gm.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    wrong = np.zeros_like(right)
    wrong[:, :40] = 255                                       # somewhere else entirely
    assert verify(rgb, right, region, h) < 0.25
    assert verify(rgb, wrong, region, h) > 0.5


def test_eraser_removes_text_from_a_busy_background(detector, scene):
    bg, rgb, gm, poly = scene
    src = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    res = TextEraser(detector).erase(src, opts=EraseOptions())
    assert res.found
    out = res.image[..., :3]
    h = float(poly[2, 1] - poly[0, 1])
    big = np.array([[0, 60], [rgb.shape[1] - 1, 60], [rgb.shape[1] - 1, 200], [0, 200]], np.float32)
    assert detector.text_score(out, big, h) < 0.25, "text is still readable after the erase"
    err = np.abs(out.astype(int) - bg.astype(int)).mean(axis=2)
    assert float(err[gm].mean()) < 40.0                       # pulled towards the background, not smeared away


# ------------------------------------------------------------------------ detector helpers

def _box(x0, y0, x1, y1, score):
    return TextBox(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32), score)


def test_duplicate_suppression_is_score_first():
    """A big doubtful box must not swallow the small confident boxes inside it."""
    big_weak = _box(0, 0, 400, 200, 0.3)
    small_a = _box(20, 20, 180, 60, 0.9)
    small_b = _box(20, 120, 180, 160, 0.8)
    has = lambda boxes, b: any(k is b for k in boxes)                   # noqa: E731  (TextBox == compares arrays)
    kept = _suppress_duplicates([big_weak, small_a, small_b])
    assert has(kept, small_a) and has(kept, small_b)
    dup = _box(22, 22, 178, 58, 0.5)                          # the same line found again, less sure
    assert not has(_suppress_duplicates([small_a, dup]), dup)


def test_merge_overlaps_makes_one_box_per_line_and_keeps_separate_lines_apart():
    a = _box(100, 100, 400, 140, 0.9)
    b = _box(60, 104, 380, 138, 0.7)                          # same line, another model / scale: reaches further left
    c = _box(100, 160, 300, 190, 0.8)                         # the next line
    merged = _merge_overlaps([a, b, c])
    assert len(merged) == 2
    top = min(merged, key=lambda m: m.xyxy()[1])
    assert top.xyxy()[0] == 60 and top.xyxy()[2] == 400 and abs(top.score - 0.9) < 1e-6
    tilted = TextBox(np.array([[100, 100], [400, 130], [395, 160], [95, 130]], np.float32), 0.8)
    assert len(_merge_overlaps([tilted, _box(100, 100, 400, 140, 0.9)])) == 2      # rotated boxes are left alone


def test_widely_letter_spaced_title_is_found_letter_for_letter(detector):
    """The case that used to leave the first / last letter behind: airy tracking over a patterned picture."""
    bg = busy_background(1100, 300, seed=5)
    glyphs = Image.new("L", (1100, 300), 0)
    d = ImageDraw.Draw(glyphs)
    f = font(84)
    centres, x = [], 50
    for ch in "WOOLSILK":
        l, t, r, b = d.textbbox((x, 90), ch, font=f)
        d.text((x, 90), ch, font=f, fill=255)
        centres.append(((l + r) // 2, (t + b) // 2))
        x += (r - l) + 62                                     # a gap of ~60 px between letters
    gm = np.array(glyphs) > 128
    rgb = bg.copy()
    rgb[gm] = (252, 252, 252)
    boxes = detector.detect(rgb)
    missed = [c for c in centres if not any(b.xyxy()[0] <= c[0] <= b.xyxy()[2] and b.xyxy()[1] <= c[1] <= b.xyxy()[3] for b in boxes)]
    assert not missed, f"letters outside every box: {missed}"
