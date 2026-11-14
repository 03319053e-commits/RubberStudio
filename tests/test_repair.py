import numpy as np

from rubberstudio.core import repair


def noisy(h=200, w=200, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, (h, w, 4), dtype=np.uint8)
    a[..., 3] = 255
    return a


def test_heal_changes_only_inside_brush():
    img = noisy()
    ref = img.copy()
    repair.heal(img, 100, 100, radius=20, hardness=0.5, flow=1.0, strength=1.0)
    changed = (img != ref).any(axis=2)
    ys, xs = np.nonzero(changed)
    assert changed.any()
    assert xs.min() >= 79 and xs.max() <= 121 and ys.min() >= 79 and ys.max() <= 121
    # blurring a noisy patch lowers its variance
    assert img[90:110, 90:110, :3].std() < ref[90:110, 90:110, :3].std()
    assert (img[..., 3] == 255).all()


def test_smudge_drags_pixels_along_the_stroke():
    img = np.zeros((120, 200, 4), np.uint8)
    img[..., 3] = 255
    img[50:70, 40:60, :3] = 255              # white square on black
    assert repair.smudge(img, (50, 60), (90, 60), radius=25, hardness=0.8, flow=1.0, smear=5.0)
    assert img[60, 80, 0] > 0                # some white was dragged to the right
    assert not repair.smudge(img, (50, 60), (50.5, 60), radius=25, hardness=0.8, flow=1.0, smear=1.0)


def test_texture_adds_grain_and_respects_borders():
    img = np.full((100, 100, 4), 128, np.uint8)
    img[..., 3] = 255
    repair.texture(img, 5, 5, radius=30, hardness=0.5, flow=1.0, strength=1.0, block=8,
                   rng=np.random.default_rng(0))     # brush hangs over the corner
    assert img[:20, :20, :3].std() > 0.5
    assert (img[..., 3] == 255).all()


def test_brush_radius_scales_with_resolution():
    assert repair.brush_radius_px(1400, 1000, 28) == 14
    assert repair.brush_radius_px(2800, 1000, 28) == 28
