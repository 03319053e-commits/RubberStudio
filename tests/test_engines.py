"""Neural engines (LaMa style / MI-GAN style), tiling, grain matching and the model manager — all with tiny dummy
networks, so nothing here needs a real model download."""
import io

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from conftest import gradient_bg
from rubberstudio.core.neural import (LamaEngine, MiganEngine, ModelError, ModelManager, MODELS, make_engine)
from rubberstudio.core.texture import grain, match_grain, noise_sigma, ring_noise_sigma
from test_neural import make_dummy_model


def make_dummy_migan(path, grey=100):
    """MI-GAN's I/O: uint8 image + uint8 mask (255 = keep, 0 = hole) -> uint8; holes come out flat ``grey``."""
    img = helper.make_tensor_value_info("image", TensorProto.UINT8, [1, 3, 512, 512])
    mask = helper.make_tensor_value_info("mask", TensorProto.UINT8, [1, 1, 512, 512])
    out = helper.make_tensor_value_info("result", TensorProto.UINT8, [1, 3, 512, 512])
    thr = helper.make_tensor("thr", TensorProto.FLOAT, [1], [127.0])
    flat = helper.make_tensor("flat", TensorProto.UINT8, [1], [grey])
    nodes = [
        helper.make_node("Cast", ["mask"], ["maskf"], to=TensorProto.FLOAT),
        helper.make_node("Greater", ["maskf", "thr"], ["keep"]),
        helper.make_node("Where", ["keep", "image", "flat"], ["result"]),
    ]
    g = helper.make_graph(nodes, "dummy_migan", [img, mask], [out], initializer=[thr, flat])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))


@pytest.fixture()
def lama_path(tmp_path):
    p = tmp_path / "dummy_lama.onnx"
    make_dummy_model(p)
    return p


@pytest.fixture()
def migan_path(tmp_path):
    p = tmp_path / "dummy_migan.onnx"
    make_dummy_migan(p)
    return p


def test_make_engine_picks_the_class_from_the_input_type(lama_path, migan_path):
    assert type(make_engine(lama_path)) is LamaEngine
    assert type(make_engine(migan_path)) is MiganEngine


def test_migan_mask_polarity_and_untouched_surroundings(migan_path):
    rgb = gradient_bg(600, 400)
    src = rgb.copy()
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[150:230, 200:420] = 255
    eng = make_engine(migan_path)
    eng.inpaint(rgb, mask, (180, 130, 440, 250))
    hole = mask > 0
    assert abs(float(rgb[hole].mean()) - 100.0) < 4.0          # the hole was filled by the network
    far = np.ones_like(hole)
    far[120:260, 170:450] = False
    assert (rgb[far] == src[far]).all()                        # nothing outside the window changed


def test_lama_engine_fills_every_masked_pixel_of_a_wide_area(lama_path):
    """A 1300 px wide area needs several windows; each masked pixel must be written by exactly the right one."""
    rgb = gradient_bg(1400, 300)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[100:160, 40:1360] = 255
    calls = []
    eng = make_engine(lama_path)
    orig = eng._infer
    eng._infer = lambda canvas, hole: (calls.append(1), orig(canvas, hole))[1]
    eng.inpaint(rgb, mask, (30, 90, 1370, 170))
    assert len(calls) >= 4
    assert abs(float(rgb[mask > 0].mean()) - 127.5) < 6.0
    assert (rgb[:80] == gradient_bg(1400, 300)[:80]).all()


def test_huge_area_is_scaled_down_instead_of_skipped(lama_path):
    rgb = gradient_bg(2600, 1300)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[100:1200, 100:2500] = 255
    eng = make_engine(lama_path)
    eng.inpaint(rgb, mask, (60, 60, 2540, 1240))
    assert abs(float(rgb[mask > 0].mean()) - 127.5) < 8.0


def test_grain_is_put_back_into_a_smooth_fill():
    rng = np.random.default_rng(3)
    base = np.clip(120 + grain((200, 200), 2.0, rng), 0, 255).astype(np.uint8)      # camera-like, slightly soft noise
    mask = np.zeros((200, 200), np.uint8)
    mask[70:130, 70:130] = 255
    flat = base.copy()
    flat[mask > 0] = 120                                         # a network's smooth fill
    out = match_grain(base, flat, mask)
    cv = pytest.importorskip("cv2")
    ring = ring_noise_sigma(base, mask)
    gray = lambda a: cv.cvtColor(a, cv.COLOR_RGB2GRAY).astype(np.float32)          # noqa: E731
    assert ring > 1.5
    assert noise_sigma(gray(flat), mask > 0) < 0.3                                 # the network's fill is plastic ...
    assert 0.7 * ring < noise_sigma(gray(out), mask > 0) < 1.4 * ring                          # ... and now as grainy as its surroundings
    assert (out[mask == 0] == flat[mask == 0]).all()                               # only masked pixels are touched
    clean = np.full((200, 200, 3), 120, np.uint8)               # a clean picture stays clean
    assert match_grain(clean, clean.copy(), mask) is not None
    assert (match_grain(clean, clean.copy(), mask) == clean).all()


def test_resolve_falls_back_to_what_is_installed(tmp_path, lama_path, migan_path):
    mgr = ModelManager(tmp_path / "models")
    mgr.min_bytes = 100
    assert mgr.resolve("auto") == (None, "")
    assert mgr.resolve("lama")[0] is None
    mgr.import_file(migan_path)
    assert mgr.installed() == ["migan"]
    eng, note = mgr.resolve("auto")
    assert isinstance(eng, MiganEngine) and note == ""
    eng, note = mgr.resolve("lama")                              # not installed -> the other one, with a note
    assert isinstance(eng, MiganEngine) and note
    assert mgr.resolve("fast") == (None, "")
    mgr.import_file(lama_path)
    assert mgr.installed() == ["lama", "migan"]
    assert isinstance(mgr.resolve("auto")[0], LamaEngine)       # LaMa is preferred when both are there
    assert isinstance(mgr.resolve("migan")[0], MiganEngine)
    mgr.remove("lama")
    assert mgr.installed() == ["migan"]


def test_import_guesses_the_slot_from_the_file(tmp_path, lama_path, migan_path):
    mgr = ModelManager(tmp_path / "models")
    mgr.min_bytes = 100
    assert mgr.import_file(migan_path) == "migan"
    assert mgr.import_file(lama_path) == "lama"
    assert mgr.import_file(migan_path, model="lama") == "lama"   # an explicit choice wins
    with pytest.raises(ModelError):
        mgr.spec("nope")


class _FakeResponse(io.BytesIO):
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_checks_the_pinned_checksum(tmp_path, lama_path, monkeypatch):
    data = lama_path.read_bytes()
    monkeypatch.setattr("rubberstudio.core.neural.urllib.request.urlopen", lambda req, timeout=0: _FakeResponse(data))
    mgr = ModelManager(tmp_path / "models")
    mgr.min_bytes = 100
    mgr.spec("lama")
    with pytest.raises(ModelError, match="checksum"):
        mgr.download(model="lama")                               # official URL, but not the official file
    assert not mgr.is_ready("lama")
    assert not any(p.name.endswith(".part") for p in mgr.dir.iterdir())
    mgr.download(["https://example.org/my-own-lama.onnx"], model="lama")   # a URL the user chose is not pinned
    assert mgr.is_ready("lama")


def test_catalogue_is_consistent():
    assert set(MODELS) == {"lama", "migan"}
    for spec in MODELS.values():
        assert all(u.startswith("https://") for u in spec.urls)
        assert len(spec.sha256) == 64 and spec.file.endswith(".onnx")
