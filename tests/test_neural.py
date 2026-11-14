import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from conftest import gradient_bg, with_text
from rubberstudio.core.eraser import EraseOptions, TextEraser
from rubberstudio.core.neural import ModelError, ModelManager, NeuralInpainter


def make_dummy_model(path):
    """Tiny 'inpainter' with LaMa's I/O: image*(1-mask) + 0.5*mask, scaled to 0‥255."""
    img = helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 512, 512])
    mask = helper.make_tensor_value_info("mask", TensorProto.FLOAT, ["batch", 1, 512, 512])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 3, 512, 512])
    one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])
    half = helper.make_tensor("half", TensorProto.FLOAT, [1], [0.5])
    k255 = helper.make_tensor("k255", TensorProto.FLOAT, [1], [255.0])
    nodes = [
        helper.make_node("Sub", ["one", "mask"], ["inv"]),
        helper.make_node("Mul", ["image", "inv"], ["keep"]),
        helper.make_node("Mul", ["mask", "half"], ["fill"]),
        helper.make_node("Add", ["keep", "fill"], ["sum"]),
        helper.make_node("Mul", ["sum", "k255"], ["output"]),
    ]
    g = helper.make_graph(nodes, "dummy", [img, mask], [out], initializer=[one, half, k255])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))


def test_manager_validate_import_remove(tmp_path):
    model = tmp_path / "dummy.onnx"
    make_dummy_model(model)
    mgr = ModelManager(tmp_path / "models")
    mgr.min_bytes = 100
    assert not mgr.is_ready() and mgr.status().state == "missing"
    mgr.import_file(model)
    assert mgr.is_ready() and mgr.status().state == "ready"
    assert mgr.inpainter() is not None
    mgr.remove()
    assert not mgr.is_ready()
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"garbage")
    with pytest.raises(ModelError):
        mgr.import_file(bad)


def test_download_rejects_plain_http(tmp_path):
    mgr = ModelManager(tmp_path / "m")
    with pytest.raises(ModelError):
        mgr.download(["http://example.com/x.onnx"])


def test_neural_path_is_used_and_composited(tmp_path, detector):
    model = tmp_path / "dummy.onnx"
    make_dummy_model(model)
    nn = NeuralInpainter(model)
    clean = gradient_bg()
    texty = with_text(clean)
    src = np.dstack([texty, np.full(texty.shape[:2], 255, np.uint8)])
    res = TextEraser(detector, nn).erase(src, opts=EraseOptions(always_neural=True))
    assert res.neural_count >= 1 and res.fast_count == 0
    out = res.image[..., :3].astype(int)
    # pixels well away from the text are not touched by the network
    assert (res.image[300:400, 600:800] == src[300:400, 600:800]).all()
    # inside the text the dummy network wrote mid-grey (127.5)
    text_px = np.abs(texty.astype(int) - clean.astype(int)).sum(axis=2) > 200
    assert abs(float(out[text_px].mean()) - 127.5) < 8
