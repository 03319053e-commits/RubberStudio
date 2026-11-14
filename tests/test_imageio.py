import numpy as np
from PIL import Image

from rubberstudio.core import imageio


def test_unicode_path_roundtrip_png_alpha(tmp_path):
    arr = np.zeros((20, 30, 4), np.uint8)
    arr[..., 0] = 200
    arr[:, :15, 3] = 255          # half transparent
    p = tmp_path / "图片 测试 ✓.png"
    imageio.save_image(p, arr)
    back, meta = imageio.load_image(p)
    assert back.shape == (20, 30, 4)
    assert (back[:, :15, 3] == 255).all() and (back[:, 15:, 3] == 0).all()
    assert meta.has_alpha and meta.format_key == "png"


def test_jpeg_flattens_alpha_and_exif_orientation(tmp_path):
    src = Image.new("RGB", (40, 20), (10, 200, 10))
    exif = Image.Exif()
    exif[0x0112] = 6                      # rotate 90° CW when displayed
    p = tmp_path / "rot.jpg"
    src.save(p, exif=exif)
    arr, _ = imageio.load_image(p)
    assert arr.shape[:2] == (40, 20)      # width/height swapped by EXIF transpose
    out = tmp_path / "o.jpg"
    imageio.save_image(out, np.dstack([arr[..., :3], np.zeros(arr.shape[:2], np.uint8)]))
    flat, _ = imageio.load_image(out)
    assert flat[..., :3].min() > 240      # fully transparent -> white


def test_16bit_gray_png(tmp_path):
    g = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 16)
    p = tmp_path / "g16.png"
    Image.fromarray(g).save(p)
    arr, _ = imageio.load_image(p)
    assert arr.dtype == np.uint8 and arr.shape == (64, 64, 4)
    assert arr[-1, -1, 0] > arr[0, 0, 0]


def test_iter_images_and_unique_path(tmp_path):
    (tmp_path / "a").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "a" / "x.png")
    Image.new("RGB", (4, 4)).save(tmp_path / "b.JPG")
    (tmp_path / "note.txt").write_text("hi")
    found = list(imageio.iter_images([tmp_path, tmp_path / "b.JPG"]))
    assert len(found) == 2
    p = tmp_path / "b.JPG"
    assert imageio.unique_path(p).name == "b (2).JPG"
