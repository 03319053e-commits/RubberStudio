import threading

import numpy as np
from PIL import Image

from conftest import gradient_bg, with_text
from rubberstudio.core import imageio
from rubberstudio.core.batch import BatchRunner, BatchSettings, Status, make_items, plan_output


def make_folder(root):
    (root / "子文件夹").mkdir(parents=True)
    files = []
    for i in range(3):
        p = root / ("图片%d.png" % i)
        Image.fromarray(with_text(gradient_bg(seed=i))).save(p)
        files.append(p)
    clean = root / "子文件夹" / "无字.jpg"
    Image.fromarray(gradient_bg(seed=9)).save(clean, quality=92)
    files.append(clean)
    bad = root / "坏文件.png"
    bad.write_bytes(b"not an image")
    files.append(bad)
    return files


def test_batch_end_to_end(tmp_path, detector):
    src = tmp_path / "in"
    files = make_folder(src)
    out = tmp_path / "out"
    s = BatchSettings(out_dir=out, workers=2)
    items = make_items(files, root=src)
    events = []
    summary = BatchRunner(items, s, detector=detector).run(lambda i, it: events.append((i, it.status)))

    st = {it.src.name: it for it in items}
    assert summary.done == 3 and summary.no_text == 1 and summary.failed == 1
    assert st["坏文件.png"].status == Status.FAILED and st["坏文件.png"].message
    # sub-folders are mirrored, names get the suffix
    assert (out / "图片0_clean.png").is_file()
    assert (out / "子文件夹" / "无字_clean.jpg").is_file()
    # image without text is copied byte-for-byte
    assert (out / "子文件夹" / "无字_clean.jpg").read_bytes() == (src / "子文件夹" / "无字.jpg").read_bytes()
    # sources untouched, outputs have the same size and no more text
    a, _ = imageio.load_image(out / "图片1_clean.png")
    assert a.shape[:2] == (600, 900)
    again = BatchRunner(make_items([out / "图片1_clean.png"]), BatchSettings(out_dir=tmp_path / "o2"),
                        detector=detector)
    again.run()
    assert again.items[0].status == Status.NO_TEXT
    assert any(s_ == Status.RUNNING for _, s_ in events)


def test_existing_files_and_source_protection(tmp_path, detector):
    src = tmp_path / "in"
    src.mkdir()
    p = src / "a.png"
    Image.fromarray(with_text(gradient_bg())).save(p)
    before = p.read_bytes()
    # empty suffix + same folder would overwrite the source -> must be renamed instead
    s = BatchSettings(out_dir=src, suffix="", workers=1)
    BatchRunner(make_items([p]), s, detector=detector).run()
    assert p.read_bytes() == before
    assert (src / "a (2).png").is_file()
    # skip policy
    s2 = BatchSettings(out_dir=tmp_path / "o", existing="skip", workers=1)
    BatchRunner(make_items([p]), s2, detector=detector).run()
    it = make_items([p])
    BatchRunner(it, s2, detector=detector).run()
    assert it[0].out_path is None


def test_format_conversion_and_cancel(tmp_path, detector):
    src = tmp_path / "in"
    src.mkdir()
    paths = []
    for i in range(4):
        p = src / f"{i}.png"
        Image.fromarray(with_text(gradient_bg(seed=i))).save(p)
        paths.append(p)
    s = BatchSettings(out_dir=tmp_path / "o", fmt="jpg", quality=90, workers=1)
    items = make_items(paths)
    runner = BatchRunner(items, s, detector=detector)
    done = threading.Event()

    def on_update(i, it):
        if it.status == Status.DONE:
            runner.cancel.set()           # stop after the first finished image
            done.set()

    summary = runner.run(on_update)
    assert items[0].status == Status.DONE and items[0].out_path.suffix == ".jpg"
    assert summary.canceled >= 1
    assert plan_output(items[0], s).suffix == ".jpg"


def test_same_region_for_all_images(tmp_path, detector):
    src = tmp_path / "in"
    src.mkdir()
    p = src / "r.png"
    texty = with_text(gradient_bg())
    Image.fromarray(texty).save(p)
    s = BatchSettings(out_dir=tmp_path / "o", region=(0.0, 0.0, 1.0, 0.2), workers=1)
    it = make_items([p])
    BatchRunner(it, s, detector=detector).run()
    out, _ = imageio.load_image(it[0].out_path)
    assert (out[480:540, 500:800, :3] == texty[480:540, 500:800]).all()   # outside region: untouched
    assert (out[40:130, 40:900, :3] != texty[40:130, 40:900]).any()        # inside region: erased
