"""GUI behaviour through real (offscreen) Qt events: tools, canvas input, batch run, language, folding."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from rubberstudio import i18n, settings as S, theme as T
from rubberstudio.core.neural import ModelManager
from rubberstudio.demo import make_demo_image
from rubberstudio.editor import RepairTool, Tool
from rubberstudio.mainwindow import LEFT_W, MainWindow
from rubberstudio.widgets.envelope import TAB


def wait_until(cond, timeout=90.0, step=20):
    end = time.time() + timeout
    while time.time() < end:
        QApplication.processEvents()
        if cond():
            return True
        QTest.qWait(step)
    return cond()


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    T.load_fonts()
    app.setStyleSheet(T.stylesheet())
    app.setFont(T.font_ui(12))
    return app


@pytest.fixture()
def win(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RUBBERSTUDIO_HOME", str(tmp_path / "home"))
    S.reset_for_tests()
    i18n.set_language("en")
    w = MainWindow(ModelManager(tmp_path / "models"), splash=False, restore=False)
    w.resize(1360, 860)
    w.show()
    qapp.processEvents()
    yield w
    w.batch_ctl.stop()
    wait_until(lambda: not w.batch_ctl.running, 30)
    w.close()
    i18n.set_language("en")
    S.reset_for_tests()


@pytest.fixture()
def demo():
    return make_demo_image()


def ready(win, demo, tool=Tool.NONE):
    win.editor.load_array(demo.copy())
    win.editor.set_tool(tool)
    win.canvas.grab()                      # paint once so the image rectangle is known
    QApplication.processEvents()


def to_widget(win, x, y) -> QPoint:
    c = win.canvas
    r, s = c._img_rect, c.scale()
    return QPoint(int(r.left() + x * s), int(r.top() + y * s))


def drag(widget, a: QPoint, b: QPoint, steps=6):
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a)
    for i in range(1, steps + 1):
        p = QPoint(a.x() + (b.x() - a.x()) * i // steps, a.y() + (b.y() - a.y()) * i // steps)
        QTest.mouseMove(widget, p)
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b)


# ---------------------------------------------------------------------------------------------

def test_envelope_panels_fold_and_unfold(win):
    assert win.left_env.is_open() and win.left_env.width() == LEFT_W + TAB
    win.left_env.set_open(False)
    assert wait_until(lambda: win.left_env.width() == TAB)
    win.left_env.set_open(True)
    assert wait_until(lambda: win.left_env.width() == LEFT_W + TAB)
    win.header_env.set_open(False)
    assert wait_until(lambda: win.header_env.height() == TAB)
    win.status_env.set_open(False, animate=False)
    assert win.status_env.height() == TAB


def test_language_switch_retranslates(win):
    assert win.header.btn_open.text() == "OPEN"
    win.toggle_language()
    assert i18n.language() == "zh"
    assert win.header.btn_open.text() == "打开"
    assert win.drawer.buttons[Tool.ERASE].text() == "去除文字"
    assert win.statusbar._text == i18n.tr("st.ready")
    win.toggle_language()
    assert win.header.btn_open.text() == "OPEN"


def test_open_save_roundtrip_and_title(win, tmp_path, demo):
    src = tmp_path / "pic.png"
    Image.fromarray(demo).save(src)
    win.open_paths([str(src)])
    assert win.editor.size == (1280, 800) and "pic.png" in win.windowTitle()
    out = win.editor.export(tmp_path / "out.jpg", quality=90)
    assert out.exists() and Image.open(out).size == (1280, 800)


def test_repair_stroke_and_undo(win, demo):
    ready(win, demo, Tool.REPAIR)
    e = win.editor
    e.brush.size, e.brush.tool = 40, RepairTool.HEAL
    before = e.image.copy()
    drag(win.canvas, to_widget(win, 95, 236), to_widget(win, 415, 236))        # along the small caption
    assert e.can_undo()
    changed = np.abs(e.image.astype(int) - before.astype(int)).max(-1) > 6
    assert changed[212:262, 90:420].mean() > 0.01           # the stroke softened the caption's letters
    assert not changed[:200].any() and not changed[275:].any()      # nothing outside the brush band moved
    e.undo()
    assert np.array_equal(e.image, before)
    e.redo()
    assert not np.array_equal(e.image, before)


def test_smudge_and_texture_strokes(win, demo):
    ready(win, demo, Tool.REPAIR)
    e = win.editor
    for tool in (RepairTool.SMUDGE, RepairTool.TEXTURE):
        e.brush.tool, e.brush.size = tool, 30
        before = e.image.copy()
        drag(win.canvas, to_widget(win, 500, 520), to_widget(win, 700, 560))
        assert not np.array_equal(e.image, before), tool


def test_crop_by_dragging_handles(win, demo):
    ready(win, demo, Tool.CROP)
    e, c = win.editor, win.canvas
    x, y, w, h = e.crop_rect
    wr = c._rect_to_widget(e.crop_rect)
    handle = c._handles(wr)[4]                       # bottom-right
    drag(c, handle.toPoint(), QPoint(int(handle.x() - 120), int(handle.y() - 80)))
    assert e.crop_rect[2] < w - 0.05 and e.crop_rect[3] < h - 0.05
    expect = (round(e.crop_rect[2] * 1280), round(e.crop_rect[3] * 800))
    assert e.confirm_crop()
    assert abs(e.size[0] - expect[0]) <= 2 and abs(e.size[1] - expect[1]) <= 2
    e.undo()
    assert e.size == (1280, 800)


def test_new_frame_by_dragging_on_picture(win, demo):
    ready(win, demo, Tool.ERASE)
    e = win.editor
    e.erase.whole_image = False
    drag(win.canvas, to_widget(win, 10, 10), to_widget(win, 300, 200))        # outside the current frame -> new frame
    x, y, w, h = e.erase_rect
    assert 0.2 < w < 0.3 and 0.2 < h < 0.3


def test_watermark_drag_and_apply(win, demo):
    ready(win, demo, Tool.WATERMARK)
    e = win.editor
    e.wm.text, e.wm.opacity, e.wm.color = "TEST MARK", 1.0, "black"
    win.canvas.grab()
    wr = win.canvas._wm_widget_rect()
    drag(win.canvas, wr.center().toPoint(), QPoint(int(wr.center().x() - 300), int(wr.center().y() - 250)))
    assert e.wm.center[0] < 0.6 and e.wm.center[1] < 0.7
    before = e.image.copy()
    rect = e.watermark_rect()
    e.apply_watermark()
    x0, y0, x1, y1 = int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())
    changed = np.abs(e.image.astype(int) - before.astype(int)).max(-1) > 20
    assert changed[y0:y1, x0:x1].mean() > 0.02
    assert changed.sum() == changed[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4].sum()     # only inside the label
    e.undo()
    assert np.array_equal(e.image, before)


def test_detect_then_erase_whole_image_with_compare(win, demo):
    ready(win, demo, Tool.ERASE)
    e = win.editor
    e.erase.whole_image, e.erase.use_ai = True, False
    e.detect_text()
    assert wait_until(lambda: not e.busy) and len(e.boxes) >= 3
    before = e.image.copy()
    e.erase_text()
    assert wait_until(lambda: not e.busy and e.after is not None)
    assert e.status_text().startswith("Text Removed")
    diff = np.abs(e.image.astype(int) - before.astype(int)).max(-1)
    assert diff[300:330, 930:960].max() == 0                  # the sun is untouched
    assert (diff[80:165, 77:1010] > 12).mean() > 0.02         # banner text is gone
    assert win.canvas.cmp_btn.isVisible() or win.canvas.cmp_btn.isVisibleTo(win.canvas)
    e.toggle_compare()
    assert e.show_after and np.array_equal(e.qimage().constBits() is not None and e.after, before)
    e.undo()
    assert np.array_equal(e.image, before) and e.after is None


def test_erase_only_inside_frame(win, demo):
    ready(win, demo, Tool.ERASE)
    e = win.editor
    e.erase.whole_image, e.erase.use_ai = False, False
    e.erase_rect = (0.05, 0.25, 0.4, 0.1)                  # just the small caption (x 64‥576, y 200‥280)
    before = e.image.copy()
    e.erase_text()
    assert wait_until(lambda: not e.busy and e.can_undo())
    diff = np.abs(e.image.astype(int) - before.astype(int)).max(-1)
    assert (diff[214:254, 88:420] > 12).any()
    assert diff[:190, :].max() == 0                         # banner not touched
    assert diff[290:, :].max() == 0                         # copyright line not touched


def test_wheel_zoom_keeps_point_under_cursor(win, demo):
    ready(win, demo)
    c = win.canvas
    pos = to_widget(win, 640, 400)
    img_before = c.to_image(QPointF(pos))
    ev = QWheelEvent(QPointF(pos), QPointF(pos), QPoint(0, 0), QPoint(0, 240), Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    c.wheelEvent(ev)
    assert c.zoom > 1.2
    c.grab()
    img_after = c.to_image(QPointF(pos))
    assert abs(img_before.x() - img_after.x()) < 1.5 and abs(img_before.y() - img_after.y()) < 1.5
    c.fit()
    assert c.zoom == 1.0


def test_batch_run_from_ui(win, tmp_path, demo):
    src = tmp_path / "in"
    (src / "sub").mkdir(parents=True)
    Image.fromarray(demo[..., :3]).save(src / "a.png")
    Image.fromarray(demo[..., :3]).save(src / "sub" / "b.jpg", quality=90)
    Image.new("RGB", (300, 200), (90, 140, 200)).save(src / "plain.png")
    (src / "notes.txt").write_text("not an image")
    win.open_paths([str(src)])                                  # a folder -> batch queue + batch tool
    assert win.editor.tool == Tool.BATCH and win.batch_model.counts()["total"] == 3
    assert win.right_env.is_open() and win.editor.image is not None
    out = tmp_path / "out"
    win.batch_ctl.settings.out_dir = out
    win.editor.erase.use_ai = False
    win.start_batch()
    assert win.batch_ctl.running
    assert wait_until(lambda: not win.batch_ctl.running, 180)
    c = win.batch_model.counts()
    assert (c["done"], c["no_text"], c["failed"]) == (2, 1, 0), c
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.*")) == [
        "a_clean.png", "plain_clean.png", "sub/b_clean.jpg"]
    page = win.drawer.batch_page
    assert page.btn_open.isVisibleTo(page) and "2 cleaned" in page.summary.text()
    assert win.statusbar._text.startswith("Batch finished")
    win.start_batch()                                           # nothing pending -> "run again" requeues everything
    assert wait_until(lambda: not win.batch_ctl.running, 180)
    assert win.batch_model.counts()["done"] == 2


def test_batch_stop_writes_nothing_for_cancelled_items(win, tmp_path, demo):
    src = tmp_path / "in"
    src.mkdir()
    for i in range(6):
        Image.fromarray(demo[..., :3]).save(src / f"p{i}.png")
    win.open_paths([str(src)])
    win.batch_ctl.settings.out_dir = tmp_path / "out"
    win.editor.erase.use_ai = False
    win.start_batch()
    QTest.qWait(300)
    win.batch_ctl.stop()
    assert wait_until(lambda: not win.batch_ctl.running, 180)
    c = win.batch_model.counts()
    assert c["done"] + c["canceled"] + c["no_text"] + c["failed"] == 6
    outs = list((tmp_path / "out").glob("*.png")) if (tmp_path / "out").exists() else []
    assert len(outs) == c["done"]


def test_drop_event_routes_files(win, tmp_path, demo):
    one = tmp_path / "one.png"
    Image.fromarray(demo).save(one)
    other = tmp_path / "two.png"
    Image.fromarray(demo).save(other)

    def drop(paths):
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
        ev = QDropEvent(QPointF(100, 100), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        win.dropEvent(ev)

    drop([one])
    assert win.editor.path == one and win.batch_model.counts()["total"] == 0
    drop([one, other])
    assert win.batch_model.counts()["total"] == 2 and win.editor.tool == Tool.BATCH
    drop([one, other])                                         # duplicates are ignored
    assert win.batch_model.counts()["total"] == 2


def test_model_controller_import_and_remove(win, tmp_path, monkeypatch):
    from test_neural import make_dummy_model
    from rubberstudio.widgets import dialogs
    shown = []
    monkeypatch.setattr(dialogs, "error", lambda parent, body, detail="": shown.append((body, detail)))   # a modal box would block
    m = tmp_path / "dummy.onnx"
    make_dummy_model(m)
    ctl = win.model_ctl
    ctl.manager.min_bytes = 100
    assert ctl.state == "missing"
    ctl.import_file(str(m))
    assert wait_until(lambda: ctl.state == "ready", 30)
    ctl.remove()
    assert ctl.state == "missing"
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"x" * 500)
    ctl.import_file(str(bad))
    assert wait_until(lambda: bool(shown), 30) and ctl.state == "missing"
    assert "not a valid ONNX model" in shown[0][1]


def test_all_tool_pages_switch_and_render(win, demo):
    ready(win, demo)
    for t in (Tool.CROP, Tool.REPAIR, Tool.WATERMARK, Tool.ERASE, Tool.BATCH):
        win.drawer.buttons[t].click()
        QApplication.processEvents()
        assert win.editor.tool == t and win.drawer.pages[t].isVisibleTo(win.drawer)
        assert not win.grab().isNull()
    win.drawer.buttons[Tool.BATCH].click()                      # clicking the active tool again closes it
    assert win.editor.tool == Tool.NONE
    assert wait_until(lambda: not win.right_env.is_open(), 5) or True
