"""Entry point: logging, Qt setup, command line (--selftest / --screenshot) and the main loop."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Sequence

from . import APP_NAME, __version__

log = logging.getLogger("rubberstudio")
_SHOW_ERROR_DIALOG = True          # off for --selftest / --screenshot, where nobody can click a dialog away


def _fix_std_streams() -> None:
    """A windowed (no-console) build has ``sys.stdout is None`` — make print() and argparse harmless."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def _setup_logging() -> None:
    from .core.paths import logs_dir
    root = logging.getLogger("rubberstudio")
    root.setLevel(logging.INFO)
    if root.handlers:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        fh = RotatingFileHandler(logs_dir() / "rubberstudio.log", maxBytes=512 * 1024, backupCount=2, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass
    if sys.stderr is not None and getattr(sys.stderr, "isatty", lambda: False)():
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)


def _install_excepthook() -> None:
    def hook(exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log.critical("Unhandled exception:\n%s", text)
        if not _SHOW_ERROR_DIALOG:
            try:
                sys.stderr.write(text)
            except Exception:
                pass
            return
        try:
            from PySide6.QtWidgets import QApplication
            if QApplication.instance() is not None:
                from .i18n import tr
                from .widgets import dialogs
                dialogs.error(None, tr("dlg.error"), text[-1800:])
        except Exception:
            pass

    sys.excepthook = hook


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="RubberStudio", description=f"{APP_NAME} {__version__}")
    ap.add_argument("files", nargs="*", help="images (or folders, for the batch queue) to open")
    ap.add_argument("--selftest", metavar="REPORT.json", nargs="?", const="-", help="run the built-in checks and exit")
    ap.add_argument("--screenshot", metavar="OUT.png", help="render the window with a demo image to a PNG and exit")
    ap.add_argument("--tool", default="erase", help="tool shown in --screenshot (crop|repair|watermark|erase|batch)")
    ap.add_argument("--lang", choices=("en", "zh"), help="user interface language")
    ap.add_argument("--no-splash", action="store_true", help="skip the opening animation")
    ap.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    return ap.parse_args(None if argv is None else list(argv))


# ====================================================================== self test

def run_selftest(report_path: Optional[str]) -> int:
    """Exercise the bundled models, the engine, the batch runner and the GUI. Returns a process exit code."""
    import numpy as np

    results: dict = {}

    def step(name: str, fn) -> None:
        t0 = time.time()
        try:
            info = fn() or {}
            results[name] = {"ok": True, "seconds": round(time.time() - t0, 2), **info}
        except Exception:
            results[name] = {"ok": False, "error": traceback.format_exc()}

    from .core.eraser import TextEraser
    from .core.textdet import TextDetector
    from .demo import make_demo_image

    demo = make_demo_image()
    state: dict = {}

    def detect() -> dict:
        det = TextDetector()
        state["det"] = det
        boxes = det.detect(demo[..., :3].copy())
        assert len(boxes) >= 2, f"expected >= 2 text boxes, found {len(boxes)}"
        return {"boxes": len(boxes)}

    def erase() -> dict:
        res = TextEraser(state["det"]).erase(demo.copy())
        assert res.found, "nothing erased"
        assert res.image.shape == demo.shape
        diff = np.abs(res.image.astype(np.int16) - demo.astype(np.int16)).max(-1)
        changed = float((diff > 12).mean())
        assert 0.002 < changed < 0.35, f"implausible changed area {changed:.4f}"
        # the sun, far from any text, must be untouched
        assert int(diff[300:330, 930:960].max()) == 0, "pixels outside the text changed"
        # heavy banner type and the watermark (with its "©") must be gone, not just smudged
        rgb = res.image[..., :3].astype(np.int16)
        band = rgb[80:165, 70:1015]
        assert band[..., 0].min() >= 200 and band[..., 1].max() <= 110 and band[..., 2].max() <= 95, \
            "text left in the banner"
        assert int(rgb[735:775, 925:1240].max()) <= 125, "watermark left behind"
        return {"changed_fraction": round(changed, 4), "fast": res.fast_count, "neural": res.neural_count}

    def batch() -> dict:
        from PIL import Image
        from .core.batch import BatchRunner, BatchSettings, Status, make_items
        with tempfile.TemporaryDirectory(prefix="rubberstudio-selftest-") as tmp:
            src, out = Path(tmp) / "in", Path(tmp) / "out"
            src.mkdir()
            Image.fromarray(demo[..., :3]).save(src / "a.png")
            Image.fromarray(demo[..., :3]).save(src / "b.jpg", quality=92)
            Image.new("RGB", (320, 200), (120, 160, 200)).save(src / "plain.png")
            items = make_items(sorted(src.iterdir()), src)
            summary = BatchRunner(items, BatchSettings(out_dir=out, workers=2), detector=state["det"]).run()
            assert summary.failed == 0, [i.message for i in items]
            assert summary.done == 2 and summary.no_text == 1, (summary.done, summary.no_text)
            names = sorted(p.name for p in out.iterdir())
            assert names == ["a_clean.png", "b_clean.jpg", "plain_clean.png"], names
            return {"done": summary.done, "no_text": summary.no_text}

    def gui() -> dict:
        from PySide6.QtGui import QFontDatabase
        from PySide6.QtWidgets import QApplication
        from .core.neural import ModelManager
        from .mainwindow import MainWindow
        from . import theme as T
        app = QApplication.instance()
        assert app is not None
        T.load_fonts()
        fams = set(QFontDatabase.families())
        assert "Liberation Sans" in fams, "bundled font did not load"
        with tempfile.TemporaryDirectory(prefix="rubberstudio-selftest-") as tmp:
            win = MainWindow(ModelManager(Path(tmp)), splash=False, restore=False)
            win.resize(1280, 800)
            win.show()
            win.editor.load_array(demo.copy())
            win.editor.set_tool(win.editor.tool.ERASE)
            for _ in range(5):
                app.processEvents()
            pm = win.grab()
            ok = pm.width() > 100 and not pm.toImage().isNull()
            win.close()
        assert ok, "window did not render"
        return {"fonts": True}

    step("engine.detect", detect)
    if results["engine.detect"]["ok"]:
        step("engine.erase", erase)
        step("batch", batch)
    step("gui", gui)

    ok = all(v["ok"] for v in results.values())
    report = {"app": APP_NAME, "version": __version__, "python": sys.version.split()[0], "platform": sys.platform,
              "ok": ok, "results": results}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if report_path and report_path != "-":
        Path(report_path).write_text(text, encoding="utf-8")
    else:
        print(text)
    log.info("selftest %s", "passed" if ok else "FAILED")
    return 0 if ok else 1


# ====================================================================== main

def main(argv: Optional[Sequence[str]] = None) -> int:
    _fix_std_streams()
    args = parse_args(argv)
    _setup_logging()
    _install_excepthook()
    log.info("%s %s starting (python %s, %s)", APP_NAME, __version__, sys.version.split()[0], sys.platform)
    global _SHOW_ERROR_DIALOG
    _SHOW_ERROR_DIALOG = args.selftest is None and not args.screenshot

    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QGuiApplication, QIcon
    from PySide6.QtWidgets import QApplication

    from . import i18n, theme as T
    from .core.neural import ModelManager
    from .core.paths import resource_path

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    T.apply_platform_tweaks()
    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("RubberStudio")
    app.setApplicationVersion(__version__)
    T.load_fonts()
    app.setStyleSheet(T.stylesheet())
    app.setFont(T.font_ui(12))
    icon = resource_path("assets", "icon.png")
    if icon.is_file():
        app.setWindowIcon(QIcon(str(icon)))

    from . import settings as S
    i18n.set_language(args.lang or S.get_str("ui/lang", "") or i18n.detect_language())

    if args.selftest is not None:
        return run_selftest(args.selftest)

    from .mainwindow import MainWindow

    shot = args.screenshot
    win = MainWindow(ModelManager(), splash=not (args.no_splash or shot), restore=not shot)
    if shot:
        return _screenshot(app, win, shot, args.tool)
    win.show()
    if args.files:
        QTimer.singleShot(0, lambda: win.open_paths(args.files))
    return app.exec()


def _screenshot(app, win, out: str, tool: str) -> int:
    """Demo picture + chosen tool, detection preview, then save the window as PNG."""
    from .demo import make_demo_image
    from .editor import Tool

    win.resize(1360, 860)
    win.show()
    win.editor.load_array(make_demo_image())
    t = {"crop": Tool.CROP, "repair": Tool.REPAIR, "watermark": Tool.WATERMARK, "erase": Tool.ERASE,
         "batch": Tool.BATCH}.get(tool, Tool.ERASE)
    win.editor.set_tool(t)

    def spin(ms: int) -> None:
        end = time.time() + ms / 1000.0
        while time.time() < end:
            app.processEvents()
            time.sleep(0.01)

    spin(500)
    if t == Tool.ERASE:
        win.editor.erase.whole_image = True
        win.drawer.pages[Tool.ERASE].scope.setCurrent("whole", emit=True)
        win.editor.detect_text()
        deadline = time.time() + 60
        while win.editor.busy and time.time() < deadline:
            spin(50)
    spin(700)
    ok = win.grab().save(out)
    log.info("screenshot %s -> %s", "saved" if ok else "FAILED", out)
    win.close()
    return 0 if ok else 1
