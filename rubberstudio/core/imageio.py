"""Image loading / saving. Unicode-path safe (Pillow only, never cv2.imread), EXIF-aware.

Images are held as ``H x W x 4`` ``uint8`` RGBA arrays everywhere in the app.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 500_000_000

PathLike = Union[str, "os.PathLike[str]"]

READ_EXTS = (".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".webp", ".bmp", ".tif", ".tiff")

# short name -> (Pillow format, canonical extension)
FORMATS = {
    "png": ("PNG", ".png"),
    "jpg": ("JPEG", ".jpg"),
    "webp": ("WEBP", ".webp"),
    "bmp": ("BMP", ".bmp"),
    "tiff": ("TIFF", ".tif"),
}
_EXT_TO_KEY = {
    ".png": "png", ".jpg": "jpg", ".jpeg": "jpg", ".jpe": "jpg", ".jfif": "jpg",
    ".webp": "webp", ".bmp": "bmp", ".tif": "tiff", ".tiff": "tiff",
}


class ImageIOError(Exception):
    """Raised for unreadable / unsupported images."""


@dataclass
class ImageMeta:
    format_key: str = "png"          # png / jpg / webp / bmp / tiff
    icc_profile: Optional[bytes] = None
    dpi: Optional[tuple] = None
    has_alpha: bool = False


def is_supported(path: PathLike) -> bool:
    return Path(path).suffix.lower() in READ_EXTS


def iter_images(paths: Iterable[PathLike], recursive: bool = True) -> Iterator[Path]:
    """Expand files and folders into a de-duplicated, sorted stream of supported image paths."""
    seen = set()
    for raw in paths:
        p = Path(raw)
        found = []
        if p.is_dir():
            walker = p.rglob("*") if recursive else p.glob("*")
            found = sorted((q for q in walker if q.is_file() and is_supported(q)),
                           key=lambda q: [s.lower() for s in q.parts])
        elif p.is_file() and is_supported(p):
            found = [p]
        for q in found:
            key = os.path.normcase(os.path.abspath(q))
            if key not in seen:
                seen.add(key)
                yield q


def _to_8bit_rgba(im: Image.Image) -> Image.Image:
    if im.mode in ("I;16", "I;16L", "I;16B", "I"):
        arr = np.asarray(im, dtype=np.uint32)
        top = int(arr.max()) if arr.size else 0
        arr8 = (arr >> 8).astype(np.uint8) if top > 255 else arr.astype(np.uint8)
        return Image.fromarray(arr8, "L").convert("RGBA")
    if im.mode == "F":
        arr = np.asarray(im, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr8 = ((arr - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
        return Image.fromarray(arr8, "L").convert("RGBA")
    if im.mode == "CMYK":
        return im.convert("RGB").convert("RGBA")
    return im.convert("RGBA")


def load_image(path: PathLike) -> tuple[np.ndarray, ImageMeta]:
    """Read an image into an RGBA array (EXIF orientation applied)."""
    p = Path(path)
    try:
        with Image.open(p) as im:
            fmt_key = _EXT_TO_KEY.get(p.suffix.lower()) or {
                "PNG": "png", "JPEG": "jpg", "WEBP": "webp", "BMP": "bmp", "TIFF": "tiff",
            }.get(im.format or "", "png")
            if getattr(im, "n_frames", 1) > 1 and (im.format or "") in ("WEBP", "GIF"):
                raise ImageIOError("animated images are not supported")
            im.load()
            meta = ImageMeta(
                format_key=fmt_key,
                icc_profile=im.info.get("icc_profile"),
                dpi=im.info.get("dpi"),
                has_alpha=im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info),
            )
            im = ImageOps.exif_transpose(im)
            rgba = _to_8bit_rgba(im)
            arr = np.array(rgba, dtype=np.uint8)
    except ImageIOError:
        raise
    except Exception as exc:  # Pillow raises many different types
        raise ImageIOError(f"{p.name}: {exc}") from exc
    return np.ascontiguousarray(arr), meta


def key_for_path(path: PathLike) -> str:
    return _EXT_TO_KEY.get(Path(path).suffix.lower(), "png")


def save_image(path: PathLike, rgba: np.ndarray, *, fmt: Optional[str] = None, quality: int = 95,
               meta: Optional[ImageMeta] = None) -> Path:
    """Write ``rgba`` to ``path``; ``fmt`` is a key of :data:`FORMATS` (default: from extension)."""
    p = Path(path)
    key = fmt or key_for_path(p)
    if key not in FORMATS:
        raise ImageIOError(f"unsupported output format: {key}")
    pil_fmt = FORMATS[key][0]
    p.parent.mkdir(parents=True, exist_ok=True)

    rgba = np.ascontiguousarray(rgba)
    opaque = bool(rgba.shape[2] < 4 or (rgba[..., 3] == 255).all())
    im = Image.fromarray(rgba[..., :3] if opaque else rgba, "RGB" if opaque else "RGBA")

    kwargs: dict = {}
    if meta is not None and meta.icc_profile:
        kwargs["icc_profile"] = meta.icc_profile
    if meta is not None and meta.dpi:
        kwargs["dpi"] = meta.dpi
    q = int(max(1, min(100, quality)))
    if pil_fmt == "JPEG":
        if not opaque:  # flatten on white
            flat = Image.new("RGB", im.size, (255, 255, 255))
            flat.paste(im, mask=im.getchannel("A"))
            im = flat
        kwargs.update(quality=q, optimize=True, subsampling=0 if q >= 90 else 2)
    elif pil_fmt == "WEBP":
        kwargs.update(quality=q, method=4, lossless=(q >= 100))
    elif pil_fmt == "PNG":
        kwargs.update(optimize=False, compress_level=6)
    elif pil_fmt == "BMP" and not opaque:
        flat = Image.new("RGB", im.size, (255, 255, 255))
        flat.paste(im, mask=im.getchannel("A"))
        im = flat
    elif pil_fmt == "TIFF":
        kwargs.update(compression="tiff_lzw")

    tmp = p.with_name(p.name + ".part")
    try:
        im.save(tmp, format=pil_fmt, **kwargs)
        os.replace(tmp, p)
    except Exception as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise ImageIOError(f"{p.name}: {exc}") from exc
    return p


def unique_path(path: Path) -> Path:
    """``name.png`` -> ``name (2).png`` … until the path is free."""
    if not path.exists():
        return path
    n = 2
    while True:
        cand = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if not cand.exists():
            return cand
        n += 1
