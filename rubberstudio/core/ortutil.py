"""ONNX Runtime helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import onnxruntime as ort


def open_session(path, so: Optional[ort.SessionOptions] = None) -> ort.InferenceSession:
    """CPU session for the model at ``path``.

    A path containing non-ASCII characters (a Chinese Windows user name in ``%LOCALAPPDATA%``, an install folder such
    as ``D:\\软件\\RubberStudio``) is loaded from bytes, so no code-page conversion can get in the way of the file name.
    """
    p = Path(path)
    try:
        str(p).encode("ascii")
        source: object = str(p)
    except UnicodeEncodeError:
        source = p.read_bytes()
    return ort.InferenceSession(source, sess_options=so, providers=["CPUExecutionProvider"])
