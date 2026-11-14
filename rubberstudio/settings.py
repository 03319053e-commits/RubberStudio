"""Tiny persistent settings store (an .ini file in the user data folder — nothing in the registry)."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSettings

from .core.paths import user_data_dir

_store: Optional[QSettings] = None


def store() -> QSettings:
    global _store
    if _store is None:
        _store = QSettings(str(user_data_dir() / "settings.ini"), QSettings.Format.IniFormat)
    return _store


def reset_for_tests() -> None:
    global _store
    _store = None


def get_str(key: str, default: str = "") -> str:
    v = store().value(key, default)
    return default if v is None else str(v)


def get_float(key: str, default: float) -> float:
    try:
        return float(store().value(key, default))
    except (TypeError, ValueError):
        return float(default)


def get_int(key: str, default: int) -> int:
    try:
        return int(float(store().value(key, default)))
    except (TypeError, ValueError):
        return int(default)


def get_bool(key: str, default: bool) -> bool:
    v = store().value(key, default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def put(key: str, value) -> None:
    store().setValue(key, value)
