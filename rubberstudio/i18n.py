"""Tiny two-language (English / 简体中文) string table with live switching."""
from __future__ import annotations

import locale
from typing import Dict

from PySide6.QtCore import QObject, Signal

from .strings import STRINGS

_lang = "en"


def detect_language() -> str:
    try:
        name = (locale.getlocale()[0] or locale.getdefaultlocale()[0] or "")  # type: ignore[attr-defined]
    except Exception:
        name = ""
    return "zh" if name.lower().startswith(("zh", "chinese")) else "en"


def language() -> str:
    return _lang


def set_language(lang: str) -> None:
    global _lang
    _lang = "zh" if lang == "zh" else "en"
    bus.changed.emit(_lang)


def tr(key: str, **fmt) -> str:
    table: Dict[str, str] = STRINGS.get(_lang, {})
    s = table.get(key)
    if s is None:
        s = STRINGS["en"].get(key, key)
    if fmt:
        try:
            return s.format(**fmt)
        except (KeyError, IndexError, ValueError):
            return s
    return s


class _Bus(QObject):
    changed = Signal(str)


bus = _Bus()


class Translatable:
    """Mixin for widgets whose text comes from a key: ``retranslate()`` is called on language change."""
    _key: str = ""
    _fmt: dict = {}

    def init_key(self, key: str = "", **fmt) -> None:
        self._key = key
        self._fmt = fmt

    def set_key(self, key: str, **fmt) -> None:
        self._key = key
        self._fmt = fmt
        self.retranslate()

    def text_now(self) -> str:
        return tr(self._key, **self._fmt) if self._key else ""

    def retranslate(self) -> None:   # overridden by widgets
        pass
