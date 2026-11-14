"""Hop from a worker thread onto the GUI thread."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, Signal, Slot


class Dispatcher(QObject):
    """Create it on the GUI thread; ``post(fn)`` may then be called from any thread."""
    _call = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._call.connect(self._run)

    def post(self, fn: Callable[[], None]) -> None:
        self._call.emit(fn)

    @Slot(object)
    def _run(self, fn) -> None:
        try:
            fn()
        except Exception:                      # a failing UI callback must not kill the worker's signal queue
            import logging
            logging.getLogger("rubberstudio").exception("GUI callback failed")
