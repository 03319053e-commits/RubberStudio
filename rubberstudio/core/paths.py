"""Locations of bundled resources and of the per-user data folder."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def resource_path(*parts: str) -> Path:
    """Path of a bundled resource, both from source and from a PyInstaller build."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    base = Path(frozen_root) if frozen_root else Path(__file__).resolve().parents[2]
    return base.joinpath(*parts)


def user_data_dir() -> Path:
    """Per-user folder for downloaded models and logs (created on demand)."""
    override = os.environ.get("RUBBERSTUDIO_HOME")
    if override:
        base = Path(override)
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "RubberStudio"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "RubberStudio"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "RubberStudio"
    base.mkdir(parents=True, exist_ok=True)
    return base


def models_dir() -> Path:
    d = user_data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = user_data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d
