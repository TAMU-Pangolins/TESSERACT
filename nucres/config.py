"""
Configuration helpers for locating nucres auxiliary data (e.g., HFB tables).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

ENV_VAR = "NUCRES_DATA_ROOT"
CONFIG_DIR = Path.home() / ".nucres"
CONFIG_PATH = CONFIG_DIR / "config.json"
CACHE_DATA_ROOT = (
    Path.home()
    / ".cache"
    / "nucres"
    / "densities"
    / "level-densities-hfb"
)
CHECKOUT_DATA_ROOT = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "densities"
    / "level-densities-hfb"
)


def store_data_root(path: Path) -> None:
    """Persist the configured auxiliary-data root in the user config file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump({"data_root": str(Path(path).resolve())}, fh, indent=2)


def _load_config_path() -> Optional[Path]:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        root = data.get("data_root")
        if root:
            p = Path(root).expanduser()
            if p.exists():
                return p
    return None


def _default_data_root() -> Path:
    """
    Return checkout-local data when present, otherwise use the user cache.

    Source checkouts with an existing HFB dataset retain their current behavior.
    Installed packages use a writable cache instead of attempting to modify
    site-packages.
    """
    if CHECKOUT_DATA_ROOT.exists():
        return CHECKOUT_DATA_ROOT
    try:
        CACHE_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - depends on host permissions
        raise FileNotFoundError(
            f"Cannot create default data directory at {CACHE_DATA_ROOT}"
        ) from exc
    return CACHE_DATA_ROOT


def resolve_data_root() -> Path:
    """Resolve the active auxiliary-data directory from env, config, or default."""
    env_value = os.environ.get(ENV_VAR)
    if env_value:
        env_path = Path(env_value).expanduser()
        if env_path.exists():
            return env_path
    cfg_path = _load_config_path()
    if cfg_path:
        return cfg_path
    try:
        return _default_data_root()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "nucres data root not found and the user cache could not be created. "
            f"Set the {ENV_VAR} environment variable to a writable directory."
        ) from exc


def ensure_data_root_env() -> Path:
    """Resolve the data root and mirror it into `NUCRES_DATA_ROOT`."""
    root = resolve_data_root()
    os.environ[ENV_VAR] = str(root)
    return root
