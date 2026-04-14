"""Persistent state management backed by SQLite."""

import os

from state_db import StateDB
from utils import setup_logger

log = setup_logger("state")

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")

_DEFAULTS = {
    "peak_equity": 0.0,
    "high_watermarks": {},
}

# Module-level singleton — avoids opening a new SQLite connection per save/load call.
_db_instance: StateDB | None = None


def _get_db() -> StateDB:
    global _db_instance
    if _db_instance is None:
        _db_instance = StateDB(base_dir=os.path.dirname(STATE_FILE))
        _db_instance.migrate_from_json()
    return _db_instance


def load_state() -> dict:
    """Load saved state from SQLite, with one-time JSON migration."""
    db = _get_db()
    watermarks = db.get_watermarks()
    positions = {pos["symbol"]: pos for pos in db.get_all_positions()}
    state = dict(_DEFAULTS)
    state["peak_equity"] = db.get_state("peak_equity", _DEFAULTS["peak_equity"])
    state["high_watermarks"] = {
        symbol: row["high_watermark"]
        for symbol, row in watermarks.items()
        if row.get("high_watermark") is not None
    }
    state["low_watermarks"] = {
        symbol: row["low_watermark"]
        for symbol, row in watermarks.items()
        if row.get("low_watermark") is not None
    }
    state["position_meta"] = positions
    try:
        log.info(
            f"Loaded state: peak_equity=${state.get('peak_equity', 0):,.2f}, "
            f"{len(state.get('high_watermarks', {}))} watermarks"
        )
        return state
    except Exception as e:
        log.error(f"Failed to load state DB: {e} - starting fresh")
        return dict(_DEFAULTS)


def save_state(state: dict):
    """Persist state to SQLite."""
    try:
        db = _get_db()
        db.set_state("peak_equity", state.get("peak_equity", _DEFAULTS["peak_equity"]))
        db.replace_watermarks(
            state.get("high_watermarks", {}) or {},
            state.get("low_watermarks", {}) or {},
        )
        db.replace_positions(state.get("position_meta", {}) or {})
    except Exception as e:
        log.error(f"Failed to save state: {e}")
