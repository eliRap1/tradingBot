"""Persistent state management — survives bot restarts."""

import json
import os
from utils import setup_logger

log = setup_logger("state")

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")

_DEFAULTS = {
    "peak_equity": 0.0,
    "high_watermarks": {},
}


def load_state() -> dict:
    """Load saved state from disk, or return defaults."""
    if not os.path.exists(STATE_FILE):
        log.info("No state file found — starting fresh")
        return dict(_DEFAULTS)

    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        log.info(f"Loaded state: peak_equity=${data.get('peak_equity', 0):,.2f}, "
                 f"{len(data.get('high_watermarks', {}))} watermarks")
        # Merge with defaults so new keys are always present
        merged = dict(_DEFAULTS)
        merged.update(data)
        return merged
    except Exception as e:
        log.error(f"Failed to load state file: {e} — starting fresh")
        return dict(_DEFAULTS)


def save_state(state: dict):
    """Persist state to disk."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log.error(f"Failed to save state: {e}")
