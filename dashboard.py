"""
Web dashboard for the trading bot.
Run: python dashboard.py
Open: http://localhost:5000

Shows per-stock watcher threads, their status, which strategy each is using,
candle patterns detected, and live signals.
"""

import json
import os
import threading
import time
from datetime import datetime

import numpy as np
from flask import Flask, render_template, jsonify

from utils import load_config, setup_logger
from broker import Broker
from data import DataFetcher
from tracker import TradeTracker
from state import load_state
from regime import RegimeFilter
from watcher import StockWatcher
from strategy_selector import select_strategies
from candles import detect_patterns, bullish_score, bearish_score
from trend import get_trend_context

log = setup_logger("dashboard")

app = Flask(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


# Global state
_state = {
    "account": {},
    "positions": [],
    "regime": {},
    "watchers": [],
    "stats": {},
    "trades": [],
    "logs": [],
    "last_update": None,
    "thread_count": 0,
}

_config = None
_broker = None
_data = None
_tracker = None
_regime = None
_watchers: dict[str, StockWatcher] = {}

MAX_LOGS = 200


def _add_log(msg: str, level: str = "info"):
    _state["logs"].insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "level": level,
        "msg": msg
    })
    if len(_state["logs"]) > MAX_LOGS:
        _state["logs"] = _state["logs"][:MAX_LOGS]


def init_connections():
    global _config, _broker, _data, _tracker, _regime
    _config = load_config()
    _broker = Broker(_config)
    _data = DataFetcher(_broker)
    _tracker = TradeTracker()
    _regime = RegimeFilter(_data)
    _add_log("Dashboard connected to Alpaca")


def start_watchers():
    """Start a watcher thread per stock for the dashboard."""
    global _watchers
    universe = _config["screener"]["universe"]

    for sym in universe:
        if sym not in _watchers:
            w = StockWatcher(
                symbol=sym,
                config=_config,
                data_fetcher=_data,
                interval=120  # Dashboard watchers refresh every 2 min
            )
            _watchers[sym] = w
            w.start()
            time.sleep(0.3)  # Stagger to avoid rate limits

    _add_log(f"Started {len(_watchers)} watcher threads")


def refresh_data():
    """Fetch account, positions, regime, and watcher states."""
    try:
        # Account
        account = _broker.get_account()
        _state["account"] = {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "pnl_today": float(account.equity) - float(account.last_equity),
            "pnl_today_pct": ((float(account.equity) - float(account.last_equity))
                              / float(account.last_equity) * 100
                              if float(account.last_equity) > 0 else 0),
        }

        # Positions
        positions = []
        for pos in _broker.get_positions():
            positions.append({
                "symbol": pos.symbol,
                "qty": int(pos.qty),
                "entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "pnl": float(pos.unrealized_pl),
                "pnl_pct": float(pos.unrealized_plpc) * 100,
                "side": pos.side,
            })
        _state["positions"] = positions

        # Regime
        _state["regime"] = _regime.get_regime()

        # Trades + stats
        _state["trades"] = list(reversed(_tracker.trades[-50:]))
        _state["stats"] = _tracker.get_stats()

        # Peak equity
        state = load_state()
        _state["peak_equity"] = state.get("peak_equity", 0)

        # Watcher thread states
        watcher_states = []
        for sym, w in sorted(_watchers.items()):
            s = w.state
            watcher_states.append({
                "symbol": s.symbol,
                "status": s.status,
                "score": float(s.score),
                "num_agreeing": s.num_agreeing,
                "strategy_scores": {k: float(v) for k, v in s.strategy_scores.items()},
                "strategy_weights": {k: float(v) for k, v in s.strategy_weights.items()},
                "regime": s.regime,
                "regime_reason": s.regime_reason,
                "candle_patterns": list(s.candle_patterns),
                "adx": float(s.adx),
                "trend": s.trend_direction,
                "above_200ema": bool(s.above_200ema),
                "above_vwap": bool(s.above_vwap),
                "weekly_trend_up": bool(s.weekly_trend_up),
                "last_price": float(s.last_price),
                "last_update": s.last_update,
                "confirmed": bool(s.confirmed),
                "error": s.error,
            })
        _state["watchers"] = watcher_states
        _state["thread_count"] = len(_watchers)

        _state["last_update"] = datetime.now().strftime("%H:%M:%S")

    except Exception as e:
        _add_log(f"Error refreshing: {e}", "error")


def background_refresh():
    while True:
        try:
            refresh_data()
        except Exception as e:
            _add_log(f"Background error: {e}", "error")
        time.sleep(30)


# ── Routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():
    safe = json.loads(json.dumps(_state, cls=NumpyEncoder))
    return jsonify(safe)


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    refresh_data()
    return jsonify({"ok": True})


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_connections()
    start_watchers()
    refresh_data()

    t = threading.Thread(target=background_refresh, daemon=True)
    t.start()

    log.info("Dashboard running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
