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
    """Start a watcher thread per stock + crypto for the dashboard."""
    global _watchers
    universe = list(_config["screener"]["universe"]) + _config["screener"].get("crypto", [])

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


@app.route("/chart/<symbol>")
def chart_page(symbol):
    # Only show chart for symbols in our universe or with active positions
    valid_symbols = set(_config["screener"]["universe"])
    valid_symbols.update(_config["screener"].get("crypto", []))
    valid_symbols.update(w.symbol for w in _watchers.values())
    # Also allow position symbols
    try:
        for pos in _broker.get_positions():
            valid_symbols.add(pos.symbol)
    except Exception:
        pass
    if symbol not in valid_symbols:
        return render_template("dashboard.html"), 404
    return render_template("chart.html", symbol=symbol)


@app.route("/api/chart/<symbol>")
def api_chart(symbol):
    """Return OHLCV + indicator data for TradingView lightweight-charts."""
    from indicators import supertrend, stochastic_rsi
    from trend import get_trend_context

    days = int(os.environ.get("CHART_DAYS", 120))

    # Fetch daily bars
    daily = _data.get_bars([symbol], timeframe="1Day", days=days)
    if symbol not in daily or daily[symbol].empty:
        return jsonify({"error": "No data"}), 404
    df = daily[symbol]

    # Fetch 5-min bars
    intraday = _data.get_intraday_bars(symbol, timeframe="5Min", days=5)

    # Build OHLCV for daily chart
    candles = []
    for ts, row in df.iterrows():
        t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(pd.Timestamp(ts).timestamp())
        candles.append({
            "time": t,
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
        })
    volumes = []
    for ts, row in df.iterrows():
        t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(pd.Timestamp(ts).timestamp())
        color = "rgba(38,166,154,0.5)" if row["close"] >= row["open"] else "rgba(239,83,80,0.5)"
        volumes.append({"time": t, "value": int(row["volume"]), "color": color})

    # Build 5-min candles
    intraday_candles = []
    if intraday is not None and not intraday.empty:
        for ts, row in intraday.iterrows():
            t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(pd.Timestamp(ts).timestamp())
            intraday_candles.append({
                "time": t,
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
            })

    # SuperTrend overlay
    st_line = []
    if len(df) >= 20:
        st_values, st_direction = supertrend(df, period=10, multiplier=3.0)
        for i, ts in enumerate(df.index):
            t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(pd.Timestamp(ts).timestamp())
            val = float(st_values.iloc[i])
            dirn = int(st_direction.iloc[i])
            if val > 0:
                st_line.append({
                    "time": t,
                    "value": round(val, 2),
                    "color": "#22c55e" if dirn == 1 else "#ef4444",
                })

    # EMAs (20, 50, 200)
    ema_data = {}
    for period in [20, 50, 200]:
        if len(df) >= period:
            ema = df["close"].ewm(span=period, adjust=False).mean()
            series = []
            for ts, val in ema.items():
                t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(pd.Timestamp(ts).timestamp())
                series.append({"time": t, "value": round(float(val), 2)})
            ema_data[f"ema{period}"] = series

    # VWAP
    vwap_line = []
    if "vwap" in df.columns:
        for ts, row in df.iterrows():
            t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(pd.Timestamp(ts).timestamp())
            vwap_val = float(row["vwap"])
            if vwap_val > 0:
                vwap_line.append({"time": t, "value": round(vwap_val, 2)})

    # StochRSI (for lower pane)
    stoch_k = []
    stoch_d = []
    if len(df) >= 30:
        k, d = stochastic_rsi(df["close"])
        for i, ts in enumerate(df.index):
            t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(pd.Timestamp(ts).timestamp())
            if i < len(k) and not np.isnan(k.iloc[i]):
                stoch_k.append({"time": t, "value": round(float(k.iloc[i]), 2)})
            if i < len(d) and not np.isnan(d.iloc[i]):
                stoch_d.append({"time": t, "value": round(float(d.iloc[i]), 2)})

    # RSI
    rsi_line = []
    if len(df) >= 15:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        for ts, val in rsi.items():
            t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(pd.Timestamp(ts).timestamp())
            if not np.isnan(val):
                rsi_line.append({"time": t, "value": round(float(val), 2)})

    # Signal markers from watcher
    markers = []
    if symbol in _watchers:
        w = _watchers[symbol]
        s = w.state
        if s.confirmed and s.last_price > 0:
            last_ts = candles[-1]["time"] if candles else 0
            if s.action.value == "buy":
                markers.append({
                    "time": last_ts, "position": "belowBar",
                    "color": "#22c55e", "shape": "arrowUp",
                    "text": f"BUY {s.score:.2f} ({s.num_agreeing}/5)"
                })
            elif s.action.value == "short":
                markers.append({
                    "time": last_ts, "position": "aboveBar",
                    "color": "#ef4444", "shape": "arrowDown",
                    "text": f"SHORT {s.score:.2f} ({s.num_agreeing}/5)"
                })

    # Trend context
    trend_ctx = {}
    try:
        trend_ctx = get_trend_context(df)
        trend_ctx = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                     for k, v in trend_ctx.items()}
    except Exception:
        pass

    # Watcher state
    watcher_info = {}
    if symbol in _watchers:
        s = _watchers[symbol].state
        watcher_info = {
            "status": s.status,
            "score": float(s.score),
            "num_agreeing": s.num_agreeing,
            "confirmed": bool(s.confirmed),
            "action": s.action.value if s.action else "hold",
            "regime": s.regime,
            "candle_patterns": list(s.candle_patterns),
            "strategy_scores": {k: float(v) for k, v in s.strategy_scores.items()},
            "strategy_weights": {k: float(v) for k, v in s.strategy_weights.items()},
            "adx": float(s.adx),
            "trend": s.trend_direction,
            "above_200ema": bool(s.above_200ema),
            "above_vwap": bool(s.above_vwap),
        }

    return jsonify({
        "symbol": symbol,
        "candles": candles,
        "volumes": volumes,
        "intraday_candles": intraday_candles,
        "supertrend": st_line,
        "emas": ema_data,
        "vwap": vwap_line,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "rsi": rsi_line,
        "markers": markers,
        "trend": trend_ctx,
        "watcher": watcher_info,
    })


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_connections()
    start_watchers()
    refresh_data()

    t = threading.Thread(target=background_refresh, daemon=True)
    t.start()

    log.info("Dashboard running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
