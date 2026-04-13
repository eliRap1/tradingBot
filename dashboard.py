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
from trend import get_trend_context, get_weekly_trend
from data import CRYPTO_SYMBOLS
from strategies import ALL_STRATEGIES
from indicators import keltner_squeeze, daily_pivot_points

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

        # Positions (with position_meta for R-multiple)
        positions = []
        state_data = load_state()
        pos_meta = state_data.get("position_meta", {})
        for pos in _broker.get_positions():
            meta = pos_meta.get(pos.symbol, {})
            initial_risk = meta.get("initial_risk", 0)
            entry_price = float(pos.avg_price)
            qty_val = float(pos.qty)
            market_value = float(pos.market_value)
            pnl = float(pos.unrealized_pl)
            current_price = (market_value / qty_val) if qty_val else entry_price
            pnl_pct = (pnl / (entry_price * abs(qty_val)) * 100) if entry_price and qty_val else 0.0
            r_multiple = None
            if initial_risk and initial_risk > 0:
                r_multiple = round(pnl / initial_risk, 2)
            positions.append({
                "symbol": pos.symbol,
                "qty": qty_val,
                "entry_price": entry_price,
                "current_price": current_price,
                "market_value": market_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "side": pos.side,
                "r_multiple": r_multiple,
                "opened_at": meta.get("opened_at"),
                "breakeven_armed": meta.get("breakeven_armed", False),
                "strategies": meta.get("strategies", []),
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
            hourly_bias = getattr(w, "_hourly_bias", {}).get("bias", "neutral")
            alpha_decay = getattr(w, "_alpha_decay", {})
            decaying = {k: round(v, 2) for k, v in alpha_decay.items() if v < 0.8}

            squeeze_info = None
            pivot_info = None
            try:
                bars = getattr(w, "_bars", None)
                if bars is not None and len(bars) >= 30:
                    squeeze_info = keltner_squeeze(bars)
                    pivot_info = daily_pivot_points(bars)
            except Exception:
                pass

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
                "prev_signal": bool(s.prev_signal),
                "error": s.error,
                "hourly_bias": hourly_bias,
                "alpha_decay": alpha_decay,
                "decaying_strategies": decaying,
                "keltner_squeeze": squeeze_info,
                "pivot_levels": pivot_info,
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


# ── Analysis Engine ──────────────────────────────────────────────

def _build_analysis(symbol: str, daily_df, intraday_df, trend_ctx: dict) -> dict:
    """Run the bot's full analysis pipeline and return step-by-step reasoning."""
    steps = []
    decision = {"action": "HOLD", "reason": "No signal", "score": 0.0, "confluence": 0}

    try:
        # Step 1: Trend Context (daily)
        ctx = trend_ctx or {}
        direction = ctx.get("trend_direction", "unknown")
        adx = ctx.get("adx", 0)
        trending = ctx.get("trending", False)
        strong_trend = ctx.get("strong_trend", False)
        above_ema200 = ctx.get("above_ema_200", False)
        above_vwap = ctx.get("above_vwap", False)

        trend_bias = "BULLISH" if direction == "up" and above_ema200 else \
                     "BEARISH" if direction == "down" and not above_ema200 else "NEUTRAL"

        steps.append({
            "step": 1,
            "name": "Daily Trend Context",
            "status": "bullish" if trend_bias == "BULLISH" else "bearish" if trend_bias == "BEARISH" else "neutral",
            "details": [
                {"label": "Trend Direction", "value": direction, "signal": direction == "up"},
                {"label": "ADX", "value": f"{adx:.1f}", "signal": adx > 25},
                {"label": "Trending", "value": str(trending), "signal": trending},
                {"label": "Strong Trend", "value": str(strong_trend), "signal": strong_trend},
                {"label": "Above 200 EMA", "value": str(above_ema200), "signal": above_ema200},
                {"label": "Above VWAP", "value": str(above_vwap), "signal": above_vwap},
            ],
            "summary": f"Trend is {direction.upper()} (ADX={adx:.0f}) -- Overall bias: {trend_bias}",
        })

        # Step 2: Weekly Trend
        try:
            wk = get_weekly_trend(daily_df)
            weekly_up = wk.get("weekly_trend_up", False)
            steps.append({
                "step": 2,
                "name": "Weekly Trend Confirmation",
                "status": "bullish" if weekly_up else "bearish",
                "details": [
                    {"label": "Weekly EMA Trend Up", "value": str(weekly_up), "signal": weekly_up},
                ],
                "summary": f"Weekly trend {'confirms upside' if weekly_up else 'warns downside'}",
            })
        except Exception:
            steps.append({"step": 2, "name": "Weekly Trend", "status": "neutral",
                          "details": [], "summary": "Could not compute weekly trend"})

        # Step 3: Candle Patterns (5-min)
        entry_df = intraday_df if intraday_df is not None and len(intraday_df) >= 10 else daily_df
        patterns = detect_patterns(entry_df)
        bullish_pats = [k for k, v in patterns.items() if v and k in
            ("hammer", "bullish_engulfing", "morning_star", "three_white_soldiers",
             "dragonfly_doji", "piercing_line", "tweezer_bottom")]
        bearish_pats = [k for k, v in patterns.items() if v and k in
            ("shooting_star", "bearish_engulfing", "evening_star", "three_black_crows",
             "gravestone_doji", "dark_cloud_cover", "tweezer_top")]
        bull_sc = bullish_score(patterns)
        bear_sc = bearish_score(patterns)

        pat_status = "bullish" if bull_sc > bear_sc and bull_sc > 0 else \
                     "bearish" if bear_sc > bull_sc and bear_sc > 0 else "neutral"
        steps.append({
            "step": 3,
            "name": "Candlestick Patterns (Entry TF)",
            "status": pat_status,
            "details": [
                {"label": "Bullish Patterns", "value": ", ".join(bullish_pats) if bullish_pats else "None",
                 "signal": len(bullish_pats) > 0},
                {"label": "Bearish Patterns", "value": ", ".join(bearish_pats) if bearish_pats else "None",
                 "signal": False},
                {"label": "Bullish Score", "value": str(bull_sc), "signal": bull_sc > 0},
                {"label": "Bearish Score", "value": str(bear_sc), "signal": False},
            ],
            "summary": f"Candles: {len(bullish_pats)} bullish, {len(bearish_pats)} bearish patterns",
        })

        # Step 4: Regime Classification & Strategy Selection
        selection = select_strategies(daily_df, symbol)
        regime = selection["regime"]
        reason = selection["reason"]
        strat_weights = selection["strategies"]

        active_strats = {k: v for k, v in strat_weights.items() if v > 0}
        inactive_strats = {k: v for k, v in strat_weights.items() if v <= 0}

        strat_details = []
        for name, weight in sorted(strat_weights.items(), key=lambda x: -x[1]):
            strat_details.append({
                "label": name.replace("_", " ").title(),
                "value": f"Weight: {weight*100:.0f}%",
                "signal": weight > 0,
            })

        steps.append({
            "step": 4,
            "name": "Regime & Strategy Selection",
            "status": "bullish" if regime in ("trending", "breakout") else "neutral",
            "details": strat_details,
            "summary": f"Regime: {regime.upper()} -- {reason}. "
                       f"Active: {list(active_strats.keys())}"
                       + (f", OFF: {list(inactive_strats.keys())}" if inactive_strats else ""),
        })

        # Step 5: Run each strategy and show scores
        strategies = {name: cls(_config) for name, cls in ALL_STRATEGIES.items()}
        strat_scores = {}
        num_bullish = 0
        num_bearish = 0
        weighted_sum = 0.0
        total_weight = 0.0
        strat_step_details = []

        for name, weight in strat_weights.items():
            strat = strategies.get(name)
            if not strat or weight <= 0:
                strat_step_details.append({
                    "label": name.replace("_", " ").title(),
                    "value": "OFF (weight=0)",
                    "signal": None,
                })
                continue

            try:
                signals = strat.generate_signals({symbol: entry_df})
                score = signals.get(symbol, 0.0)
            except Exception as e:
                score = 0.0

            strat_scores[name] = round(float(score), 4)
            weighted_sum += score * weight
            total_weight += weight

            is_bull = score > 0.1
            is_bear = score < -0.1
            if is_bull:
                num_bullish += 1
            elif is_bear:
                num_bearish += 1

            verdict = "BULLISH" if is_bull else "BEARISH" if is_bear else "NEUTRAL"
            strat_step_details.append({
                "label": f"{name.replace('_', ' ').title()} (wt={weight*100:.0f}%)",
                "value": f"Score: {score:+.4f} -> {verdict}",
                "signal": is_bull,
            })

        composite = weighted_sum / total_weight if total_weight > 0 else 0.0
        total_strats = len(strat_weights)
        steps.append({
            "step": 5,
            "name": "Strategy Signals (Entry TF)",
            "status": "bullish" if num_bullish >= 2 else "bearish" if num_bearish >= 2 else "neutral",
            "details": strat_step_details,
            "summary": f"Composite: {composite:+.4f} | "
                       f"Bullish: {num_bullish}/{total_strats}, Bearish: {num_bearish}/{total_strats}",
        })

        # Step 6: Confluence Check
        is_crypto = symbol in CRYPTO_SYMBOLS
        min_agreeing = _config["signals"].get("min_crypto_agreeing", 2) if is_crypto \
            else _config["signals"].get("min_agreeing_strategies", 3)
        min_score = _config["signals"].get("min_crypto_score", 0.15) if is_crypto \
            else _config["signals"]["min_composite_score"]

        has_long = num_bullish >= min_agreeing and composite >= min_score
        has_short = num_bearish >= min_agreeing and composite <= -min_score

        confluence_details = [
            {"label": "Bullish Strategies", "value": f"{num_bullish}/{total_strats} (need {min_agreeing})",
             "signal": num_bullish >= min_agreeing},
            {"label": "Bearish Strategies", "value": f"{num_bearish}/{total_strats} (need {min_agreeing})",
             "signal": num_bearish >= min_agreeing},
            {"label": "Composite Score", "value": f"{composite:+.4f} (need {'+' if composite >= 0 else ''}{min_score})",
             "signal": abs(composite) >= min_score},
            {"label": "Long Signal", "value": "YES" if has_long else "NO", "signal": has_long},
            {"label": "Short Signal", "value": "YES" if has_short else "NO", "signal": has_short},
        ]

        # Check confirmation from watcher
        confirmed = False
        pending = False
        if symbol in _watchers:
            ws = _watchers[symbol].state
            confirmed = bool(ws.confirmed)
            pending = bool(ws.prev_signal) and not confirmed
            confluence_details.append(
                {"label": "Confirmation (2 cycles)", "value":
                 "CONFIRMED" if confirmed else "PENDING (1/2)" if pending else "NOT YET",
                 "signal": confirmed})

        if has_long:
            signal_str = "LONG SIGNAL"
        elif has_short:
            signal_str = "SHORT SIGNAL"
        else:
            blockers = []
            if num_bullish < min_agreeing and num_bearish < min_agreeing:
                blockers.append(f"not enough strategies agree (best: {max(num_bullish, num_bearish)}/{min_agreeing})")
            if abs(composite) < min_score:
                blockers.append(f"composite score too weak ({composite:+.3f}, need {min_score})")
            signal_str = "NO SIGNAL -- " + "; ".join(blockers) if blockers else "NO SIGNAL"

        steps.append({
            "step": 6,
            "name": "Confluence Filter",
            "status": "bullish" if has_long else "bearish" if has_short else "blocked",
            "details": confluence_details,
            "summary": signal_str + (" -> CONFIRMED" if confirmed else " -> PENDING" if pending and (has_long or has_short) else ""),
        })

        # Step 7: Final Decision
        if has_long and confirmed:
            decision = {"action": "BUY", "reason": f"{num_bullish}/{total_strats} agree, score={composite:+.3f}, confirmed",
                        "score": round(composite, 4), "confluence": num_bullish}
        elif has_short and confirmed:
            decision = {"action": "SHORT", "reason": f"{num_bearish}/{total_strats} agree, score={composite:+.3f}, confirmed",
                        "score": round(composite, 4), "confluence": num_bearish}
        elif has_long or has_short:
            direction_word = "LONG" if has_long else "SHORT"
            decision = {"action": "WAIT", "reason": f"{direction_word} signal detected, waiting for confirmation",
                        "score": round(composite, 4), "confluence": max(num_bullish, num_bearish)}
        else:
            decision = {"action": "HOLD", "reason": signal_str,
                        "score": round(composite, 4), "confluence": max(num_bullish, num_bearish)}

        # Add regime gate info
        regime_data = _state.get("regime", {})
        if regime_data:
            allow_longs = regime_data.get("allow_longs", True)
            mkt_regime = regime_data.get("regime", "unknown")
            steps.append({
                "step": 7,
                "name": "Market Regime Gate (SPY)",
                "status": "bullish" if allow_longs else "bearish",
                "details": [
                    {"label": "SPY Regime", "value": mkt_regime.upper(), "signal": mkt_regime == "bull"},
                    {"label": "Longs Allowed", "value": str(allow_longs),
                     "signal": allow_longs},
                    {"label": "Crypto Bypass", "value": str(is_crypto),
                     "signal": is_crypto},
                ],
                "summary": f"SPY regime: {mkt_regime.upper()} -- "
                    + (f"Crypto bypasses SPY gate" if is_crypto else
                       f"Longs {'ALLOWED' if allow_longs else 'BLOCKED'}")
            })

    except Exception as e:
        steps.append({"step": 0, "name": "Error", "status": "error",
                      "details": [{"label": "Error", "value": str(e), "signal": False}],
                      "summary": f"Analysis error: {e}"})

    return {"steps": steps, "decision": decision}


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


@app.route("/chart/<path:symbol>")
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


@app.route("/api/chart/<path:symbol>")
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

    # Deduplicate daily data
    df = df[~df.index.duplicated(keep='last')].sort_index()

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

    # Build 5-min candles + volume + indicators
    intraday_candles = []
    intraday_volumes = []
    intraday_emas = {}
    intraday_st = []
    if intraday is not None and not intraday.empty:
        # Deduplicate and sort
        idf = intraday[~intraday.index.duplicated(keep='last')].sort_index()

        for ts, row in idf.iterrows():
            t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(pd.Timestamp(ts).timestamp())
            intraday_candles.append({
                "time": t,
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
            })
            color = "rgba(38,166,154,0.5)" if row["close"] >= row["open"] else "rgba(239,83,80,0.5)"
            intraday_volumes.append({"time": t, "value": int(row["volume"]), "color": color})

        # 5-min EMAs
        for period in [9, 21, 50]:
            if len(idf) >= period:
                ema = idf["close"].ewm(span=period, adjust=False).mean()
                series = []
                for ts, val in ema.items():
                    t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(pd.Timestamp(ts).timestamp())
                    series.append({"time": t, "value": round(float(val), 2)})
                intraday_emas[f"ema{period}"] = series

        # 5-min SuperTrend
        if len(idf) >= 20:
            try:
                st_vals, st_dir = supertrend(idf, period=10, multiplier=3.0)
                for i, ts in enumerate(idf.index):
                    t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(pd.Timestamp(ts).timestamp())
                    val = float(st_vals.iloc[i])
                    dirn = int(st_dir.iloc[i])
                    if val > 0:
                        intraday_st.append({
                            "time": t, "value": round(val, 2),
                            "color": "#22c55e" if dirn == 1 else "#ef4444",
                        })
            except Exception:
                pass

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

    # Signal markers from watcher - CURRENT signals only
    markers = []
    current_time = int(datetime.now().timestamp())
    
    if symbol in _watchers:
        w = _watchers[symbol]
        s = w.state
        if s.confirmed and s.last_price > 0:
            # Use current time for LIVE signal marker (not historical bar time)
            if s.action.value == "buy":
                markers.append({
                    "time": current_time, "position": "belowBar",
                    "color": "#00ff00", "shape": "arrowUp",
                    "text": f">>> LIVE BUY {s.score:.2f} ({s.num_agreeing}/{len(ALL_STRATEGIES)}) <<<"
                })
            elif s.action.value == "short":
                markers.append({
                    "time": current_time, "position": "aboveBar",
                    "color": "#ff0000", "shape": "arrowDown",
                    "text": f">>> LIVE SHORT {s.score:.2f} ({s.num_agreeing}/{len(ALL_STRATEGIES)}) <<<"
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

    # Trade history markers (actual buys/sells from tracker) - Last 7 days only
    seven_days_ago = current_time - (7 * 24 * 60 * 60)
    
    for trade in _tracker.trades:
        if trade.get("symbol") != symbol:
            continue
        # Entry marker
        entry_time = trade.get("opened_at")
        if entry_time:
            try:
                if isinstance(entry_time, str):
                    entry_ts = int(pd.Timestamp(entry_time).timestamp())
                else:
                    entry_ts = int(entry_time.timestamp())
                
                # Only show recent trades (last 7 days)
                if entry_ts < seven_days_ago:
                    continue
                    
                side = trade.get("side", "buy")
                is_buy = side == "buy"
                markers.append({
                    "time": entry_ts,
                    "position": "belowBar" if is_buy else "aboveBar",
                    "color": "#22c55e" if is_buy else "#ef4444",
                    "shape": "arrowUp" if is_buy else "arrowDown",
                    "text": f"{'BUY' if is_buy else 'SHORT'} @ ${trade.get('entry_price', 0):.2f}",
                })
            except Exception:
                pass
        # Exit marker
        exit_time = trade.get("closed_at")
        if exit_time:
            try:
                if isinstance(exit_time, str):
                    exit_ts = int(pd.Timestamp(exit_time).timestamp())
                else:
                    exit_ts = int(exit_time.timestamp())
                
                # Only show recent exits
                if exit_ts < seven_days_ago:
                    continue
                    
                pnl = trade.get("pnl", 0)
                markers.append({
                    "time": exit_ts,
                    "position": "aboveBar" if pnl >= 0 else "belowBar",
                    "color": "#22c55e" if pnl >= 0 else "#ef4444",
                    "shape": "circle",
                    "text": f"EXIT ${pnl:+,.2f} ({trade.get('reason', '')})",
                })
            except Exception:
                pass

    # Sort markers by time (required by lightweight-charts)
    markers.sort(key=lambda m: m["time"])

    # ── Detailed bot analysis (step-by-step reasoning) ──
    analysis = _build_analysis(symbol, df, intraday, trend_ctx)

    # Use NumpyEncoder to handle numpy types (bool_, int64, float64, etc.)
    
    # Add signal timing info for clarity
    last_data_time = candles[-1]["time"] if candles else 0
    last_intraday_time = intraday_candles[-1]["time"] if intraday_candles else 0
    
    signal_info = {
        "current_time": current_time,
        "current_time_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_daily_bar": datetime.fromtimestamp(last_data_time).strftime("%Y-%m-%d %H:%M") if last_data_time else "N/A",
        "last_5min_bar": datetime.fromtimestamp(last_intraday_time).strftime("%Y-%m-%d %H:%M") if last_intraday_time else "N/A",
        "has_live_signal": len([m for m in markers if "LIVE" in m.get("text", "")]) > 0,
    }
    
    result = {
        "symbol": symbol,
        "candles": candles,
        "volumes": volumes,
        "intraday_candles": intraday_candles,
        "intraday_volumes": intraday_volumes,
        "intraday_emas": intraday_emas,
        "intraday_st": intraday_st,
        "supertrend": st_line,
        "emas": ema_data,
        "vwap": vwap_line,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "rsi": rsi_line,
        "markers": markers,
        "trend": trend_ctx,
        "watcher": watcher_info,
        "analysis": analysis,
        "signal_timing": signal_info,
    }
    safe = json.loads(json.dumps(result, cls=NumpyEncoder))
    return jsonify(safe)


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_connections()
    start_watchers()
    refresh_data()

    t = threading.Thread(target=background_refresh, daemon=True)
    t.start()

    log.info("Dashboard running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
