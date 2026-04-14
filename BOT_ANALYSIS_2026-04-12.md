# Trading Bot Analysis — April 12, 2026

## Current Performance (Live/Paper)

- **9 closed trades** | Win rate: **100%** | Total PnL: **+$650.25**
- Average win: $72.25 | No losses recorded yet
- Peak equity: $101,209 (started ~$100,000)
- 2 open positions: **BTCUSD** (opened Apr 5) and **SOLUSD** (opened Apr 11)
- All exits via trailing stop or time stop — bracket TP orders have not triggered once

---

## 🔴 Critical Bugs

### 1. `initial_risk` is Always 0.0 — Partial Exits Never Fire

This is the most impactful bug in the whole bot. In `state.json`, both open positions show:
```json
"initial_risk": 0.0
```
In `portfolio.py`, the partial exit logic checks:
```python
if initial_risk > 0 and qty > 1:
```
Since `initial_risk` is never set from actual risk data (the `set_position_risk` call in coordinator updates it, but `position_meta` is initialized with `initial_risk: 0.0` and may get overwritten on the next `get_current_positions()` call), **the two-tier partial exit system (1.2R and 2.5R) has never triggered**. The R-multiple can't be calculated either, so the ML model has no useful training signal.

BTCUSD has been open 7 days with `check_count: 773` — it was never partially exited despite being profitable.

**Fix** — in `portfolio.py`, `get_current_positions()` should NOT overwrite `initial_risk` if it's already set:
```python
if pos.symbol not in self.position_meta:
    self.position_meta[pos.symbol] = {
        "opened_at": datetime.now().isoformat(),
        "entry_price": float(pos.avg_entry_price),
        "initial_risk": 0.0,
    }
# DON'T overwrite an existing meta entry
```
The coordinator already calls `set_position_risk`, but `get_current_positions` runs right after and can create a fresh meta entry that wipes initial_risk if there was a restart.

### 2. Crypto OCO Leg Failure — Qty Mismatch

From Apr 11 log:
```
Failed to place crypto TP order: insufficient balance for SOL (requested: 66.641844988, available: 66.475240375)
Order failed for ETH/USD: insufficient balance for ETH (requested: 2.520993507, available: 0)
```

The entry order fills, but the TP/SL orders use `filled_qty` which can exceed the actual available balance due to fees or rounding on Alpaca's side. The fix:

```python
# In broker.py submit_crypto_order, before placing exit orders:
filled_qty = float(order_status.filled_qty) * 0.999  # 99.9% to avoid balance edge case
```

For the ETH case — the ETH/USD short went through despite the coordinator's check (`if is_crypto: continue` for shorts). This means either the watcher state was already `confirmed=True` from a prior cycle, or the crypto-short block has a timing gap. Add a second check in the broker layer:

```python
# In submit_crypto_order:
if side == "sell":  # crypto long-only on Alpaca
    log.error(f"Crypto short rejected: {symbol} — Alpaca does not support crypto shorts")
    return None
```

### 3. Strategy Attribution Missing in All Trades

Every trade record has `"strategies": []` — the strategy list is empty for all 9 trades. This means:
- The ML model has no training data (it needs `strategies` to learn)
- Alpha decay can't be calculated
- Performance attribution is impossible

In `coordinator.py`, `contributing` is computed from `watcher.state.strategy_scores`, but by the time the trade is recorded (at exit), the watcher may have already reset or the coordinator stored the strategies under a different key format. Verify that `position_meta[symbol]["strategies"]` is being saved and read back correctly at trade recording time.

### 4. AMZN Duplicate in Universe

`config.yaml` has AMZN listed twice (mega-cap tech AND consumer/retail). This spawns two watcher threads for the same symbol, wastes API quota, and could double-signal.

**Fix** — add a dedup line in `coordinator.py`:
```python
universe = list(dict.fromkeys(universe))  # preserve order, remove dupes
```

---

## 🟡 Significant Issues

### 5. Crypto Bars Going Stale Repeatedly

SOL/USD bars were stale 11–14 minutes old during three consecutive evening cycles:
```
Skipping SOL/USD: bars_stale_14min_old
Skipping SOL/USD: bars_stale_11min_old  
Skipping SOL/USD: bars_stale_13min_old
```

Crypto trades 24/7 and Alpaca's REST polling is fine for stocks but misses crypto price moves during off-peak hours. The 10-minute freshness threshold for 5Min bars is too tight when the coordinator's cache priming (`prime_intraday_cache`) takes time for 176 symbols.

**Options:**
- Raise crypto staleness limit to 20 minutes for non-peak hours
- Switch to Alpaca's WebSocket streaming for crypto prices (see improvement section)

### 6. SOL Position Immediately Underwater

SOLUSD was bought at $85.05 (Apr 11) and the state shows `high_watermarks: SOLUSD: 86.06`, meaning it barely moved in the bot's favor before the `check_count` counter shows it's being held. If `initial_risk = 0`, the trailing stop is using the fallback `%` trail rather than ATR, which may be the wrong distance.

### 7. ML Model Cannot Train — Needs Strategy Data

`MLMetaModel` requires `min_trades=50` and trades with `strategies` populated. Currently: 9 trades, 0 with strategy data. The ML filter is therefore OFF for the entire early operation of the bot, providing no benefit. Reduce `min_trades` to 20, and fix the attribution bug (#3) so it can actually learn.

### 8. Large Universe = High Thread + API Pressure

176 watcher threads (170 stocks + 6 crypto), each running every 5 minutes. The `prime_intraday_cache` bulk fetches help, but each watcher still calls `get_intraday_bars` for 3 timeframes. At scale this creates API rate limit pressure and slows down the cycle. During the Apr 12 log, cycles are running every ~5 minutes cleanly, but as more watchers activate errors may spike.

---

## 🟢 What's Working Well

- **Regime detection** is solid — HMM + EMA layering + ADX + RSI + breadth is production-grade
- **Sector regime layer** is a meaningful edge — biasing strategy weights by sector
- **Hourly bias confirmation** (1H timeframe gate) is reducing false entries
- **Chandelier ATR trailing stop** is the right exit mechanic — all 9 wins came from it
- **Correlation filter** and **sector cap** properly preventing over-concentration
- **Market hours logic** (stock vs crypto separation) is working correctly
- **2-cycle signal confirmation** (prev_signal + confirmed) is a good noise filter
- **Order rate limiter** (20/hour) is a solid safety valve

---

## 🚀 Recommended Improvements

### Improvement 1 — WebSocket Streaming for Crypto Prices (High Impact)

Replace REST polling for crypto with Alpaca's WebSocket stream. This eliminates the staleness problem entirely and gives real-time fills/quotes.

```python
# In a new file: crypto_stream.py
from alpaca_trade_api.stream import Stream

class CryptoStreamer:
    def __init__(self, api_key, secret_key, symbols):
        self.stream = Stream(api_key, secret_key, crypto_exchanges=["CBSE"])
        self._prices = {}
        
    async def _on_crypto_quote(self, q):
        self._prices[q.symbol] = float(q.ask_price)
    
    def subscribe(self, symbols):
        for sym in symbols:
            self.stream.subscribe_crypto_quotes(self._on_crypto_quote, sym)
    
    def get_price(self, symbol):
        return self._prices.get(symbol)
```

This gives sub-second crypto price updates, replaces the `get_live_price()` polling for crypto, and eliminates all staleness skips.

### Improvement 2 — Dynamic Universe Shrinking

Instead of watching 170 stocks all the time, score the universe during pre-market and only activate watchers for the top 40–60 most active / most likely to signal. This cuts API load by 65% and focuses compute on the best setups.

```python
# In screener.py — add a pre-market ranking step
def get_active_universe(self, top_n=50):
    universe = self.config["screener"]["universe"]
    # Rank by: yesterday's volume spike + ATR percentile + recent momentum
    scored = []
    bars = self.data.get_bars(universe, timeframe="1Day", days=30)
    for sym, df in bars.items():
        if df is None or len(df) < 20: continue
        rvol_score = df["volume"].iloc[-1] / df["volume"].iloc[-20:].mean()
        atr_pct = (df["high"].iloc[-1] - df["low"].iloc[-1]) / df["close"].iloc[-1]
        scored.append((sym, rvol_score * atr_pct))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:top_n]]
```

### Improvement 3 — SQLite Instead of JSON State Files

`state.json`, `trades.json`, and `watcher_pending.json` are write-concurrent JSON files. Under high load (176 threads + coordinator writing), corruption is possible. Replace with SQLite:

```python
import sqlite3

class StateDB:
    def __init__(self, path="bot_state.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()
    
    def record_trade(self, trade: dict):
        with self._lock:
            self.conn.execute(
                "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?)",
                (trade["symbol"], trade["side"], trade["qty"], ...)
            )
            self.conn.commit()
```

This gives atomic writes, concurrent reads, and query capability for analytics.

### Improvement 4 — Entry Timeframe Upgrade (1-min confirmation)

Currently using 5-minute bars for entry signals. Adding a 1-minute "entry refinement" step after a 5-min signal confirms could improve fill prices significantly:

```python
# After watcher confirms signal on 5-min:
# Fetch 1-min bars and wait for a micro-structure entry:
# - Pullback to VWAP or 8-EMA on 1-min
# - Volume dry-up (low volume on pullback = accumulation)
# - Then enter on the next green candle
```

This is how professional day traders refine entries — the strategy fires on 5-min, but you enter on the 1-min dip. Reduces average entry cost.

### Improvement 5 — Portfolio Heat Metric

Add a real-time "portfolio heat" metric — the total unrealized loss across all positions as a % of equity. If heat exceeds 3%, don't open new positions until it cools:

```python
def get_portfolio_heat(self, positions: dict) -> float:
    """Total unrealized risk as % of equity."""
    total_risk = sum(
        abs(meta.get("initial_risk", 0))
        for meta in self.position_meta.values()
    )
    return total_risk / self.broker.get_equity()
```

This is a standard professional risk overlay that prevents "position piling" in losing environments.

### Improvement 6 — Regime-Specific Strategy Veto

Currently the strategy selector uses weights but never sets a weight to 0 for a strategy that's actively harmful in the current regime. For example, in a strong bear market, `gap` and `momentum` long signals are noise. Add hard vetoes:

```python
if regime == "bear" and adx > 30:
    strategies["momentum"] = 0.0
    strategies["gap"] = 0.0
    strategies["breakout"] = 0.0
    # Only allow mean_reversion (for shorts) and liquidity_sweep
```

### Improvement 7 — Adaptive Partial Exit Thresholds

The current 1.2R / 2.5R partials are fixed. In a high-volatility regime (ATR elevated), set them wider (2R / 4R) to avoid being shaken out. In low-vol, tighten them (1R / 2R) to lock profits faster:

```python
atr_regime = coordinator.regime.get("atr_regime", "normal")
if atr_regime == "high_vol":
    partial_r = 2.0
    second_partial_r = 4.0
elif atr_regime == "low_vol":
    partial_r = 1.0
    second_partial_r = 2.0
```

---

## Summary Priority Table

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| P0 | `initial_risk=0` — partials never fire | Critical | 30 min |
| P0 | Crypto OCO qty mismatch | Critical | 20 min |
| P0 | Strategy attribution empty | High | 1 hr |
| P1 | AMZN duplicate in universe | Medium | 5 min |
| P1 | Crypto short bypass fix | High | 15 min |
| P1 | ML model min_trades reduce (50→20) | Medium | 5 min |
| P2 | WebSocket crypto streaming | High | 3-4 hrs |
| P2 | Dynamic universe shrinking | High | 2 hrs |
| P2 | SQLite state storage | Medium | 4 hrs |
| P3 | Portfolio heat metric | Medium | 1 hr |
| P3 | Adaptive partial exit thresholds | Medium | 1 hr |
| P3 | Entry refinement on 1-min | Medium | 2 hrs |

---

*Generated: 2026-04-12 | Bot version: v2.0 | Paper trading mode*
