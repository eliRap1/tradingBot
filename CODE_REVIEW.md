# Trading Bot Code Review & Bug Fixes

## Summary

Your trading bot is **well-architected** with solid foundations:
- Multi-strategy confluence system
- Per-symbol watcher threads
- Regime-based strategy weighting
- Proper R-multiple tracking

However, I've identified and **fixed several critical issues** that could cause real trading losses.

---

## Issues Fixed

### CRITICAL (Could cause real money loss)

| Issue | Location | Fix Applied |
|-------|----------|-------------|
| **Crypto TP/SL not OCO-linked** | `broker.py:85-136` | Added `_crypto_exit_orders` tracking with `check_crypto_exit_fills()` to cancel the other order when one fills |
| **No partial fill handling** | `broker.py` | Now waits for entry fill confirmation and uses actual `filled_qty` for exit orders |
| **No idempotency keys** | `broker.py` | Added `client_order_id` to prevent duplicate orders on network retries |
| **Crypto OCO not checked** | `coordinator.py` | Added `broker.check_crypto_exit_fills()` at start of each cycle |

### IMPORTANT (Could degrade performance)

| Issue | Location | Fix Applied |
|-------|----------|-------------|
| **No API rate limiting** | `data.py` | Added token bucket rate limiter (150 req/min) with exponential backoff |
| **Kelly sizing edge case** | `risk.py:112-134` | Capped win/loss ratio at 5:1 to prevent extreme position sizing |
| **Watcher memory leak** | `watcher.py` | Added `_bars_cache_time` with 5-min expiry to clear old data |

---

## Issues NOT Fixed (Require Your Decision)

### 1. Smart Order Blocking (`broker.py:281-380`)
The `submit_smart_order` function blocks for up to 30 seconds polling for fills. This delays other trades.

**Options:**
- A) Move to async/threading (complex)
- B) Reduce timeout to 10 seconds (trade-off: more market orders)
- C) Accept current behavior (simple)

### 2. Trailing Stop Only On Cycle (`portfolio.py`)
Trailing stops are checked every 5 minutes during coordinator cycles. Rapid moves between cycles won't adjust the trail.

**Options:**
- A) Use Alpaca's native trailing stop orders (they handle this server-side)
- B) Decrease cycle time (more API calls)
- C) Accept current behavior (brackets catch most moves)

### 3. Correlation Cache Staleness (`filters.py`)
30-minute cache for correlation data could allow highly correlated positions during volatile periods.

**Options:**
- A) Reduce cache to 10 minutes
- B) Force recalculate before new positions
- C) Accept (correlation is for filtering, not critical)

---

## Strategy Evaluation

### Strengths
- Multi-strategy confluence reduces false signals
- Regime-based weighting adapts to market conditions
- R-multiple tracking for proper risk measurement
- Confirmation bar filter reduces whipsaws
- Weekly trend alignment adds context

### Weaknesses & Recommendations

| Weakness | Recommendation |
|----------|----------------|
| **Overfitting risk**: Many hardcoded parameters | Implement walk-forward optimization |
| **Same params for all assets** | Consider per-sector parameter sets |
| **Mean reversion in trends** | Add trend strength check before MR signals |
| **No slippage model variation** | Add stochastic slippage based on ATR |

---

## Backtesting Quality Check

| Bias | Status | Notes |
|------|--------|-------|
| Lookahead bias | **Check weekly trend** | `dropna()` should exclude incomplete weeks |
| Survivorship bias | **OK** | Using current universe is fine for short backtests |
| Data leakage | **OK** | Separate daily/intraday timeframes |
| Overfitting | **Risk** | Many fixed parameters - use walk-forward validation |

---

## Testing Recommendations

### Backtest Validation
```python
# 1. Walk-forward optimization (6mo train / 2mo test, rolling)
# 2. Monte Carlo simulation of trade sequence
# 3. Stress test: 2020 March, 2022 bear market
# 4. Compare in-sample vs out-of-sample Sharpe ratios
```

### Paper Trading Checklist
- [ ] Run paper trading for minimum 30 trades
- [ ] Verify fills match expected prices (slippage)
- [ ] Check bracket orders actually fire on TP/SL
- [ ] Monitor crypto OCO cancellation works
- [ ] Validate position sizing calculations

### Live Trading Checklist
- [ ] Start with 1/4 position size
- [ ] Set max daily loss limit in Alpaca
- [ ] Enable SMS/email alerts
- [ ] Have manual override plan ready

---

## Files Modified

1. **`broker.py`** - Crypto OCO tracking, idempotency keys, partial fill handling
2. **`coordinator.py`** - Added crypto exit fill check to cycle
3. **`data.py`** - Rate limiting with exponential backoff
4. **`risk.py`** - Kelly criterion edge case fix
5. **`watcher.py`** - Memory leak prevention

---

## Running the Bot

### Paper Trading
```bash
# Set environment variables
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
export TRADING_MODE=paper

# Run bot
python main.py

# Or run dashboard only (for monitoring without trading)
python dashboard.py
```

### Dashboard
Open http://localhost:5000 to see:
- Live watcher threads and their signals
- Account equity and P&L
- Open positions with trailing stop status
- Per-symbol charts with indicator overlays
- Bot's reasoning for each signal (step-by-step analysis)

---

## Questions?

If you want me to:
1. Implement any of the "not fixed" options
2. Add more sophisticated slippage modeling
3. Build a backtesting report generator
4. Add parameter optimization framework

Just let me know!
