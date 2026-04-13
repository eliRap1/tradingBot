# Strategy Audit, Bot Health Review & Edge Layer Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit every strategy's real performance per sector/asset class, identify the top bot improvements, and add a proven edge layer (cross-asset signals, news/earnings avoidance, microstructure gate, ML meta-filter) that gives the bot an institutional-grade information advantage.

**Architecture:** Three parallel workstreams — (1) offline research harness that produces a scorecard and auto-updates strategy weights, (2) edge module wired into the coordinator as a layered pre-trade gate, (3) strategy router upgrade to per-sector × per-regime weight maps. No production code is touched during the research phase. The edge layer ships each component independently so it can be validated in isolation.

**Tech Stack:** Python 3.12, pandas, numpy, lightgbm (meta-model), vaderSentiment (news), ib_insync (IB quotes), alpaca-trade-api (historical data fallback), existing strategy classes, existing AlpacaDataFetcher/IBDataFetcher.

---

## Workstream 1 — Research Harness

### 1.1 Strategy Audit (`research/strategy_audit.py`)

**Purpose:** Run each strategy in isolation against 6 months of historical data, sector by sector. Measure signal quality, compute alpha over baseline, and generate per-sector × per-regime weight maps.

**Sector groups — survivorship bias fix: each sector includes 2 weaker names (marked †):**
```python
SECTOR_GROUPS = {
    "tech_mega":    ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "HPQ†", "INTC†"],
    "semis":        ["NVDA", "AMD", "AVGO", "MU", "QCOM", "ON†", "SWKS†"],
    "software":     ["CRM", "NOW", "PANW", "CRWD", "DDOG", "OKTA†", "ZM†"],
    "financials":   ["JPM", "GS", "MS", "V", "MA", "COF†", "AFRM†"],
    "healthcare":   ["UNH", "LLY", "ISRG", "VRTX", "AMGN", "BIIB†", "IDXX†"],
    "energy":       ["XOM", "CVX", "COP", "SLB", "EOG", "HAL†", "OXY†"],
    "industrials":  ["CAT", "HON", "GE", "RTX", "DE", "MMM†", "CSX†"],
    "consumer":     ["COST", "HD", "NKE", "MCD", "SBUX", "TGT†", "LYFT†"],
    "crypto":       ["BTC/USD", "ETH/USD", "SOL/USD"],
    "futures":      ["NQ", "ES", "CL", "GC"],
}
```

**4-state regime classification (replaces binary bull/bear):**
```python
def classify_regime(daily_bars) -> str:
    # Trend strength: ADX-14 on daily bars
    adx = compute_adx(daily_bars, period=14)
    trending = adx.iloc[-1] > 25

    # Direction: price vs 50-day EMA
    ema50 = daily_bars["close"].ewm(span=50).mean()
    bull = daily_bars["close"].iloc[-1] > ema50.iloc[-1]

    # Volatility regime: ATR percentile over 60 days
    atr = compute_atr(daily_bars, period=14)
    atr_pct = atr.iloc[-1] / atr.rolling(60).mean().iloc[-1]
    high_vol = atr_pct > 1.2

    if bull and trending:   return "bull_trending"
    if bull and not trending: return "bull_choppy"
    if not bull and trending: return "bear_trending"
    return "bear_choppy"
    # high_vol is stored separately as an additional feature
```

**Signal evaluation — what is recorded per trade:**
- symbol, sector, strategy, score, direction, outcome (HIT_TP / HIT_SL / TIME_STOP)
- R_multiple, entry_price, exit_price
- regime_4state (bull_trending / bull_choppy / bear_trending / bear_choppy)
- high_vol (bool — ATR percentile > 1.2)
- **hour_of_day** (0–23), **session_bucket** ("open" = 9:30–10:00, "mid" = 10:00–15:00, "close" = 15:00–16:00)
- stock_vs_spy_corr: rolling 20-day correlation of daily returns vs SPY
- days_to_earnings (from earnings calendar)
- futures_overnight_move: for NQ/ES only — `(today_open - yesterday_close) / yesterday_close`

**Baseline comparison — alpha calculation:**
```python
# For each symbol over the test period
baseline_buy_hold = (last_close / first_close) - 1

# Simple EMA baseline: enter on 20/50 EMA cross, exit on reverse
ema_baseline_return = simulate_ema_cross_strategy(bars, fast=20, slow=50)

# Strategy return (simulated)
strategy_return = sum(trade["R_multiple"] * risk_per_trade for trade in trades)

# Alpha = what we add above the dumb baseline
alpha_vs_hold   = strategy_return - baseline_buy_hold
alpha_vs_ema    = strategy_return - ema_baseline_return
```

Strategies with negative alpha vs buy & hold are flagged `⚠ NO_ALPHA` regardless of win rate.

**Upgraded quality score formula:**
```python
import math
# Penalizes thin sample sizes naturally; rewards consistency
quality_score = max(0, (win_rate * avg_R) * math.log(max(signals, 1)))
# If win_rate < 0.40 OR avg_R < 0.5 → quality_score = 0 (removed from sector)
```

**Scorecard output per strategy × sector:**
```
| Strategy       | Sector     | Signals | Win% | Avg R | Alpha | Best Session | Best Regime     | Flag      |
|----------------|------------|---------|------|-------|-------|--------------|-----------------|-----------|
| momentum       | tech_mega  | 87      | 61%  | 1.4   | +4.2% | close        | bull_trending   | ✅ STRONG |
| mean_reversion | financials | 34      | 58%  | 0.9   | -1.1% | mid          | bear_choppy     | ⚠ NO_ALPHA|
```

Flags:
- `⚠ LOW_SAMPLE` — fewer than 30 signals
- `❌ REMOVE` — win% < 40% AND avg R < 0.5
- `⚠ NO_ALPHA` — alpha vs buy-and-hold is negative
- `✅ STRONG` — win% > 55% AND avg R > 1.2 AND positive alpha

---

**Per-sector × per-regime weight map (new architecture):**

The output is a 2-level dict: `sector_weights[sector][regime]`. This is how institutional strategy allocation actually works.

```python
# Example output written to research/sector_weights.json
{
  "tech_mega": {
    "bull_trending":  {"momentum": 0.45, "breakout": 0.30, "supertrend": 0.25},
    "bull_choppy":    {"momentum": 0.20, "mean_reversion": 0.45, "stoch_rsi": 0.35},
    "bear_trending":  {"liquidity_sweep": 0.50, "supertrend": 0.30, "stoch_rsi": 0.20},
    "bear_choppy":    {"mean_reversion": 0.55, "stoch_rsi": 0.30, "vwap_reclaim": 0.15},
    "_fallback":      {"momentum": 0.20, ...}   # used if cell has < 20 samples
  },
  ...
}
```

**Cell sample gate:** If a sector × regime cell has fewer than 20 signals, its weight map falls back to the sector-level `_fallback` (weighted average across all regimes for that sector). If the sector itself has < 30 total signals, falls back to global `_STOCK_WEIGHTS`. This prevents overfitting sparse cells.

**Auto-apply logic:**
- Write `research/sector_weights.json` always (full output for review)
- Auto-patch `strategy_router.py` only if: cell has ≥ 20 samples AND new weights improve quality score ≥ 15% vs current

---

### 1.2 Bot Health Audit (`research/bot_health_audit.py`)

**Purpose:** Evaluate current config parameters against actual trade history and known failure modes. Produces prioritised improvement backlog.

**Six audit checks:**

**1. Signal threshold calibration**
- Replay historical signals at min_score = [0.15, 0.20, 0.25, 0.30, 0.35]
- Measure: trade count, win rate, avg R at each threshold
- Output: recommended min_composite_score per asset class

**2. Risk parameter calibration**
- Analyze `trades.json` — compare actual MAE (max adverse excursion) vs SL placement
- Check: were TP targets reachable given the ATR mult?
- Output: recommended stop_loss_atr_mult, take_profit_atr_mult

**3. Missing safeguards check**
- Earnings avoidance: count signals that fired within 2 days of earnings
- VIX gating: compare signal quality when VIX > 25 vs normal
- Time-of-day: open vs mid vs close session quality
- Market breadth: signal quality when < 40% of S&P stocks are above 50 EMA (weak internals)
- Output: yes/no + estimated win-rate impact for each safeguard

**4. Execution gap analysis**
- Estimate slippage of market orders vs limit orders using config.backtest.spread_pct
- Output: annual slippage saving from enabling smart_orders

**5. Regime coverage gaps**
- Identify HMM/EMA conflict periods
- Measure signal quality during conflict vs agreement
- Output: whether 4-state regime resolves the conflicts

**6. Correlation clustering risk**
- On days with 3+ open positions: compute pairwise correlations
- Flag if > 2 positions have correlation > 0.7 (concentrated risk)
- Output: recommended max_correlation and sector cap adjustments

**Output:** `research/audit_report.md` — unified findings + prioritised improvements:
```
| # | Improvement                    | Est. Impact      | Effort  | Priority |
|---|--------------------------------|------------------|---------|----------|
| 1 | Earnings avoidance             | -8% drawdown     | 1 day   | HIGH     |
| 2 | Per-sector × regime weights    | +12% win rate    | 3 days  | HIGH     |
| 3 | Time-of-day filter (open/close)| +5-10% win rate  | 1 day   | HIGH     |
| 4 | VIX / cross-asset sizing       | -15% drawdown    | 1 day   | HIGH     |
| 5 | Market breadth gate            | -5% bad trades   | 1 day   | MED      |
| 6 | Limit order execution          | +0.3%/trade      | 2 days  | MED      |
| 7 | ML signal filter               | +8% win rate     | 1 week  | MED      |
```

---

## Workstream 2 — Edge Layer (`edge/` module)

### 2.1 Cross-Asset Signals (`edge/cross_asset.py`)

**Purpose:** Macro context that individual stock strategies can't see. Proven institutional signals.

**Signals computed:**
```python
@dataclass
class CrossAssetSignals:
    vix_regime: str           # "low"(<15) "normal"(15-25) "elevated"(25-35) "panic"(>35)
    vix_term_structure: str   # "contango" (VIX < VIX3M) or "backwardation"
    bond_trend: str           # "risk_on" (TLT falling) or "risk_off" (TLT rising)
    dxy_trend: str            # "strong" or "weak"
    market_breadth: float     # % of SPY components above 50 EMA (proxy: breadth ETF or computed)
    breadth_signal: str       # "healthy"(>60%) "neutral"(40-60%) "weak"(<40%)
    sector_momentum: dict     # {sector: "leading"|"lagging"|"neutral"} vs SPY
    nq_overnight_move: float  # (NQ open - prev NQ close) / prev close — futures signal
    size_multiplier: float    # combined scalar: 0.25 (panic) to 1.25 (ideal)
```

**Market breadth implementation:**
- Proxy: fetch daily bars for ["SPY", "IWM", "QQQ", "RSP"] + sector ETFs already in cache
- Compute % of sector ETFs above their 50 EMA → breadth proxy
- If breadth < 40%: suppress all new longs by 40%, allow shorts freely
- If breadth > 60%: normal operation

**NQ overnight move:**
- Fetch NQ front-month previous close and today's open from IBDataFetcher
- If overnight_move > +0.5%: boost momentum/breakout weights for first session bucket
- If overnight_move < -0.5%: suppress long signals in first session bucket

**Idiosyncratic opportunity signal (per-symbol):**
- Computed in `microstructure.py` alongside spread gate
- `stock_vs_spy_corr`: rolling 20-day correlation of daily returns vs SPY
- Low correlation (<0.3) → stock is trading on its own catalyst → boost signal confidence by 15%
- High correlation (>0.8) → stock is just following the market → apply market breadth gate strictly

**Size multiplier logic:**
```python
mult = 1.0
if vix_regime == "panic":     mult *= 0.25
elif vix_regime == "elevated": mult *= 0.60
if vix_term_structure == "backwardation": mult *= 0.80
if bond_trend == "risk_off":  mult *= 0.85
if breadth_signal == "weak":  mult *= 0.70
# Cap: never above 1.25, never below 0.15
size_multiplier = max(0.15, min(1.25, mult))
```

**Update frequency:** Once per cycle (5 min), cached 5 min. Sector ETFs reuse `SectorRegimeFilter` cache.

---

### 2.2 News & Earnings (`edge/news_sentiment.py`)

**Purpose:** Avoid trading into binary events. Add news catalyst awareness.

**Component A — Earnings avoidance (always active):**
- Fetch earnings calendar from Alpaca `/v1beta1/corporate_actions/announcements`
- Cache per symbol for 24h
- Block new entries: 2 calendar days before AND 1 day after earnings
- Cannot be disabled — earnings destroy stop-based strategies unconditionally

**Component B — News sentiment (optional):**
- Fetch last 4h headlines per symbol from NewsAPI or Polygon.io
- Score with VADER (no GPU needed)
- Negative (compound < -0.5): suppress buy signals by 30%
- Positive (compound > 0.5): boost buy score by 10% (asymmetric — downside worse than upside)
- Neutral: no effect

**Config:**
```yaml
edge:
  earnings_avoidance: true
  news_sentiment: false        # requires NEWSAPI_KEY in .env
  news_lookback_hours: 4
  news_sentiment_threshold: 0.5
```

---

### 2.3 Microstructure Gate (`edge/microstructure.py`)

**Purpose:** Execution quality — don't enter in thin markets. Use IB live quotes already available.

**Spread gate (blocks entry):**
```python
quote = ib_broker.get_quote(symbol)
spread_pct = (quote.ask - quote.bid) / quote.mid
if spread_pct > config["edge"]["max_spread_pct"]:  # default 0.15%
    return False  # skip this entry
```

**OFI nudge (does NOT block — only small adjustment):**
- `(close - open) / (high - low)` on last 3 × 5-min bars
- Acknowledged limitation: not real order flow, noisy at 5-min
- Max adjustment: ±0.05 on composite score
- Weight kept intentionally small — do not rely on this signal

**Idiosyncratic correlation:**
- `stock_vs_spy_corr` rolling 20-day — if < 0.3, stock has its own catalyst, boost confidence 15%
- If > 0.8, apply breadth gate strictly (stock is just beta exposure)

**Config:**
```yaml
edge:
  max_spread_pct: 0.0015
  ofi_weight: 0.05
```

---

### 2.4 ML Signal Filter (`edge/ml_filter.py`)

**Purpose:** Meta-model that learns which setups actually work. Passthrough until 100 trades exist.

**Features (per signal):**
```python
features = {
    # Strategy layer
    "strategy_scores":    [...],   # 8 floats
    "composite_score":    float,
    "num_agreeing":       int,

    # Regime layer (4-state)
    "regime":             str,     # one-hot encoded: bull_trending/choppy, bear_trending/choppy
    "high_vol":           bool,

    # Cross-asset layer
    "vix_regime":         str,     # encoded
    "bond_trend":         str,     # encoded
    "breadth_signal":     str,     # encoded
    "size_multiplier":    float,

    # Microstructure layer
    "spread_pct":         float,
    "ofi_score":          float,
    "spy_corr":           float,

    # Time features
    "hour_of_day":        int,
    "day_of_week":        int,
    "session_bucket":     str,     # encoded: open/mid/close
    "days_to_earnings":   int,

    # Futures-specific
    "nq_overnight_move":  float,

    # Context
    "sector":             str,     # one-hot
}
```

**Target:** 1 if trade hit TP or R ≥ 1.0, else 0.

**Model design:**
- LightGBM classifier — handles small datasets, categorical features natively, no scaling
- Train/test split: **first 70% chronologically = train, last 30% = test** (never random split)
- Threshold: predict_proba ≥ 0.55 to pass
- Output: **confidence float (0–1), not pass/fail** → `position_size *= model_confidence`
- Feature importance logged at each training run — if `hour_of_day` or `sector` dominate over strategy scores, flag as suspect

**Training lifecycle:**
- Auto-trains when `trades.json` reaches 100 records
- Retrains every 50 new trades
- Saves to `research/ml_filter.pkl` with metadata (train date, n_samples, test AUC)
- Passthrough mode: when pkl missing or n_trades < 100 → always returns 1.0

**Coordinator integration:**
```python
confidence = ml_filter.predict_quality(features)  # 0.0–1.0 (or 1.0 in passthrough)
final_size = base_size * confidence                # scales position, never blocks entirely
# Log confidence with every order for future analysis
```

---

## Workstream 3 — Strategy Router Upgrade

**Current:** 3 flat weight maps (stock/crypto/futures).

**New:** 2-level lookup: `sector_weights[sector][regime]` with fallback chain:
1. `sector_weights[sector][regime]` — if cell has ≥ 20 samples
2. `sector_weights[sector]["_fallback"]` — weighted average across regimes for sector
3. `_stock_weights` — global stock fallback if sector unknown

```python
def get_strategies(self, instrument_type: str,
                   sector: str = None,
                   regime: str = None) -> dict[str, float]:
    if instrument_type == "crypto":
        return dict(self._crypto_weights)
    if instrument_type == "futures":
        return dict(self._futures_weights)

    if sector and sector in self._sector_weights:
        sector_map = self._sector_weights[sector]
        if regime and regime in sector_map:
            return dict(sector_map[regime])      # best: sector + regime
        if "_fallback" in sector_map:
            return dict(sector_map["_fallback"]) # sector only
    return dict(self._stock_weights)             # global fallback
```

Coordinator computes `current_regime` once per cycle using the same 4-state classifier as the research harness, then passes it to `start_watchers()` → `StockWatcher` → `get_strategies()`.

---

## Coordinator Integration

```python
# Once per cycle at top of _coordinator_cycle():
edge_ctx = EdgeContext(
    cross_asset   = self.edge_cross_asset.get_signals(),
    blocked_syms  = self.edge_news.get_blocked_symbols(),
    ml_filter     = self.edge_ml.predict_quality,   # callable → float
    current_regime= self.regime.classify_4state(),  # new method
)

# Per order:
# 1. Skip if symbol in edge_ctx.blocked_syms (earnings)
# 2. regime_size_mult *= edge_ctx.cross_asset.size_multiplier
# 3. spread gate in microstructure.py before submit_order()
# 4. position_size *= edge_ctx.ml_filter(features)   # confidence scale
```

**New config block:**
```yaml
edge:
  enabled: true
  earnings_avoidance: true
  news_sentiment: false
  max_spread_pct: 0.0015
  ofi_weight: 0.05
  ml_filter_threshold: 0.55
  ml_min_trades: 100
```

---

## Implementation Order

1. **Research harness** — offline, no production risk, produces everything else
2. **Earnings avoidance** — highest impact, lowest risk, 1 day
3. **4-state regime classifier** — needed by strategy router + ML features
4. **Cross-asset signals** — VIX/TLT/DXY/breadth/overnight NQ
5. **Microstructure spread gate + correlation** — tiny, uses existing IB quotes
6. **Strategy router per-sector × per-regime weights** — needs research output first
7. **ML filter scaffold** — passthrough now, trains automatically when data grows
8. **News sentiment** — optional, only if NEWSAPI_KEY set

---

## Files Created / Modified

| File | Action | Purpose |
|------|--------|---------|
| `research/strategy_audit.py` | Create | Per-strategy × sector × regime scorecard + alpha vs baseline |
| `research/bot_health_audit.py` | Create | Config calibration + safeguard gap analysis |
| `research/audit_report.md` | Auto-generated | Unified findings + prioritised improvements |
| `research/sector_weights.json` | Auto-generated | `sector_weights[sector][regime]` maps |
| `research/ml_filter.pkl` | Auto-generated | Trained ML model (activates at 100 trades) |
| `edge/__init__.py` | Create | EdgeContext dataclass + package init |
| `edge/cross_asset.py` | Create | VIX/TLT/DXY/breadth/NQ overnight signals |
| `edge/news_sentiment.py` | Create | Earnings calendar + optional news sentiment |
| `edge/microstructure.py` | Create | Spread gate + OFI nudge + SPY correlation |
| `edge/ml_filter.py` | Create | LightGBM meta-model, confidence output, passthrough |
| `strategy_router.py` | Modify | 2-level `sector_weights[sector][regime]` + fallback chain |
| `regime.py` | Modify | Add `classify_4state()` (ADX + ATR percentile + EMA direction) |
| `coordinator.py` | Modify | EdgeContext wired into cycle, 4-state regime passed to watchers |
| `watcher.py` | Modify | Accept sector + regime params, pass to strategy router |
| `config.yaml` | Modify | Add `edge:` block |
