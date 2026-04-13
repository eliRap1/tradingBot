# Strategy Audit, Bot Health Review & Edge Layer Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit every strategy's real performance per sector/asset class, identify the top bot improvements, and add a proven edge layer (cross-asset signals, news/earnings avoidance, microstructure gate, ML meta-filter) that gives the bot an institutional-grade information advantage.

**Architecture:** Three parallel workstreams — (1) offline research harness that produces a scorecard and auto-updates strategy weights, (2) edge module wired into the coordinator as a layered pre-trade gate, (3) strategy router upgrade to per-sector weight maps. No production code is touched during the research phase. The edge layer ships each component independently so it can be validated in isolation.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn (ML filter), lightgbm (meta-model), FinBERT/VADER (news sentiment), ib_insync (IB quotes for microstructure), alpaca-trade-api (historical data fallback), existing strategy classes, existing AlpacaDataFetcher/IBDataFetcher.

---

## Workstream 1 — Research Harness

### 1.1 Strategy Audit (`research/strategy_audit.py`)

**Purpose:** Run each strategy in isolation against 6 months of historical data, sector by sector. Measure signal quality and compute new per-sector weights.

**Sector groups and representative symbols:**
```python
SECTOR_GROUPS = {
    "tech_mega":    ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
    "semis":        ["NVDA", "AMD", "AVGO", "MU", "QCOM"],
    "software":     ["CRM", "NOW", "PANW", "CRWD", "DDOG"],
    "financials":   ["JPM", "GS", "MS", "V", "MA"],
    "healthcare":   ["UNH", "LLY", "ISRG", "VRTX", "AMGN"],
    "energy":       ["XOM", "CVX", "COP", "SLB", "EOG"],
    "industrials":  ["CAT", "HON", "GE", "RTX", "DE"],
    "consumer":     ["COST", "HD", "NKE", "MCD", "SBUX"],
    "crypto":       ["BTC/USD", "ETH/USD", "SOL/USD"],
    "futures":      ["NQ", "ES", "CL", "GC"],
}
```

**Signal evaluation method:**
- For each symbol, fetch 6 months of 5-min bars + 1-day bars (IB first, Alpaca fallback)
- Run `strategy.generate_signals({"SYMBOL": bars})` over rolling 100-bar windows (step 5 bars)
- At each signal with `|score| >= min_composite_score`:
  - Determine entry direction and simulated entry price (next bar open)
  - ATR-based SL and TP using config risk params
  - Walk forward bars to determine outcome: HIT_TP, HIT_SL, or TIME_STOP (60 bars)
  - Record: symbol, sector, strategy, score, direction, outcome, R_multiple, regime (HMM/EMA at time of signal)

**Scorecard metrics per strategy × sector:**
```
| Strategy        | Sector      | Signals | Win% | Avg R | Best Regime  | Worst Regime |
|-----------------|-------------|---------|------|-------|--------------|--------------|
| momentum        | tech_mega   | 87      | 61%  | 1.4   | bull+trend   | bear+ranging |
| mean_reversion  | financials  | 34      | 58%  | 0.9   | ranging      | trending     |
...
```

Flags:
- `⚠ LOW_SAMPLE` — fewer than 30 signals (don't trust result)
- `❌ REMOVE` — win% < 40% AND avg R < 0.5 (strategy is net negative for this sector)
- `✅ STRONG` — win% > 55% AND avg R > 1.2

**Weight generation:**
```python
# For each sector, compute new weights proportional to quality score
quality_score = max(0, win_rate - 0.40) * avg_R  # zero if below threshold
new_weight = quality_score / sum(quality_scores)  # normalize
# Only auto-apply if sector has >= 30 signals per strategy AND
# quality_score improvement > 15% vs current weights
```

**Auto-apply logic:**
- If a sector has enough data AND new weights differ meaningfully: write `_SECTOR_WEIGHTS[sector]` to `strategy_router.py`
- Otherwise: write recommended weights to `research/audit_report.md` for manual review

---

### 1.2 Bot Health Audit (`research/bot_health_audit.py`)

**Purpose:** Evaluate current config parameters against actual trade history and known failure modes. Produces top 5 improvements ranked by impact/effort.

**Five audit checks:**

**1. Signal threshold calibration**
- Replay historical signals at min_score = [0.15, 0.20, 0.25, 0.30, 0.35]
- Measure: trade count, win rate, avg R at each threshold
- Output: recommended min_composite_score per asset class

**2. Risk parameter calibration**
- Analyze existing trades in `trades.json`
- Check: were SL hits within 1× ATR? Were TP hits possible given the ATR mult?
- Compare actual MAE (max adverse excursion) vs SL placement
- Output: recommended stop_loss_atr_mult, take_profit_atr_mult

**3. Missing safeguards check**
- Earnings avoidance: count how many signals fired within 2 days of earnings on historical data
- VIX gating: compare signal quality on high-VIX days (VIX > 25) vs normal
- Time-of-day: compare signal quality in first/last 30 min vs mid-session
- Output: yes/no + estimated impact for each safeguard

**4. Execution gap analysis**
- Estimate slippage cost of market-only orders vs limit orders
- Use bid-ask spread from config.backtest.spread_pct
- Output: estimated annual slippage saving from enabling smart_orders

**5. Regime coverage gaps**
- Identify periods where both HMM says BEAR and EMA says BULL (conflict)
- Measure signal quality during conflicted regime periods
- Output: whether a third regime signal (e.g. VIX) would resolve conflicts

**Output:** `research/audit_report.md` with unified scorecard + top 5 improvements table:
```
| # | Improvement              | Est. Impact | Effort | Priority |
|---|--------------------------|-------------|--------|----------|
| 1 | Earnings avoidance       | -8% drawdown | 1 day  | HIGH     |
| 2 | Per-sector weights       | +12% win rate | 3 days | HIGH    |
| 3 | VIX size scaling         | -15% drawdown | 1 day | HIGH     |
| 4 | Limit order execution    | +0.3% per trade | 2 days | MED   |
| 5 | ML signal filter         | +8% win rate | 1 week | MED     |
```

---

## Workstream 2 — Edge Layer (`edge/` module)

### 2.1 Cross-Asset Signals (`edge/cross_asset.py`)

**Purpose:** Provide macro context that individual stock strategies can't see. Proven institutional signals.

**Signals computed:**
```python
@dataclass
class CrossAssetSignals:
    vix_regime: str          # "low" (<15), "normal" (15-25), "elevated" (25-35), "panic" (>35)
    vix_term_structure: str  # "contango" (VIX < VIX3M = complacent) or "backwardation" (fearful)
    bond_trend: str          # "risk_on" (TLT falling = yields rising = growth favored)
                             # "risk_off" (TLT rising = yields falling = defensives favored)
    dxy_trend: str           # "strong" (bad for commodities/EM) or "weak" (good for commodities)
    sector_momentum: dict    # {sector_etf: "leading"|"lagging"|"neutral"} vs SPY
    size_multiplier: float   # combined sizing scalar: 0.25 (panic) to 1.25 (ideal conditions)
```

**Data sources:**
- VIX, VIX3M: IB historical bars (`VIX` index, `VIX3M` index) or Alpaca
- TLT: standard stock bars
- DXY: IB forex or ETF proxy (UUP)
- Sector ETFs: already fetched by `SectorRegimeFilter` — reuse that cache

**Update frequency:** Once per coordinator cycle (5 min), cached for 5 min.

**How it's used in coordinator:**
```python
signals = self.edge_cross_asset.get_signals()
# Applied in _coordinator_cycle():
regime_size_mult *= signals.size_multiplier
# Passed to strategy_selector to bias weights:
# vix_regime="panic" → suppress all trend strategies by 50%
# bond_trend="risk_off" → boost defensive sectors, suppress growth
```

---

### 2.2 News & Earnings (`edge/news_sentiment.py`)

**Purpose:** Avoid the single biggest retail trap — trading into earnings. Add news catalyst awareness.

**Component A — Earnings avoidance (must-have):**
- Fetch earnings calendar from Alpaca `/v1beta1/corporate_actions/announcements` or IB `reqFundamentalData`
- Cache earnings dates per symbol for 24h
- Block new entries within 2 calendar days before earnings AND 1 day after
- Always active — not configurable. Earnings are binary events that destroy stop-based strategies.

**Component B — News sentiment (optional, activates if API key present):**
- Fetch last 4h of news headlines per symbol from NewsAPI or Polygon.io free tier
- Score with VADER (fast, no GPU needed) — positive/negative/neutral
- If strong negative sentiment (compound < -0.5): suppress buy signals by 30%
- If strong positive: boost buy signal scores by 10% (smaller boost than suppress, asymmetric)
- Neutral: no effect

**Config:**
```yaml
edge:
  earnings_avoidance: true           # always recommended
  news_sentiment: false              # requires NEWSAPI_KEY in .env
  news_lookback_hours: 4
  news_sentiment_threshold: 0.5
```

---

### 2.3 Microstructure Gate (`edge/microstructure.py`)

**Purpose:** Don't enter when liquidity is thin. Uses IB live quotes already available.

**Two checks:**

**Spread gate:**
```python
quote = ib_broker.get_quote(symbol)
spread_pct = (quote.ask - quote.bid) / quote.mid
if spread_pct > config["edge"]["max_spread_pct"]:  # default 0.15%
    return False  # skip entry
```

**Order flow imbalance (OFI) score:**
- On the last 3 completed 5-min bars: compute `(close - open) / (high - low)` — positive = buyers in control
- OFI score = mean of last 3 bars' directional pressure
- Used as a small weight boost/penalty (±0.05) on the composite signal score
- Does NOT block trades, only nudges — it's noisy at 5-min resolution

**Config:**
```yaml
edge:
  max_spread_pct: 0.0015   # 0.15% — blocks entry if spread wider than this
  ofi_weight: 0.05         # max OFI adjustment to composite score
```

---

### 2.4 ML Signal Filter (`edge/ml_filter.py`)

**Purpose:** A meta-model that learns which setups actually work, filtering out low-quality signals before they reach execution. Ships in passthrough mode until enough trade history exists.

**Architecture:**
```
Features (per signal):
  - strategy scores (8 floats)
  - composite score, num_agreeing
  - HMM regime (0/1), EMA regime (0/1)
  - cross-asset: vix_regime (encoded), bond_trend (encoded), size_multiplier
  - microstructure: spread_pct, ofi_score
  - sector (one-hot)
  - time features: hour_of_day, day_of_week, days_to_earnings

Target:
  - 1 if trade hit TP or partial TP (R >= 1.0)
  - 0 if trade hit SL or time-stopped with loss

Model:
  - LightGBM classifier (fast, handles small datasets, no scaling needed)
  - Threshold: predict_proba >= 0.55 to pass filter (conservative)
  - Min training samples: 100 trades before activating

Training:
  - Auto-trains when trades.json reaches 100 records
  - Retrains every 50 new trades
  - Saves model to research/ml_filter.pkl
```

**Passthrough mode:** When `research/ml_filter.pkl` does not exist OR trade count < 100, `predict_quality()` always returns 1.0 (let all signals through). Bot behavior is identical to today.

**Activation:** Once trained, coordinator calls `ml_filter.predict_quality(features)` before submitting an order. If score < 0.55, signal is skipped and logged as `ml_filtered`.

---

## Workstream 3 — Strategy Router Upgrade

**Current state:** `strategy_router.py` has 3 flat weight maps (stock/crypto/futures).

**New state:** 3 flat maps remain as fallbacks. Added `_SECTOR_WEIGHTS` dict populated by the research harness. `get_strategies()` accepts an optional `sector` parameter.

```python
def get_strategies(self, instrument_type: str, sector: str = None) -> dict[str, float]:
    if instrument_type == "crypto":
        return dict(self._crypto_weights)
    if instrument_type == "futures":
        return dict(self._futures_weights)
    # Stock: use sector-specific if available, else fall back to stock default
    if sector and sector in self._sector_weights:
        return dict(self._sector_weights[sector])
    return dict(self._stock_weights)
```

`_sector_weights` is populated from `research/sector_weights.json` (written by the audit script). If the file doesn't exist, falls back silently — no behaviour change.

Coordinator passes sector to `start_watchers()` → `StockWatcher.__init__()` → `_strategy_router.get_strategies(instrument_type, sector)`.

---

## Coordinator Integration

The edge layer is wired in as a single call at the top of each `_coordinator_cycle()`:

```python
# Gather edge signals once per cycle
edge_ctx = EdgeContext(
    cross_asset=self.edge_cross_asset.get_signals(),
    earnings_blocked=self.edge_news.get_blocked_symbols(),
    # microstructure checked per-symbol at order time
    ml_filter=self.edge_ml.predict_quality,  # callable
)

# Applied in order submission:
# 1. Skip if symbol in earnings_blocked
# 2. Multiply regime_size_mult by cross_asset.size_multiplier
# 3. Check microstructure spread gate before submit_order()
# 4. Check ml_filter score before submit_order()
```

New config block:
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

1. **Research harness first** — no production risk, produces data to guide everything else
2. **Earnings avoidance** — highest impact, lowest risk, 1 day of work
3. **Cross-asset signals** — VIX/TLT/DXY, proven, straightforward
4. **Microstructure spread gate** — tiny, uses existing IB quotes
5. **Strategy router per-sector weights** — requires research harness output first
6. **ML filter scaffold** — design + passthrough now, train when data exists
7. **News sentiment** — optional, only if NEWSAPI_KEY configured

---

## Files Created / Modified

| File | Action | Purpose |
|------|--------|---------|
| `research/strategy_audit.py` | Create | Per-strategy × sector scorecard + weight generator |
| `research/bot_health_audit.py` | Create | Config calibration + safeguard gap analysis |
| `research/audit_report.md` | Auto-generated | Unified findings + top 5 improvements |
| `research/sector_weights.json` | Auto-generated | Per-sector weights for strategy router |
| `research/ml_filter.pkl` | Auto-generated | Trained ML model (when 100+ trades exist) |
| `edge/__init__.py` | Create | Package init |
| `edge/cross_asset.py` | Create | VIX/TLT/DXY/sector momentum signals |
| `edge/news_sentiment.py` | Create | Earnings calendar + news sentiment |
| `edge/microstructure.py` | Create | Spread gate + OFI score |
| `edge/ml_filter.py` | Create | Meta-model scaffold + passthrough |
| `strategy_router.py` | Modify | Add `_sector_weights` + sector param to `get_strategies()` |
| `coordinator.py` | Modify | Wire EdgeContext into cycle, pass sector to watchers |
| `watcher.py` | Modify | Accept sector param, pass to strategy router |
| `config.yaml` | Modify | Add `edge:` block |
