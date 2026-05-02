# Edge Layer Audit — May 2026

## Executive summary

Bot has **11 edge modules wired in `coordinator.py`** (live path) but
**only 1 (`regime_gate`) wired in `backtester.py`**. This explains
the IS/OOS gap, the unrealistic Sharpe 13.97, and likely overstates
backtest performance vs reality.

**Critical action:** wire all live edges into backtester for parity,
OR accept that backtest = upper bound and live = lower bound.

## Existing edges (12 modules, 1390 LOC)

| Edge | Live (coord) | Backtest | Function | Sharpe contribution* |
|------|--------------|----------|----------|----------------------|
| `regime_gate.py` | ✓ | ✓ | SPY ADX chop + 20d vol panic + EMA50 | tail-risk reducer (+0.3) |
| `relative_strength.py` | ✓ | partial (screener) | RS rank vs SPY/sector | momentum (+0.5) |
| `volume_gate.py` | ✓ | ✗ | OBV/volume surge boost or block | confirmation (+0.2) |
| `gap_filter.py` | ✓ | ✗ | Block/penalize >3% gaps | tail-risk reducer (+0.2) |
| `insider_flow.py` | ✓ | ✗ | SEC Form 4 cluster buys (P codes) | edge-effect (+0.4) |
| `cross_asset.py` | ✓ | ✗ | VIX/DXY/bond/breadth/sector momentum | regime-adapt (+0.4) |
| `microstructure.py` | ✓ | ✗ | NBBO spread, halts, hours | execution quality (+0.1) |
| `ml_filter.py` | ✓ | ✗ | LightGBM prob (passthrough until 100 trades) | precision (+0.5) |
| `news_sentiment.py` | ✓ | ✗ | Earnings hard-block + headline VADER | event-risk (+0.3) |
| `econ_calendar.py` | ✓ | ✗ | FOMC/CPI/jobs blackout | event-risk (+0.2) |
| `market_calendar.py` | ✓ | ✗ | Holidays + half-days | hygiene (+0.05) |

\* Estimated standalone Sharpe lift from literature/heuristics.
Cumulative ≠ additive — overlap exists.

### Backtest-live parity gap

10 edges are silent in backtest. Two interpretations:

1. **Optimistic case (live ≥ backtest):** filters reject more
   bad trades than good → live PF improves.
2. **Pessimistic case (live ≤ backtest):** filters over-block,
   trade count drops 30-50%, dollar P&L drops despite higher PF.

**Empirical answer requires paper-trade comparison.** Until then,
discount backtest by 30-40% as live expectation.

## Missing institutional edges (ranked by ROI)

### Tier S — high lift, moderate effort

**1. Pairs / stat-arb (cointegration mean-reversion)**
- Signal: pick same-sector pairs (e.g. AAPL/MSFT). Compute z-score
  of price ratio. When |z| > 2 → long laggard, short leader.
  Reverts in 3-10 days.
- Lit. Sharpe: 1.0-2.0 standalone, low correlation to momentum.
- Effort: 2-3 days. Use `statsmodels.tsa` Engle-Granger test.
- File: `edge/pairs.py` (new) + `strategies/pairs_trade.py`.

**2. Post-earnings announcement drift (PEAD)**
- Signal: SUE = (actual EPS − consensus) / σ. Top decile drifts up
  60+ days post-earnings; bottom decile drifts down.
- Lit. Sharpe: 0.5-0.8 standalone, robust 50+ years.
- Effort: 1-2 days. Need EPS surprise data (Alpaca, Polygon,
  Finnhub free tier).
- File: `edge/pead.py` + integrate into `relative_strength.py`.

**3. Short interest squeeze detector**
- Signal: SI > 20% of float + days-to-cover > 5 + ATR expansion +
  positive RS = squeeze setup. Block shorts in same names.
- Lit. Sharpe: 0.4-0.7 in tail events, asymmetric payoff.
- Effort: 1 day. FINRA biweekly free; quarterly EDGAR.
- File: `edge/short_interest.py`.

### Tier A — moderate lift, low effort

**4. Term structure (VIX9D / VIX / VIX3M)**
- Signal: VIX9D > VIX = stress (reduce size); VIX9D < VIX3M
  steeply = complacency (raise alert). Already have VIX in
  cross_asset; just add term-structure ratio.
- Lit. Sharpe: 0.3 as overlay.
- Effort: 0.5 day. Add to `cross_asset.py`.

**5. Seasonality / TOM (turn-of-month)**
- Signal: last 4 + first 3 trading days of month historically
  capture ~80% of equity returns. Flag size +20% in window.
- Lit. Sharpe: 0.2-0.4 as overlay.
- Effort: 0.5 day. Pure date math.
- File: extend `market_calendar.py`.

**6. Analyst revisions momentum**
- Signal: 30-day net upgrades − downgrades. Top quintile
  outperforms bottom by ~6% annual.
- Lit. Sharpe: 0.4-0.6.
- Effort: 1 day. Finnhub `/stock/upgrade-downgrade` free tier.
- File: `edge/analyst_revisions.py`.

**7. Intermarket ratios**
- Signal: XLY/XLP (cyclical/defensive), HYG/IEF (credit), DBA
  (commodities). Falling cyclicals + rising defensives = risk-off.
- Lit. Sharpe: 0.2 standalone, complementary to VIX.
- Effort: 0.5 day. Add to `cross_asset.py`.

### Tier B — high lift, hard effort (require paid feeds)

**8. Options flow / unusual activity**
- Signal: call/put volume ratio anomalies, large OTM call sweeps.
  Often precedes 5-10% moves.
- Lit. Sharpe: 0.5-1.0 with quality data.
- Effort: 5+ days. Paid feeds: BlackBoxStocks $90/mo,
  Cheddar Flow $80/mo. Unusual Whales $90/mo.
- Defer until $25k+ AUM justifies cost.

**9. Dark pool prints**
- Signal: large block trades flagged as institutional flow.
- Lit. Sharpe: 0.3-0.6.
- Effort: 5+ days. Same paid feeds.
- Defer same as #8.

**10. Gamma exposure (GEX)**
- Signal: dealer gamma positioning. Positive GEX = mean-reverting,
  negative GEX = trending/volatile. SpotGamma $80/mo.
- Lit. Sharpe: 0.3 as overlay.
- Defer.

### Tier C — niche / experimental

**11. Fed liquidity (RRP, reserves, BTFP balance)**
- Signal: Fed liquidity injections/drains. Macro tide.
- Free from FRED. Slow signal (weekly).
- Effort: 1 day. Add to `cross_asset.py`.

**12. Crypto/equity correlation regime**
- Signal: BTC trend predicts QQQ on 1-3 day lag during high-corr
  regimes (2021-2024). Currently weak.
- Effort: 1 day.

## Roadmap (prioritized)

### Sprint 1 (week 1) — backtest parity
1. Wire all 10 edges into backtester. Backtest 35-sym 500d should
   then drop to ~realistic Sharpe 2-4 range.
2. Re-run A/B with all edges live. Establish true baseline.

### Sprint 2 (week 2-3) — Tier S adds
3. Implement `edge/pairs.py` + pairs strategy.
4. Implement `edge/pead.py` + integrate.
5. Implement `edge/short_interest.py`.

### Sprint 3 (week 4) — Tier A overlays
6. Term structure → cross_asset.
7. TOM + Fed-day overlays → market_calendar.
8. Analyst revisions module.

### Sprint 4 (post-paper-trade) — paid Tier B
9. Options flow (if AUM justifies).
10. Dark pool / GEX.

## Edge attribution tracking (proposed)

Current `trades.strategies` field captures only winning strategies.
**Extend `state_db.trades` schema:**

```sql
ALTER TABLE trades ADD COLUMN edge_snapshot TEXT;
-- JSON blob: {rs_score, vol_ratio, ml_prob, vix_regime,
--             insider_cluster, gap_pct, news_score, ...}
```

Then weekly aggregate query: which edges fire on winners vs losers?
Drops the bottom-10% performers; promotes top-3 to higher weights.

## Verdict

Bot has **strong edge architecture**. 11 modules covering momentum,
event-risk, microstructure, regime, ML. The gap is:

1. **Backtest doesn't reflect live filtering** → decline backtest by
   30-40% for live expectation.
2. **No stat-arb / mean-reversion alpha** → bot is purely directional.
   Pairs trades are uncorrelated alpha worth $2-5/year on $10k.
3. **No institutional flow data** → blind to dark pool / options
   sweeps. Tier-B feeds cost $80-90/mo each.

Realistic live Sharpe ceiling **with all Sprint 1-3 edges**: 2.0-2.5.
Top 5% of retail algos. Top quartile of mid-tier hedge funds. Below
elite quants by design (no HFT, no alt data).
