# Bot Health Audit

Generated: 2026-04-16T20:35:35.797740 UTC

## Prioritised Improvements

| # | Improvement | Impact | Effort | Priority |
|---|-------------|--------|--------|----------|
| 1 | Enable execution.smart_orders (limit + improve) | ~0.3% saved per round-trip | 1 day | MED |
| 2 | Signal threshold looks calibrated | No tuning needed | — | LOW |
| 3 | Risk params balanced | No tuning needed | — | LOW |
| 4 | Safeguards configured | No gaps found | — | LOW |
| 5 | Regime coverage balanced | No action | — | LOW |
| 6 | Live correlation cap at 0.7 | Prevents beta clustering | — | LOW |

## Current Snapshot

- Trades in DB: `1`
- Win rate: `100.0%`
- Avg win: `$67.63`
- Avg loss: `$0.00`
- Config min score: `0.25`
- Config min agreeing: `3`
- Config stop ATR: `2.0`
- Config target ATR: `5.0`
- Edge enabled: `True`
- Earnings avoidance: `True`
- Max spread %: `0.0015`

## Check Details

### Enable execution.smart_orders (limit + improve) (`execution_gap`)

Currently using market orders — paying spread every trade.

### Signal threshold looks calibrated (`threshold_calibration`)

Win rate per threshold:
- 0.15: insufficient (0 trades)
- 0.2: insufficient (0 trades)
- 0.25: insufficient (0 trades)
- 0.3: insufficient (0 trades)
- 0.35: insufficient (0 trades)

### Risk params balanced (`risk_params`)

Exit mix: SL=0.0% TP=0.0% other=100.0% (R:R 2.50:1)

### Safeguards configured (`safeguards`)

earnings_avoidance OK, max_spread_pct <= 0.5% OK, edge.enabled OK

### Regime coverage balanced (`regime_coverage`)

Win rate per regime:
- unknown: insufficient (1)

### Live correlation cap at 0.7 (`correlation_clustering`)

risk.max_correlation=0.7. Full pairwise analysis requires position snapshots.
