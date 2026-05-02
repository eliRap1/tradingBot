# Daily Health — 2026-05-02

**Severity:** CRITICAL

## Snapshot

- Total closed trades: **26**
- Last trade: 2026-05-02T23:33:06.056048 (-1d ago)
- Open positions: **1** — CLM6
- Realized P&L: **$-695020.8**

## Performance vs Backtest Baseline

| Metric | Live (rolling 20) | Live (all) | Baseline | Verdict |
|--------|-------------------|------------|----------|---------|
| Profit factor | 0.0 | 0.019 | 2.36 | WARN |
| Win rate | 0.0% | 19.23% | 47.2% | WARN |
| Expectancy | $-35406.83 | $-26731.57 | $524.0 | WARN |
| Drawdown | 624.79% | — | 7.02% (max) | CRITICAL |
| Loss streak | 20 (current) | 20 (max) | — | WARN |
| Drift PF | +0.0% | — | within 35% | OK |

## Alerts

- **CRITICAL**: PF20=0.00 < 1.20 (baseline 2.36)
- **CRITICAL**: WR20=0.0% < 35.0% (baseline 47.2%)
- **CRITICAL**: Loss streak=20 >= 5
- **CRITICAL**: Drawdown=624.79% > 10.0% (baseline MDD 7.02%)
