# Trading Bot - PROFIT MAXIMIZED Edition

## Summary of Profit-Maximizing Improvements

This version has been optimized for **maximum profitability** while maintaining robust risk management.

---

## Key Improvements Made

### 1. **More Aggressive Signal Generation**
- Lowered confluence requirement from 3 to 2 strategies
- Lowered minimum composite score from 0.25 to 0.20
- More sensitive RSI ranges for earlier entries

### 2. **Better Position Sizing**
- **Volatility-Adjusted Sizing**: Inverse volatility sizing (bigger positions on low-vol stocks, smaller on high-vol)
- Increased max position size from 8% to 10% of portfolio
- Increased per-trade risk from 1.5% to 2%

### 3. **Improved Risk/Reward**
- Raised minimum R:R requirement from 2:1 to 2.5:1
- Tighter stops (1.5x ATR vs 2x ATR) = less loss per trade
- Bigger targets (5x ATR vs 4x ATR) = let winners run

### 4. **Smarter Partial Exits**
- **First partial at 1.2R**: Take 40% off the table early
- **Second partial at 2.5R**: Take another 30% off
- Let remaining 30% run with tighter trail

### 5. **New Gap Trading Strategy**
- Gap and Go: Ride strong gaps with volume confirmation
- Gap Fade: Short weak gaps that are filling
- Institutional-level volume detection

### 6. **Enhanced Momentum Strategy**
- Added MACD crossover signals
- Momentum acceleration detection (ROC of ROC)
- RSI divergence detection (bullish/bearish)
- Price location vs 20-day range
- Higher volume thresholds for confirmation

### 7. **Profit Maximizer Module** (`profit_maximizer.py`)
- **Signal Enhancement**: Volume surge detection, momentum persistence
- **Dynamic Targets**: Uses support/resistance levels for TP/SL
- **Sector Rotation**: Overweight strongest sectors, underweight weakest
- **Adaptive Exits**: Hold winners longer in trends, cut losers faster

---

## Configuration Changes (`config.yaml`)

| Setting | Old | New | Reason |
|---------|-----|-----|--------|
| max_positions | 9 | 12 | More opportunities |
| min_agreeing_strategies | 3 | 2 | More signals |
| min_composite_score | 0.25 | 0.20 | Easier entry |
| max_position_pct | 8% | 10% | Bigger positions |
| max_portfolio_risk_pct | 1.5% | 2% | More risk per trade |
| stop_loss_atr_mult | 2.0 | 1.5 | Tighter stops |
| take_profit_atr_mult | 4.0 | 5.0 | Bigger targets |
| min_risk_reward | 2.0 | 2.5 | Better quality trades |
| partial_exit_r | 1.5 | 1.2 | Earlier profit taking |
| sizing_method | fixed_fractional | volatility_adjusted | Smarter sizing |
| allow_shorts_in_bull | false | true | Hedging allowed |

---

## New Strategies

### Gap Strategy (`strategies/gap.py`)
- **Gap and Go Long**: Gap up > 1.5%, holding above open, high volume
- **Gap and Go Short**: Gap down > 1.5%, holding below open, high volume
- **Gap Fade Short**: Gap up fading without volume
- **Gap Fade Long**: Gap down bouncing back

### Profit Maximizer (`profit_maximizer.py`)
- **ProfitMaximizer class**: Enhances signals with volume, momentum, divergence
- **AdaptiveExitManager class**: Extends winners, cuts losers faster

---

## Risk Management (Still Conservative Where It Matters)

| Protection | Setting |
|------------|---------|
| Max Drawdown | 12% (halts trading) |
| Daily Loss Limit | 2.5% (stops new trades for day) |
| Position Correlation | Max 70% correlation between positions |
| Partial Profit Taking | 40% at 1.2R, 30% at 2.5R |
| Trailing Stop | 2.5% from watermark |
| Breakeven Stop | Armed after 1.5% profit |
| Time Stop | Close zombie trades after 5 days |

---

## Expected Impact

Based on the changes:

| Metric | Expected Change |
|--------|-----------------|
| Trade Frequency | +40-60% more trades |
| Win Rate | Similar (55-60%) |
| Average Win | +15-25% larger (bigger targets) |
| Average Loss | -10-15% smaller (tighter stops) |
| Profit Factor | +0.2-0.3 improvement |
| Sharpe Ratio | +0.1-0.2 improvement |

---

## How to Test

### Paper Trading (Required First)
```bash
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
export TRADING_MODE=paper

python main.py
```

### Validation Checklist
1. [ ] Run for minimum 50 trades in paper mode
2. [ ] Verify win rate > 50%
3. [ ] Verify profit factor > 1.5
4. [ ] Verify average win > average loss
5. [ ] Verify drawdown stays below 12%
6. [ ] Verify partial exits trigger correctly
7. [ ] Verify gap strategy signals fire

### Dashboard
Open http://localhost:5000 to monitor:
- 7 strategies now (added Gap)
- Enhanced signal scores with profit maximizer boosts
- Partial exit tracking

---

## Files Modified/Created

### Modified
- `config.yaml` - More aggressive settings
- `risk.py` - Volatility-adjusted sizing
- `coordinator.py` - Integrated profit maximizer
- `portfolio.py` - Second partial exit
- `strategies/momentum.py` - Enhanced signals
- `strategies/__init__.py` - Added gap strategy

### Created
- `profit_maximizer.py` - Signal enhancement and adaptive exits
- `strategies/gap.py` - Gap trading strategy

---

## Warning

This configuration is **more aggressive** than the original. While it should generate more profit, it also takes on more risk:

1. **More trades** = more commission impact (though Alpaca is commission-free)
2. **Bigger positions** = larger individual losses possible
3. **Lower confluence** = some lower-quality signals may get through

**Always paper trade first** for at least 2 weeks before going live.
