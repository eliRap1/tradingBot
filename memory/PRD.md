# Trading Bot PRD - PROFIT MAXIMIZED Edition

## Original Problem Statement
Build the most profitable trading bot possible for stocks and crypto using Alpaca API. The bot should:
- Make the most profit
- Have robust risk management
- Support both stocks and crypto
- Have a monitoring dashboard

## Tech Stack
- **Backend**: Python (FastAPI for API if needed)
- **Broker API**: Alpaca (paper & live trading)
- **Dashboard**: Flask + HTML/JS (TradingView Lightweight Charts)
- **Storage**: JSON files for state persistence

## Architecture

### Core Components
1. **Coordinator** - Main orchestrator running 5-min cycles
2. **Watchers** - Per-symbol threads monitoring price action (40+ stocks, 2 crypto)
3. **Strategies** (7 total):
   - SuperTrend (trend following)
   - Momentum (EMA + RSI + MACD)
   - Mean Reversion (Bollinger Bands)
   - Breakout (range breakout)
   - Stoch RSI (oscillator)
   - VWAP Reclaim (institutional levels)
   - **NEW: Gap** (gap and go / fade)
4. **Profit Maximizer** - Signal enhancement and adaptive exits
5. **Risk Manager** - Volatility-adjusted sizing, Kelly criterion
6. **Portfolio Manager** - Multi-tier partial exits, trailing stops

### Signal Flow
```
Watcher → Strategy Signals → Confluence Filter → Profit Enhancement → Risk Sizing → Execution
```

## What's Been Implemented (2026-01-27)

### Session 1: Bug Fixes
- ✅ Crypto OCO tracking (TP/SL properly linked)
- ✅ Partial fill handling
- ✅ Idempotency keys for orders
- ✅ API rate limiting with exponential backoff
- ✅ Kelly criterion edge case fix
- ✅ Memory leak fix in watcher threads

### Session 2: Profit Maximization
- ✅ **Gap Trading Strategy** - New high-probability setup
- ✅ **Volatility-Adjusted Sizing** - Inverse vol positioning
- ✅ **Enhanced Momentum Strategy** - MACD cross, divergence detection
- ✅ **Profit Maximizer Module** - Volume surge, sector rotation
- ✅ **Two-Tier Partial Exits** - 40% at 1.2R, 30% at 2.5R
- ✅ **More Aggressive Config** - Lowered confluence, bigger targets
- ✅ Updated documentation with profit expectations

## User Personas
1. **Aggressive Trader**: Wants max returns, comfortable with higher risk
2. **Part-time Investor**: Uses dashboard for monitoring, hands-off approach
3. **Quant Developer**: Wants to understand and modify strategy logic

## Risk Management Layers
1. **Trade Level**: 2.5:1 minimum R:R, volatility-adjusted sizing
2. **Position Level**: 10% max per position, correlation limits
3. **Portfolio Level**: 12% max drawdown, 2.5% daily loss limit
4. **Exit Management**: Partial profits, trailing stops, time stops

## Configuration Summary

| Category | Key Settings |
|----------|--------------|
| Positions | Max 12 stocks + 2 crypto |
| Risk | 2% per trade, 12% max drawdown |
| Entries | 2+ strategies agree, 0.20+ score |
| Exits | 1.2R partial (40%), 2.5R partial (30%), trail rest |
| Sizing | Volatility-adjusted (0.6x-1.4x multiplier) |

## Prioritized Backlog

### P0 (Completed)
- ✅ Critical bug fixes
- ✅ Profit maximization improvements

### P1 (Next Session)
- [ ] Walk-forward optimization framework
- [ ] Monte Carlo trade sequence simulation
- [ ] WebSocket streaming for real-time updates
- [ ] Email/SMS alerting integration

### P2 (Future)
- [ ] Machine learning signal enhancement
- [ ] Options strategy integration
- [ ] Multi-broker support
- [ ] Mobile app dashboard

## Testing Requirements

### Paper Trading (Required)
- Minimum 50 trades before live
- Win rate target: > 50%
- Profit factor target: > 1.5
- Max drawdown: < 10%

### Validation Checklist
- [ ] All 7 strategies generating signals
- [ ] Partial exits triggering correctly
- [ ] Gap strategy working on market open
- [ ] Correlation filter blocking similar positions
- [ ] Volatility sizing working as expected

## Next Tasks
1. User provides Alpaca API keys
2. Run paper trading validation
3. Monitor for 50+ trades
4. Adjust config based on results
5. Consider live deployment with 1/4 size
