# Trading Bot PRD & Implementation Log

## Original Problem Statement
Review, debug, and improve a trading bot code for stocks and crypto using Alpaca API. Goals included:
- Identify bugs, logical errors, edge cases
- Evaluate trading strategy (entries, exits, risk management)
- Detect overfitting/unrealistic assumptions
- Improve performance and real-world handling
- Ensure proper risk management
- Evaluate backtesting quality
- Provide improvements and testing recommendations

## Tech Stack
- **Language**: Python
- **Broker API**: Alpaca (paper & live trading)
- **Dashboard**: Flask + HTML/JS (TradingView Lightweight Charts)
- **Database**: JSON files for state persistence

## Core Architecture
- **Coordinator**: Main orchestrator running 5-min cycles
- **Watchers**: Per-symbol threads monitoring price action
- **Strategies**: SuperTrend, Momentum, Mean Reversion, Breakout, Stoch RSI, VWAP Reclaim
- **Risk Manager**: ATR-based position sizing, Kelly criterion, max drawdown
- **Portfolio Manager**: Trailing stops, partial exits, position reconciliation

## What's Been Implemented (2026-01-27)

### Critical Bug Fixes
1. ✅ **Crypto OCO Tracking** (`broker.py`) - TP/SL orders now properly linked; one cancels the other
2. ✅ **Partial Fill Handling** (`broker.py`) - Waits for fill confirmation, uses actual filled_qty
3. ✅ **Idempotency Keys** (`broker.py`) - Prevents duplicate orders on network retries
4. ✅ **API Rate Limiting** (`data.py`) - Token bucket rate limiter with exponential backoff
5. ✅ **Kelly Edge Case** (`risk.py`) - Capped win/loss ratio at 5:1
6. ✅ **Memory Leak Fix** (`watcher.py`) - Bars cache expiry to prevent memory buildup
7. ✅ **Coordinator Crypto Check** (`coordinator.py`) - Added OCO fill check to each cycle

### Documentation
- ✅ `CODE_REVIEW.md` - Comprehensive analysis with all issues and fixes

## User Personas
- **Algorithmic Trader**: Wants reliable automated execution with proper risk management
- **Quant Developer**: Needs to understand and modify strategy logic
- **Part-time Trader**: Uses dashboard for monitoring without coding

## Prioritized Backlog

### P0 (Critical - Deferred to User Decision)
- [ ] Smart order blocking (30s timeout) - affects latency
- [ ] Trailing stop intra-bar updates - consider native Alpaca trails

### P1 (Important)
- [ ] Correlation cache freshness (currently 30min)
- [ ] Per-sector parameter optimization
- [ ] Walk-forward backtesting framework
- [ ] Monte Carlo trade sequence simulation

### P2 (Nice to Have)
- [ ] Stochastic slippage model based on ATR
- [ ] WebSocket streaming for real-time updates
- [ ] Email/SMS alerting integration
- [ ] Performance attribution by strategy

## Next Tasks
1. User to provide Alpaca API keys for live testing
2. Run paper trading for 30+ trades validation
3. Implement user's choice on deferred issues
4. Add walk-forward optimization if requested
