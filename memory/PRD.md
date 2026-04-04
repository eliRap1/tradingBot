# Trading Bot PRD - LIVE TRADING Edition

## Original Problem Statement
Build the most profitable trading bot for stocks and crypto using Alpaca API that:
- Trades LIVE with real-time data (not historical/stale)
- Uses walk-forward optimized parameters
- Maximizes profit with robust risk management

## Tech Stack
- **Backend**: Python 
- **Broker API**: Alpaca (paper & live trading)
- **Dashboard**: Flask + HTML/JS (TradingView charts)
- **Storage**: JSON files for state persistence

## Architecture

### Core Components
1. **Coordinator** - Main orchestrator with live trading validation
2. **LiveTradingManager** - Real-time data validation, market hours
3. **WalkForwardOptimizer** - Parameter optimization framework
4. **ProfitMaximizer** - Signal enhancement and adaptive exits
5. **7 Trading Strategies** - SuperTrend, Momentum, Mean Reversion, Breakout, Stoch RSI, VWAP, Gap

### Live Trading Flow
```
Market Open Check → Fresh Price Validation → Signal Generation → 
Profit Enhancement → Risk Sizing → Order Execution → Position Management
```

## What's Been Implemented (2026-01-27)

### Session 1: Bug Fixes
- ✅ Crypto OCO tracking
- ✅ Partial fill handling  
- ✅ API rate limiting
- ✅ Kelly criterion fix
- ✅ Memory leak fix

### Session 2: Profit Maximization
- ✅ Gap Trading Strategy
- ✅ Enhanced Momentum signals
- ✅ Volatility-adjusted sizing
- ✅ Two-tier partial exits
- ✅ Aggressive config tuning

### Session 3: Live Trading & Optimization
- ✅ **LiveTradingManager** - Market hours awareness
- ✅ **DataFreshnessValidator** - Rejects stale price data
- ✅ **WalkForwardOptimizer** - 6mo train / 2mo test rolling windows
- ✅ **Auto-apply optimized params** - Bot uses best parameters automatically
- ✅ **Startup validation** - Checks API keys, trading mode, market status

## Live Trading Features

### Market Hours Handling
| Asset | Trading Hours |
|-------|---------------|
| US Stocks | 9:30 AM - 4:00 PM ET (regular), skip first/last 15 min |
| Crypto | 24/7 (always tradeable) |

### Data Freshness Rules
| Timeframe | Max Age Before Stale |
|-----------|---------------------|
| 1Min bars | 2 minutes |
| 5Min bars | 10 minutes |
| Stock prices | 60 seconds |
| Crypto prices | 30 seconds |

### Automatic Safeguards
- Skip opening 15 min (high volatility)
- Skip closing 15 min (wide spreads)
- Validate price freshness before every trade
- Check market status before stock trades

## Walk-Forward Optimization

### How It Works
1. Train on 6 months of historical data
2. Test parameters on next 2 months (out-of-sample)
3. Roll forward 1 month, repeat
4. Only use parameters that work BOTH in-sample AND out-of-sample

### Parameters Optimized
- `min_composite_score`: 0.15 - 0.30
- `min_agreeing_strategies`: 2 - 3
- `stop_loss_atr_mult`: 1.0 - 2.0
- `take_profit_atr_mult`: 3.0 - 6.0
- `trailing_stop_pct`: 0.02 - 0.04
- `max_portfolio_risk_pct`: 0.015 - 0.025

### Running Optimization
```bash
python walk_forward_optimizer.py --start 2024-01-01 --end 2025-12-31
```

## Configuration Summary

| Category | Key Settings |
|----------|--------------|
| Positions | Max 12 stocks + 2 crypto |
| Risk | 2% per trade, 12% max drawdown |
| Entries | 2+ strategies agree, 0.20+ score |
| Exits | 1.2R partial (40%), 2.5R partial (30%), trail rest |
| Sizing | Volatility-adjusted (0.6x-1.4x) |
| Live Trading | 60s max price age, market hours enforced |

## Files Created

### New Files
- `live_trading.py` - Market hours, price freshness validation
- `walk_forward_optimizer.py` - Parameter optimization framework
- `profit_maximizer.py` - Signal enhancement
- `strategies/gap.py` - Gap trading strategy

### Modified Files
- `main.py` - Startup validation
- `coordinator.py` - Live trading integration
- `config.yaml` - Optimized settings
- `risk.py` - Volatility-adjusted sizing
- `portfolio.py` - Two-tier partial exits

## Running the Bot

### Quick Start
```bash
# Set environment variables
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
export TRADING_MODE=paper  # or 'live' for real money

# Run the bot
python main.py
```

### With Optimization (Recommended)
```bash
# First, optimize parameters (takes ~30 min)
python walk_forward_optimizer.py

# Then run with optimized params
python main.py
```

### Dashboard
```bash
python dashboard.py
# Open http://localhost:5000
```

## Prioritized Backlog

### P0 (Completed)
- ✅ Bug fixes
- ✅ Profit maximization
- ✅ Live trading validation
- ✅ Walk-forward optimization

### P1 (Next)
- [ ] Email/SMS alerts
- [ ] WebSocket streaming
- [ ] Performance attribution

### P2 (Future)
- [ ] Machine learning signals
- [ ] Options integration
- [ ] Multi-broker support

## Testing Requirements

### Before Live Trading
1. Paper trade for 50+ trades
2. Verify win rate > 50%
3. Verify profit factor > 1.5
4. Verify max drawdown < 10%
5. Verify live price validation works
6. Run during market hours AND outside

## Important Notes

⚠️ **Always paper trade first!**
⚠️ **The bot validates live prices** - stale data is rejected
⚠️ **Market hours enforced** - stocks won't trade outside hours
⚠️ **Crypto trades 24/7** - no market hour restrictions
