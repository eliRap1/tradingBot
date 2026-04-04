# Trading Bot - Quick Start Guide

## What This Bot Does

This is a **real-time algorithmic trading bot** for stocks and crypto using Alpaca API.

**Key Features:**
- Trades at **LIVE market prices** (rejects stale data)
- 7 trading strategies with confluence voting
- Volatility-adjusted position sizing
- Two-tier partial profit taking (40% at 1.2R, 30% at 2.5R)
- Market hours awareness (stocks 9:30-4:00 ET, crypto 24/7)
- Max 12% drawdown protection

---

## Quick Start (Your Local Machine)

### 1. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# .env
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
TRADING_MODE=paper
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Bot

```bash
python main.py
```

The bot will:
- Connect to Alpaca
- Check market status
- Start watchers for 40+ stocks and 2 crypto
- Trade in real-time when signals align

### 4. Monitor via Dashboard

```bash
python dashboard.py
```
Open http://localhost:5000

---

## Live Trading Safeguards

The bot **automatically ensures real-time trading**:

| Check | Description |
|-------|-------------|
| **Price Freshness** | Rejects prices older than 60s (stocks) / 30s (crypto) |
| **Market Hours** | Stocks only trade 9:30 AM - 4:00 PM ET |
| **Opening Skip** | Skips first 15 min (high volatility) |
| **Closing Skip** | Skips last 15 min (wide spreads) |
| **Crypto 24/7** | No restrictions on BTC/ETH |

---

## Optional: Walk-Forward Optimization

Run this to find optimal parameters (takes ~30 min):

```bash
python walk_forward_optimizer.py
```

The bot will automatically use optimized params if available.

---

## Configuration (config.yaml)

Key settings you can tune:

```yaml
signals:
  min_composite_score: 0.20    # Lower = more trades
  min_agreeing_strategies: 2   # 2-3 strategies must agree

risk:
  max_position_pct: 0.10       # 10% max per position
  max_portfolio_risk_pct: 0.02 # 2% risk per trade
  stop_loss_atr_mult: 1.5      # Tighter stops
  take_profit_atr_mult: 5.0    # Let winners run
  max_drawdown_pct: 0.12       # 12% max drawdown halts bot
```

---

## Troubleshooting

### "No API keys found"
- Make sure `.env` file exists in the project root
- Check the keys are correct (no extra spaces)

### "Market closed"
- Stocks only trade during US market hours
- Crypto trades 24/7

### "Stale price rejected"
- This is normal - the bot skips trades when prices are too old
- Ensures you never trade on outdated data

---

## Files Overview

| File | Purpose |
|------|---------|
| `main.py` | Entry point |
| `coordinator.py` | Main orchestrator |
| `live_trading.py` | Real-time validation |
| `profit_maximizer.py` | Signal enhancement |
| `strategies/` | 7 trading strategies |
| `dashboard.py` | Web monitoring UI |
| `config.yaml` | All settings |

---

## Support

Check logs at:
- Console output
- `logs/` directory (if configured)

Dashboard shows real-time:
- Account equity
- Open positions
- Recent signals
- Strategy performance
