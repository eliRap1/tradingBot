import pandas as pd
import numpy as np
import ta
from dataclasses import dataclass
from utils import setup_logger
from state import load_state, save_state

log = setup_logger("risk")


@dataclass
class SizedOrder:
    symbol: str
    qty: int
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    score: float
    size_boost: float = 1.0  # From profit maximizer


class RiskManager:
    def __init__(self, config: dict):
        self._config = config
        self.cfg = config["risk"]
        # Restore peak equity from persisted state
        state = load_state()
        self.peak_equity = state.get("peak_equity", 0.0)
        self.daily_pnl = 0.0
        self.starting_equity = 0.0
        
        # Track recent volatility for adaptive sizing
        self._vol_cache = {}

    def set_starting_equity(self, equity: float):
        """Call at start of day to track daily loss limit."""
        if self.starting_equity == 0:
            self.starting_equity = equity

    def size_orders(self, opportunities, bars: dict[str, pd.DataFrame],
                    prices: dict[str, float], equity: float,
                    num_existing: int,
                    regime_size_mult: float = 1.0,
                    tracker_stats: dict = None,
                    profit_enhancements: dict = None) -> list[SizedOrder]:
        """
        Convert opportunities into sized orders with stops.

        Now includes:
        - Risk:Reward filter (minimum 2.5:1)
        - Regime-adjusted position sizing
        - Volatility-adjusted sizing (inverse vol)
        - Daily loss limit check
        - Profit maximizer size boosts
        """
        orders = []
        max_new = max(0, self.cfg.get("max_positions", 12) - num_existing)
        profit_enhancements = profit_enhancements or {}

        # Daily loss limit check
        if self.starting_equity > 0:
            daily_loss_pct = (self.starting_equity - equity) / self.starting_equity
            daily_limit = self.cfg.get("daily_loss_limit_pct", 0.025)
            if daily_loss_pct >= daily_limit:
                log.warning(f"DAILY LOSS LIMIT HIT ({daily_loss_pct:.1%}) — no new trades")
                return []

        for opp in opportunities[:max_new]:
            if opp.symbol not in prices or opp.symbol not in bars:
                continue

            entry_price = prices[opp.symbol]
            df = bars[opp.symbol]

            # Calculate ATR for stop/target placement
            atr = self._get_atr(df)
            if atr is None or atr <= 0:
                continue

            # Direction: positive score = long, negative = short
            is_short = opp.direction == "sell" or opp.score < 0

            # Use crypto-specific risk params if applicable
            from broker import CRYPTO_SYMBOLS
            crypto_cfg = self.cfg
            if opp.symbol in CRYPTO_SYMBOLS:
                # Merge crypto overrides from screener.crypto_risk
                crypto_overrides = self._config.get("screener", {}).get("crypto_risk", {})
                if crypto_overrides:
                    crypto_cfg = {**self.cfg, **crypto_overrides}

            # Get profit enhancement for this symbol
            enhancement = profit_enhancements.get(opp.symbol, {})
            target_mult = enhancement.get("target_multiplier", 1.0)
            stop_mult = enhancement.get("stop_multiplier", 1.0)
            size_boost = enhancement.get("size_boost", 1.0)

            if is_short:
                # SHORT: stop above entry, target below entry
                stop_loss = entry_price + (atr * crypto_cfg["stop_loss_atr_mult"] * stop_mult)
                take_profit = entry_price - (atr * crypto_cfg["take_profit_atr_mult"] * target_mult)
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
                side = "sell"
            else:
                # LONG: stop below entry, target above entry
                stop_loss = entry_price - (atr * crypto_cfg["stop_loss_atr_mult"] * stop_mult)
                take_profit = entry_price + (atr * crypto_cfg["take_profit_atr_mult"] * target_mult)
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
                side = "buy"

            # ── Risk:Reward filter ───────────────────────────
            # Use crypto-specific min R:R if available
            min_rr = crypto_cfg.get("min_risk_reward", self.cfg.get("min_risk_reward", 2.5))
            if risk <= 0:
                continue
            rr_ratio = reward / risk
            if rr_ratio < min_rr:
                log.info(f"Skipping {opp.symbol}: R:R={rr_ratio:.1f} < {min_rr} minimum")
                continue

            # ── Position sizing ───────────────────────────────
            risk_per_share = risk
            sizing_method = self.cfg.get("sizing_method", "volatility_adjusted")

            if sizing_method == "volatility_adjusted":
                # Volatility-adjusted sizing: smaller positions for volatile stocks
                vol_factor = self._get_volatility_factor(df, entry_price)
                base_risk_pct = self.cfg["max_portfolio_risk_pct"]
                # Adjust risk based on volatility (inverse relationship)
                adjusted_risk_pct = base_risk_pct * vol_factor
                max_risk_dollars = equity * adjusted_risk_pct
                log.debug(f"{opp.symbol}: vol_factor={vol_factor:.2f} risk_pct={adjusted_risk_pct:.3f}")
                
            elif sizing_method == "kelly" and tracker_stats:
                # Kelly Criterion: f* = p - q/b
                kelly_min_trades = self.cfg.get("kelly_min_trades", 30)
                total_trades = tracker_stats.get("total_trades", 0)

                if total_trades >= kelly_min_trades:
                    p = tracker_stats.get("win_pct", 50) / 100
                    avg_win = abs(tracker_stats.get("avg_win", 1))
                    avg_loss = abs(tracker_stats.get("avg_loss", 1))
                    
                    # Cap the win/loss ratio to prevent extreme sizing
                    b = avg_win / avg_loss if avg_loss > 0 else 1.0
                    b = min(b, 5.0)
                    
                    kelly_f = p - ((1 - p) / b) if b > 0 else 0
                    kelly_f *= self.cfg.get("kelly_fraction", 0.5)
                    kelly_f = max(0.005, min(kelly_f, self.cfg["max_portfolio_risk_pct"]))

                    max_risk_dollars = equity * kelly_f
                    log.info(f"Kelly sizing: f={kelly_f:.4f} (p={p:.2f} b={b:.2f})")
                else:
                    max_risk_dollars = equity * self.cfg["max_portfolio_risk_pct"]
            else:
                max_risk_dollars = equity * self.cfg["max_portfolio_risk_pct"]

            qty_by_risk = int(max_risk_dollars / risk_per_share)

            max_position_dollars = equity * self.cfg["max_position_pct"]
            qty_by_size = int(max_position_dollars / entry_price)

            qty = min(qty_by_risk, qty_by_size)

            # Apply regime size multiplier
            qty = int(qty * regime_size_mult)
            
            # Apply profit maximizer size boost
            qty = int(qty * size_boost)

            if qty <= 0:
                continue

            orders.append(SizedOrder(
                symbol=opp.symbol,
                qty=qty,
                side=side,
                entry_price=entry_price,
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                score=opp.score,
                size_boost=size_boost
            ))

            log.info(
                f"Sized: {side.upper()} {opp.symbol} qty={qty} entry={entry_price:.2f} "
                f"SL={stop_loss:.2f} TP={take_profit:.2f} "
                f"R:R={rr_ratio:.1f} risk=${risk_per_share * qty:.2f}"
                + (f" boost={size_boost:.2f}" if size_boost != 1.0 else "")
            )

        return orders
    
    def _get_volatility_factor(self, df: pd.DataFrame, price: float) -> float:
        """
        Calculate inverse volatility factor for position sizing.
        
        Returns 0.6-1.4 multiplier:
        - High vol stocks: smaller factor (0.6-0.8) = smaller positions
        - Low vol stocks: higher factor (1.2-1.4) = larger positions
        """
        if len(df) < 20:
            return 1.0
        
        # Calculate 20-day volatility (annualized)
        returns = df["close"].pct_change().dropna()
        if len(returns) < 10:
            return 1.0
        
        vol_20d = returns.tail(20).std() * np.sqrt(252)  # Annualized
        
        # Target volatility for "average" stock
        target_vol = 0.30  # 30% annualized is typical
        
        # Inverse relationship: high vol = smaller size, low vol = larger size
        # Clamp between 0.6 and 1.4
        vol_factor = target_vol / vol_20d if vol_20d > 0 else 1.0
        vol_factor = max(0.6, min(1.4, vol_factor))
        
        return vol_factor

    def check_drawdown(self, current_equity: float) -> bool:
        """Returns True if max drawdown exceeded (should stop trading)."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            # Persist new peak
            state = load_state()
            state["peak_equity"] = self.peak_equity
            save_state(state)

        if self.peak_equity == 0:
            return False

        drawdown = (self.peak_equity - current_equity) / self.peak_equity

        if drawdown >= self.cfg["max_drawdown_pct"]:
            log.critical(
                f"MAX DRAWDOWN BREACHED: {drawdown:.1%} "
                f"(peak={self.peak_equity:.2f}, current={current_equity:.2f})"
            )
            return True

        if drawdown > self.cfg["max_drawdown_pct"] * 0.7:
            log.warning(f"Drawdown warning: {drawdown:.1%}")

        return False

    def _get_atr(self, df: pd.DataFrame, period: int = 14) -> float | None:
        if len(df) < period + 1:
            return None
        atr = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"],
            window=period
        ).average_true_range()
        return atr.iloc[-1]
