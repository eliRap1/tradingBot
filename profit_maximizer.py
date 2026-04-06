"""
PROFIT MAXIMIZER - Advanced signal enhancement and profit optimization.

This module adds profit-enhancing features:
1. Volume Profile Analysis - Find high-probability price levels
2. Momentum Persistence - Hold winners longer, cut losers faster
3. Volatility-Adjusted Targets - Bigger targets in high-vol, tighter in low-vol
4. Smart Entry Timing - Wait for pullbacks in trends
5. Sector Rotation - Overweight strongest sectors
6. Gap Analysis - Trade gap-and-go setups
"""

import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from utils import setup_logger

log = setup_logger("profit_maximizer")


class ProfitMaximizer:
    """
    Enhances signals and exits for maximum profitability.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self._sector_cache = {}
        self._sector_cache_time = None
        
        # Sector mappings
        self.SECTORS = {
            "tech": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMD", "INTC", "CSCO", 
                     "AVGO", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "MRVL"],
            "consumer": ["AMZN", "TSLA", "HD", "COST", "PEP", "KO", "DIS", "NFLX"],
            "fintech": ["V", "MA", "PYPL", "SQ", "COIN"],
            "software": ["CRM", "ADBE", "NOW", "PANW", "SNOW", "PLTR"],
            "healthcare": ["UNH"],
            "financials": ["JPM"],
            "travel": ["UBER", "ABNB"],
            "crypto": ["BTC/USD", "ETH/USD"],
        }
    
    def enhance_signal(self, symbol: str, df: pd.DataFrame, 
                       base_score: float, side: str) -> dict:
        """
        Enhance a trading signal with additional profit-maximizing factors.
        
        Returns:
            {
                "enhanced_score": float,  # Modified signal score
                "entry_modifier": str,    # "immediate", "pullback", "breakout"
                "target_multiplier": float,  # Multiply default TP by this
                "stop_multiplier": float,    # Multiply default SL by this
                "size_boost": float,         # 0.8-1.2x position size
                "reasons": list[str],
            }
        """
        enhanced_score = base_score
        entry_mod = "immediate"
        target_mult = 1.0
        stop_mult = 1.0
        size_boost = 1.0
        reasons = []
        
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        
        # 1. VOLUME SURGE DETECTION
        vol_sma = volume.rolling(20).mean()
        current_vol_ratio = volume.iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1.0
        
        if current_vol_ratio > 2.0:
            # Massive volume = institutional activity
            enhanced_score *= 1.25
            size_boost = 1.15
            reasons.append(f"volume_surge_{current_vol_ratio:.1f}x")
        elif current_vol_ratio > 1.5:
            enhanced_score *= 1.1
            reasons.append(f"above_avg_volume_{current_vol_ratio:.1f}x")
        elif current_vol_ratio < 0.7:
            # Low volume = weak conviction
            enhanced_score *= 0.8
            size_boost = 0.85
            reasons.append("low_volume_weak")
        
        # 2. MOMENTUM PERSISTENCE
        roc_5 = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100
        roc_10 = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] * 100
        
        if side == "long":
            if roc_5 > 0 and roc_10 > 0 and roc_5 > roc_10/2:
                # Accelerating momentum
                target_mult = 1.3
                reasons.append("accelerating_momentum")
            elif roc_5 > 3:
                # Strong recent move, wait for pullback
                entry_mod = "pullback"
                reasons.append("wait_pullback")
        else:  # short
            if roc_5 < 0 and roc_10 < 0 and roc_5 < roc_10/2:
                target_mult = 1.3
                reasons.append("accelerating_decline")
            elif roc_5 < -3:
                entry_mod = "pullback"
                reasons.append("wait_bounce_to_short")
        
        # ATR REGIME - Adjust targets to volatility
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        atr_sma = atr.rolling(20).mean().iloc[-1]
        
        if atr.iloc[-1] > atr_sma * 1.3:
            # High volatility regime - wider targets, tighter stops
            target_mult *= 1.2
            stop_mult = 0.9  # Tighter stop in vol (use ATR-based anyway)
            reasons.append("high_vol_regime")
        elif atr.iloc[-1] < atr_sma * 0.7:
            # Low volatility - tighter targets, might breakout soon
            target_mult *= 0.85
            reasons.append("low_vol_compression")
        
        # 4. TREND STRENGTH - Hold stronger trends longer
        adx = ta.trend.ADXIndicator(high, low, close, window=14).adx()
        if adx.iloc[-1] > 35:
            # Strong trend - extend target
            target_mult *= 1.15
            reasons.append(f"strong_trend_adx_{adx.iloc[-1]:.0f}")
        elif adx.iloc[-1] < 20:
            # Weak trend - be conservative
            target_mult *= 0.9
            size_boost *= 0.9
            reasons.append("weak_trend")
        
        # 5. GAP ANALYSIS
        prev_close = close.iloc[-2]
        today_open = df["open"].iloc[-1]
        gap_pct = (today_open - prev_close) / prev_close * 100
        
        if side == "long" and gap_pct > 1.5:
            # Gap up and hold - bullish
            enhanced_score *= 1.15
            reasons.append(f"gap_up_{gap_pct:.1f}%")
        elif side == "long" and gap_pct < -2:
            # Gap down on long signal - could be reversal or more downside
            enhanced_score *= 0.85
            reasons.append("gap_down_caution")
        elif side == "short" and gap_pct < -1.5:
            enhanced_score *= 1.15
            reasons.append(f"gap_down_{gap_pct:.1f}%")
        
        # 6. PRICE LOCATION vs RANGE
        range_20_high = high.rolling(20).max().iloc[-1]
        range_20_low = low.rolling(20).min().iloc[-1]
        price_location = (close.iloc[-1] - range_20_low) / (range_20_high - range_20_low) if range_20_high > range_20_low else 0.5
        
        if side == "long":
            if price_location < 0.3:
                # Near range low - good risk/reward for long
                size_boost *= 1.1
                reasons.append("near_range_low")
            elif price_location > 0.9:
                # Near range high - breakout or fade?
                if adx.iloc[-1] > 25:
                    entry_mod = "breakout"
                    reasons.append("breakout_setup")
                else:
                    enhanced_score *= 0.85
                    reasons.append("near_resistance")
        else:  # short
            if price_location > 0.7:
                size_boost *= 1.1
                reasons.append("near_range_high")
            elif price_location < 0.1:
                enhanced_score *= 0.85
                reasons.append("near_support")
        
        # 7. RSI DIVERGENCE
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        price_higher = close.iloc[-1] > close.iloc[-5]
        rsi_higher = rsi.iloc[-1] > rsi.iloc[-5]
        
        if side == "long" and not price_higher and rsi_higher:
            # Bullish divergence
            enhanced_score *= 1.2
            reasons.append("bullish_divergence")
        elif side == "short" and price_higher and not rsi_higher:
            # Bearish divergence
            enhanced_score *= 1.2
            reasons.append("bearish_divergence")
        
        return {
            "enhanced_score": round(float(enhanced_score), 4),
            "entry_modifier": entry_mod,
            "target_multiplier": round(target_mult, 2),
            "stop_multiplier": round(stop_mult, 2),
            "size_boost": round(size_boost, 2),
            "reasons": reasons,
        }
    
    def get_sector_strength(self, data_fetcher) -> dict:
        """
        Calculate relative sector strength for rotation strategy.
        Overweight strongest sectors, underweight weakest.
        """
        now = datetime.now()
        
        # Cache for 1 hour
        if self._sector_cache_time and (now - self._sector_cache_time).total_seconds() < 3600:
            return self._sector_cache
        
        sector_scores = {}
        
        for sector, symbols in self.SECTORS.items():
            try:
                bars = data_fetcher.get_bars(symbols[:5], timeframe="1Day", days=30)
                if not bars:
                    continue
                
                returns = []
                for sym, df in bars.items():
                    if len(df) >= 20:
                        ret_20d = (df["close"].iloc[-1] - df["close"].iloc[-20]) / df["close"].iloc[-20]
                        returns.append(ret_20d)
                
                if returns:
                    sector_scores[sector] = {
                        "avg_return": np.mean(returns),
                        "symbols": symbols,
                    }
            except Exception as e:
                log.error(f"Sector analysis failed for {sector}: {e}")
        
        # Rank sectors
        sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1]["avg_return"], reverse=True)
        
        for i, (sector, data) in enumerate(sorted_sectors):
            if i < len(sorted_sectors) // 3:
                data["weight"] = 1.2  # Top third: overweight
            elif i >= len(sorted_sectors) * 2 // 3:
                data["weight"] = 0.8  # Bottom third: underweight
            else:
                data["weight"] = 1.0  # Middle: neutral
        
        self._sector_cache = dict(sorted_sectors)
        self._sector_cache_time = now
        
        return self._sector_cache
    
    def get_symbol_sector_weight(self, symbol: str) -> float:
        """Get the sector-based weight multiplier for a symbol."""
        for sector, data in self._sector_cache.items():
            if symbol in data.get("symbols", []):
                return data.get("weight", 1.0)
        return 1.0
    
    def calculate_dynamic_targets(self, symbol: str, df: pd.DataFrame,
                                  entry_price: float, side: str) -> dict:
        """
        Calculate dynamic TP/SL based on support/resistance and volatility.
        
        Returns better risk/reward by finding actual price levels.
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]
        
        # ATR for baseline
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        
        # Find recent swing highs/lows
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(df) - 5):
            # Swing high: higher than 5 bars on each side
            if high.iloc[i] == high.iloc[i-5:i+6].max():
                swing_highs.append(high.iloc[i])
            if low.iloc[i] == low.iloc[i-5:i+6].min():
                swing_lows.append(low.iloc[i])
        
        if side == "long":
            # Target: next resistance (swing high above entry)
            targets_above = [h for h in swing_highs if h > entry_price * 1.01]
            if targets_above:
                natural_target = min(targets_above)
            else:
                natural_target = entry_price + atr * 4
            
            # Stop: below recent swing low
            stops_below = [low_val for low_val in swing_lows if low_val < entry_price * 0.99]
            if stops_below:
                natural_stop = max(stops_below)
            else:
                natural_stop = entry_price - atr * 2
        else:
            # Short
            targets_below = [low_val for low_val in swing_lows if low_val < entry_price * 0.99]
            if targets_below:
                natural_target = max(targets_below)
            else:
                natural_target = entry_price - atr * 4
            
            stops_above = [h for h in swing_highs if h > entry_price * 1.01]
            if stops_above:
                natural_stop = min(stops_above)
            else:
                natural_stop = entry_price + atr * 2
        
        # Calculate risk/reward
        if side == "long":
            risk = entry_price - natural_stop
            reward = natural_target - entry_price
        else:
            risk = natural_stop - entry_price
            reward = entry_price - natural_target
        
        rr_ratio = reward / risk if risk > 0 else 0
        
        return {
            "take_profit": round(natural_target, 2),
            "stop_loss": round(natural_stop, 2),
            "risk_reward": round(rr_ratio, 2),
            "atr": round(atr, 2),
        }


class AdaptiveExitManager:
    """
    Manages exits adaptively based on market conditions.
    """
    
    def __init__(self, config: dict):
        self.config = config
    
    def should_hold_winner(self, position: dict, df: pd.DataFrame) -> dict:
        """
        Determine if we should extend a winning position's target.
        
        Winners should run in trends, but lock profits in choppy markets.
        """
        entry_price = position["entry_price"]
        current_price = position["current_price"]
        side = position.get("side", "long")
        
        if side == "long":
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price
        
        # Not in profit - don't extend
        if profit_pct < 0.02:
            return {"action": "normal", "reason": "not_enough_profit"}
        
        close = df["close"]
        high = df["high"]
        low = df["low"]
        
        # Check trend strength
        adx = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]
        
        # Check momentum
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        
        # Strong trend + room to run = hold
        if adx > 30:
            if side == "long" and rsi < 75:
                return {
                    "action": "extend_target",
                    "reason": f"strong_trend_adx_{adx:.0f}_rsi_{rsi:.0f}",
                    "new_trail_pct": 0.02,  # Tighter trail to lock more profit
                }
            elif side == "short" and rsi > 25:
                return {
                    "action": "extend_target",
                    "reason": f"strong_downtrend",
                    "new_trail_pct": 0.02,
                }
        
        # Overbought/oversold - take profit
        if (side == "long" and rsi > 80) or (side == "short" and rsi < 20):
            return {
                "action": "close_now",
                "reason": f"extreme_rsi_{rsi:.0f}",
            }
        
        # Weak trend - standard exit
        return {"action": "normal", "reason": "standard_exit"}
    
    def should_cut_loser(self, position: dict, df: pd.DataFrame) -> dict:
        """
        Determine if we should close a losing position early.
        
        Cut losers that show no sign of recovery.
        """
        entry_price = position["entry_price"]
        current_price = position["current_price"]
        side = position.get("side", "long")
        
        if side == "long":
            loss_pct = (entry_price - current_price) / entry_price
        else:
            loss_pct = (current_price - entry_price) / entry_price
        
        # Not losing enough to worry
        if loss_pct < 0.01:
            return {"action": "hold", "reason": "small_loss"}
        
        close = df["close"]
        
        # Check if making new lows (for long) or new highs (for short)
        if side == "long":
            making_new_lows = current_price < close.rolling(5).min().iloc[-2]
            if making_new_lows and loss_pct > 0.02:
                return {
                    "action": "close_early",
                    "reason": "making_new_lows_cut_loss",
                }
        else:
            making_new_highs = current_price > close.rolling(5).max().iloc[-2]
            if making_new_highs and loss_pct > 0.02:
                return {
                    "action": "close_early",
                    "reason": "making_new_highs_cut_loss",
                }
        
        return {"action": "hold", "reason": "allow_recovery"}
