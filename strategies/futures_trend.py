"""FuturesTrendStrategy — designed for NQ, ES, CL, GC on 5-min bars.

Signals:
  1. Opening Range Breakout (ORB): first 30-min range; breakout with volume
  2. Session VWAP reclaim: price crosses VWAP from below/above with volume
  3. Trend filter: ADX > 25, EMA 8/21 alignment gating
  4. ATR volatility gate: skip if ATR spike > 2.5× 20-bar average (news/event noise)

Score range: -1.0 to +1.0.
High-conviction filter: only scores with |score| >= 0.40 pass through.
"""
import numpy as np
import pandas as pd
from utils import setup_logger

log = setup_logger("futures_trend")

HIGH_CONVICTION_THRESHOLD = 0.40


class FuturesTrendStrategy:
    def __init__(self, config: dict):
        self.config = config

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Return {symbol: score} for each symbol in data."""
        results = {}
        for symbol, df in data.items():
            try:
                score = self._score(df)
                results[symbol] = score
            except Exception as e:
                log.error(f"FuturesTrend error for {symbol}: {e}")
                results[symbol] = 0.0
        return results

    def _score(self, df: pd.DataFrame) -> float:
        if df is None or len(df) < 50:
            return 0.0

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ── 1. ATR volatility gate ────────────────────────────────
        atr = self._atr(high, low, close, period=14)
        atr_20ma = atr.rolling(20).mean()
        if len(atr) > 0 and len(atr_20ma) > 0:
            latest_atr = atr.iloc[-1]
            avg_atr = atr_20ma.iloc[-1]
            if pd.notna(avg_atr) and avg_atr > 0 and latest_atr > 2.5 * avg_atr:
                return 0.0  # news/event noise — skip

        # ── 2. Trend filter (ADX + EMA alignment) ────────────────
        adx = self._adx(high, low, close, period=14)
        ema8 = close.ewm(span=8).mean()
        ema21 = close.ewm(span=21).mean()

        trend_score = 0.0
        latest_adx = adx.iloc[-1] if len(adx) > 0 else 0
        ema_bullish = ema8.iloc[-1] > ema21.iloc[-1]
        ema_bearish = ema8.iloc[-1] < ema21.iloc[-1]

        if latest_adx > 25:
            if ema_bullish:
                trend_score = 0.40
            elif ema_bearish:
                trend_score = -0.40

        if abs(trend_score) < 0.01:
            return 0.0  # No trend confirmed — don't trade

        # ── 3. Session VWAP reclaim ───────────────────────────────
        vwap = self._session_vwap(df)
        vwap_score = 0.0
        if vwap is not None and len(vwap) >= 3:
            prev_below = close.iloc[-3] < vwap.iloc[-3]
            now_above = close.iloc[-1] > vwap.iloc[-1]
            prev_above = close.iloc[-3] > vwap.iloc[-3]
            now_below = close.iloc[-1] < vwap.iloc[-1]

            vol_avg = volume.rolling(20).mean().iloc[-1]
            vol_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 else 1.0

            if prev_below and now_above and vol_ratio > 1.2:
                vwap_score = 0.30  # bullish VWAP reclaim
            elif prev_above and now_below and vol_ratio > 1.2:
                vwap_score = -0.30  # bearish VWAP breakdown

        # ── 4. Opening Range Breakout (ORB) ───────────────────────
        orb_score = self._orb_score(df, volume)

        # ── Composite score ────────────────────────────────────────
        # Trend direction gates: only add same-direction signal components
        if trend_score > 0:
            total = trend_score + max(0, vwap_score) + max(0, orb_score)
        else:
            total = trend_score + min(0, vwap_score) + min(0, orb_score)

        # Clamp to [-1, 1]
        total = max(-1.0, min(1.0, total))

        # High-conviction filter: return 0 if score doesn't meet threshold
        if abs(total) < HIGH_CONVICTION_THRESHOLD:
            return 0.0

        return round(float(total), 3)

    def _orb_score(self, df: pd.DataFrame, volume: pd.Series) -> float:
        """Opening Range Breakout: 30-min range from session open (first 6 x 5-min bars)."""
        try:
            if not hasattr(df.index, 'date'):
                return 0.0

            today = df.index[-1].date()
            today_bars = df[pd.Series(df.index.date, index=df.index) == today]
            if len(today_bars) < 8:
                return 0.0

            # Opening range = first 6 bars (30 min)
            orb_bars = today_bars.iloc[:6]
            orb_high = orb_bars["high"].max()
            orb_low = orb_bars["low"].min()

            if orb_high <= orb_low:
                return 0.0

            # Current price relative to ORB
            current_close = df["close"].iloc[-1]
            vol_avg = volume.rolling(20).mean().iloc[-1]
            current_vol = volume.iloc[-1]
            vol_confirm = current_vol > 1.5 * vol_avg if vol_avg > 0 else False

            if current_close > orb_high and vol_confirm:
                return 0.30  # bullish ORB breakout
            elif current_close < orb_low and vol_confirm:
                return -0.30  # bearish ORB breakdown
        except Exception:
            pass
        return 0.0

    def _session_vwap(self, df: pd.DataFrame) -> pd.Series | None:
        """Calculate session VWAP from today's open."""
        try:
            if not hasattr(df.index, 'date'):
                return None
            today = df.index[-1].date()
            today_mask = pd.Series(df.index.date, index=df.index) == today
            session_df = df[today_mask]
            if len(session_df) < 3:
                return None
            typical_price = (session_df["high"] + session_df["low"] + session_df["close"]) / 3
            cumvol = session_df["volume"].cumsum()
            cumtpv = (typical_price * session_df["volume"]).cumsum()
            vwap = cumtpv / cumvol
            return vwap
        except Exception:
            return None

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        plus_dm = high - prev_high
        minus_dm = prev_low - low
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()
        return adx.fillna(0)
