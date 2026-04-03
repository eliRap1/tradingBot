"""
Smart filters that eliminate losing trades before entry.

These are the "free win rate" improvements — they don't make winners bigger,
they just cut out trades that are statistically more likely to lose.
"""

import json
import os
from datetime import datetime
from utils import setup_logger

log = setup_logger("filters")

# Sector mapping for correlation filtering (max 2 per sector)
SECTOR_MAP = {
    # Tech
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "META": "tech",
    "ADBE": "tech", "CRM": "tech", "NOW": "tech", "INTC": "tech", "CSCO": "tech",
    # Semiconductors
    "NVDA": "semi", "AMD": "semi", "AVGO": "semi", "QCOM": "semi",
    "TXN": "semi", "MU": "semi", "AMAT": "semi", "LRCX": "semi",
    "KLAC": "semi", "MRVL": "semi",
    # Consumer Internet / E-commerce
    "AMZN": "ecom", "TSLA": "auto", "NFLX": "media", "DIS": "media",
    "UBER": "transport", "ABNB": "travel", "SHOP": "ecom",
    # Fintech / Crypto
    "PYPL": "fintech", "SQ": "fintech", "COIN": "fintech",
    # Cloud / SaaS
    "SNOW": "cloud", "PLTR": "cloud", "PANW": "cyber",
    # Finance
    "JPM": "finance", "V": "finance", "MA": "finance",
    # Healthcare
    "UNH": "health",
    # Consumer
    "HD": "retail", "COST": "retail", "PEP": "consumer", "KO": "consumer",
}

CONFIRMATION_FILE = os.path.join(os.path.dirname(__file__), "pending_signals.json")
MAX_PER_SECTOR = 2


class SmartFilters:
    def __init__(self, tracker=None):
        self.tracker = tracker
        self._pending_signals = self._load_pending()

    # ═══════════════════════════════════════════════════════════
    # Confirmation bar filter
    # ═══════════════════════════════════════════════════════════

    def filter_confirmed(self, opportunities: list, bars: dict) -> list:
        """
        Only allow entries for symbols that had a signal in the PREVIOUS cycle too.
        This filters ~30-40% of false signals. First time a signal appears, it gets
        saved but not traded. Second time = confirmed.
        """
        confirmed = []
        new_pending = {}

        for opp in opportunities:
            sym = opp.symbol
            new_pending[sym] = {
                "score": opp.score,
                "timestamp": datetime.now().isoformat()
            }

            if sym in self._pending_signals:
                # Signal appeared before — it's confirmed
                confirmed.append(opp)
                log.info(f"CONFIRMED: {sym} (signal persisted across cycles)")
            else:
                log.info(f"PENDING: {sym} score={opp.score:.3f} — waiting for confirmation")

        # Save current signals for next cycle
        self._pending_signals = new_pending
        self._save_pending(new_pending)

        return confirmed

    # ═══════════════════════════════════════════════════════════
    # Gap filter
    # ═══════════════════════════════════════════════════════════

    def filter_gaps(self, opportunities: list, bars: dict,
                    max_gap_pct: float = 0.02) -> list:
        """
        Skip stocks that gapped > max_gap_pct from previous close.
        Gaps create unpredictable price action — the move is already priced in.
        """
        filtered = []
        for opp in opportunities:
            if opp.symbol not in bars:
                filtered.append(opp)
                continue

            df = bars[opp.symbol]
            if len(df) < 2 or "open" not in df.columns:
                filtered.append(opp)
                continue

            prev_close = df["close"].iloc[-2]
            today_open = df["open"].iloc[-1]

            if prev_close == 0:
                filtered.append(opp)
                continue

            gap_pct = abs(today_open - prev_close) / prev_close

            if gap_pct > max_gap_pct:
                log.info(f"GAP FILTER: Skipping {opp.symbol} — "
                         f"gapped {gap_pct:.1%} (limit {max_gap_pct:.0%})")
            else:
                filtered.append(opp)

        return filtered

    # ═══════════════════════════════════════════════════════════
    # Sector correlation cap
    # ═══════════════════════════════════════════════════════════

    def filter_sector_cap(self, opportunities: list,
                          held_symbols: list[str]) -> list:
        """
        Max 2 positions per sector. Correlated positions act as one big bet.
        """
        # Count current sector exposure
        sector_count = {}
        for sym in held_symbols:
            sector = SECTOR_MAP.get(sym, "other")
            sector_count[sector] = sector_count.get(sector, 0) + 1

        filtered = []
        for opp in opportunities:
            sector = SECTOR_MAP.get(opp.symbol, "other")
            current = sector_count.get(sector, 0)

            if current >= MAX_PER_SECTOR:
                log.info(f"SECTOR CAP: Skipping {opp.symbol} — "
                         f"already {current} positions in '{sector}'")
            else:
                filtered.append(opp)
                sector_count[sector] = current + 1

        return filtered

    # ═══════════════════════════════════════════════════════════
    # Consecutive loss cooldown
    # ═══════════════════════════════════════════════════════════

    def get_loss_cooldown_mult(self) -> float:
        """
        Reduce position size after consecutive losses.
        3 losses = 50% size, 5+ = 25% size.
        Returns a multiplier (0.25 to 1.0).
        """
        if self.tracker is None or not self.tracker.trades:
            return 1.0

        # Count consecutive losses from most recent
        consecutive_losses = 0
        for trade in reversed(self.tracker.trades):
            if trade["pnl"] < 0:
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= 5:
            log.warning(f"COOLDOWN: {consecutive_losses} consecutive losses — 25% size")
            return 0.25
        elif consecutive_losses >= 3:
            log.warning(f"COOLDOWN: {consecutive_losses} consecutive losses — 50% size")
            return 0.5
        return 1.0

    # ═══════════════════════════════════════════════════════════
    # Persistence
    # ═══════════════════════════════════════════════════════════

    def _load_pending(self) -> dict:
        if not os.path.exists(CONFIRMATION_FILE):
            return {}
        try:
            with open(CONFIRMATION_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_pending(self, data: dict):
        try:
            with open(CONFIRMATION_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save pending signals: {e}")
