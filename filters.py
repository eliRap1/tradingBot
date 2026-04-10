"""
Smart filters that eliminate losing trades before entry.

These are the "free win rate" improvements — they don't make winners bigger,
they just cut out trades that are statistically more likely to lose.
"""

import json
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from utils import setup_logger

log = setup_logger("filters")

# Sector mapping for correlation filtering (max 2 per sector)
# Labels aligned with ETF proxies used by SectorRegimeFilter.
SECTOR_MAP = {
    # ── Tech / Mega-cap (XLK) ─────────────────────────────────
    "AAPL": "tech",  "MSFT": "tech",  "GOOGL": "tech", "META": "tech",
    "ADBE": "tech",  "CRM":  "tech",  "NOW":   "tech", "INTC": "tech",
    "CSCO": "tech",  "ORCL": "tech",  "IBM":   "tech", "HPQ":  "tech",
    "DELL": "tech",
    # ── Semiconductors (XLK) ──────────────────────────────────
    "NVDA": "semi",  "AMD":  "semi",  "AVGO": "semi",  "QCOM": "semi",
    "TXN":  "semi",  "MU":   "semi",  "AMAT": "semi",  "LRCX": "semi",
    "KLAC": "semi",  "MRVL": "semi",  "ASML": "semi",  "NXPI": "semi",
    "ON":   "semi",  "SWKS": "semi",  "MPWR": "semi",
    # ── Cloud / SaaS (XLK) ────────────────────────────────────
    "SNOW": "cloud", "PLTR": "cloud", "DDOG": "cloud", "WDAY": "cloud",
    "TEAM": "cloud", "ZS":   "cloud", "NET":  "cloud", "OKTA": "cloud",
    "VEEV": "cloud", "MDB":  "cloud", "BILL": "cloud", "HUBS": "cloud",
    "GTLB": "cloud", "PATH": "cloud", "DOCU": "cloud",
    # ── Cybersecurity (XLK) ───────────────────────────────────
    "PANW": "cyber", "CRWD": "cyber",
    # ── Energy (XLE) ──────────────────────────────────────────
    "XOM":  "energy", "CVX": "energy", "COP": "energy", "SLB": "energy",
    "OXY":  "energy", "HAL": "energy", "EOG": "energy", "MPC": "energy",
    "VLO":  "energy", "PSX": "energy",
    # ── Financials (XLF) ──────────────────────────────────────
    "JPM":  "finance", "BAC":  "finance", "GS":   "finance", "MS":   "finance",
    "WFC":  "finance", "C":    "finance", "V":    "finance", "MA":   "finance",
    "AXP":  "finance", "BLK":  "finance", "SCHW": "finance", "COF":  "finance",
    "PYPL": "finance", "SQ":   "finance", "AFRM": "finance", "COIN": "finance",
    # ── Healthcare (XLV) ──────────────────────────────────────
    "UNH":  "health", "LLY":  "health", "ABT":  "health", "TMO":  "health",
    "ISRG": "health", "AMGN": "health", "GILD": "health", "VRTX": "health",
    "DHR":  "health", "SYK":  "health", "BSX":  "health", "REGN": "health",
    "BIIB": "health", "DXCM": "health", "IDXX": "health", "PFE":  "health",
    "MRK":  "health", "CVS":  "health", "ELV":  "health",
    # ── Consumer Discretionary (XLY) ──────────────────────────
    "AMZN": "consumer_disc", "TSLA": "consumer_disc", "HD":   "consumer_disc",
    "LOW":  "consumer_disc", "TGT":  "consumer_disc", "NKE":  "consumer_disc",
    "MCD":  "consumer_disc", "SBUX": "consumer_disc", "CMG":  "consumer_disc",
    "YUM":  "consumer_disc", "TJX":  "consumer_disc", "LULU": "consumer_disc",
    "ETSY": "consumer_disc", "NFLX": "consumer_disc", "DIS":  "consumer_disc",
    "ROKU": "consumer_disc", "SNAP": "consumer_disc", "PINS": "consumer_disc",
    "RBLX": "consumer_disc", "DASH": "consumer_disc", "LYFT": "consumer_disc",
    "ABNB": "consumer_disc", "SHOP": "consumer_disc", "MELI": "consumer_disc",
    "ZM":   "consumer_disc", "UBER": "consumer_disc",
    # ── Consumer Staples (XLP) ────────────────────────────────
    "PEP":  "consumer_stap", "KO":   "consumer_stap", "PG":   "consumer_stap",
    "CL":   "consumer_stap", "MDLZ": "consumer_stap", "PM":   "consumer_stap",
    "MO":   "consumer_stap", "COST": "consumer_stap", "WMT":  "consumer_stap",
    # ── Industrials (XLI) ─────────────────────────────────────
    "CAT":  "industrial", "HON": "industrial", "GE":  "industrial",
    "RTX":  "industrial", "BA":  "industrial", "LMT": "industrial",
    "NOC":  "industrial", "UPS": "industrial", "FDX": "industrial",
    "DE":   "industrial", "MMM": "industrial", "CSX": "industrial",
    "NSC":  "industrial", "WM":  "industrial",
    # ── REITs (XLRE) ──────────────────────────────────────────
    "AMT":  "reit", "EQIX": "reit", "PLD":  "reit",
    # ── Utilities (XLU) ───────────────────────────────────────
    "NEE":  "utility", "DUK": "utility", "SO": "utility",
    # ── Crypto (BTC/USD direct proxy) ─────────────────────────
    "BTC/USD":  "crypto", "ETH/USD":  "crypto", "SOL/USD":  "crypto",
    "AVAX/USD": "crypto", "LINK/USD": "crypto", "DOGE/USD": "crypto",
}

# Maps internal sector label → ETF ticker used by SectorRegimeFilter
SECTOR_ETF_MAP = {
    "tech":          "XLK",
    "semi":          "XLK",
    "cloud":         "XLK",
    "cyber":         "XLK",
    "energy":        "XLE",
    "finance":       "XLF",
    "health":        "XLV",
    "consumer_disc": "XLY",
    "consumer_stap": "XLP",
    "industrial":    "XLI",
    "reit":          "XLRE",
    "utility":       "XLU",
    "crypto":        None,   # crypto handled via BTC/USD directly
}

CONFIRMATION_FILE = os.path.join(os.path.dirname(__file__), "pending_signals.json")
MAX_PER_SECTOR = 2


class SmartFilters:
    def __init__(self, tracker=None, config: dict = None):
        self.tracker = tracker
        self.config = config or {}
        self._pending_signals = self._load_pending()

        # Correlation cache
        self._corr_cache: dict = {}
        self._corr_cache_time: float = 0
        self._corr_cache_ttl = (config or {}).get("filters", {}).get(
            "correlation_cache_minutes", 30) * 60

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
    # Dynamic correlation filter
    # ═══════════════════════════════════════════════════════════

    def filter_correlated(self, candidate_symbols: list[str],
                          held_symbols: list[str],
                          bars: dict[str, pd.DataFrame],
                          max_correlation: float = 0.70,
                          lookback: int = 60) -> list[str]:
        """
        Correlation-adjusted filter: instead of binary block, scales position
        size based on correlation with held positions.

        Returns candidates that pass (with size adjustments stored in self.corr_size_mult).
        Hard block only at >0.85 correlation.
        """
        self.corr_size_mult = {}  # {symbol: multiplier 0.0-1.0}

        if not held_symbols:
            for sym in candidate_symbols:
                self.corr_size_mult[sym] = 1.0
            return candidate_symbols

        max_corr = self.config.get("filters", {}).get(
            "max_correlation", max_correlation)
        hard_block_corr = 0.85  # Only hard-block above this

        # Check cache
        now = time.time()
        if now - self._corr_cache_time < self._corr_cache_ttl and self._corr_cache:
            corr_matrix = self._corr_cache
        else:
            # Build returns matrix
            all_syms = list(set(candidate_symbols + held_symbols))
            returns = {}
            for sym in all_syms:
                if sym in bars and bars[sym] is not None and len(bars[sym]) >= lookback:
                    closes = bars[sym]["close"].tail(lookback)
                    ret = closes.pct_change().dropna()
                    if len(ret) >= lookback - 5:
                        returns[sym] = ret.values[:lookback - 1]

            if len(returns) < 2:
                for sym in candidate_symbols:
                    self.corr_size_mult[sym] = 1.0
                return candidate_symbols

            # Align lengths
            min_len = min(len(v) for v in returns.values())
            aligned = {k: v[-min_len:] for k, v in returns.items()}

            # Calculate correlation matrix
            syms = list(aligned.keys())
            matrix = np.array([aligned[s] for s in syms])
            corr = np.corrcoef(matrix)

            corr_matrix = {}
            for i, s1 in enumerate(syms):
                for j, s2 in enumerate(syms):
                    if i != j:
                        corr_matrix[(s1, s2)] = corr[i][j]

            self._corr_cache = corr_matrix
            self._corr_cache_time = now

        # Filter candidates with proportional sizing
        filtered = []
        for sym in candidate_symbols:
            max_pair_corr = 0.0
            has_data = False

            for held in held_symbols:
                pair_corr = corr_matrix.get((sym, held), None)
                if pair_corr is not None:
                    has_data = True
                    max_pair_corr = max(max_pair_corr, abs(pair_corr))

            if has_data:
                if max_pair_corr > hard_block_corr:
                    log.info(f"CORRELATION BLOCK: Skipping {sym} "
                             f"(corr={max_pair_corr:.2f} > {hard_block_corr})")
                    continue
                elif max_pair_corr > max_corr:
                    # Proportional reduction: 0.70 corr = 70% size, 0.80 = 50% size
                    reduction = (max_pair_corr - max_corr) / (hard_block_corr - max_corr)
                    mult = max(0.3, 1.0 - reduction * 0.7)
                    self.corr_size_mult[sym] = round(mult, 2)
                    log.info(f"CORRELATION ADJUST: {sym} corr={max_pair_corr:.2f} "
                             f"→ size multiplier={mult:.2f}")
                    filtered.append(sym)
                else:
                    self.corr_size_mult[sym] = 1.0
                    filtered.append(sym)
            else:
                # Fallback: sector check
                sym_sector = SECTOR_MAP.get(sym, "other")
                sector_held = sum(
                    1 for h in held_symbols
                    if SECTOR_MAP.get(h, "x") == sym_sector
                )
                if sector_held >= MAX_PER_SECTOR:
                    log.info(f"SECTOR FALLBACK: Skipping {sym} "
                             f"(sector '{sym_sector}' full)")
                    continue
                self.corr_size_mult[sym] = 1.0
                filtered.append(sym)

        return filtered

    def get_corr_size_mult(self, symbol: str) -> float:
        """Get correlation-adjusted size multiplier for a symbol."""
        return self.corr_size_mult.get(symbol, 1.0)

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
