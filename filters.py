"""
Smart filters that eliminate losing trades before entry.

These are the "free win rate" improvements - they don't make winners bigger,
they just cut out trades that are statistically more likely to lose.
"""

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils import setup_logger

log = setup_logger("filters")

SECTOR_MAP = {
    "AAPL": "tech",  "MSFT": "tech",  "GOOGL": "tech", "META": "tech",
    "ADBE": "tech",  "CRM":  "tech",  "NOW":   "tech", "INTC": "tech",
    "CSCO": "tech",  "ORCL": "tech",  "IBM":   "tech", "HPQ":  "tech",
    "DELL": "tech",
    "NVDA": "semi",  "AMD":  "semi",  "AVGO": "semi",  "QCOM": "semi",
    "TXN":  "semi",  "MU":   "semi",  "AMAT": "semi",  "LRCX": "semi",
    "KLAC": "semi",  "MRVL": "semi",  "ASML": "semi",  "NXPI": "semi",
    "ON":   "semi",  "SWKS": "semi",  "MPWR": "semi",
    "SNOW": "cloud", "PLTR": "cloud", "DDOG": "cloud", "WDAY": "cloud",
    "TEAM": "cloud", "ZS":   "cloud", "NET":  "cloud", "OKTA": "cloud",
    "VEEV": "cloud", "MDB":  "cloud", "BILL": "cloud", "HUBS": "cloud",
    "GTLB": "cloud", "PATH": "cloud", "DOCU": "cloud",
    "PANW": "cyber", "CRWD": "cyber",
    "XOM":  "energy", "CVX": "energy", "COP": "energy", "SLB": "energy",
    "OXY":  "energy", "HAL": "energy", "EOG": "energy", "MPC": "energy",
    "VLO":  "energy", "PSX": "energy",
    "JPM":  "finance", "BAC":  "finance", "GS":   "finance", "MS":   "finance",
    "WFC":  "finance", "C":    "finance", "V":    "finance", "MA":   "finance",
    "AXP":  "finance", "BLK":  "finance", "SCHW": "finance", "COF":  "finance",
    "PYPL": "finance", "SQ":   "finance", "AFRM": "finance", "COIN": "finance",
    "UNH":  "health", "LLY":  "health", "ABT":  "health", "TMO":  "health",
    "ISRG": "health", "AMGN": "health", "GILD": "health", "VRTX": "health",
    "DHR":  "health", "SYK":  "health", "BSX":  "health", "REGN": "health",
    "BIIB": "health", "DXCM": "health", "IDXX": "health", "PFE":  "health",
    "MRK":  "health", "CVS":  "health", "ELV":  "health",
    "AMZN": "consumer_disc", "TSLA": "consumer_disc", "HD":   "consumer_disc",
    "LOW":  "consumer_disc", "TGT":  "consumer_disc", "NKE":  "consumer_disc",
    "MCD":  "consumer_disc", "SBUX": "consumer_disc", "CMG":  "consumer_disc",
    "YUM":  "consumer_disc", "TJX":  "consumer_disc", "LULU": "consumer_disc",
    "ETSY": "consumer_disc", "NFLX": "consumer_disc", "DIS":  "consumer_disc",
    "ROKU": "consumer_disc", "SNAP": "consumer_disc", "PINS": "consumer_disc",
    "RBLX": "consumer_disc", "DASH": "consumer_disc", "LYFT": "consumer_disc",
    "ABNB": "consumer_disc", "SHOP": "consumer_disc", "MELI": "consumer_disc",
    "ZM":   "consumer_disc", "UBER": "consumer_disc",
    "PEP":  "consumer_stap", "KO":   "consumer_stap", "PG":   "consumer_stap",
    "CL":   "consumer_stap", "MDLZ": "consumer_stap", "PM":   "consumer_stap",
    "MO":   "consumer_stap", "COST": "consumer_stap", "WMT":  "consumer_stap",
    "CAT":  "industrial", "HON": "industrial", "GE":  "industrial",
    "RTX":  "industrial", "BA":  "industrial", "LMT": "industrial",
    "NOC":  "industrial", "UPS": "industrial", "FDX": "industrial",
    "DE":   "industrial", "MMM": "industrial", "CSX": "industrial",
    "NSC":  "industrial", "WM":  "industrial",
    "AMT":  "reit", "EQIX": "reit", "PLD":  "reit",
    "NEE":  "utility", "DUK": "utility", "SO": "utility",
    "BTC/USD":  "crypto", "ETH/USD":  "crypto", "SOL/USD":  "crypto",
    "AVAX/USD": "crypto", "LINK/USD": "crypto", "DOGE/USD": "crypto",
}

SECTOR_ETF_MAP = {
    "tech": "XLK",
    "semi": "XLK",
    "cloud": "XLK",
    "cyber": "XLK",
    "energy": "XLE",
    "finance": "XLF",
    "health": "XLV",
    "consumer_disc": "XLY",
    "consumer_stap": "XLP",
    "industrial": "XLI",
    "reit": "XLRE",
    "utility": "XLU",
    "crypto": None,
}

MAX_PER_SECTOR = 2


@dataclass(frozen=True)
class RegimeGuardDecision:
    mode: str
    size_mult: float
    max_positions: int
    min_score: float
    min_agreeing: int
    reason: str


def compute_regime_guard_decision(
    trades: list[dict],
    config: dict,
    base_max_positions: int | None = None,
    base_min_score: float | None = None,
    base_min_agreeing: int | None = None,
    trading_mode: str | None = None,
) -> RegimeGuardDecision:
    """Return the current portfolio-level regime guard decision.

    This function is intentionally pure so live trading and backtesting can
    share the same guard policy.
    """
    signals_cfg = config.get("signals", {})
    if base_max_positions is None:
        base_max_positions = signals_cfg.get("max_positions", 7)
    if base_min_score is None:
        base_min_score = signals_cfg.get("min_composite_score", 0.23)
    if base_min_agreeing is None:
        base_min_agreeing = signals_cfg.get("min_agreeing_strategies", 3)
    base_max_positions = int(base_max_positions)
    base_min_score = float(base_min_score)
    base_min_agreeing = int(base_min_agreeing)

    def normal(reason: str) -> RegimeGuardDecision:
        return RegimeGuardDecision(
            mode="normal",
            size_mult=1.0,
            max_positions=base_max_positions,
            min_score=base_min_score,
            min_agreeing=base_min_agreeing,
            reason=reason,
        )

    guard_cfg = config.get("filters", {}).get("regime_guard", {})
    if not guard_cfg.get("enabled", True):
        return normal("regime_guard disabled")
    if guard_cfg.get("paper_only", False) and str(trading_mode or "").lower() == "live":
        return normal("regime_guard paper_only in live mode")

    lookback = int(guard_cfg.get("lookback_trades", 20))
    min_trades = int(guard_cfg.get("min_trades", 8))
    recent_wr_trades = int(guard_cfg.get("recent_wr_trades", 5))
    recent = list(trades or [])[-lookback:]
    if len(recent) < min_trades:
        return normal(f"only {len(recent)}/{min_trades} closed trades")

    pnls = [float(t.get("pnl", 0.0) or 0.0) for t in recent]
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    rolling_pf = float("inf") if gross_loss <= 1e-9 else gross_profit / gross_loss

    wr_sample = pnls[-recent_wr_trades:] if len(pnls) >= recent_wr_trades else pnls
    recent_wr = (sum(1 for p in wr_sample if p > 0) / len(wr_sample)) if wr_sample else 1.0

    caution_pf = float(guard_cfg.get("caution_pf", 1.3))
    defensive_pf = float(guard_cfg.get("defensive_pf", 1.0))
    defensive_recent_wr = float(guard_cfg.get("defensive_recent_wr", 0.30))
    reason = (
        f"rolling_pf={rolling_pf:.2f} over {len(recent)} trades, "
        f"recent_wr={recent_wr:.0%} over {len(wr_sample)} trades"
    )

    if rolling_pf < defensive_pf or recent_wr < defensive_recent_wr:
        max_positions = (
            0 if base_max_positions <= 0 else max(
                1,
                math.ceil(base_max_positions * float(guard_cfg.get("defensive_max_positions_mult", 0.50))),
            )
        )
        return RegimeGuardDecision(
            mode="defensive",
            size_mult=float(guard_cfg.get("defensive_size_mult", 0.50)),
            max_positions=max_positions,
            min_score=round(base_min_score + float(guard_cfg.get("defensive_min_score_add", 0.04)), 3),
            min_agreeing=max(
                int(guard_cfg.get("defensive_min_agreeing", 5)),
                base_min_agreeing + int(guard_cfg.get("defensive_min_agreeing_add", 2)),
            ),
            reason=reason,
        )

    if rolling_pf < caution_pf:
        max_positions = (
            0 if base_max_positions <= 0 else max(
                1,
                math.ceil(base_max_positions * float(guard_cfg.get("caution_max_positions_mult", 0.75))),
            )
        )
        return RegimeGuardDecision(
            mode="caution",
            size_mult=float(guard_cfg.get("caution_size_mult", 0.75)),
            max_positions=max_positions,
            min_score=round(base_min_score + float(guard_cfg.get("caution_min_score_add", 0.02)), 3),
            min_agreeing=base_min_agreeing + int(guard_cfg.get("caution_min_agreeing_add", 1)),
            reason=reason,
        )

    return normal(reason)


def sector_cap_for(universe_size: int) -> int:
    """Scale sector cap with universe size. 20 syms -> 2, 35 -> 3, 75 -> 5."""
    return max(2, math.ceil(universe_size / 15))


class SmartFilters:
    def __init__(self, tracker=None, config: dict = None):
        self.tracker = tracker
        self.config = config or {}
        self._corr_cache: dict = {}
        self._corr_cache_time: float = 0
        self._corr_cache_ttl = (config or {}).get("filters", {}).get(
            "correlation_cache_minutes", 30
        ) * 60
        self.corr_size_mult: dict[str, float] = {}

    def filter_gaps(self, opportunities: list, bars: dict,
                    max_gap_pct: float = 0.02) -> list:
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
                log.info(f"GAP FILTER: Skipping {opp.symbol} - gapped {gap_pct:.1%} (limit {max_gap_pct:.0%})")
            else:
                filtered.append(opp)

        return filtered

    def filter_sector_cap(self, opportunities: list,
                          held_symbols: list[str],
                          universe_size: int | None = None) -> list:
        cap = sector_cap_for(universe_size) if universe_size else MAX_PER_SECTOR
        sector_count = {}
        for sym in held_symbols:
            sector = SECTOR_MAP.get(sym, "other")
            sector_count[sector] = sector_count.get(sector, 0) + 1

        filtered = []
        for opp in opportunities:
            sector = SECTOR_MAP.get(opp.symbol, "other")
            current = sector_count.get(sector, 0)
            if current >= cap:
                log.info(f"SECTOR CAP: Skipping {opp.symbol} - already {current} positions in '{sector}' (cap={cap})")
            else:
                filtered.append(opp)
                sector_count[sector] = current + 1

        return filtered

    def filter_correlated(self, candidate_symbols: list[str],
                          held_symbols: list[str],
                          bars: dict[str, pd.DataFrame],
                          max_correlation: float = 0.70,
                          lookback: int = 60) -> list[str]:
        self.corr_size_mult = {}

        if not held_symbols:
            for sym in candidate_symbols:
                self.corr_size_mult[sym] = 1.0
            return candidate_symbols

        max_corr = self.config.get("filters", {}).get("max_correlation", max_correlation)
        hard_block_corr = 0.85
        now = time.time()
        if now - self._corr_cache_time < self._corr_cache_ttl and self._corr_cache:
            corr_matrix = self._corr_cache
        else:
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

            min_len = min(len(v) for v in returns.values())
            aligned = {k: v[-min_len:] for k, v in returns.items()}
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
                    log.info(f"CORRELATION BLOCK: Skipping {sym} (corr={max_pair_corr:.2f} > {hard_block_corr})")
                    continue
                if max_pair_corr > max_corr:
                    reduction = (max_pair_corr - max_corr) / (hard_block_corr - max_corr)
                    mult = max(0.3, 1.0 - reduction * 0.7)
                    self.corr_size_mult[sym] = round(mult, 2)
                    log.info(f"CORRELATION ADJUST: {sym} corr={max_pair_corr:.2f} -> size multiplier={mult:.2f}")
                    filtered.append(sym)
                else:
                    self.corr_size_mult[sym] = 1.0
                    filtered.append(sym)
            else:
                sym_sector = SECTOR_MAP.get(sym, "other")
                sector_held = sum(1 for h in held_symbols if SECTOR_MAP.get(h, "x") == sym_sector)
                if sector_held >= MAX_PER_SECTOR:
                    log.info(f"SECTOR FALLBACK: Skipping {sym} (sector '{sym_sector}' full)")
                    continue
                self.corr_size_mult[sym] = 1.0
                filtered.append(sym)

        return filtered

    def get_corr_size_mult(self, symbol: str) -> float:
        return self.corr_size_mult.get(symbol, 1.0)

    def get_regime_guard_decision(self) -> RegimeGuardDecision:
        trades = self.tracker.trades if self.tracker is not None else []
        trading_mode = os.getenv("TRADING_MODE", "paper").lower()
        return compute_regime_guard_decision(
            trades,
            self.config,
            trading_mode=trading_mode,
        )

    def get_loss_cooldown_mult(self) -> float:
        if self.tracker is None or not self.tracker.trades:
            return 1.0

        consecutive_losses = 0
        for trade in reversed(self.tracker.trades):
            if trade["pnl"] < 0:
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= 5:
            log.warning(f"COOLDOWN: {consecutive_losses} consecutive losses - 25% size")
            return 0.25
        if consecutive_losses >= 3:
            log.warning(f"COOLDOWN: {consecutive_losses} consecutive losses - 50% size")
            return 0.5
        if consecutive_losses >= 2:
            log.warning(f"COOLDOWN: {consecutive_losses} consecutive losses - 75% size")
            return 0.75
        return 1.0

    def get_regime_throttle_mult(self, lookback: int = 20) -> float:
        """Compatibility wrapper for older callers that only use size."""
        return self.get_regime_guard_decision().size_mult
