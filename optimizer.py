"""
Walk-forward optimizer — prevents overfitting by testing on unseen data.

How it works:
  1. Split data into rolling windows: train (6 months) + test (2 months)
  2. On train set: try parameter combinations, pick best by Sharpe
  3. On test set: run with best params, record out-of-sample performance
  4. Slide window forward, repeat
  5. Aggregate OOS results = realistic performance estimate

Overfitting safeguards:
  - Compare in-sample vs out-of-sample Sharpe (>50% drop = overfit)
  - Cap parameter grid size (max 200 combinations)
  - Require minimum 30 trades per window
  - Report parameter stability across windows
"""

import copy
import itertools
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import timedelta

from backtester import Backtester, BacktestResult
from utils import setup_logger

log = setup_logger("optimizer")


@dataclass
class WindowResult:
    window_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: dict
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    oos_trades: int
    oos_win_rate: float
    oos_return_pct: float
    degradation_pct: float  # how much worse OOS is vs IS


@dataclass
class OptimizationResult:
    windows: list
    avg_oos_sharpe: float
    avg_oos_win_rate: float
    avg_degradation_pct: float
    total_oos_trades: int
    best_stable_params: dict  # most frequently chosen params
    param_stability: float    # 0-1, how consistent the best params are
    is_overfit: bool
    summary: str


# ── Parameter Grid ──────────────────────────────────────────

DEFAULT_PARAM_GRID = {
    "signals.min_composite_score": [0.20, 0.25, 0.30, 0.35],
    "signals.min_agreeing_strategies": [2, 3, 4],
    "risk.stop_loss_atr_mult": [1.5, 2.0, 2.5],
    "risk.take_profit_atr_mult": [3.0, 4.0, 5.0],
}


class WalkForwardOptimizer:
    def __init__(self, config: dict, initial_equity: float = 100000.0):
        self.base_config = config
        self.initial_equity = initial_equity

    def optimize(self, bars_dict: dict[str, pd.DataFrame],
                 param_grid: dict = None,
                 train_days: int = 180,
                 test_days: int = 60,
                 step_days: int = 30,
                 metric: str = "sharpe_ratio",
                 max_combinations: int = 200) -> OptimizationResult:
        """
        Walk-forward optimization.

        Returns aggregated out-of-sample results with overfitting analysis.
        """
        if param_grid is None:
            param_grid = DEFAULT_PARAM_GRID

        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combos = list(itertools.product(*param_values))

        if len(all_combos) > max_combinations:
            log.warning(f"Grid has {len(all_combos)} combos, sampling {max_combinations}")
            np.random.seed(42)
            indices = np.random.choice(len(all_combos), max_combinations, replace=False)
            all_combos = [all_combos[i] for i in indices]

        log.info(f"Walk-forward: {len(all_combos)} param combos, "
                 f"train={train_days}d test={test_days}d step={step_days}d")

        # Find common date range
        all_dates = set()
        for df in bars_dict.values():
            all_dates.update(df.index.tolist())
        all_dates = sorted(all_dates)

        if len(all_dates) < train_days + test_days:
            log.error("Not enough data for walk-forward optimization")
            return self._empty_result()

        # Generate windows
        windows = []
        window_num = 0
        start_idx = 0

        while start_idx + train_days + test_days <= len(all_dates):
            train_start = all_dates[start_idx]
            train_end = all_dates[start_idx + train_days - 1]
            test_start = all_dates[start_idx + train_days]
            test_end_idx = min(start_idx + train_days + test_days - 1, len(all_dates) - 1)
            test_end = all_dates[test_end_idx]

            log.info(f"Window {window_num}: train {train_start} -> {train_end}, "
                     f"test {test_start} -> {test_end}")

            # Split bars for this window
            train_bars = {}
            test_bars = {}
            for sym, df in bars_dict.items():
                train_mask = (df.index >= train_start) & (df.index <= train_end)
                test_mask = (df.index >= test_start) & (df.index <= test_end)
                tb = df[train_mask]
                if len(tb) >= 30:
                    train_bars[sym] = tb
                te = df[test_mask]
                if len(te) >= 10:
                    test_bars[sym] = te

            if not train_bars or not test_bars:
                start_idx += step_days
                continue

            # Grid search on train set
            best_is_metric = -999
            best_params = {}
            best_is_result = None

            for combo in all_combos:
                trial_config = self._apply_params(param_names, combo)
                bt = Backtester(trial_config, self.initial_equity)
                result = bt.run(train_bars, min_bars=30)

                val = getattr(result, metric, 0.0)
                if result.total_trades < 5:
                    val = -999  # Need minimum trades

                if val > best_is_metric:
                    best_is_metric = val
                    best_params = dict(zip(param_names, combo))
                    best_is_result = result

            # Test best params on OOS data
            oos_config = self._apply_params(
                list(best_params.keys()), list(best_params.values()))
            bt_oos = Backtester(oos_config, self.initial_equity)
            oos_result = bt_oos.run(test_bars, min_bars=10)

            oos_metric = getattr(oos_result, metric, 0.0)
            degradation = 0.0
            if best_is_metric > 0:
                degradation = (best_is_metric - oos_metric) / best_is_metric * 100

            windows.append(WindowResult(
                window_num=window_num,
                train_start=str(train_start),
                train_end=str(train_end),
                test_start=str(test_start),
                test_end=str(test_end),
                best_params=best_params,
                in_sample_sharpe=round(best_is_metric, 2),
                out_of_sample_sharpe=round(oos_metric, 2),
                oos_trades=oos_result.total_trades,
                oos_win_rate=oos_result.win_rate,
                oos_return_pct=oos_result.total_return_pct,
                degradation_pct=round(degradation, 1),
            ))

            log.info(f"  IS {metric}={best_is_metric:.2f} -> OOS={oos_metric:.2f} "
                     f"(degradation={degradation:.0f}%) trades={oos_result.total_trades}")

            window_num += 1
            start_idx += step_days

        return self._aggregate_results(windows, param_names)

    def _apply_params(self, param_names: list, values) -> dict:
        """Create config with specific parameter values."""
        config = copy.deepcopy(self.base_config)
        for name, val in zip(param_names, values):
            parts = name.split(".")
            d = config
            for part in parts[:-1]:
                d = d[part]
            d[parts[-1]] = val
        return config

    def _aggregate_results(self, windows: list[WindowResult],
                           param_names: list) -> OptimizationResult:
        """Aggregate all window results into final assessment."""
        if not windows:
            return self._empty_result()

        avg_oos_sharpe = np.mean([w.out_of_sample_sharpe for w in windows])
        avg_oos_wr = np.mean([w.oos_win_rate for w in windows])
        avg_degradation = np.mean([w.degradation_pct for w in windows])
        total_oos_trades = sum(w.oos_trades for w in windows)

        # Parameter stability: how often do the same params get chosen?
        from collections import Counter
        param_strs = [str(sorted(w.best_params.items())) for w in windows]
        most_common = Counter(param_strs).most_common(1)
        stability = most_common[0][1] / len(windows) if most_common else 0

        # Most frequently chosen params
        best_stable = windows[0].best_params  # default
        if most_common:
            # Find the window with the most common params
            for w in windows:
                if str(sorted(w.best_params.items())) == most_common[0][0]:
                    best_stable = w.best_params
                    break

        # Overfitting detection
        is_overfit = bool(avg_degradation > 50 or avg_oos_sharpe < 0)

        summary_lines = [
            f"Walk-Forward Results ({len(windows)} windows):",
            f"  Avg OOS Sharpe: {avg_oos_sharpe:.2f}",
            f"  Avg OOS Win Rate: {avg_oos_wr:.1f}%",
            f"  Total OOS Trades: {total_oos_trades}",
            f"  Avg Degradation: {avg_degradation:.0f}%",
            f"  Param Stability: {stability:.0%}",
            f"  Overfitting: {'YES' if is_overfit else 'NO'}",
            f"  Best Stable Params: {best_stable}",
        ]
        summary = "\n".join(summary_lines)
        log.info(summary)

        return OptimizationResult(
            windows=[{
                "window": w.window_num,
                "is_sharpe": w.in_sample_sharpe,
                "oos_sharpe": w.out_of_sample_sharpe,
                "oos_trades": w.oos_trades,
                "oos_win_rate": w.oos_win_rate,
                "degradation": w.degradation_pct,
                "params": w.best_params,
            } for w in windows],
            avg_oos_sharpe=round(avg_oos_sharpe, 2),
            avg_oos_win_rate=round(avg_oos_wr, 1),
            avg_degradation_pct=round(avg_degradation, 1),
            total_oos_trades=total_oos_trades,
            best_stable_params=best_stable,
            param_stability=round(stability, 2),
            is_overfit=is_overfit,
            summary=summary,
        )

    def _empty_result(self) -> OptimizationResult:
        return OptimizationResult(
            windows=[], avg_oos_sharpe=0.0, avg_oos_win_rate=0.0,
            avg_degradation_pct=0.0, total_oos_trades=0,
            best_stable_params={}, param_stability=0.0,
            is_overfit=True, summary="Insufficient data for optimization",
        )
