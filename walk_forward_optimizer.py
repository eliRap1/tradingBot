"""
Walk-Forward Optimization Framework

This module implements professional-grade parameter optimization:

1. WALK-FORWARD ANALYSIS
   - Train on 6 months, test on 2 months
   - Roll forward by 1 month, repeat
   - Only use parameters that work OUT-OF-SAMPLE

2. PARAMETER SEARCH
   - Grid search over key parameters
   - Optimize for Sharpe ratio (risk-adjusted returns)
   - Cross-validate to avoid overfitting

3. ADAPTIVE PARAMETERS
   - Parameters update monthly based on recent performance
   - Bot uses optimized params automatically

Usage:
    python walk_forward_optimizer.py --start 2024-01-01 --end 2025-12-31
"""

import os
import json
import itertools
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from utils import setup_logger, load_config

log = setup_logger("optimizer")

# Parameter search space
PARAM_GRID = {
    "signals": {
        "min_composite_score": [0.15, 0.20, 0.25, 0.30],
        "min_agreeing_strategies": [2, 3],
    },
    "risk": {
        "stop_loss_atr_mult": [1.0, 1.5, 2.0],
        "take_profit_atr_mult": [3.0, 4.0, 5.0, 6.0],
        "trailing_stop_pct": [0.02, 0.025, 0.03, 0.04],
        "max_portfolio_risk_pct": [0.015, 0.02, 0.025],
    },
    "strategies": {
        "momentum": {
            "ema_fast": [5, 8, 12],
            "ema_slow": [15, 21, 30],
        },
        "supertrend": {
            "multiplier": [2.0, 2.5, 3.0, 3.5],
        },
    },
}

OPTIMIZED_PARAMS_FILE = os.path.join(os.path.dirname(__file__), "optimized_params.json")


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    params: dict
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    start_date: str
    end_date: str


class WalkForwardOptimizer:
    """
    Walk-forward optimization engine.
    
    Splits data into train/test windows, optimizes on train, validates on test.
    Only parameters that work out-of-sample are used.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.results_history = []
        
    def run_optimization(self, 
                         start_date: str = None,
                         end_date: str = None,
                         train_months: int = 6,
                         test_months: int = 2,
                         step_months: int = 1) -> dict:
        """
        Run full walk-forward optimization.
        
        Args:
            start_date: Start of optimization period (default: 18 months ago)
            end_date: End of optimization period (default: today)
            train_months: Training window size
            test_months: Testing window size
            step_months: How far to roll forward each iteration
            
        Returns:
            Best parameters and their performance metrics
        """
        # Default dates
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=18*30)).strftime("%Y-%m-%d")
        
        log.info(f"Starting walk-forward optimization: {start_date} to {end_date}")
        log.info(f"Windows: train={train_months}mo, test={test_months}mo, step={step_months}mo")
        
        # Generate windows
        windows = self._generate_windows(
            start_date, end_date, train_months, test_months, step_months
        )
        
        log.info(f"Generated {len(windows)} train/test windows")

        # Generate all param combos once
        param_combos = self._generate_param_combinations()
        log.info(f"Testing {len(param_combos)} param combos × {len(windows)} windows")

        # For each param combo, find which ones pass training, then test OOS
        # This ensures aggregation has multiple data points per param set
        all_test_results = []  # (params, [test_results across windows])

        for pi, params in enumerate(param_combos):
            if pi % 20 == 0:
                log.info(f"Param combo {pi}/{len(param_combos)}...")

            test_results_for_param = []

            for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
                # Train: does this param set work in-sample?
                train_result = self._backtest_params(params, train_start, train_end)
                if not train_result or train_result.total_trades < 10:
                    continue
                if train_result.sharpe_ratio < 0.5 or train_result.max_drawdown > 0.15:
                    continue  # Doesn't pass training bar

                # Test: validate out-of-sample
                test_result = self._backtest_params(params, test_start, test_end)
                if test_result and test_result.total_trades >= 5:
                    test_results_for_param.append(test_result)

            # Only keep params that worked in multiple windows
            if len(test_results_for_param) >= max(2, len(windows) // 2):
                all_test_results.append((params, test_results_for_param))

        if not all_test_results:
            log.error("No valid test results - optimization failed")
            return None

        best_params = self._aggregate_results(all_test_results)

        # Save optimized parameters
        self._save_optimized_params(best_params)

        return best_params
    
    def _generate_windows(self, start_date: str, end_date: str,
                          train_months: int, test_months: int,
                          step_months: int) -> list:
        """Generate train/test window pairs."""
        windows = []
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        current = start
        while True:
            train_start = current
            train_end = current + timedelta(days=train_months * 30)
            test_start = train_end
            test_end = test_start + timedelta(days=test_months * 30)
            
            if test_end > end:
                break
            
            windows.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            ))
            
            current += timedelta(days=step_months * 30)
        
        return windows
    
    def _generate_param_combinations(self) -> list:
        """Generate parameter combinations to test (random sample if too many)."""
        import random

        signal_combos = list(itertools.product(
            PARAM_GRID["signals"]["min_composite_score"],
            PARAM_GRID["signals"]["min_agreeing_strategies"],
        ))

        risk_combos = list(itertools.product(
            PARAM_GRID["risk"]["stop_loss_atr_mult"],
            PARAM_GRID["risk"]["take_profit_atr_mult"],
            PARAM_GRID["risk"]["trailing_stop_pct"],
            PARAM_GRID["risk"]["max_portfolio_risk_pct"],
        ))

        # Build full cartesian product
        all_combos = []
        for sig in signal_combos:
            for risk in risk_combos:
                all_combos.append({
                    "signals": {
                        "min_composite_score": sig[0],
                        "min_agreeing_strategies": sig[1],
                    },
                    "risk": {
                        "stop_loss_atr_mult": risk[0],
                        "take_profit_atr_mult": risk[1],
                        "trailing_stop_pct": risk[2],
                        "max_portfolio_risk_pct": risk[3],
                    },
                })

        # Random sample to cap compute, avoiding bias toward early grid points
        max_combos = 200
        if len(all_combos) > max_combos:
            random.seed(42)  # Reproducible
            all_combos = random.sample(all_combos, max_combos)

        return all_combos
    
    def _fetch_all_data(self):
        """Fetch full date range of data once, cache for all windows."""
        if hasattr(self, '_all_bars') and self._all_bars:
            return self._all_bars

        try:
            from data import DataFetcher
            from broker import Broker

            broker = Broker(self.config)
            data_fetcher = DataFetcher(broker)

            # Fetch maximum history needed (18 months + buffer)
            symbols = self.config["screener"]["universe"][:15]
            raw_bars = data_fetcher.get_bars(symbols, timeframe="1Day", days=600)

            bars_dict = {}
            for sym, bar_list in raw_bars.items():
                if isinstance(bar_list, list) and len(bar_list) >= 30:
                    df = pd.DataFrame(bar_list)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    bars_dict[sym] = df
                elif isinstance(bar_list, pd.DataFrame) and len(bar_list) >= 30:
                    bars_dict[sym] = bar_list

            self._all_bars = bars_dict
            log.info(f"Fetched data for {len(bars_dict)} symbols")
            return bars_dict
        except Exception as e:
            log.warning(f"Could not fetch data: {e}")
            return {}

    def _slice_bars(self, bars_dict: dict, start_date: str, end_date: str) -> dict:
        """Slice cached bars to a specific date range."""
        sliced = {}
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        for sym, df in bars_dict.items():
            idx = df.index
            # Handle timezone-aware indexes
            if hasattr(idx, 'tz') and idx.tz is not None:
                start_tz = start.tz_localize(idx.tz)
                end_tz = end.tz_localize(idx.tz)
            else:
                start_tz = start
                end_tz = end

            window = df.loc[(idx >= start_tz) & (idx <= end_tz)]
            if len(window) >= 30:
                sliced[sym] = window

        return sliced

    def _backtest_params(self, params: dict, start_date: str,
                         end_date: str) -> Optional[BacktestResult]:
        """Run backtest with specific parameters on date-sliced data."""
        try:
            config = self._merge_params(params)

            from backtester import Backtester

            all_bars = self._fetch_all_data()
            if not all_bars:
                return None

            # Slice to the specific window date range
            bars_dict = self._slice_bars(all_bars, start_date, end_date)
            if not bars_dict:
                return None

            bt = Backtester(config)
            result = bt.run(bars_dict, min_bars=30)
            
            if result is None or result.total_trades == 0:
                return None
            
            return BacktestResult(
                params=params,
                total_return=result.total_return_pct,
                sharpe_ratio=result.sharpe_ratio,
                max_drawdown=result.max_drawdown_pct / 100,  # Convert to decimal
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                total_trades=result.total_trades,
                avg_trade_duration=result.avg_bars_held,
                start_date=start_date,
                end_date=end_date,
            )
            
        except Exception as e:
            log.error(f"Backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _merge_params(self, params: dict) -> dict:
        """Merge optimization params with base config."""
        import copy
        config = copy.deepcopy(self.config)
        
        for section, values in params.items():
            if section in config:
                config[section].update(values)
        
        return config
    
    def _aggregate_results(self, results: list) -> dict:
        """Aggregate results from all test windows to find robust params.

        Args:
            results: list of (params_dict, [BacktestResult, ...]) tuples.
                     Each tuple has one param set and its OOS results across windows.
        """
        best_score = -float('inf')
        best_params = None

        for params, test_results in results:
            sharpes = [r.sharpe_ratio for r in test_results]
            win_rates = [r.win_rate for r in test_results]
            drawdowns = [r.max_drawdown for r in test_results]

            avg_sharpe = np.mean(sharpes)
            std_sharpe = np.std(sharpes) if len(sharpes) > 1 else 999.0
            avg_win_rate = np.mean(win_rates)
            max_dd = max(drawdowns)
            n_windows = len(test_results)

            # Robustness score: consistent performance across windows
            # Penalize: high variance, high drawdown, few passing windows
            score = (avg_sharpe
                     - 0.5 * std_sharpe
                     + 0.01 * avg_win_rate
                     - 0.5 * max_dd
                     + 0.1 * n_windows)  # Bonus for working across more windows

            if score > best_score:
                best_score = score
                best_params = params

        if best_params:
            log.info("\n=== OPTIMIZED PARAMETERS ===")
            log.info(f"Score: {best_score:.3f}")
            log.info(json.dumps(best_params, indent=2))

        return {
            "params": best_params,
            "score": best_score,
            "optimized_at": datetime.now().isoformat(),
        }
    
    def _save_optimized_params(self, result: dict):
        """Save optimized parameters to file."""
        try:
            with open(OPTIMIZED_PARAMS_FILE, "w") as f:
                json.dump(result, f, indent=2)
            log.info(f"Saved optimized params to {OPTIMIZED_PARAMS_FILE}")
        except Exception as e:
            log.error(f"Failed to save params: {e}")


def load_optimized_params() -> Optional[dict]:
    """Load optimized parameters if they exist and are recent."""
    if not os.path.exists(OPTIMIZED_PARAMS_FILE):
        log.info("No optimized params file found - using default config")
        return None
    
    try:
        with open(OPTIMIZED_PARAMS_FILE, "r") as f:
            data = json.load(f)
        
        # Check if params are recent (< 30 days old)
        optimized_at = datetime.fromisoformat(data.get("optimized_at", "2000-01-01"))
        age_days = (datetime.now() - optimized_at).days
        
        if age_days > 30:
            log.warning(f"Optimized params are {age_days} days old - consider re-running optimization")
        
        return data.get("params")
        
    except Exception as e:
        log.error(f"Failed to load optimized params: {e}")
        return None


def apply_optimized_params(config: dict) -> dict:
    """Apply optimized parameters to config if available."""
    try:
        params = load_optimized_params()
        
        if not params:
            log.info("Using default config (no optimization applied)")
            return config
        
        log.info("Applying optimized parameters from walk-forward analysis")
        
        for section, values in params.items():
            if section in config and isinstance(values, dict):
                config[section].update(values)
        
        return config
    except Exception as e:
        log.warning(f"Could not apply optimized params: {e} - using defaults")
        return config


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Walk-forward parameter optimization")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=2)
    parser.add_argument("--step-months", type=int, default=1)
    
    args = parser.parse_args()
    
    optimizer = WalkForwardOptimizer()
    result = optimizer.run_optimization(
        start_date=args.start,
        end_date=args.end,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )
    
    if result:
        print("\n" + "=" * 50)
        print("OPTIMIZATION COMPLETE")
        print("=" * 50)
        print(json.dumps(result, indent=2))
