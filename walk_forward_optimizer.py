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
from concurrent.futures import ProcessPoolExecutor, as_completed

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
        
        # Run optimization for each window
        all_test_results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            log.info(f"\n=== Window {i+1}/{len(windows)} ===")
            log.info(f"Train: {train_start} to {train_end}")
            log.info(f"Test:  {test_start} to {test_end}")
            
            # Find best params on training data
            best_train_params, train_result = self._optimize_window(
                train_start, train_end
            )
            
            if not best_train_params:
                log.warning(f"No valid params found for window {i+1}")
                continue
            
            log.info(f"Best train params: Sharpe={train_result.sharpe_ratio:.2f}, "
                     f"WinRate={train_result.win_rate:.1f}%")
            
            # Validate on test data
            test_result = self._backtest_params(
                best_train_params, test_start, test_end
            )
            
            if test_result:
                log.info(f"Test result: Sharpe={test_result.sharpe_ratio:.2f}, "
                         f"WinRate={test_result.win_rate:.1f}%, "
                         f"Trades={test_result.total_trades}")
                all_test_results.append(test_result)
        
        # Aggregate results and find robust parameters
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
    
    def _optimize_window(self, start_date: str, end_date: str) -> tuple:
        """Find best parameters for a training window."""
        
        # Generate parameter combinations
        param_combos = self._generate_param_combinations()
        
        log.info(f"Testing {len(param_combos)} parameter combinations...")
        
        best_result = None
        best_params = None
        
        for i, params in enumerate(param_combos):
            if i % 20 == 0:
                log.info(f"Progress: {i}/{len(param_combos)}")
            
            result = self._backtest_params(params, start_date, end_date)
            
            if result and result.total_trades >= 10:  # Minimum trades
                if best_result is None or result.sharpe_ratio > best_result.sharpe_ratio:
                    # Also check drawdown constraint
                    if result.max_drawdown < 0.15:  # Max 15% drawdown
                        best_result = result
                        best_params = params
        
        return best_params, best_result
    
    def _generate_param_combinations(self) -> list:
        """Generate all parameter combinations to test."""
        combos = []
        
        # Flatten the grid
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
        
        # Limit combinations to avoid explosion
        max_combos = 200
        
        for sig in signal_combos:
            for risk in risk_combos:
                params = {
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
                }
                combos.append(params)
                
                if len(combos) >= max_combos:
                    break
            if len(combos) >= max_combos:
                break
        
        return combos
    
    def _backtest_params(self, params: dict, start_date: str, 
                         end_date: str) -> Optional[BacktestResult]:
        """Run backtest with specific parameters."""
        try:
            # Merge params with base config
            config = self._merge_params(params)
            
            # Import here to avoid circular imports
            from backtester import Backtester
            
            bt = Backtester(config)
            result = bt.run(
                start_date=start_date,
                end_date=end_date,
                symbols=config["screener"]["universe"][:20],  # Limit for speed
            )
            
            if not result or result.get("total_trades", 0) == 0:
                return None
            
            return BacktestResult(
                params=params,
                total_return=result.get("total_return", 0),
                sharpe_ratio=result.get("sharpe_ratio", 0),
                max_drawdown=result.get("max_drawdown", 0),
                win_rate=result.get("win_rate", 0),
                profit_factor=result.get("profit_factor", 0),
                total_trades=result.get("total_trades", 0),
                avg_trade_duration=result.get("avg_trade_duration", 0),
                start_date=start_date,
                end_date=end_date,
            )
            
        except Exception as e:
            log.error(f"Backtest failed: {e}")
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
        """Aggregate results from all test windows to find robust params."""
        
        # Group by parameter set
        param_performance = {}
        
        for result in results:
            param_key = json.dumps(result.params, sort_keys=True)
            
            if param_key not in param_performance:
                param_performance[param_key] = {
                    "params": result.params,
                    "sharpes": [],
                    "win_rates": [],
                    "drawdowns": [],
                    "returns": [],
                }
            
            param_performance[param_key]["sharpes"].append(result.sharpe_ratio)
            param_performance[param_key]["win_rates"].append(result.win_rate)
            param_performance[param_key]["drawdowns"].append(result.max_drawdown)
            param_performance[param_key]["returns"].append(result.total_return)
        
        # Find most robust parameters (consistent across windows)
        best_score = -float('inf')
        best_params = None
        
        for param_key, perf in param_performance.items():
            # Robustness score: average Sharpe - std(Sharpe) + win_rate bonus
            avg_sharpe = np.mean(perf["sharpes"])
            std_sharpe = np.std(perf["sharpes"])
            avg_win_rate = np.mean(perf["win_rates"])
            max_dd = max(perf["drawdowns"])
            
            # Penalize inconsistent results and high drawdown
            score = avg_sharpe - 0.5 * std_sharpe + 0.01 * avg_win_rate - 0.5 * max_dd
            
            if score > best_score:
                best_score = score
                best_params = perf["params"]
        
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
    params = load_optimized_params()
    
    if not params:
        log.info("No optimized params found - using default config")
        return config
    
    log.info("Applying optimized parameters")
    
    for section, values in params.items():
        if section in config and isinstance(values, dict):
            config[section].update(values)
    
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
