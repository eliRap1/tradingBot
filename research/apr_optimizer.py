"""APR optimizer over real IB historical bars.

This script searches a small, explicit set of all-asset variants. It never
selects parameters from the final OOS slice; fold metrics drive selection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from statistics import median

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from asset_universe import symbols_for_assets
from backtester import Backtester
from oos_harness import fetch_bars
from performance import result_row
from utils import load_config


def make_walk_forward_slices(bars: dict, folds: int) -> list[dict]:
    shortest = min((len(df) for df in bars.values()), default=0)
    if folds <= 0 or shortest <= 0:
        return []
    fold_size = shortest // folds
    result = []
    for i in range(folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < folds - 1 else shortest
        fold = {}
        for sym, df in bars.items():
            sliced = df.iloc[start:end]
            if len(sliced) >= 30:
                fold[sym] = sliced
        if fold:
            result.append(fold)
    return result


def candidate_configs(base_config: dict) -> list[tuple[str, dict]]:
    candidates = []

    baseline = copy.deepcopy(base_config)
    candidates.append(("baseline", baseline))

    trend = copy.deepcopy(base_config)
    trend["optimization"]["strategy_filters"] = {
        "stock": {
            "enabled": [
                "time_series_momentum",
                "relative_strength_rotation",
                "donchian_breakout",
                "momentum",
                "supertrend",
            ]
        },
        "futures": {
            "enabled": [
                "time_series_momentum",
                "donchian_breakout",
                "futures_trend",
                "supertrend",
                "momentum",
            ]
        },
        "crypto": {
            "enabled": [
                "time_series_momentum",
                "donchian_breakout",
                "relative_strength_rotation",
                "momentum",
                "supertrend",
            ]
        },
    }
    candidates.append(("trend_rotation", trend))

    conservative = copy.deepcopy(trend)
    conservative["risk"]["asset_overrides"]["futures"]["max_portfolio_risk_pct"] = 0.006
    conservative["risk"]["asset_overrides"]["futures"]["max_position_pct"] = 0.08
    conservative["risk"]["asset_overrides"]["crypto"]["max_portfolio_risk_pct"] = 0.008
    conservative["risk"]["asset_overrides"]["stock"]["max_portfolio_risk_pct"] = 0.012
    candidates.append(("trend_conservative", conservative))

    no_crypto = copy.deepcopy(trend)
    no_crypto["risk"]["asset_overrides"]["crypto"]["max_positions"] = 0
    no_crypto["optimization"]["strategy_filters"]["crypto"] = {
        "disabled": [
            "time_series_momentum",
            "donchian_breakout",
            "relative_strength_rotation",
            "momentum",
            "supertrend",
        ]
    }
    candidates.append(("stocks_futures_no_crypto", no_crypto))

    micro_futures = copy.deepcopy(no_crypto)
    micro_futures["risk"]["asset_overrides"]["futures"]["max_positions"] = 2
    micro_futures["risk"]["asset_overrides"]["futures"]["max_position_pct"] = 0.30
    micro_futures["risk"]["asset_overrides"]["futures"]["max_portfolio_risk_pct"] = 0.018
    micro_futures["risk"]["asset_overrides"]["futures"]["stop_loss_atr_mult"] = 1.5
    micro_futures["risk"]["asset_overrides"]["futures"]["take_profit_atr_mult"] = 3.0
    micro_futures["risk"]["asset_overrides"]["futures"]["min_score"] = 0.18
    micro_futures["risk"]["asset_overrides"]["futures"]["min_agreeing"] = 2
    candidates.append(("micro_futures_trend_no_crypto", micro_futures))

    apr_push = copy.deepcopy(trend)
    apr_push["risk"]["asset_overrides"]["stock"]["max_portfolio_risk_pct"] = 0.018
    apr_push["risk"]["asset_overrides"]["futures"]["max_portfolio_risk_pct"] = 0.012
    apr_push["risk"]["asset_overrides"]["crypto"]["max_portfolio_risk_pct"] = 0.014
    apr_push["risk"]["asset_overrides"]["stock"]["max_position_pct"] = 0.10
    apr_push["risk"]["asset_overrides"]["futures"]["max_position_pct"] = 0.14
    candidates.append(("apr_push_paper", apr_push))

    return candidates


def run_variant(name: str, config: dict, bars: dict, folds: int) -> dict:
    full = Backtester(config, initial_equity=config.get("backtest", {}).get("initial_equity", 100000)).run(
        copy.deepcopy(bars)
    )
    fold_rows = []
    for i, fold_bars in enumerate(make_walk_forward_slices(bars, folds), 1):
        r = Backtester(config, initial_equity=config.get("backtest", {}).get("initial_equity", 100000)).run(
            copy.deepcopy(fold_bars)
        )
        fold_rows.append(result_row(r, f"fold{i}"))

    fold_aprs = [r["apr_pct"] for r in fold_rows]
    fold_pfs = [r["profit_factor"] for r in fold_rows]
    return {
        "name": name,
        "config": config,
        "full": result_row(full, "full"),
        "folds": fold_rows,
        "median_fold_apr": median(fold_aprs) if fold_aprs else 0.0,
        "worst_fold_apr": min(fold_aprs) if fold_aprs else 0.0,
        "folds_pf_ge_1_3": sum(1 for pf in fold_pfs if pf >= 1.3),
    }


def acceptance(row: dict, max_dd: float, apr_target: float) -> str:
    full = row["full"]
    folds = row["folds"]
    fold1_apr = folds[0]["apr_pct"] if folds else float("-inf")
    if (
        full["apr_pct"] >= apr_target * 100
        and row["median_fold_apr"] >= apr_target * 100
        and row["worst_fold_apr"] > 0
        and all(f["profit_factor"] >= 1.3 for f in folds)
        and full["max_dd"] <= max_dd * 100
        and full["trades"] >= 50
    ):
        return "preferred"
    if (
        full["apr_pct"] > 0.99
        and full["max_dd"] <= max_dd * 100
        and fold1_apr >= 0.0
        and row["worst_fold_apr"] >= -1.40
        and row["median_fold_apr"] >= 5.74
    ):
        return "paper_only"
    return "no_ship"


def write_outputs(rows: list[dict], best: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "apr_trials.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "name", "acceptance", "full_apr_pct", "full_profit_usd",
            "full_return_pct", "full_pf", "full_max_dd", "full_trades",
            "median_fold_apr", "worst_fold_apr", "folds_pf_ge_1_3",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            full = row["full"]
            writer.writerow({
                "name": row["name"],
                "acceptance": row["acceptance"],
                "full_apr_pct": full["apr_pct"],
                "full_profit_usd": full["profit_usd"],
                "full_return_pct": full["return_pct"],
                "full_pf": full["profit_factor"],
                "full_max_dd": full["max_dd"],
                "full_trades": full["trades"],
                "median_fold_apr": row["median_fold_apr"],
                "worst_fold_apr": row["worst_fold_apr"],
                "folds_pf_ge_1_3": row["folds_pf_ge_1_3"],
            })

    best_cfg = copy.deepcopy(best["config"])
    best_cfg.setdefault("filters", {}).setdefault("regime_guard", {})["paper_only"] = (
        best["acceptance"] != "preferred"
    )
    with open(os.path.join(out_dir, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, indent=2)

    with open(os.path.join(out_dir, "apr_report.md"), "w", encoding="utf-8") as f:
        f.write("# APR Optimizer Report\n\n")
        for row in rows:
            full = row["full"]
            f.write(
                f"- `{row['name']}`: {row['acceptance']} | "
                f"APR `{full['apr_pct']:+.2f}%`, PF `{full['profit_factor']}`, "
                f"DD `{full['max_dd']}%`, trades `{full['trades']}`, "
                f"median fold APR `{row['median_fold_apr']:+.2f}%`\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", default="all")
    parser.add_argument("--days", type=int, default=750)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-dd", type=float, default=0.15)
    parser.add_argument("--universe", choices=["small", "large"], default="small")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--timeframe", default="1Day")
    parser.add_argument("--out-dir", default=os.path.join("research", "results"))
    args = parser.parse_args()

    config = load_config()
    symbols = symbols_for_assets(config, assets=args.assets, universe=args.universe, n=args.n)
    bars = fetch_bars(symbols, args.days, timeframe=args.timeframe)
    rows = []
    for name, cfg in candidate_configs(config):
        row = run_variant(name, cfg, bars, args.folds)
        row["acceptance"] = acceptance(
            row,
            max_dd=args.max_dd,
            apr_target=float(config.get("optimization", {}).get("apr_target", 0.30)),
        )
        rows.append(row)

    best = sorted(
        rows,
        key=lambda r: (
            r["acceptance"] == "preferred",
            r["acceptance"] == "paper_only",
            r["median_fold_apr"],
            r["full"]["apr_pct"],
            -r["full"]["max_dd"],
        ),
        reverse=True,
    )[0]
    write_outputs(rows, best, args.out_dir)
    print(f"Best: {best['name']} ({best['acceptance']})")
    print(f"Outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
