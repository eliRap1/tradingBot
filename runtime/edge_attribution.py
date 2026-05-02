"""Edge attribution analysis — reads trades.edge_snapshot JSON,
correlates each edge signal with trade P&L to identify which edges
actually contribute to winners vs losers.

Use this weekly to:
- Promote edges that consistently fire on winners (boost weights)
- Demote/retire edges that fire on losers
- Detect edge drift (was profitable, now isn't)

Run:
    python -m runtime.edge_attribution
    python -m runtime.edge_attribution --since 2026-04-01
    python -m runtime.edge_attribution --report

Edge snapshot fields (set by coordinator before each entry):
  rs_score        float [-100..100]  — relative strength rank
  vol_ratio       float (>0)         — volume / 20d avg
  ml_prob         float [0..1]       — ML model probability
  vix_regime      str                — low / normal / elevated / panic
  insider_cluster bool               — Form 4 cluster buy
  gap_pct         float              — overnight gap %
  news_score      float [-1..1]      — sentiment
  sector_momentum str                — leading / neutral / lagging
  bond_trend      str                — risk_on / risk_off
  composite_score float              — confluence engine score
  confluence      int                — # agreeing strategies
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics as stats
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EdgeStat:
    name: str
    n_winners: int = 0
    n_losers: int = 0
    pnl_winners: float = 0.0
    pnl_losers: float = 0.0
    avg_pnl_when_present: float = 0.0
    avg_pnl_when_absent: float = 0.0
    delta: float = 0.0  # avg_present - avg_absent (lift)


def _connect(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"db not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_trades(conn: sqlite3.Connection, since: str | None = None) -> list[dict]:
    sql = "SELECT * FROM trades WHERE closed_at IS NOT NULL"
    args: list[Any] = []
    if since:
        sql += " AND closed_at >= ?"
        args.append(since)
    sql += " ORDER BY id ASC"
    rows = conn.execute(sql, args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("edge_snapshot"):
            try:
                d["edge_snapshot"] = json.loads(d["edge_snapshot"])
            except Exception:
                d["edge_snapshot"] = {}
        else:
            d["edge_snapshot"] = {}
        out.append(d)
    return out


def _present(value: Any, threshold: dict[str, Any] | None = None) -> bool:
    """Return True iff edge signal is meaningfully 'on' for this trade."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if threshold and "min" in threshold:
            return value >= threshold["min"]
        if threshold and "max" in threshold:
            return value <= threshold["max"]
        return abs(value) > 1e-9
    if isinstance(value, str):
        if threshold and "in" in threshold:
            return value in threshold["in"]
        return value not in ("", "neutral", "normal", "off")
    return bool(value)


# Map edge name → predicate that returns True when edge is "on"
EDGE_PREDICATES = {
    "rs_high": lambda v: isinstance(v, (int, float)) and v >= 60,
    "rs_low": lambda v: isinstance(v, (int, float)) and v <= 40,
    "vol_surge": lambda v: isinstance(v, (int, float)) and v >= 1.5,
    "vol_below": lambda v: isinstance(v, (int, float)) and v < 0.7,
    "ml_high": lambda v: isinstance(v, (int, float)) and v >= 0.6,
    "ml_low": lambda v: isinstance(v, (int, float)) and v < 0.4,
    "vix_panic": lambda v: v == "panic",
    "vix_elevated": lambda v: v == "elevated",
    "insider_cluster": lambda v: bool(v),
    "gap_up": lambda v: isinstance(v, (int, float)) and v >= 0.02,
    "gap_down": lambda v: isinstance(v, (int, float)) and v <= -0.02,
    "news_pos": lambda v: isinstance(v, (int, float)) and v >= 0.3,
    "news_neg": lambda v: isinstance(v, (int, float)) and v <= -0.3,
    "sector_leading": lambda v: v == "leading",
    "sector_lagging": lambda v: v == "lagging",
    "bond_risk_off": lambda v: v == "risk_off",
}

# Map edge predicate → which snapshot key feeds it
EDGE_KEYS = {
    "rs_high": "rs_score",
    "rs_low": "rs_score",
    "vol_surge": "vol_ratio",
    "vol_below": "vol_ratio",
    "ml_high": "ml_prob",
    "ml_low": "ml_prob",
    "vix_panic": "vix_regime",
    "vix_elevated": "vix_regime",
    "insider_cluster": "insider_cluster",
    "gap_up": "gap_pct",
    "gap_down": "gap_pct",
    "news_pos": "news_score",
    "news_neg": "news_score",
    "sector_leading": "sector_momentum",
    "sector_lagging": "sector_momentum",
    "bond_risk_off": "bond_trend",
}


def attribute(trades: list[dict]) -> dict[str, EdgeStat]:
    """For each edge predicate, compute lift = avg_pnl_present - avg_pnl_absent."""
    closed = [t for t in trades if t.get("pnl") is not None]
    if not closed:
        return {}

    stats_out: dict[str, EdgeStat] = {}
    for edge_name, predicate in EDGE_PREDICATES.items():
        key = EDGE_KEYS[edge_name]
        present_pnls = []
        absent_pnls = []
        n_win = 0
        n_los = 0
        win_pnl = 0.0
        los_pnl = 0.0

        for t in closed:
            snap = t.get("edge_snapshot", {}) or {}
            value = snap.get(key)
            pnl = t["pnl"] or 0
            if value is None:
                continue
            if predicate(value):
                present_pnls.append(pnl)
                if pnl > 0:
                    n_win += 1
                    win_pnl += pnl
                elif pnl < 0:
                    n_los += 1
                    los_pnl += pnl
            else:
                absent_pnls.append(pnl)

        avg_present = stats.mean(present_pnls) if present_pnls else 0.0
        avg_absent = stats.mean(absent_pnls) if absent_pnls else 0.0
        stats_out[edge_name] = EdgeStat(
            name=edge_name,
            n_winners=n_win,
            n_losers=n_los,
            pnl_winners=round(win_pnl, 2),
            pnl_losers=round(los_pnl, 2),
            avg_pnl_when_present=round(avg_present, 2),
            avg_pnl_when_absent=round(avg_absent, 2),
            delta=round(avg_present - avg_absent, 2),
        )
    return stats_out


def render_text(stats_map: dict[str, EdgeStat], total_trades: int) -> str:
    if not stats_map:
        return "No closed trades available for attribution."

    sorted_edges = sorted(stats_map.values(), key=lambda s: s.delta, reverse=True)
    lines = [
        f"=== EDGE ATTRIBUTION (n={total_trades} closed trades) ===",
        "",
        f"{'Edge':<18}{'N_win':>6}{'N_los':>6}{'WinPnl':>10}{'LosPnl':>10}{'AvgPres':>10}{'AvgAbs':>10}{'Lift':>10}",
        "-" * 80,
    ]
    for s in sorted_edges:
        lines.append(
            f"{s.name:<18}{s.n_winners:>6}{s.n_losers:>6}"
            f"{s.pnl_winners:>10.2f}{s.pnl_losers:>10.2f}"
            f"{s.avg_pnl_when_present:>10.2f}{s.avg_pnl_when_absent:>10.2f}"
            f"{s.delta:>+10.2f}"
        )
    lines += [
        "",
        "Lift = avg PnL when edge fires - avg PnL when edge absent.",
        "Positive lift = edge contributes. Negative = edge hurts.",
        "Promote top-3 lift edges. Demote/investigate negative-lift edges.",
    ]
    return "\n".join(lines)


def render_markdown(stats_map: dict[str, EdgeStat], total_trades: int, since: str | None) -> str:
    if not stats_map:
        return "# Edge Attribution\n\nNo trades available."
    sorted_edges = sorted(stats_map.values(), key=lambda s: s.delta, reverse=True)
    md = [
        "# Edge Attribution",
        "",
        f"- Total closed trades analyzed: **{total_trades}**",
        f"- Window: {since or 'all-time'}",
        "",
        "## Per-edge lift (avg PnL present - avg PnL absent)",
        "",
        "| Edge | N win | N los | Win $ | Loss $ | Avg present | Avg absent | **Lift** |",
        "|------|-------|-------|-------|--------|-------------|------------|----------|",
    ]
    for s in sorted_edges:
        md.append(
            f"| {s.name} | {s.n_winners} | {s.n_losers} | "
            f"{s.pnl_winners:.2f} | {s.pnl_losers:.2f} | "
            f"{s.avg_pnl_when_present:.2f} | {s.avg_pnl_when_absent:.2f} | "
            f"**{s.delta:+.2f}** |"
        )
    md += [
        "",
        "## Recommendations",
        "",
        "- Top 3 lift → consider boosting weight in `coordinator.py` size logic.",
        "- Negative lift → demote, investigate (data quality? regime change? overfit?).",
        "- Edges with N=0 winners/losers → not enough data, keep collecting.",
    ]
    return "\n".join(md)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=os.environ.get("BOT_STATE_DB", "bot_state.db"))
    p.add_argument("--since", default=None, help="ISO date filter (e.g. 2026-04-01)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--report-dir", default="docs/reports/health")
    args = p.parse_args()

    try:
        conn = _connect(args.db)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    try:
        trades = _fetch_trades(conn, since=args.since)
    finally:
        conn.close()

    stats_map = attribute(trades)
    total = sum(1 for t in trades if t.get("pnl") is not None)

    if args.json:
        from dataclasses import asdict
        print(json.dumps({k: asdict(v) for k, v in stats_map.items()}, indent=2))
    else:
        print(render_text(stats_map, total))

    if args.report:
        out_dir = Path(args.report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        path = out_dir / f"edge_attribution_{date_str}.md"
        path.write_text(render_markdown(stats_map, total, args.since), encoding="utf-8")
        print(f"\nReport written to: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
