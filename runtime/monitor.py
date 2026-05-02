"""Bot health monitor — reads bot_state.db, computes live metrics,
compares to backtest baseline, posts alert to Discord on drift.

Run via cron (e.g. daily at 22:30 local) or directly:
    python -m runtime.monitor               # human-readable stdout
    python -m runtime.monitor --json        # JSON to stdout
    python -m runtime.monitor --discord     # post alert if needed
    python -m runtime.monitor --report      # write daily_health.md

Baseline (backtest, 35-sym 300d real IB, post-overhaul May 2026):
  PF=2.36  WR=47.2%  expectancy=$524/trade  MDD=7.02%

Alert thresholds (live):
  PF20 < 1.2       — performance halved vs baseline
  WR20 < 35%       — clearly below baseline
  Loss streak >= 5 — discipline check
  DD > 10%         — exceeds backtest MDD ceiling
  Days idle > 14   — regime-blocked or broken
  IS-OOS gap > 35% — possible overfit drifting
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics as stats
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


BASELINE = {
    "pf": 2.36,
    "win_rate": 47.2,
    "expectancy": 524.0,
    "mdd_pct": 7.02,
    "annual_return_pct": 33.8,
}

THRESHOLDS = {
    "pf_min": 1.2,
    "wr_min": 35.0,
    "loss_streak_max": 5,
    "dd_max": 10.0,
    "idle_days_max": 14,
    "rolling_window": 20,
}


@dataclass
class HealthReport:
    timestamp: str
    db_path: str
    total_trades: int
    last_trade_at: str | None
    days_since_last_trade: int | None

    pf_all: float
    pf_rolling: float
    win_rate_all: float
    win_rate_rolling: float
    expectancy_all: float
    expectancy_rolling: float
    avg_r_rolling: float
    loss_streak_current: int
    loss_streak_max: int

    realized_pnl: float
    open_position_count: int
    open_positions: list[str] = field(default_factory=list)

    drawdown_pct: float = 0.0
    peak_equity: float = 0.0
    trough_equity: float = 0.0

    drift_vs_backtest_pct: float = 0.0
    alerts: list[str] = field(default_factory=list)
    severity: str = "ok"  # ok / warn / critical


def _connect(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"state db not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_trades(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, symbol, side, qty, entry_price, exit_price, pnl, "
        "pnl_pct, reason, risk_dollars, r_multiple, strategies, "
        "opened_at, closed_at FROM trades ORDER BY id ASC"
    ).fetchall()
    return [dict(r) for r in rows]


def _fetch_open_positions(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT symbol, side, opened_at, entry_price FROM positions "
        "ORDER BY opened_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def _profit_factor(trades: list[dict]) -> float:
    gross_win = sum(t["pnl"] for t in trades if (t["pnl"] or 0) > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if (t["pnl"] or 0) < 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return round(gross_win / gross_loss, 3)


def _win_rate(trades: list[dict]) -> float:
    closed = [t for t in trades if t["pnl"] is not None and t["pnl"] != 0]
    if not closed:
        return 0.0
    wins = sum(1 for t in closed if t["pnl"] > 0)
    return round(100.0 * wins / len(closed), 2)


def _expectancy(trades: list[dict]) -> float:
    pnls = [t["pnl"] for t in trades if t["pnl"] is not None and t["pnl"] != 0]
    if not pnls:
        return 0.0
    return round(stats.mean(pnls), 2)


def _avg_r(trades: list[dict]) -> float:
    rs = [t["r_multiple"] for t in trades if t.get("r_multiple") is not None]
    if not rs:
        return 0.0
    return round(stats.mean(rs), 3)


def _loss_streaks(trades: list[dict]) -> tuple[int, int]:
    """Returns (current_streak, max_streak) on losses."""
    cur = 0
    longest = 0
    for t in trades:
        pnl = t["pnl"] or 0
        if pnl < 0:
            cur += 1
            longest = max(longest, cur)
        elif pnl > 0:
            cur = 0
    return cur, longest


def _drawdown(trades: list[dict], starting_equity: float = 100_000.0) -> tuple[float, float, float]:
    """Returns (peak_equity, trough_equity, max_dd_pct)."""
    equity = starting_equity
    peak = starting_equity
    trough = starting_equity
    max_dd = 0.0
    for t in trades:
        equity += (t["pnl"] or 0)
        if equity > peak:
            peak = equity
            trough = equity
        if equity < trough:
            trough = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return peak, trough, round(max_dd, 2)


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def compute_health(db_path: str, starting_equity: float = 100_000.0,
                   window: int = THRESHOLDS["rolling_window"]) -> HealthReport:
    conn = _connect(db_path)
    try:
        trades = _fetch_trades(conn)
        positions = _fetch_open_positions(conn)
    finally:
        conn.close()

    closed = [t for t in trades if t["closed_at"] is not None]
    rolling = closed[-window:] if len(closed) >= 1 else []

    last_trade = closed[-1] if closed else None
    last_at = last_trade["closed_at"] if last_trade else None
    last_dt = _parse_iso(last_at)
    days_idle = None
    if last_dt is not None:
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        days_idle = (datetime.now(timezone.utc) - last_dt).days

    pf_all = _profit_factor(closed)
    pf_roll = _profit_factor(rolling)
    wr_all = _win_rate(closed)
    wr_roll = _win_rate(rolling)
    exp_all = _expectancy(closed)
    exp_roll = _expectancy(rolling)
    avg_r_roll = _avg_r(rolling)
    streak_cur, streak_max = _loss_streaks(closed)
    peak, trough, max_dd = _drawdown(closed, starting_equity=starting_equity)
    realized = sum(t["pnl"] or 0 for t in closed)

    drift = 0.0
    if BASELINE["pf"] > 0 and pf_roll > 0:
        drift = round(100.0 * (pf_roll - BASELINE["pf"]) / BASELINE["pf"], 1)

    alerts = []
    sev_rank = {"ok": 0, "warn": 1, "critical": 2}
    severity = "ok"

    def _bump(level: str):
        nonlocal severity
        if sev_rank[level] > sev_rank[severity]:
            severity = level

    if len(rolling) >= window:
        if pf_roll < THRESHOLDS["pf_min"]:
            alerts.append(f"PF{window}={pf_roll:.2f} < {THRESHOLDS['pf_min']:.2f} (baseline {BASELINE['pf']:.2f})")
            _bump("warn")
        if wr_roll < THRESHOLDS["wr_min"]:
            alerts.append(f"WR{window}={wr_roll:.1f}% < {THRESHOLDS['wr_min']:.1f}% (baseline {BASELINE['win_rate']:.1f}%)")
            _bump("warn")

    if streak_cur >= THRESHOLDS["loss_streak_max"]:
        alerts.append(f"Loss streak={streak_cur} >= {THRESHOLDS['loss_streak_max']}")
        _bump("warn")

    if max_dd > THRESHOLDS["dd_max"]:
        alerts.append(f"Drawdown={max_dd:.2f}% > {THRESHOLDS['dd_max']:.1f}% (baseline MDD {BASELINE['mdd_pct']:.2f}%)")
        _bump("critical")

    if days_idle is not None and days_idle > THRESHOLDS["idle_days_max"]:
        alerts.append(f"No trade in {days_idle}d > {THRESHOLDS['idle_days_max']}d threshold")
        _bump("warn")

    if abs(drift) > 35.0 and len(rolling) >= window:
        alerts.append(f"Live drift {drift:+.1f}% from backtest PF baseline")
        _bump("warn")

    return HealthReport(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        db_path=str(db_path),
        total_trades=len(closed),
        last_trade_at=last_at,
        days_since_last_trade=days_idle,
        pf_all=pf_all,
        pf_rolling=pf_roll,
        win_rate_all=wr_all,
        win_rate_rolling=wr_roll,
        expectancy_all=exp_all,
        expectancy_rolling=exp_roll,
        avg_r_rolling=avg_r_roll,
        loss_streak_current=streak_cur,
        loss_streak_max=streak_max,
        realized_pnl=round(realized, 2),
        open_position_count=len(positions),
        open_positions=[p["symbol"] for p in positions],
        drawdown_pct=max_dd,
        peak_equity=round(peak, 2),
        trough_equity=round(trough, 2),
        drift_vs_backtest_pct=drift,
        alerts=alerts,
        severity=severity,
    )


def render_text(r: HealthReport) -> str:
    lines = [
        f"=== BOT HEALTH @ {r.timestamp} ({r.severity.upper()}) ===",
        f"Total closed trades: {r.total_trades}",
        f"Last trade: {r.last_trade_at} ({r.days_since_last_trade}d ago)" if r.last_trade_at else "Last trade: <none>",
        f"Open positions: {r.open_position_count} {r.open_positions}",
        "",
        "--- Performance ---",
        f"  PF (all): {r.pf_all}    PF (last {THRESHOLDS['rolling_window']}): {r.pf_rolling}",
        f"  WR (all): {r.win_rate_all}%    WR (last {THRESHOLDS['rolling_window']}): {r.win_rate_rolling}%",
        f"  Expectancy: ${r.expectancy_all} (all) / ${r.expectancy_rolling} (rolling)",
        f"  Avg R: {r.avg_r_rolling}    Realized P&L: ${r.realized_pnl}",
        f"  Drawdown: {r.drawdown_pct}%    Peak: ${r.peak_equity}    Trough: ${r.trough_equity}",
        f"  Loss streak: current={r.loss_streak_current}    longest={r.loss_streak_max}",
        f"  Drift vs backtest PF: {r.drift_vs_backtest_pct:+.1f}%",
        "",
        "--- Baseline ---",
        f"  PF {BASELINE['pf']}    WR {BASELINE['win_rate']}%    "
        f"Exp ${BASELINE['expectancy']}    MDD {BASELINE['mdd_pct']}%",
    ]
    if r.alerts:
        lines.append("")
        lines.append("--- ALERTS ---")
        for a in r.alerts:
            lines.append(f"  [{r.severity.upper()}] {a}")
    else:
        lines.append("")
        lines.append("All thresholds nominal. No action required.")
    return "\n".join(lines)


def render_markdown(r: HealthReport) -> str:
    md = [
        f"# Daily Health — {r.timestamp[:10]}",
        "",
        f"**Severity:** {r.severity.upper()}",
        "",
        "## Snapshot",
        "",
        f"- Total closed trades: **{r.total_trades}**",
        f"- Last trade: {r.last_trade_at} ({r.days_since_last_trade}d ago)" if r.last_trade_at else "- No trades yet.",
        f"- Open positions: **{r.open_position_count}** — {', '.join(r.open_positions) or 'none'}",
        f"- Realized P&L: **${r.realized_pnl}**",
        "",
        "## Performance vs Backtest Baseline",
        "",
        "| Metric | Live (rolling 20) | Live (all) | Baseline | Verdict |",
        "|--------|-------------------|------------|----------|---------|",
        f"| Profit factor | {r.pf_rolling} | {r.pf_all} | {BASELINE['pf']} | "
        f"{'OK' if r.pf_rolling >= THRESHOLDS['pf_min'] else 'WARN'} |",
        f"| Win rate | {r.win_rate_rolling}% | {r.win_rate_all}% | {BASELINE['win_rate']}% | "
        f"{'OK' if r.win_rate_rolling >= THRESHOLDS['wr_min'] else 'WARN'} |",
        f"| Expectancy | ${r.expectancy_rolling} | ${r.expectancy_all} | ${BASELINE['expectancy']} | "
        f"{'OK' if r.expectancy_rolling > 0 else 'WARN'} |",
        f"| Drawdown | {r.drawdown_pct}% | — | {BASELINE['mdd_pct']}% (max) | "
        f"{'OK' if r.drawdown_pct <= THRESHOLDS['dd_max'] else 'CRITICAL'} |",
        f"| Loss streak | {r.loss_streak_current} (current) | {r.loss_streak_max} (max) | — | "
        f"{'OK' if r.loss_streak_current < THRESHOLDS['loss_streak_max'] else 'WARN'} |",
        f"| Drift PF | {r.drift_vs_backtest_pct:+.1f}% | — | within 35% | "
        f"{'OK' if abs(r.drift_vs_backtest_pct) <= 35 else 'WARN'} |",
        "",
    ]
    if r.alerts:
        md += ["## Alerts", ""]
        for a in r.alerts:
            md.append(f"- **{r.severity.upper()}**: {a}")
        md.append("")
    return "\n".join(md)


def post_discord(report: HealthReport) -> bool:
    """Post to Discord webhook iff DISCORD_WEBHOOK_URL is set and severity != ok."""
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        return False
    if report.severity == "ok":
        return False
    try:
        import requests
    except ImportError:
        return False

    color = {"ok": 0x2ECC71, "warn": 0xF39C12, "critical": 0xE74C3C}.get(report.severity, 0x95A5A6)
    fields = [
        {"name": "PF (rolling)", "value": str(report.pf_rolling), "inline": True},
        {"name": "WR (rolling)", "value": f"{report.win_rate_rolling}%", "inline": True},
        {"name": "Drawdown", "value": f"{report.drawdown_pct}%", "inline": True},
        {"name": "Loss streak", "value": str(report.loss_streak_current), "inline": True},
        {"name": "Total trades", "value": str(report.total_trades), "inline": True},
        {"name": "Drift vs BT", "value": f"{report.drift_vs_backtest_pct:+.1f}%", "inline": True},
    ]
    embed = {
        "title": f"Bot Health — {report.severity.upper()}",
        "description": "\n".join(f"- {a}" for a in report.alerts) or "Nominal.",
        "color": color,
        "fields": fields,
        "timestamp": report.timestamp,
    }
    try:
        resp = requests.post(webhook, json={"embeds": [embed]}, timeout=10)
        return resp.status_code in (200, 204)
    except Exception:
        return False


def write_report(report: HealthReport, out_dir: str) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    date_str = report.timestamp[:10]
    path = out / f"daily_health_{date_str}.md"
    path.write_text(render_markdown(report), encoding="utf-8")
    return str(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=os.environ.get("BOT_STATE_DB", "bot_state.db"))
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--window", type=int, default=THRESHOLDS["rolling_window"])
    p.add_argument("--json", action="store_true")
    p.add_argument("--discord", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--report-dir", default="docs/reports/health")
    args = p.parse_args()

    try:
        report = compute_health(args.db, starting_equity=args.equity, window=args.window)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(asdict(report), indent=2, default=str))
    else:
        print(render_text(report))

    if args.report:
        path = write_report(report, args.report_dir)
        print(f"\nReport written to: {path}")

    if args.discord:
        sent = post_discord(report)
        if sent:
            print("\nDiscord alert sent.")
        elif report.severity == "ok":
            print("\nDiscord skipped (severity=ok).")
        else:
            print("\nDiscord NOT sent (no webhook or send failed).", file=sys.stderr)

    return 1 if report.severity in ("warn", "critical") else 0


if __name__ == "__main__":
    sys.exit(main())
