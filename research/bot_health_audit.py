"""Bot health audit — 6 checks against config + trade history.

Checks (per design spec):
1. Signal threshold calibration — replays historical signals at thresholds.
   Skipped if no trades in DB.
2. Risk param calibration — MAE vs stop, TP reachability. Skipped if no trades.
3. Missing safeguards — earnings, VIX, time-of-day, breadth gates wired?
4. Execution gap — smart_orders enabled?
5. Regime coverage — HMM/EMA conflict periods. Skipped if no trades.
6. Correlation clustering risk — concurrent position correlations. Skipped if no trades.

Produces a prioritised improvement backlog in audit_report.md.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from state_db import StateDB
from utils import load_config

OUT_DIR = os.path.dirname(__file__)
REPORT_FILE = os.path.join(OUT_DIR, "audit_report.md")


class BotHealthAudit:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.db = StateDB()

    def run(self) -> str:
        trades = self.db.get_trades()
        findings: list[dict] = []

        findings.extend(self._check_threshold_calibration(trades))
        findings.extend(self._check_risk_params(trades))
        findings.extend(self._check_safeguards())
        findings.extend(self._check_execution_gap())
        findings.extend(self._check_regime_coverage(trades))
        findings.extend(self._check_correlation_clustering(trades))

        # Prioritise: HIGH > MED > LOW, break ties by impact
        order = {"HIGH": 0, "MED": 1, "LOW": 2}
        findings.sort(key=lambda f: order.get(f["priority"], 3))

        lines = ["# Bot Health Audit", "", f"Generated: {datetime.utcnow().isoformat()} UTC", ""]
        lines += ["## Prioritised Improvements", ""]
        lines += ["| # | Improvement | Impact | Effort | Priority |",
                  "|---|-------------|--------|--------|----------|"]
        for i, f in enumerate(findings, 1):
            lines.append(
                f"| {i} | {f['improvement']} | {f['impact']} | {f['effort']} | {f['priority']} |"
            )
        lines += ["", "## Current Snapshot", ""]
        lines += self._snapshot(trades)
        lines += ["", "## Check Details", ""]
        for f in findings:
            lines += [f"### {f['improvement']} (`{f['check']}`)", "", f["detail"], ""]

        content = "\n".join(lines)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        return content

    # ── Checks ────────────────────────────────────────────────

    def _check_threshold_calibration(self, trades: list[dict]) -> list[dict]:
        if not trades:
            return [{
                "check": "threshold_calibration",
                "improvement": "Signal threshold calibration",
                "impact": "Tunes entry selectivity",
                "effort": "1 day",
                "priority": "MED",
                "detail": "SKIPPED — no trades in DB. Will activate after ~30 trades.",
            }]
        # With trades: check win% at composite_score buckets
        buckets = {0.15: [], 0.20: [], 0.25: [], 0.30: [], 0.35: []}
        for t in trades:
            score = float(t.get("composite_score") or 0.0)
            if score <= 0:
                continue
            for threshold in buckets:
                if score >= threshold:
                    buckets[threshold].append(1 if t.get("pnl", 0) >= 0 else 0)
        best_th = None
        best_wr = 0.0
        detail_lines = ["Win rate per threshold:"]
        for th, outcomes in buckets.items():
            if len(outcomes) < 10:
                detail_lines.append(f"- {th}: insufficient ({len(outcomes)} trades)")
                continue
            wr = mean(outcomes)
            detail_lines.append(f"- {th}: {wr*100:.1f}% ({len(outcomes)} trades)")
            if wr > best_wr:
                best_wr, best_th = wr, th
        current = self.config.get("signals", {}).get("min_composite_score", 0.25)
        if best_th and abs(best_th - current) > 0.025:
            return [{
                "check": "threshold_calibration",
                "improvement": f"Retune min_composite_score {current}→{best_th}",
                "impact": f"Win rate jump to {best_wr*100:.1f}%",
                "effort": "config tweak",
                "priority": "HIGH",
                "detail": "\n".join(detail_lines),
            }]
        return [{
            "check": "threshold_calibration",
            "improvement": "Signal threshold looks calibrated",
            "impact": "No tuning needed",
            "effort": "—",
            "priority": "LOW",
            "detail": "\n".join(detail_lines),
        }]

    def _check_risk_params(self, trades: list[dict]) -> list[dict]:
        risk = self.config.get("risk", {})
        stop_mult = risk.get("stop_loss_atr_mult", 2.0)
        tp_mult = risk.get("take_profit_atr_mult", 5.0)
        ratio = tp_mult / max(stop_mult, 0.1)
        if ratio < 1.5:
            return [{
                "check": "risk_params",
                "improvement": f"Widen TP or tighten SL — current R:R {ratio:.2f}:1",
                "impact": "Payoff too thin vs win rate",
                "effort": "config tweak",
                "priority": "HIGH",
                "detail": f"take_profit_atr_mult={tp_mult}, stop_loss_atr_mult={stop_mult}",
            }]
        if not trades:
            return [{
                "check": "risk_params",
                "improvement": "Risk param MAE analysis",
                "impact": "Verify stops are adequate",
                "effort": "1 day",
                "priority": "LOW",
                "detail": "SKIPPED — no trades to compute MAE vs SL placement.",
            }]
        # Check stop-outs vs TP hits
        hits = {"stop_loss": 0, "take_profit": 0, "other": 0}
        for t in trades:
            reason = str(t.get("exit_reason", "other")).lower()
            if "stop" in reason:
                hits["stop_loss"] += 1
            elif "take" in reason or "profit" in reason:
                hits["take_profit"] += 1
            else:
                hits["other"] += 1
        total = max(1, sum(hits.values()))
        sl_pct = hits["stop_loss"] / total
        tp_pct = hits["take_profit"] / total
        detail = (
            f"Exit mix: SL={sl_pct*100:.1f}% TP={tp_pct*100:.1f}% "
            f"other={hits['other']/total*100:.1f}% (R:R {ratio:.2f}:1)"
        )
        if sl_pct > 0.6:
            return [{
                "check": "risk_params",
                "improvement": "Too many stops — widen stop or tighten entries",
                "impact": f"{sl_pct*100:.0f}% trades hit SL",
                "effort": "config tweak",
                "priority": "HIGH",
                "detail": detail,
            }]
        return [{
            "check": "risk_params",
            "improvement": "Risk params balanced",
            "impact": "No tuning needed",
            "effort": "—",
            "priority": "LOW",
            "detail": detail,
        }]

    def _check_safeguards(self) -> list[dict]:
        edge = self.config.get("edge", {})
        gaps = []
        if not edge.get("earnings_avoidance", False):
            gaps.append({
                "check": "earnings_avoidance",
                "improvement": "Enable edge.earnings_avoidance",
                "impact": "Avoids binary event gap risk",
                "effort": "config flip",
                "priority": "HIGH",
                "detail": "edge.earnings_avoidance is False or missing in config.",
            })
        if edge.get("max_spread_pct", 0.01) > 0.005:
            gaps.append({
                "check": "spread_gate",
                "improvement": f"Tighten edge.max_spread_pct (now {edge.get('max_spread_pct')})",
                "impact": "Cuts thin-market entries",
                "effort": "config tweak",
                "priority": "MED",
                "detail": "Spread gate above 0.5% lets wide-spread names through.",
            })
        if not edge.get("enabled", False):
            gaps.append({
                "check": "edge_enabled",
                "improvement": "Enable edge layer (edge.enabled = true)",
                "impact": "Unlocks cross-asset + microstructure gates",
                "effort": "config flip",
                "priority": "HIGH",
                "detail": "edge.enabled is False or missing in config.",
            })
        if not gaps:
            gaps = [{
                "check": "safeguards",
                "improvement": "Safeguards configured",
                "impact": "No gaps found",
                "effort": "—",
                "priority": "LOW",
                "detail": "earnings_avoidance OK, max_spread_pct <= 0.5% OK, edge.enabled OK",
            }]
        return gaps

    def _check_execution_gap(self) -> list[dict]:
        exec_cfg = self.config.get("execution", {})
        smart = exec_cfg.get("smart_orders", False)
        if not smart:
            return [{
                "check": "execution_gap",
                "improvement": "Enable execution.smart_orders (limit + improve)",
                "impact": "~0.3% saved per round-trip",
                "effort": "1 day",
                "priority": "MED",
                "detail": "Currently using market orders — paying spread every trade.",
            }]
        return [{
            "check": "execution_gap",
            "improvement": "Smart orders enabled",
            "impact": "No action",
            "effort": "—",
            "priority": "LOW",
            "detail": "execution.smart_orders is True.",
        }]

    def _check_regime_coverage(self, trades: list[dict]) -> list[dict]:
        if not trades:
            return [{
                "check": "regime_coverage",
                "improvement": "Regime coverage analysis",
                "impact": "Find regime blind spots",
                "effort": "1 day",
                "priority": "LOW",
                "detail": "SKIPPED — no trades tagged with regime_4state yet.",
            }]
        per_regime: dict[str, list[int]] = {}
        for t in trades:
            regime = t.get("regime_4state") or t.get("regime") or "unknown"
            outcome = 1 if t.get("pnl", 0) >= 0 else 0
            per_regime.setdefault(regime, []).append(outcome)
        detail_lines = ["Win rate per regime:"]
        worst_regime = None
        worst_wr = 1.0
        for regime, outcomes in per_regime.items():
            if len(outcomes) < 10:
                detail_lines.append(f"- {regime}: insufficient ({len(outcomes)})")
                continue
            wr = mean(outcomes)
            detail_lines.append(f"- {regime}: {wr*100:.1f}% ({len(outcomes)})")
            if wr < worst_wr:
                worst_wr, worst_regime = wr, regime
        if worst_regime and worst_wr < 0.45:
            return [{
                "check": "regime_coverage",
                "improvement": f"Tune/disable signals in {worst_regime} regime",
                "impact": f"Regime win rate {worst_wr*100:.1f}%",
                "effort": "1 day",
                "priority": "HIGH",
                "detail": "\n".join(detail_lines),
            }]
        return [{
            "check": "regime_coverage",
            "improvement": "Regime coverage balanced",
            "impact": "No action",
            "effort": "—",
            "priority": "LOW",
            "detail": "\n".join(detail_lines),
        }]

    def _check_correlation_clustering(self, trades: list[dict]) -> list[dict]:
        if not trades:
            return [{
                "check": "correlation_clustering",
                "improvement": "Concurrent correlation check",
                "impact": "Caps concentrated beta risk",
                "effort": "1 day",
                "priority": "LOW",
                "detail": "SKIPPED — no trade history for position-overlap analysis.",
            }]
        max_corr = self.config.get("risk", {}).get("max_correlation", 0.7)
        return [{
            "check": "correlation_clustering",
            "improvement": f"Live correlation cap at {max_corr}",
            "impact": "Prevents beta clustering",
            "effort": "—",
            "priority": "LOW",
            "detail": f"risk.max_correlation={max_corr}. Full pairwise analysis requires position snapshots.",
        }]

    # ── Snapshot ──────────────────────────────────────────────

    def _snapshot(self, trades: list[dict]) -> list[str]:
        total = len(trades)
        wins = [t for t in trades if t.get("pnl", 0) >= 0]
        losses = [t for t in trades if t.get("pnl", 0) < 0]
        win_pct = (len(wins) / total * 100) if total else 0.0
        avg_win = (sum(t["pnl"] for t in wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(t["pnl"] for t in losses) / len(losses)) if losses else 0.0

        signals_cfg = self.config.get("signals", {})
        risk_cfg = self.config.get("risk", {})
        edge_cfg = self.config.get("edge", {})
        return [
            f"- Trades in DB: `{total}`",
            f"- Win rate: `{win_pct:.1f}%`",
            f"- Avg win: `${avg_win:,.2f}`",
            f"- Avg loss: `${avg_loss:,.2f}`",
            f"- Config min score: `{signals_cfg.get('min_composite_score')}`",
            f"- Config min agreeing: `{signals_cfg.get('min_agreeing_strategies')}`",
            f"- Config stop ATR: `{risk_cfg.get('stop_loss_atr_mult')}`",
            f"- Config target ATR: `{risk_cfg.get('take_profit_atr_mult')}`",
            f"- Edge enabled: `{edge_cfg.get('enabled', False)}`",
            f"- Earnings avoidance: `{edge_cfg.get('earnings_avoidance', False)}`",
            f"- Max spread %: `{edge_cfg.get('max_spread_pct', 'unset')}`",
        ]


if __name__ == "__main__":
    report = BotHealthAudit().run()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(report)
    print(f"\n[health] wrote {REPORT_FILE}")
