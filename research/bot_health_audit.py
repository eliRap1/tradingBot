"""Bot health audit using existing trade history and config."""

from __future__ import annotations

import os
import sys
from datetime import datetime

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
        total = len(trades)
        win_pct = (sum(1 for t in trades if t["pnl"] >= 0) / total * 100) if total else 0.0
        avg_win = sum(t["pnl"] for t in trades if t["pnl"] >= 0) / max(1, sum(1 for t in trades if t["pnl"] >= 0))
        avg_loss = sum(t["pnl"] for t in trades if t["pnl"] < 0) / max(1, sum(1 for t in trades if t["pnl"] < 0))

        lines = [
            "# Bot Health Audit",
            "",
            f"Generated: {datetime.utcnow().isoformat()} UTC",
            "",
            "| # | Improvement | Est. Impact | Effort | Priority |",
            "|---|-------------|-------------|--------|----------|",
            "| 1 | Earnings avoidance | Lower event gap risk | 1 day | HIGH |",
            "| 2 | Per-sector x regime weights | Better regime fit | 3 days | HIGH |",
            "| 3 | Time-of-day filter | Cleaner entries | 1 day | HIGH |",
            "| 4 | Cross-asset sizing | Lower macro drawdown | 1 day | HIGH |",
            "| 5 | Spread gate | Better execution | 1 day | MED |",
            "| 6 | ML confidence scaling | Better sizing selectivity | 2 days | MED |",
            "",
            "## Current Snapshot",
            "",
            f"- Trades in DB: `{total}`",
            f"- Win rate: `{win_pct:.1f}%`",
            f"- Avg win: `${avg_win:,.2f}`",
            f"- Avg loss: `${avg_loss:,.2f}`",
            f"- Config min score: `{self.config['signals']['min_composite_score']}`",
            f"- Config stop ATR: `{self.config['risk']['stop_loss_atr_mult']}`",
            f"- Config target ATR: `{self.config['risk']['take_profit_atr_mult']}`",
        ]
        os.makedirs(OUT_DIR, exist_ok=True)
        content = "\n".join(lines)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        return content


if __name__ == "__main__":
    print(BotHealthAudit().run())
