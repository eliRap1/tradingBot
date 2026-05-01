"""Unit tests for insider_flow and relative_strength edges."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from edge.insider_flow import InsiderFlow, InsiderSignal
from edge.relative_strength import RelativeStrength


# ── InsiderFlow ──────────────────────────────────────────────


class TestInsiderFlow:
    def test_disabled(self):
        ef = InsiderFlow({"edge": {"insider_flow": False}})
        sig = ef.evaluate("AAPL")
        assert not sig.cluster
        assert sig.size_mult == 1.0

    def test_missing_cik_returns_neutral(self):
        ef = InsiderFlow({"edge": {"insider_flow": True}})
        ef._cik_loaded = True  # skip network load
        ef._symbol_cik = {}
        sig = ef.evaluate("UNKNOWN")
        assert sig.buys == 0
        assert not sig.cluster

    def test_cluster_detection(self):
        """Mock Atom feed + inner Form 4 XML with purchase codes."""
        ef = InsiderFlow({"edge": {"insider_flow": True, "insider_min_cluster": 2}})
        ef._cik_loaded = True
        ef._symbol_cik = {"AAPL": "0000320193"}

        from datetime import datetime, timedelta
        days = [(datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
                for i in (1, 3, 5)]
        acc_nos = ["0001234567-26-000001", "0001234567-26-000002", "0001234567-26-000003"]
        entries = "\n".join(
            f"""<entry>
                <title>4 - Statement of changes in beneficial ownership</title>
                <summary type="html">&lt;b&gt;Filed:&lt;/b&gt; {d} &lt;b&gt;AccNo:&lt;/b&gt; {acc} &lt;b&gt;Size:&lt;/b&gt; 10 KB</summary>
                <updated>{d}T12:00:00Z</updated>
            </entry>"""
            for d, acc in zip(days, acc_nos)
        )
        atom_xml = f"""<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>"""

        inner_xml = "<ownershipDocument><nonDerivativeTable><nonDerivativeTransaction><transactionAmounts><transactionCode>P</transactionCode></transactionAmounts></nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"

        class _AtomResp:
            ok = True
            text = atom_xml

        class _IndexResp:
            ok = True
            def json(self):
                return {"directory": {"item": [{"name": "form4.xml"}]}}

        class _InnerResp:
            ok = True
            text = inner_xml

        def fake_get(url, **kwargs):
            if "browse-edgar" in url:
                return _AtomResp()
            if url.endswith("index.json"):
                return _IndexResp()
            return _InnerResp()

        with patch("edge.insider_flow.requests.get", side_effect=fake_get):
            sig = ef.evaluate("AAPL")
        assert sig.buys == 3
        assert sig.sells == 0
        assert sig.cluster
        assert sig.size_mult == 1.20
        assert sig.block_short

    def test_cache_hit(self):
        ef = InsiderFlow({"edge": {"insider_flow": True}})
        ef._cache["AAPL"] = (9e18, InsiderSignal(cluster=True, size_mult=1.5))
        # Future timestamp = always fresh; no network call
        with patch("edge.insider_flow.requests.get") as mock_get:
            sig = ef.evaluate("AAPL")
            mock_get.assert_not_called()
        assert sig.size_mult == 1.5


# ── RelativeStrength ─────────────────────────────────────────


def _bars(returns_map: dict[str, float], periods: int = 25) -> dict:
    """Build {symbol: df} where each df's close yields the given total return."""
    out = {}
    for sym, ret in returns_map.items():
        start = 100.0
        end = start * (1.0 + ret)
        closes = [start + (end - start) * i / (periods - 1) for i in range(periods)]
        out[sym] = pd.DataFrame({
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000_000] * periods,
        })
    return out


class TestRelativeStrength:
    def test_disabled(self):
        data = MagicMock()
        rs = RelativeStrength(data, {"edge": {"relative_strength": False}})
        sig = rs.evaluate("AAPL", ["AAPL", "MSFT"])
        assert sig.bucket == "mid"
        assert sig.long_size_mult == 1.0

    def test_top_rank_boost(self):
        data = MagicMock()
        returns = {
            "AAPL": 0.15,  # strongest
            "MSFT": 0.10,
            "NVDA": 0.05,
            "TSLA": 0.00,
            "META": -0.05,  # weakest
        }
        data.get_bars.return_value = _bars(returns)
        rs = RelativeStrength(data, {"edge": {"relative_strength": True, "rs_lookback_days": 20}})
        sig = rs.evaluate("AAPL", list(returns))
        assert sig.bucket == "top"
        assert sig.long_size_mult == 1.20

    def test_bottom_rank_block_long(self):
        data = MagicMock()
        returns = {
            "AAPL": 0.15,
            "MSFT": 0.10,
            "NVDA": 0.05,
            "TSLA": 0.00,
            "META": -0.05,
        }
        data.get_bars.return_value = _bars(returns)
        rs = RelativeStrength(data, {"edge": {"relative_strength": True, "rs_lookback_days": 20}})
        sig = rs.evaluate("META", list(returns))
        assert sig.bucket == "bottom"
        assert sig.block_long
        assert sig.allow_short

    def test_mid_rank_neutral(self):
        data = MagicMock()
        returns = {f"S{i}": i * 0.01 for i in range(10)}  # S0..S9
        data.get_bars.return_value = _bars(returns)
        rs = RelativeStrength(data, {"edge": {"relative_strength": True, "rs_lookback_days": 20}})
        sig = rs.evaluate("S5", list(returns))
        assert sig.bucket in ("mid", "upper", "lower")
