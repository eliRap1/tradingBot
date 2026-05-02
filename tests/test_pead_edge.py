"""Unit tests for edge.pead PEADEdge."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from edge.pead import PEADEdge, EarningsEvent


@pytest.fixture
def cfg(tmp_path):
    cache = tmp_path / "earnings_cache.json"
    return {"edge": {"pead": {
        "enabled": True,
        "drift_window_days": 60,
        "top_threshold_sue": 1.5,
        "bottom_threshold_sue": -1.5,
        "size_mult": 1.15,
        "cache_path": str(cache),
    }}}


def test_compute_sue_basic():
    history = [-0.05, 0.02, -0.03, 0.04]  # past surprises
    sue = PEADEdge.compute_sue(actual=1.20, consensus=1.00, history=history)
    assert sue is not None and sue > 0


def test_compute_sue_too_few_returns_none():
    sue = PEADEdge.compute_sue(actual=1.0, consensus=0.9, history=[0.05])
    assert sue is None


def test_surprise_pct_basic():
    sp = PEADEdge.compute_surprise_pct(1.20, 1.00)
    assert sp == pytest.approx(0.20)


def test_surprise_pct_handles_zero_consensus():
    sp = PEADEdge.compute_surprise_pct(0.05, 0.0)
    assert sp is not None


def test_evaluate_disabled(cfg):
    cfg["edge"]["pead"]["enabled"] = False
    e = PEADEdge(cfg)
    assert e.evaluate("AAPL").active is False


def test_evaluate_no_data(cfg):
    e = PEADEdge(cfg)
    sig = e.evaluate("XYZ")
    assert sig.active is False
    assert "no earnings" in sig.reason


def test_evaluate_top_tier_within_window(cfg, tmp_path):
    cache = tmp_path / "earnings_cache.json"
    report = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    cache.write_text(json.dumps({
        "AAPL": {
            "report_date": report,
            "actual_eps": 1.30,
            "consensus_eps": 1.00,
            "surprise_pct": 0.30,
            "sue": 2.50,
        }
    }))
    cfg["edge"]["pead"]["cache_path"] = str(cache)
    e = PEADEdge(cfg)
    sig = e.evaluate("AAPL")
    assert sig.active is True
    assert sig.tier == "top"
    assert sig.allow_short is False
    assert sig.size_mult == 1.15


def test_evaluate_bottom_tier(cfg, tmp_path):
    cache = tmp_path / "earnings_cache.json"
    report = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    cache.write_text(json.dumps({
        "AAPL": {
            "report_date": report,
            "actual_eps": 0.70,
            "consensus_eps": 1.00,
            "surprise_pct": -0.30,
            "sue": -2.0,
        }
    }))
    cfg["edge"]["pead"]["cache_path"] = str(cache)
    e = PEADEdge(cfg)
    sig = e.evaluate("AAPL")
    assert sig.tier == "bottom"
    assert sig.allow_long is False


def test_evaluate_outside_window(cfg, tmp_path):
    cache = tmp_path / "earnings_cache.json"
    report = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    cache.write_text(json.dumps({
        "AAPL": {
            "report_date": report,
            "actual_eps": 1.30,
            "consensus_eps": 1.00,
            "sue": 2.5,
        }
    }))
    cfg["edge"]["pead"]["cache_path"] = str(cache)
    e = PEADEdge(cfg)
    sig = e.evaluate("AAPL")
    assert sig.active is False
    assert "outside" in sig.reason
