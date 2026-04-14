import pytest
import json

CONFIG = {
    "futures": {"contracts": [{"root": "NQ"}, {"root": "ES"}, {"root": "CL"}, {"root": "GC"}]},
    "screener": {"crypto": ["BTC/USD", "ETH/USD", "SOL/USD"]},
    "strategies": {
        "momentum": {"weight": 0.25},
        "mean_reversion": {"weight": 0.15},
        "breakout": {"weight": 0.20},
        "supertrend": {"weight": 0.25},
        "stoch_rsi": {"weight": 0.15},
        "vwap_reclaim": {"weight": 0.15},
        "gap": {"weight": 0.15},
        "liquidity_sweep": {"weight": 0.20},
    }
}


@pytest.fixture
def router():
    from strategy_router import StrategyRouter
    return StrategyRouter(CONFIG)


def test_stocks_include_mean_reversion(router):
    strats = router.get_strategies("stock")
    assert "mean_reversion" in strats


def test_stocks_include_gap(router):
    strats = router.get_strategies("stock")
    assert "gap" in strats


def test_crypto_excludes_mean_reversion(router):
    strats = router.get_strategies("crypto")
    assert "mean_reversion" not in strats


def test_crypto_excludes_gap(router):
    strats = router.get_strategies("crypto")
    assert "gap" not in strats


def test_crypto_excludes_vwap_reclaim(router):
    strats = router.get_strategies("crypto")
    assert "vwap_reclaim" not in strats


def test_futures_includes_futures_trend(router):
    strats = router.get_strategies("futures")
    assert "futures_trend" in strats


def test_futures_excludes_mean_reversion(router):
    strats = router.get_strategies("futures")
    assert "mean_reversion" not in strats


def test_futures_excludes_gap(router):
    strats = router.get_strategies("futures")
    assert "gap" not in strats


def test_weights_sum_to_one_stock(router):
    strats = router.get_strategies("stock")
    total = sum(strats.values())
    assert abs(total - 1.0) < 0.001


def test_weights_sum_to_one_crypto(router):
    strats = router.get_strategies("crypto")
    total = sum(strats.values())
    assert abs(total - 1.0) < 0.001


def test_weights_sum_to_one_futures(router):
    strats = router.get_strategies("futures")
    total = sum(strats.values())
    assert abs(total - 1.0) < 0.001


def test_all_weights_positive(router):
    for itype in ("stock", "crypto", "futures"):
        strats = router.get_strategies(itype)
        for name, w in strats.items():
            assert w > 0, f"{itype}/{name} weight is not positive"


def test_unknown_type_defaults_to_stock(router):
    strats = router.get_strategies("unknown")
    assert "mean_reversion" in strats  # stock has mean_reversion


def test_sector_regime_override(tmp_path, monkeypatch):
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    (research_dir / "sector_weights.json").write_text(json.dumps({
        "tech": {
            "bull_trending": {"momentum": 0.6, "breakout": 0.4},
            "_fallback": {"supertrend": 1.0}
        }
    }))
    import strategy_router
    monkeypatch.setattr(strategy_router.os.path, "dirname", lambda _: str(tmp_path))
    router = strategy_router.StrategyRouter(CONFIG)
    assert router.get_strategies("stock", sector="tech", regime="bull_trending") == {
        "momentum": 0.6, "breakout": 0.4
    }
    assert router.get_strategies("stock", sector="tech", regime="bear_choppy") == {
        "supertrend": 1.0
    }
