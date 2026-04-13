import pytest
from instrument_classifier import InstrumentClassifier

CONFIG = {
    "futures": {"contracts": [
        {"root": "NQ"}, {"root": "ES"}, {"root": "CL"}, {"root": "GC"}
    ]},
    "screener": {"crypto": ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD", "DOGE/USD"]}
}


@pytest.fixture
def clf():
    return InstrumentClassifier(CONFIG)


def test_futures_roots(clf):
    assert clf.classify("NQ") == "futures"
    assert clf.classify("ES") == "futures"
    assert clf.classify("CL") == "futures"
    assert clf.classify("GC") == "futures"


def test_crypto_symbols(clf):
    assert clf.classify("BTC/USD") == "crypto"
    assert clf.classify("ETH/USD") == "crypto"
    assert clf.classify("DOGE/USD") == "crypto"


def test_crypto_no_slash(clf):
    assert clf.classify("BTCUSD") == "crypto"
    assert clf.classify("ETHUSD") == "crypto"


def test_stock_fallback(clf):
    assert clf.classify("AAPL") == "stock"
    assert clf.classify("NVDA") == "stock"
    assert clf.classify("TSLA") == "stock"


def test_unknown_is_stock(clf):
    assert clf.classify("ZZZZ") == "stock"


def test_helpers(clf):
    assert clf.is_futures("NQ") is True
    assert clf.is_crypto("BTC/USD") is True
    assert clf.is_stock("AAPL") is True
    assert clf.is_futures("AAPL") is False
