import sys
import types
from unittest.mock import MagicMock

from contract_manager import ContractManager


CONFIG = {
    "futures": {"contracts": [
        {"root": "NQ"},
        {"root": "ES"},
        {"root": "CL"},
        {"root": "GC"},
    ]}
}


def test_get_contract_resolves_month_coded_future(monkeypatch):
    class FakeFuture:
        def __init__(
            self,
            symbol,
            lastTradeDateOrContractMonth="",
            exchange="",
            currency="USD",
            multiplier="",
        ):
            self.symbol = symbol
            self.lastTradeDateOrContractMonth = lastTradeDateOrContractMonth
            self.exchange = exchange
            self.currency = currency
            self.multiplier = multiplier

    class FakeContFuture:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "ib_insync",
        types.SimpleNamespace(Future=FakeFuture, ContFuture=FakeContFuture),
    )

    ib = MagicMock()
    ib.qualifyContracts.side_effect = lambda contract: [contract]

    cm = ContractManager(ib, CONFIG)
    contract = cm.get_contract("CLK6")

    assert contract.symbol == "CL"
    assert contract.lastTradeDateOrContractMonth == "202605"
    assert contract.exchange == "NYMEX"
    assert contract.currency == "USD"
    assert contract.multiplier == "1000"
