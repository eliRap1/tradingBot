# TODO(audit): This file is a dead-code shim — no production module imports it.
# Production code uses ib_broker.IBBroker directly. Safe to delete once confirmed
# no external tooling or scripts depend on the `Broker` alias.
"""Backward-compat shim. Import AlpacaBroker as Broker for existing code."""
from alpaca_broker import AlpacaBroker as Broker, CRYPTO_SYMBOLS

__all__ = ["Broker", "CRYPTO_SYMBOLS"]
