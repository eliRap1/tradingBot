"""Backward-compat shim. Import AlpacaBroker as Broker for existing code."""
from alpaca_broker import AlpacaBroker as Broker, CRYPTO_SYMBOLS

__all__ = ["Broker", "CRYPTO_SYMBOLS"]
