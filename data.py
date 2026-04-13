"""Backward-compat shim. Import AlpacaDataFetcher as DataFetcher for existing code."""
from alpaca_data import AlpacaDataFetcher as DataFetcher, CRYPTO_SYMBOLS

__all__ = ["DataFetcher", "CRYPTO_SYMBOLS"]
