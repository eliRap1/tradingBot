"""Shared performance metric helpers."""

TRADING_DAYS_PER_YEAR = 252


def profit_usd(equity_curve: list, initial_equity: float) -> float:
    """Return ending-equity profit in dollars."""
    if not equity_curve:
        return 0.0
    return float(equity_curve[-1][1]) - float(initial_equity)


def apr_pct(equity_curve: list, initial_equity: float, bars: int | None = None) -> float:
    """Compounded annualized return percentage from an equity curve."""
    if not equity_curve or initial_equity <= 0:
        return 0.0
    periods = int(bars or max(1, len(equity_curve) - 1))
    if periods <= 0:
        return 0.0
    ending_equity = float(equity_curve[-1][1])
    if ending_equity <= 0:
        return -100.0
    annualized = (ending_equity / float(initial_equity)) ** (TRADING_DAYS_PER_YEAR / periods) - 1
    return round(annualized * 100, 2)


def result_row(result, label: str = "") -> dict:
    """Compact serializable metrics row for reports."""
    return {
        "label": label,
        "trades": result.total_trades,
        "win_rate": result.win_rate,
        "return_pct": result.total_return_pct,
        "profit_usd": result.profit_usd,
        "apr_pct": result.apr_pct,
        "profit_factor": result.profit_factor,
        "sharpe": result.sharpe_ratio,
        "max_dd": result.max_drawdown_pct,
        "expectancy": result.expectancy,
    }
