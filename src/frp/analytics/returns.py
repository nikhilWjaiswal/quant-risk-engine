"""Return and volatility calculations."""

import numpy as np
import pandas as pd

from frp.config import TRADING_DAYS_PER_YEAR


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute daily log returns from a price series."""
    clean_prices = pd.to_numeric(pd.Series(prices), errors="coerce").dropna()
    return np.log(clean_prices / clean_prices.shift(1)).dropna()


def rolling_volatility(returns: pd.Series, window: int = 30, annualize: bool = True) -> pd.Series:
    """
    Compute rolling volatility of a returns series.

    Args:
        returns: daily log returns
        window: rolling window size in trading days
        annualize: if True, scale by sqrt(TRADING_DAYS_PER_YEAR)
    """
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol.dropna()


if __name__ == "__main__":
    from frp.data.ingest import fetch_prices

    prices = fetch_prices("AAPL")["Close"]
    rets = log_returns(prices)
    vol = rolling_volatility(rets)

    print("Log returns (last 5):")
    print(rets.tail())
    print("\n30-day annualized volatility (last 5):")
    print(vol.tail())