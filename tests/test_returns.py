"""Sanity tests for return calculations against known values."""

import numpy as np
import pandas as pd
from frp.analytics.returns import log_returns, rolling_volatility


def test_log_returns_known_values():
    prices = pd.Series([100, 110, 121])
    rets = log_returns(prices)
    expected = np.log(110 / 100)
    assert np.isclose(rets.iloc[0], expected)


def test_log_returns_drops_first_nan():
    prices = pd.Series([100, 105, 110])
    rets = log_returns(prices)
    assert len(rets) == 2
    assert not rets.isna().any()


def test_rolling_volatility_length():
    prices = pd.Series(np.linspace(100, 150, 60))
    rets = log_returns(prices)
    vol = rolling_volatility(rets, window=30)
    assert len(vol) == len(rets) - 30 + 1