"""Tests for Kupiec VaR backtesting."""

import pandas as pd
import numpy as np

from frp.risk.backtest import kupiec_proportion_of_failures_test


def test_kupiec_test_passes_when_violations_match_expected_rate():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00, 0.04, 0.01, -0.06, 0.02, 0.01])

    passes, violations, expected_violations = kupiec_proportion_of_failures_test(
        returns,
        var=0.05,
        confidence=0.90,
        significance_level=0.05,
    )

    assert passes is True
    assert violations == 1
    assert np.isclose(expected_violations, 1.0)


def test_kupiec_test_fails_for_material_exceedance_miss():
    returns = pd.Series([-0.07, -0.08, -0.06, -0.09, 0.02, 0.03, 0.00, 0.01, 0.02, 0.04])

    passes, violations, expected_violations = kupiec_proportion_of_failures_test(
        returns,
        var=0.05,
        confidence=0.90,
        significance_level=0.05,
    )

    assert passes is False
    assert violations == 4
    assert np.isclose(expected_violations, 1.0)
