"""Tests for VaR and Expected Shortfall calculations."""

from statistics import NormalDist

import numpy as np
import pandas as pd

from frp.risk.var import conditional_var, historical_var, monte_carlo_var, parametric_var


def test_historical_var_matches_lower_empirical_tail():
    returns = pd.Series([-0.10, -0.08, -0.05, -0.02, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05])

    result = historical_var(returns, confidence=0.90)

    assert np.isclose(result, 0.08)


def test_parametric_var_matches_closed_form_normal_formula():
    returns = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
    confidence = 0.99

    result = parametric_var(returns, confidence=confidence)
    mean_return = float(returns.mean())
    return_std = float(returns.std(ddof=1))
    expected = max(0.0, -mean_return + return_std * NormalDist().inv_cdf(confidence))

    assert np.isclose(result, expected)


def test_monte_carlo_var_is_close_to_parametric_reference():
    returns = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
    confidence = 0.99

    result = monte_carlo_var(returns, confidence=confidence, n_simulations=10_000, seed=42)
    expected = parametric_var(returns, confidence=confidence)

    assert np.isclose(result, expected, atol=0.15)


def test_conditional_var_matches_empirical_tail_average():
    returns = pd.Series([-0.10, -0.08, -0.05, -0.02, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05])

    result = conditional_var(returns, confidence=0.90)

    assert np.isclose(result, 0.09)


def test_conditional_var_is_at_least_var_for_same_confidence():
    returns = pd.Series([-0.10, -0.08, -0.05, -0.02, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05])

    var = historical_var(returns, confidence=0.90)
    cvar = conditional_var(returns, confidence=0.90)

    assert cvar >= var
