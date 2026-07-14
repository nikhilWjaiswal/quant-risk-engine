"""Tests for the closed-form Markowitz optimizer."""

import numpy as np
import pandas as pd

from frp.optimization.markowitz import (
    efficient_frontier_weights,
    global_minimum_variance_weights,
    markowitz_portfolio_from_returns,
    portfolio_return,
    portfolio_variance,
)


def test_global_minimum_variance_weights_two_asset_case():
    covariance = np.array([[1.0, 0.0], [0.0, 4.0]])

    weights = global_minimum_variance_weights(covariance)

    assert np.allclose(weights.to_numpy(), np.array([0.8, 0.2]))


def test_efficient_frontier_weights_two_asset_target_return():
    expected_returns = np.array([0.10, 0.20])
    covariance = np.array([[1.0, 0.0], [0.0, 4.0]])

    weights = efficient_frontier_weights(expected_returns, covariance, target_return=0.15)

    assert np.allclose(weights.to_numpy(), np.array([0.5, 0.5]))
    assert np.isclose(portfolio_return(weights, expected_returns), 0.15)
    assert np.isclose(portfolio_variance(weights, covariance), 1.25)


def test_markowitz_portfolio_from_returns_matches_closed_form_target():
    returns = pd.DataFrame(
        {
            "asset_a": [0.08, 0.10, 0.12, 0.11],
            "asset_b": [0.18, 0.21, 0.19, 0.22],
        }
    )

    target_return = 0.15
    weights = markowitz_portfolio_from_returns(returns, target_return=target_return)
    expected_weights = efficient_frontier_weights(returns.mean(axis=0), returns.cov(), target_return)

    assert np.allclose(weights.to_numpy(), expected_weights.to_numpy())
    assert np.isclose(portfolio_return(weights, returns.mean(axis=0)), target_return)
