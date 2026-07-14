"""Closed-form Markowitz mean-variance optimization.

The implementation is intentionally unconstrained: it assumes short-selling is
allowed, there are no transaction costs, and the covariance matrix is usable in
the quadratic objective. That keeps the first version transparent and fully
auditable without a numerical optimization black box.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _coerce_square_matrix(covariance: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str]]:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("covariance must be a non-empty square matrix")

    if isinstance(covariance, pd.DataFrame):
        labels = [str(label) for label in covariance.index]
    else:
        labels = [str(index) for index in range(matrix.shape[0])]

    return matrix, labels


def _coerce_vector(
    vector: pd.Series | Sequence[float] | np.ndarray,
    labels: list[str],
    name: str,
) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    if values.ndim != 1 or values.shape[0] != len(labels):
        raise ValueError(f"{name} must be a one-dimensional vector with one value per asset")
    return values


def portfolio_return(weights: pd.Series | Sequence[float] | np.ndarray, expected_returns: pd.Series | Sequence[float] | np.ndarray) -> float:
    """Compute portfolio mean return.

    Assumes the return vector and weight vector are aligned and measured on the
    same horizon.
    """
    weight_values = np.asarray(weights, dtype=float)
    return_values = np.asarray(expected_returns, dtype=float)
    if weight_values.shape != return_values.shape:
        raise ValueError("weights and expected_returns must have the same shape")
    return float(weight_values @ return_values)


def portfolio_variance(weights: pd.Series | Sequence[float] | np.ndarray, covariance: pd.DataFrame | np.ndarray) -> float:
    """Compute portfolio variance.

    Assumes the covariance matrix is aligned with the weight ordering and is
    symmetric positive semidefinite.
    """
    weight_values = np.asarray(weights, dtype=float)
    covariance_matrix, _ = _coerce_square_matrix(covariance)
    if weight_values.ndim != 1 or weight_values.shape[0] != covariance_matrix.shape[0]:
        raise ValueError("weights must have one entry per asset")
    return float(weight_values @ covariance_matrix @ weight_values)


def global_minimum_variance_weights(covariance: pd.DataFrame | np.ndarray) -> pd.Series:
    """Compute the global minimum-variance portfolio.

    Assumes short-selling is allowed and that the covariance matrix is
    positive semidefinite. If the covariance matrix is singular, the
    Moore-Penrose pseudoinverse yields the minimum-norm solution.
    """
    covariance_matrix, labels = _coerce_square_matrix(covariance)
    ones = np.ones(covariance_matrix.shape[0], dtype=float)
    precision = np.linalg.pinv(covariance_matrix)
    numerator = precision @ ones
    denominator = float(ones @ numerator)
    if np.isclose(denominator, 0.0):
        raise ValueError("covariance matrix does not define a stable minimum-variance portfolio")

    weights = numerator / denominator
    return pd.Series(weights, index=labels, dtype=float)


def efficient_frontier_weights(
    expected_returns: pd.Series | Sequence[float] | np.ndarray,
    covariance: pd.DataFrame | np.ndarray,
    target_return: float,
) -> pd.Series:
    """Compute the unconstrained Markowitz portfolio for a target return.

    Assumes expected returns are known inputs, returns are jointly mean-variance
    efficient, and short-selling is allowed. The formula is the closed-form
    Lagrange multiplier solution to the two equality constraints.
    """
    covariance_matrix, labels = _coerce_square_matrix(covariance)
    expected_return_values = _coerce_vector(expected_returns, labels, "expected_returns")

    precision = np.linalg.pinv(covariance_matrix)
    ones = np.ones(covariance_matrix.shape[0], dtype=float)

    a_value = float(ones @ precision @ ones)
    b_value = float(ones @ precision @ expected_return_values)
    c_value = float(expected_return_values @ precision @ expected_return_values)
    discriminant = a_value * c_value - b_value**2
    if np.isclose(discriminant, 0.0):
        raise ValueError("expected returns do not define a unique efficient portfolio")

    weights = precision @ (
        ((c_value - b_value * target_return) / discriminant) * ones
        + ((a_value * target_return - b_value) / discriminant) * expected_return_values
    )
    return pd.Series(weights, index=labels, dtype=float)


def markowitz_portfolio_from_returns(
    returns: pd.DataFrame,
    target_return: float,
) -> pd.Series:
    """Estimate a Markowitz portfolio directly from a returns panel.

    Assumes the historical sample is representative of the next period's mean
    vector and covariance matrix. This is a standard but fragile plug-in
    estimator in real markets.
    """
    if returns.empty:
        raise ValueError("returns must contain at least one row and one asset")

    expected_returns = returns.mean(axis=0)
    covariance = returns.cov()
    return efficient_frontier_weights(expected_returns, covariance, target_return)
