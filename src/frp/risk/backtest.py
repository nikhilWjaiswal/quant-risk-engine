"""Backtesting helpers for VaR models."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from frp.config import DEFAULT_CONFIDENCE_LEVEL


def _clean_returns(returns: pd.Series) -> pd.Series:
    sample = pd.Series(returns, dtype=float).dropna()
    if sample.empty:
        raise ValueError("returns must contain at least one non-null observation")
    return sample


def kupiec_proportion_of_failures_test(
    returns: pd.Series,
    var: float,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    significance_level: float = 0.05,
) -> tuple[bool, int, float]:
    """Run the Kupiec unconditional coverage test for a fixed VaR threshold.

    Assumes each return is an independent Bernoulli trial against the same VaR
    forecast. This tests unconditional coverage only; it does not detect
    violation clustering.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if not 0.0 < significance_level < 1.0:
        raise ValueError("significance_level must be strictly between 0 and 1")
    if var < 0.0:
        raise ValueError("var must be non-negative")

    sample = _clean_returns(returns)
    observation_count = len(sample)
    tail_probability = 1.0 - confidence
    violations = int((sample < -var).sum())
    expected_violations = observation_count * tail_probability

    if violations == 0:
        log_numerator = observation_count * np.log(1.0 - tail_probability)
        log_denominator = observation_count * np.log(1.0)
    elif violations == observation_count:
        log_numerator = observation_count * np.log(tail_probability)
        log_denominator = observation_count * np.log(1.0)
    else:
        observed_rate = violations / observation_count
        log_numerator = (
            (observation_count - violations) * np.log(1.0 - tail_probability)
            + violations * np.log(tail_probability)
        )
        log_denominator = (
            (observation_count - violations) * np.log(1.0 - observed_rate)
            + violations * np.log(observed_rate)
        )

    likelihood_ratio = -2.0 * (log_numerator - log_denominator)
    critical_value = NormalDist().inv_cdf(1.0 - significance_level / 2.0) ** 2
    passes = bool(likelihood_ratio <= critical_value)
    return passes, violations, float(expected_violations)
