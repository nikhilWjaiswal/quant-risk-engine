"""Value-at-Risk and Expected Shortfall estimators.

Historical VaR is nonparametric and easy to explain, but noisy in small
samples. Parametric VaR and Monte Carlo VaR both assume returns are i.i.d.
Gaussian; the Monte Carlo version is a simulation cross-check that converges to
the parametric result as the number of simulations increases.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from frp.config import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_MONTE_CARLO_SEED,
    DEFAULT_MONTE_CARLO_SIMULATIONS,
)


def _clean_returns(returns: pd.Series) -> pd.Series:
    """Normalize a return series for downstream risk calculations."""
    sample = pd.Series(returns, dtype=float).dropna()
    if sample.empty:
        raise ValueError("returns must contain at least one non-null observation")
    return sample


def _validate_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")


def historical_var(returns: pd.Series, confidence: float = DEFAULT_CONFIDENCE_LEVEL) -> float:
    """Estimate VaR from the empirical loss distribution.

    Assumes the observed return sample is representative of the future tail.
    This is conservative and transparent, but unstable when the sample is
    small or regime changes are present.
    """
    _validate_confidence(confidence)
    sample = _clean_returns(returns)
    losses = -sample.to_numpy()
    var = np.quantile(losses, confidence, method="lower")
    return float(max(0.0, var))


def parametric_var(returns: pd.Series, confidence: float = DEFAULT_CONFIDENCE_LEVEL) -> float:
    """Estimate VaR under a Gaussian return model.

    Assumes returns are i.i.d. and normally distributed. That assumption is
    often weak in real markets, especially in the left tail.
    """
    _validate_confidence(confidence)
    sample = _clean_returns(returns)
    mean_return = float(sample.mean())
    if len(sample) > 1:
        return_std = float(sample.std(ddof=1))
    else:
        return_std = 0.0

    z_value = NormalDist().inv_cdf(confidence)
    var = -mean_return + return_std * z_value
    return float(max(0.0, var))


def monte_carlo_var(
    returns: pd.Series,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    n_simulations: int = DEFAULT_MONTE_CARLO_SIMULATIONS,
    seed: int = DEFAULT_MONTE_CARLO_SEED,
) -> float:
    """Estimate VaR by simulating a fitted Gaussian return model.

    Assumes returns are i.i.d. and normally distributed. This is a simulation
    check on the same model used by parametric VaR, so it adds sampling noise
    rather than new structural information.
    """
    _validate_confidence(confidence)
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")

    sample = _clean_returns(returns)
    mean_return = float(sample.mean())
    if len(sample) > 1:
        return_std = float(sample.std(ddof=1))
    else:
        return_std = 0.0

    rng = np.random.default_rng(seed)
    simulated_returns = rng.normal(loc=mean_return, scale=return_std, size=n_simulations)
    simulated_losses = -simulated_returns
    var = np.quantile(simulated_losses, confidence)
    return float(max(0.0, var))


def conditional_var(returns: pd.Series, confidence: float = DEFAULT_CONFIDENCE_LEVEL) -> float:
    """Estimate Expected Shortfall from the empirical loss tail.

    Assumes the observed tail of returns is a reasonable proxy for the future
    tail. This is nonparametric, but tail averages can be very noisy when the
    sample is short.
    """
    _validate_confidence(confidence)
    sample = _clean_returns(returns)
    losses = -sample.to_numpy()
    var = historical_var(sample, confidence=confidence)
    tail_losses = losses[losses >= var]
    if tail_losses.size == 0:
        return float(var)
    cvar = float(tail_losses.mean())
    return float(max(0.0, cvar))
