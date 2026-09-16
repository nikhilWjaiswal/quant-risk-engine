"""Risk analytics subpackage."""

from __future__ import annotations

from statistics import NormalDist
import numpy as np
import pandas as pd

from frp.config import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_MONTE_CARLO_SEED,
    DEFAULT_MONTE_CARLO_SIMULATIONS,
)
from frp.risk.var import (
    conditional_var,
    historical_var,
    monte_carlo_var,
    parametric_var,
)
from frp.risk.backtest import kupiec_proportion_of_failures_test


class VaR:
    """Convenience class providing VaR estimation methods."""

    @staticmethod
    def historical(returns: pd.Series | np.ndarray, confidence: float = DEFAULT_CONFIDENCE_LEVEL) -> float:
        return historical_var(pd.Series(returns), confidence=confidence)

    @staticmethod
    def parametric(returns: pd.Series | np.ndarray, confidence: float = DEFAULT_CONFIDENCE_LEVEL) -> float:
        return parametric_var(pd.Series(returns), confidence=confidence)

    @staticmethod
    def monte_carlo(
        returns: pd.Series | np.ndarray,
        confidence: float = DEFAULT_CONFIDENCE_LEVEL,
        simulations: int = DEFAULT_MONTE_CARLO_SIMULATIONS,
        seed: int = DEFAULT_MONTE_CARLO_SEED,
    ) -> float:
        return monte_carlo_var(
            pd.Series(returns),
            confidence=confidence,
            n_simulations=simulations,
            seed=seed,
        )


class CVaR:
    """Convenience class providing CVaR (Expected Shortfall) estimation methods."""

    @staticmethod
    def historical(returns: pd.Series | np.ndarray, confidence: float = DEFAULT_CONFIDENCE_LEVEL) -> float:
        return conditional_var(pd.Series(returns), confidence=confidence)

    @staticmethod
    def parametric(returns: pd.Series | np.ndarray, confidence: float = DEFAULT_CONFIDENCE_LEVEL) -> float:
        sample = pd.Series(returns, dtype=float).dropna()
        if sample.empty:
            raise ValueError("returns must contain at least one non-null observation")
        mean_return = float(sample.mean())
        return_std = float(sample.std(ddof=1)) if len(sample) > 1 else 0.0
        dist = NormalDist()
        z = dist.inv_cdf(confidence)
        pdf_z = dist.pdf(z)
        cvar = -mean_return + return_std * (pdf_z / (1.0 - confidence))
        return float(max(0.0, cvar))

    @staticmethod
    def monte_carlo(
        returns: pd.Series | np.ndarray,
        confidence: float = DEFAULT_CONFIDENCE_LEVEL,
        simulations: int = DEFAULT_MONTE_CARLO_SIMULATIONS,
        seed: int = DEFAULT_MONTE_CARLO_SEED,
    ) -> float:
        sample = pd.Series(returns, dtype=float).dropna()
        if sample.empty:
            raise ValueError("returns must contain at least one non-null observation")
        mean_return = float(sample.mean())
        return_std = float(sample.std(ddof=1)) if len(sample) > 1 else 0.0
        rng = np.random.default_rng(seed)
        simulated_returns = rng.normal(loc=mean_return, scale=return_std, size=simulations)
        simulated_losses = -simulated_returns
        var = np.quantile(simulated_losses, confidence)
        tail_losses = simulated_losses[simulated_losses >= var]
        if tail_losses.size == 0:
            return float(max(0.0, var))
        return float(max(0.0, tail_losses.mean()))


__all__ = [
    "historical_var",
    "parametric_var",
    "monte_carlo_var",
    "conditional_var",
    "kupiec_proportion_of_failures_test",
    "VaR",
    "CVaR",
]
