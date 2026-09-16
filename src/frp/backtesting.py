"""Backtesting module exposing Kupiec Proportion of Failures test."""

from __future__ import annotations

from statistics import NormalDist
import numpy as np
import pandas as pd
from scipy import stats

from frp.risk.backtest import kupiec_proportion_of_failures_test


class KupiecPOF:
    """Kupiec Proportion of Failures (POF) model backtesting validator."""

    def __init__(self, var_estimates: pd.Series | np.ndarray, realized_returns: pd.Series | np.ndarray):
        self.var_estimates = np.asarray(var_estimates, dtype=float)
        self.realized_returns = np.asarray(realized_returns, dtype=float)
        if len(self.var_estimates) != len(self.realized_returns):
            raise ValueError("var_estimates and realized_returns must have the same length")

    def test(self, alpha: float = 0.05, confidence: float = 0.95) -> dict:
        """Run Kupiec POF likelihood ratio backtest.

        Args:
            alpha: Significance level of the statistical test (e.g., 0.05).
            confidence: VaR confidence level (e.g., 0.95).

        Returns:
            Dictionary with pof, violations, expected_violations, p_value, is_valid.
        """
        n = len(self.realized_returns)
        if n == 0:
            raise ValueError("Data series cannot be empty")

        # VaR as positive loss threshold: violation is realized_return < -var
        # If var_estimates are already negative losses: check both
        thresholds = -np.abs(self.var_estimates)
        violations = int(np.sum(self.realized_returns < thresholds))
        p = 1.0 - confidence
        p_hat = violations / n
        expected_violations = n * p

        # Likelihood ratio calculation
        if violations == 0:
            log_num = n * np.log(1.0 - p)
            log_den = 0.0
        elif violations == n:
            log_num = n * np.log(p)
            log_den = 0.0
        else:
            log_num = (n - violations) * np.log(1.0 - p) + violations * np.log(p)
            log_den = (n - violations) * np.log(1.0 - p_hat) + violations * np.log(p_hat)

        lr_stat = -2.0 * (log_num - log_den)
        critical_value = NormalDist().inv_cdf(1.0 - alpha / 2.0) ** 2
        p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=1))

        return {
            "pof": float(p_hat),
            "violations": violations,
            "expected_violations": float(expected_violations),
            "lr_stat": float(lr_stat),
            "critical_value": float(critical_value),
            "p_value": float(p_value),
            "is_valid": bool(lr_stat <= critical_value),
        }


__all__ = ["KupiecPOF", "kupiec_proportion_of_failures_test"]
