"""Portfolio optimization subpackage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from frp.optimization.markowitz import (
    efficient_frontier_weights,
    global_minimum_variance_weights,
    markowitz_portfolio_from_returns,
    portfolio_return,
    portfolio_variance,
)


class PortfolioOptimizer:
    """Convenience wrapper for Markowitz Mean-Variance Portfolio Optimization."""

    def __init__(
        self,
        expected_returns: pd.Series | np.ndarray,
        cov_matrix: pd.DataFrame | np.ndarray,
    ):
        self.expected_returns = np.asarray(expected_returns, dtype=float)
        self.cov_matrix = np.asarray(cov_matrix, dtype=float)

        if isinstance(expected_returns, pd.Series):
            self.labels = [str(label) for label in expected_returns.index]
        elif isinstance(cov_matrix, pd.DataFrame):
            self.labels = [str(label) for label in cov_matrix.index]
        else:
            self.labels = [f"Asset_{i+1}" for i in range(len(self.expected_returns))]

    def minimum_variance(self) -> pd.Series:
        """Calculate weights of the global minimum variance portfolio."""
        cov_df = pd.DataFrame(self.cov_matrix, index=self.labels, columns=self.labels)
        return global_minimum_variance_weights(cov_df)

    def max_sharpe(self, risk_free_rate: float = 0.02) -> pd.Series:
        """Calculate weights of the tangency portfolio maximizing Sharpe ratio."""
        excess_returns = self.expected_returns - risk_free_rate
        pinv = np.linalg.pinv(self.cov_matrix)
        unnormalized_weights = pinv @ excess_returns
        sum_weights = float(np.sum(unnormalized_weights))

        if np.isclose(sum_weights, 0.0):
            return self.minimum_variance()

        weights = unnormalized_weights / sum_weights
        return pd.Series(weights, index=self.labels, dtype=float)

    def efficient_frontier(self, n_points: int = 100) -> list[dict]:
        """Generate points along the efficient frontier."""
        min_w = self.minimum_variance().to_numpy()
        min_ret = float(min_w @ self.expected_returns)
        max_ret = float(np.max(self.expected_returns))
        spread = max(max_ret - min_ret, 0.01)

        targets = np.linspace(min_ret - 0.1 * spread, max_ret + 0.3 * spread, n_points)
        frontier = []
        for target in targets:
            try:
                w_series = efficient_frontier_weights(self.expected_returns, self.cov_matrix, target)
                w = w_series.to_numpy()
                var = float(w @ self.cov_matrix @ w)
                vol = float(np.sqrt(max(0.0, var)))
                frontier.append({
                    "return": float(target),
                    "volatility": vol,
                    "variance": var,
                    "weights": w_series,
                })
            except Exception:
                continue
        return frontier


__all__ = [
    "portfolio_return",
    "portfolio_variance",
    "global_minimum_variance_weights",
    "efficient_frontier_weights",
    "markowitz_portfolio_from_returns",
    "PortfolioOptimizer",
]
