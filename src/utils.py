"""
utils.py

Shared risk/return metric helpers used by both the classical and quantum
optimizers, plus a common results container for fair comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualized_return(returns: pd.DataFrame) -> pd.Series:
    return returns.mean() * TRADING_DAYS_PER_YEAR


def annualized_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * TRADING_DAYS_PER_YEAR


def portfolio_performance(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.06,
) -> tuple[float, float, float]:
    """
    Returns (expected_return, volatility, sharpe_ratio) for a given weight
    vector. risk_free_rate defaults to a rough Indian government bond yield;
    adjust as needed.
    """
    weights = np.asarray(weights)
    exp_return = float(np.dot(weights, mean_returns))
    volatility = float(np.sqrt(weights.T @ cov_matrix.values @ weights))
    sharpe = (exp_return - risk_free_rate) / volatility if volatility > 0 else 0.0
    return exp_return, volatility, sharpe


@dataclass
class OptimizationResult:
    """Common container so classical and quantum results are directly comparable."""

    method: str                     # "classical" or "quantum"
    weights: dict = field(default_factory=dict)   # ticker -> weight
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    runtime_seconds: float = 0.0
    backend: str = "n/a"            # e.g. "cvxpy", "aer_simulator", "ibm_brisbane"

    def summary(self) -> str:
        held = {k: round(v, 3) for k, v in self.weights.items() if v > 1e-4}
        return (
            f"[{self.method} | backend={self.backend}]\n"
            f"  Weights: {held}\n"
            f"  Expected return: {self.expected_return:.2%}\n"
            f"  Volatility:      {self.volatility:.2%}\n"
            f"  Sharpe ratio:    {self.sharpe_ratio:.3f}\n"
            f"  Runtime:         {self.runtime_seconds:.2f}s"
        )
