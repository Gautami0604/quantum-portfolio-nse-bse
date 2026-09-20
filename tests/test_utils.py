import numpy as np
import pandas as pd

from src.utils import portfolio_performance


def test_portfolio_performance_equal_weights():
    mean_returns = pd.Series([0.10, 0.15, 0.08], index=["A", "B", "C"])
    cov_matrix = pd.DataFrame(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.05, 0.02],
            [0.00, 0.02, 0.03],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    weights = np.array([1 / 3, 1 / 3, 1 / 3])

    exp_return, volatility, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)

    assert exp_return == pytest_approx(0.11)
    assert volatility > 0
    assert isinstance(sharpe, float)


def pytest_approx(value, tol=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol

    return _Approx()


def test_portfolio_performance_zero_volatility_edge_case():
    mean_returns = pd.Series([0.05], index=["A"])
    cov_matrix = pd.DataFrame([[0.0]], index=["A"], columns=["A"])
    weights = np.array([1.0])

    exp_return, volatility, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)

    assert volatility == 0
    assert sharpe == 0.0  # guarded against division by zero
