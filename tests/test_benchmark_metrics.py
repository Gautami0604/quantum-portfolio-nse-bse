import numpy as np
import pandas as pd

from src.benchmark_metrics import (
    information_ratio,
    transfer_coefficient,
    value_at_risk,
)


def _sample_mean_cov():
    mean_returns = pd.Series([0.12, 0.09, 0.15, 0.07], index=["A", "B", "C", "D"])
    cov_matrix = pd.DataFrame(
        np.array(
            [
                [0.05, 0.01, 0.00, 0.02],
                [0.01, 0.04, 0.01, 0.00],
                [0.00, 0.01, 0.06, 0.01],
                [0.02, 0.00, 0.01, 0.03],
            ]
        ),
        index=["A", "B", "C", "D"],
        columns=["A", "B", "C", "D"],
    )
    return mean_returns, cov_matrix


def test_information_ratio_zero_when_identical_to_benchmark():
    idx = pd.date_range("2025-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, size=100), index=idx)

    active_ret, tracking_err, ir = information_ratio(returns, returns.copy())

    assert tracking_err == 0
    assert ir == 0.0
    assert active_ret == 0.0


def test_transfer_coefficient_range():
    mean_returns, cov_matrix = _sample_mean_cov()
    weights = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}

    tc = transfer_coefficient(weights, mean_returns, cov_matrix)

    assert -1.0 <= tc <= 1.0


def test_value_at_risk_historical_is_positive_for_typical_losses():
    idx = pd.date_range("2025-01-01", periods=252, freq="B")
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0003, 0.015, size=252), index=idx)

    var_95 = value_at_risk(returns, confidence=0.95, method="historical")
    var_99 = value_at_risk(returns, confidence=0.99, method="historical")

    assert var_95 > 0
    assert var_99 >= var_95  # 99% VaR should be at least as large as 95% VaR
