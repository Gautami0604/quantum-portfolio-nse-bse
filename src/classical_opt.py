"""
classical_opt.py

Classical mean-variance (Markowitz) portfolio optimization, used as the
baseline against which the quantum (QAOA) approach is compared.
"""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np
import pandas as pd

from src.utils import OptimizationResult, portfolio_performance


def max_sharpe_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.06,
    long_only: bool = True,
) -> OptimizationResult:
    """
    Solve for the maximum-Sharpe-ratio portfolio via convex optimization.

    Uses the standard trick of maximizing return for a fixed risk budget
    across a frontier, then picking the point with the best Sharpe ratio,
    since directly maximizing Sharpe is not convex.
    """
    start = time.perf_counter()

    n = len(mean_returns)
    mu = mean_returns.values
    sigma = cov_matrix.values

    best_result = None
    best_sharpe = -np.inf
    solve_attempts = 0
    solve_failures = 0

    target_vols = np.linspace(0.05, 0.6, 40)

    candidate_solvers = [cp.CLARABEL, cp.SCS, cp.ECOS]

    for target_vol in target_vols:
        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= 0)
        constraints.append(cp.quad_form(w, sigma) <= target_vol ** 2)

        objective = cp.Maximize(mu @ w)
        problem = cp.Problem(objective, constraints)

        solved = False
        for solver in candidate_solvers:
            solve_attempts += 1
            try:
                problem.solve(solver=solver)
                if w.value is not None and problem.status in ("optimal", "optimal_inaccurate"):
                    solved = True
                    break
            except (cp.error.SolverError, cp.error.DCPError):
                solve_failures += 1
                continue

        if not solved or w.value is None:
            continue

        exp_ret, vol, sharpe = portfolio_performance(
            w.value, mean_returns, cov_matrix, risk_free_rate
        )
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_result = (w.value, exp_ret, vol, sharpe)

    runtime = time.perf_counter() - start

    if best_result is None:
        raise RuntimeError(
            "Classical optimization failed to find a feasible portfolio "
            f"({solve_failures}/{solve_attempts} solver attempts failed). "
            "This usually means: (1) no installed cvxpy solver could handle "
            "the problem — try `pip install clarabel scs`, or (2) the "
            "covariance matrix has NaNs/extreme values from the price data — "
            "check that data/ prices downloaded cleanly."
        )

    weights, exp_ret, vol, sharpe = best_result
    return OptimizationResult(
        method="classical",
        weights=dict(zip(mean_returns.index, weights)),
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        runtime_seconds=runtime,
        backend="cvxpy",
    )