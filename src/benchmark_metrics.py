"""
benchmark_metrics.py

Computes portfolio performance/risk metrics against a benchmark index
(default: Nifty 50, ticker "^NSEI") and runs scenario/stress tests.

Metrics implemented
--------------------
- Sharpe ratio (annualized, vs. risk-free rate)
- Tracking error (annualized std. dev. of active returns)
- Information Ratio (IR) = active return / tracking error
- Transfer Coefficient (TC) — Grinold & Kahn's measure of how much of the
  "ideal" unconstrained alpha signal survives into the actual (constrained,
  long-only, budget-limited) portfolio. Estimated here as the correlation
  between the actual active-weight vector and the unconstrained mean-variance
  active-weight vector implied by the same alpha/covariance inputs.
- Value at Risk (VaR) — historical simulation and parametric (variance-
  covariance) methods, at 95% and 99% confidence, both in isolation and
  relative to the benchmark (relative VaR).
- Scenario / stress testing — historical scenario replay (e.g. COVID crash,
  a rate-shock scenario) and Monte Carlo shocks applied to the portfolio's
  current holdings.

All of this needs real price history to produce meaningful numbers — plug in
actual portfolio weights and a real date range before trusting the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data_loader import compute_returns, fetch_prices

TRADING_DAYS_PER_YEAR = 252
NIFTY50_TICKER = "^NSEI"


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def portfolio_return_series(
    prices: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Given a price DataFrame (columns = tickers) and a weights dict, returns
    the daily return series of the weighted portfolio. Weights are
    normalized to sum to 1 over the tickers actually present in `prices`.
    """
    tickers = [t for t in weights if t in prices.columns]
    if not tickers:
        raise ValueError("None of the portfolio tickers are present in the price data.")

    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()

    returns = compute_returns(prices[tickers])
    port_returns = returns[tickers].dot(w)
    port_returns.name = "portfolio"
    return port_returns


def fetch_benchmark_returns(
    start_date: str,
    end_date: str,
    ticker: str = NIFTY50_TICKER,
) -> pd.Series:
    """Fetches and returns the Nifty 50 (or other) benchmark's daily returns."""
    prices = fetch_prices([ticker], start_date, end_date)
    returns = compute_returns(prices)
    series = returns.iloc[:, 0]
    series.name = "benchmark"
    return series


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkMetrics:
    sharpe_ratio: float = 0.0
    active_return_annualized: float = 0.0
    tracking_error_annualized: float = 0.0
    information_ratio: float = 0.0
    transfer_coefficient: float | None = None
    var_95_historical: float = 0.0
    var_99_historical: float = 0.0
    var_95_parametric: float = 0.0
    var_99_parametric: float = 0.0
    relative_var_95: float = 0.0
    beta_vs_benchmark: float = 0.0
    scenario_results: dict = field(default_factory=dict)

    def summary(self) -> str:
        tc = f"{self.transfer_coefficient:.3f}" if self.transfer_coefficient is not None else "n/a"
        lines = [
            "=== Portfolio vs Nifty 50 — Metrics ===",
            f"Sharpe ratio (annualized):        {self.sharpe_ratio:.3f}",
            f"Active return (annualized):       {self.active_return_annualized:.2%}",
            f"Tracking error (annualized):      {self.tracking_error_annualized:.2%}",
            f"Information Ratio:                {self.information_ratio:.3f}",
            f"Transfer Coefficient:             {tc}",
            f"Beta vs Nifty 50:                 {self.beta_vs_benchmark:.3f}",
            f"VaR 95% (historical, 1-day):       {self.var_95_historical:.2%}",
            f"VaR 99% (historical, 1-day):       {self.var_99_historical:.2%}",
            f"VaR 95% (parametric, 1-day):       {self.var_95_parametric:.2%}",
            f"VaR 99% (parametric, 1-day):       {self.var_99_parametric:.2%}",
            f"Relative VaR 95% (vs benchmark):   {self.relative_var_95:.2%}",
        ]
        if self.scenario_results:
            lines.append("\n=== Scenario Test Results (portfolio P&L impact) ===")
            for name, impact in self.scenario_results.items():
                lines.append(f"  {name:35s}: {impact:.2%}")
        return "\n".join(lines)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
    ann_return = returns.mean() * TRADING_DAYS_PER_YEAR
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0


def information_ratio(
    port_returns: pd.Series,
    bench_returns: pd.Series,
) -> tuple[float, float, float]:
    """Returns (active_return_annualized, tracking_error_annualized, IR)."""
    aligned = pd.concat([port_returns, bench_returns], axis=1, join="inner").dropna()
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]

    active_return_ann = active.mean() * TRADING_DAYS_PER_YEAR
    tracking_error_ann = active.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    ir = active_return_ann / tracking_error_ann if tracking_error_ann > 0 else 0.0
    return active_return_ann, tracking_error_ann, ir


def beta_vs_benchmark(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    aligned = pd.concat([port_returns, bench_returns], axis=1, join="inner").dropna()
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
    var = np.var(aligned.iloc[:, 1])
    return cov / var if var > 0 else 0.0


def transfer_coefficient(
    actual_weights: dict[str, float],
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    benchmark_weights: dict[str, float] | None = None,
) -> float:
    """
    Estimates the transfer coefficient: the correlation between the actual
    (constrained) active-weight vector and the unconstrained mean-variance
    optimal active-weight vector implied by the same alpha (mean_returns)
    and risk model (cov_matrix). TC = 1 means no information is lost to
    constraints; TC closer to 0 means constraints (long-only, cardinality,
    QAOA's discretization, etc.) are destroying most of the alpha signal.

    If no benchmark_weights are supplied, assumes an equal-weight benchmark
    over the same universe (a reasonable default when the true benchmark,
    e.g. Nifty 50, holds different constituents than your universe).
    """
    tickers = list(mean_returns.index)
    n = len(tickers)

    if benchmark_weights is None:
        bench_w = np.ones(n) / n
    else:
        bench_w = np.array([benchmark_weights.get(t, 0.0) for t in tickers])

    actual_w = np.array([actual_weights.get(t, 0.0) for t in tickers])

    # Unconstrained optimal active weights: proportional to (cov^-1 @ alpha),
    # the standard Grinold-Kahn "ideal" active position absent constraints.
    try:
        inv_cov = np.linalg.pinv(cov_matrix.values)
    except np.linalg.LinAlgError:
        return float("nan")

    alpha = mean_returns.values - mean_returns.values.mean()
    ideal_active = inv_cov @ alpha
    if np.linalg.norm(ideal_active) == 0:
        return float("nan")
    ideal_active = ideal_active / np.linalg.norm(ideal_active)

    actual_active = actual_w - bench_w
    if np.linalg.norm(actual_active) == 0:
        return float("nan")
    actual_active_norm = actual_active / np.linalg.norm(actual_active)

    tc = float(np.dot(ideal_active, actual_active_norm))
    return tc


def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    1-day VaR as a positive number representing the loss magnitude at the
    given confidence level (e.g. 0.05 means a 5% one-day loss at 95% VaR).
    """
    if method == "historical":
        return -np.percentile(returns.dropna(), (1 - confidence) * 100)

    if method == "parametric":
        from scipy.stats import norm

        mu = returns.mean()
        sigma = returns.std()
        z = norm.ppf(1 - confidence)
        return -(mu + z * sigma)

    raise ValueError(f"Unknown VaR method: {method!r}")


def relative_value_at_risk(
    port_returns: pd.Series,
    bench_returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """VaR of the *active* (portfolio - benchmark) return series."""
    aligned = pd.concat([port_returns, bench_returns], axis=1, join="inner").dropna()
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return value_at_risk(active, confidence=confidence, method="historical")


# ---------------------------------------------------------------------------
# Scenario / stress testing
# ---------------------------------------------------------------------------

# Illustrative historical stress windows relevant to Indian equities.
# Adjust/extend these dates as needed; each is applied by replaying the
# realized benchmark move and scaling by the portfolio's estimated beta,
# then separately by replaying actual portfolio-constituent returns over
# the same window if data is available.
HISTORICAL_SCENARIOS = {
    "COVID crash (Feb–Mar 2020)": ("2020-02-01", "2020-03-23"),
    "2022 rate-hike selloff (Jan–Jun 2022)": ("2022-01-01", "2022-06-30"),
    "2018 IL&FS/NBFC crisis (Aug–Oct 2018)": ("2018-08-01", "2018-10-31"),
}


def run_historical_scenarios(
    weights: dict[str, float],
    scenarios: dict[str, tuple[str, str]] = HISTORICAL_SCENARIOS,
) -> dict[str, float]:
    """
    Replays each named historical window on the *actual* portfolio
    constituents (not just the benchmark) and reports cumulative portfolio
    return over that window. Requires network access to pull historical
    prices for each window.
    """
    results = {}
    for name, (start, end) in scenarios.items():
        try:
            prices = fetch_prices(list(weights.keys()), start, end, use_cache=False)
            port_returns = portfolio_return_series(prices, weights)
            cumulative = (1 + port_returns).prod() - 1
            results[name] = cumulative
        except Exception as exc:  # noqa: BLE001 - report and continue other scenarios
            results[name] = float("nan")
            print(f"  [warning] scenario '{name}' failed: {exc}")
    return results


def run_monte_carlo_shock(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    weights: dict[str, float],
    n_simulations: int = 10_000,
    shock_horizon_days: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """
    Simulates `n_simulations` correlated shock paths over `shock_horizon_days`
    using the historical mean/covariance (multivariate normal), and reports
    the portfolio's simulated VaR/CVaR at 95% and 99% over that horizon.
    This does not require network access — it's a forward-looking Monte
    Carlo, not a historical replay.
    """
    rng = np.random.default_rng(seed)
    tickers = list(mean_returns.index)
    w = np.array([weights.get(t, 0.0) for t in tickers])
    w = w / w.sum() if w.sum() != 0 else w

    daily_mean = mean_returns.values / TRADING_DAYS_PER_YEAR
    daily_cov = cov_matrix.values / TRADING_DAYS_PER_YEAR

    sims = rng.multivariate_normal(
        daily_mean, daily_cov, size=(n_simulations, shock_horizon_days)
    )
    horizon_returns = sims.sum(axis=1)  # simple sum approximation over horizon
    port_horizon_returns = horizon_returns @ w

    var_95 = -np.percentile(port_horizon_returns, 5)
    var_99 = -np.percentile(port_horizon_returns, 1)
    cvar_95 = -port_horizon_returns[port_horizon_returns <= -var_95].mean()

    return {
        f"Monte Carlo VaR 95% ({shock_horizon_days}d horizon)": var_95,
        f"Monte Carlo VaR 99% ({shock_horizon_days}d horizon)": var_99,
        f"Monte Carlo CVaR 95% ({shock_horizon_days}d horizon)": cvar_95,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def compute_full_benchmark_report(
    weights: dict[str, float],
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    risk_free_rate: float = 0.06,
    benchmark_ticker: str = NIFTY50_TICKER,
    run_scenarios: bool = True,
) -> BenchmarkMetrics:
    """
    Ties together all metrics above for a single portfolio over
    [start_date, end_date], against the given benchmark (default Nifty 50).
    """
    port_returns = portfolio_return_series(prices, weights)
    bench_returns = fetch_benchmark_returns(start_date, end_date, benchmark_ticker)

    active_ret, tracking_err, ir = information_ratio(port_returns, bench_returns)
    tc = transfer_coefficient(weights, mean_returns, cov_matrix)
    beta = beta_vs_benchmark(port_returns, bench_returns)

    metrics = BenchmarkMetrics(
        sharpe_ratio=sharpe_ratio(port_returns, risk_free_rate),
        active_return_annualized=active_ret,
        tracking_error_annualized=tracking_err,
        information_ratio=ir,
        transfer_coefficient=tc,
        beta_vs_benchmark=beta,
        var_95_historical=value_at_risk(port_returns, 0.95, "historical"),
        var_99_historical=value_at_risk(port_returns, 0.99, "historical"),
        var_95_parametric=value_at_risk(port_returns, 0.95, "parametric"),
        var_99_parametric=value_at_risk(port_returns, 0.99, "parametric"),
        relative_var_95=relative_value_at_risk(port_returns, bench_returns, 0.95),
    )

    if run_scenarios:
        scenario_results = {}
        scenario_results.update(run_monte_carlo_shock(mean_returns, cov_matrix, weights))
        # Historical replay is optional/slow (extra network calls per scenario);
        # comment out if you just want the fast Monte Carlo numbers.
        scenario_results.update(run_historical_scenarios(weights))
        metrics.scenario_results = scenario_results

    return metrics
