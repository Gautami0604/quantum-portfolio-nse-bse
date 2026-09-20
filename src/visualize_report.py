"""
visualize_report.py

Generates a full set of visualizations for the classical vs quantum
portfolio comparison: price trends, a correlation heatmap, the efficient
frontier with both portfolios marked, weight comparison bars, cumulative
returns vs Nifty 50, and scenario test results.

Usage:
    python -m src.visualize_report --stocks config/stocks.yaml --backend simulator
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save-to-file only — avoids Tk GUI backend cleanup errors

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.benchmark_metrics import (
    fetch_benchmark_returns,
    portfolio_return_series,
    run_historical_scenarios,
    run_monte_carlo_shock,
)
from src.classical_opt import max_sharpe_portfolio
from src.data_loader import compute_returns, fetch_prices
from src.quantum_opt import qaoa_portfolio
from src.utils import annualized_covariance, annualized_return

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate portfolio visualizations")
    parser.add_argument("--stocks", default="config/stocks.yaml")
    parser.add_argument("--backend", choices=["simulator", "ibm_hardware"], default="simulator")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Individual chart builders — each saves its own PNG and returns the path.
# ---------------------------------------------------------------------------

def plot_price_history(prices: pd.DataFrame, benchmark_prices: pd.Series | None = None) -> Path:
    """Normalized (indexed to 100) price history for every stock, plus Nifty 50 if available."""
    normalized = prices / prices.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in normalized.columns:
        ax.plot(normalized.index, normalized[col], label=col, linewidth=1.2)

    if benchmark_prices is not None:
        bench_norm = benchmark_prices / benchmark_prices.iloc[0] * 100
        ax.plot(bench_norm.index, bench_norm.values, label="Nifty 50", color="black",
                linewidth=2.2, linestyle="--")

    ax.set_title("Normalized Price History (Indexed to 100)")
    ax.set_ylabel("Indexed Price")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = RESULTS_DIR / "price_history.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(returns: pd.DataFrame) -> Path:
    """Correlation heatmap ('map') of daily returns across all stocks."""
    corr = returns.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     fontsize=6, color="black")

    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Stock Return Correlation Heatmap")
    fig.tight_layout()

    out_path = RESULTS_DIR / "correlation_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    classical_weights: dict,
    quantum_weights: dict,
    risk_free_rate: float = 0.06,
    n_random: int = 3000,
    seed: int = 42,
) -> Path:
    """Random-portfolio cloud approximating the efficient frontier, with both
    optimized portfolios marked."""
    rng = np.random.default_rng(seed)
    n = len(mean_returns)
    mu = mean_returns.values
    sigma = cov_matrix.values

    rand_returns, rand_vols, rand_sharpes = [], [], []
    for _ in range(n_random):
        w = rng.random(n)
        w /= w.sum()
        ret = float(np.dot(w, mu))
        vol = float(np.sqrt(w.T @ sigma @ w))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
        rand_returns.append(ret)
        rand_vols.append(vol)
        rand_sharpes.append(sharpe)

    def _portfolio_point(weights: dict) -> tuple[float, float]:
        w = np.array([weights.get(t, 0.0) for t in mean_returns.index])
        ret = float(np.dot(w, mu))
        vol = float(np.sqrt(w.T @ sigma @ w))
        return vol, ret

    classical_vol, classical_ret = _portfolio_point(classical_weights)
    quantum_vol, quantum_ret = _portfolio_point(quantum_weights)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(rand_vols, rand_returns, c=rand_sharpes, cmap="viridis",
                          s=8, alpha=0.5, label="Random portfolios")
    fig.colorbar(scatter, ax=ax, label="Sharpe ratio")

    ax.scatter([classical_vol], [classical_ret], color="red", marker="*", s=400,
               edgecolor="black", label="Classical (Markowitz)", zorder=5)
    ax.scatter([quantum_vol], [quantum_ret], color="orange", marker="*", s=400,
               edgecolor="black", label="Quantum (QAOA)", zorder=5)

    ax.set_xlabel("Volatility (annualized)")
    ax.set_ylabel("Expected Return (annualized)")
    ax.set_title("Efficient Frontier — Classical vs Quantum Portfolio")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = RESULTS_DIR / "efficient_frontier.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_weight_comparison(classical_weights: dict, quantum_weights: dict) -> Path:
    """Side-by-side bar chart of portfolio weights."""
    tickers = sorted(set(classical_weights) | set(quantum_weights))
    classical_vals = [classical_weights.get(t, 0) for t in tickers]
    quantum_vals = [quantum_weights.get(t, 0) for t in tickers]

    x = np.arange(len(tickers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, classical_vals, width, label="Classical", color="#4C72B0")
    ax.bar(x + width / 2, quantum_vals, width, label="Quantum (QAOA)", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Weight")
    ax.set_title("Portfolio Weight Allocation — Classical vs Quantum")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = RESULTS_DIR / "weight_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_cumulative_vs_benchmark(
    classical_returns: pd.Series,
    quantum_returns: pd.Series,
    bench_returns: pd.Series,
) -> Path:
    """Cumulative return lines for both portfolios vs. Nifty 50."""
    aligned = pd.concat(
        [classical_returns, quantum_returns, bench_returns], axis=1, join="inner"
    ).dropna()
    aligned.columns = ["Classical", "Quantum (QAOA)", "Nifty 50"]

    cumulative = (1 + aligned).cumprod() * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    for col, color, style in [
        ("Classical", "#4C72B0", "-"),
        ("Quantum (QAOA)", "#DD8452", "-"),
        ("Nifty 50", "black", "--"),
    ]:
        ax.plot(cumulative.index, cumulative[col], label=col, color=color, linestyle=style, linewidth=1.6)

    ax.set_title("Cumulative Growth of ₹100 — Portfolios vs Nifty 50")
    ax.set_ylabel("Value (₹, indexed to 100)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = RESULTS_DIR / "cumulative_vs_benchmark.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_scenario_results(classical_scenarios: dict, quantum_scenarios: dict) -> Path:
    """Bar chart of historical scenario replay results for both portfolios."""
    # Only compare the historical scenarios (skip the Monte Carlo VaR/CVaR keys,
    # which aren't directly comparable to a % P&L bar).
    historical_keys = [
        k for k in classical_scenarios
        if "Monte Carlo" not in k
    ]

    classical_vals = [classical_scenarios.get(k, np.nan) * 100 for k in historical_keys]
    quantum_vals = [quantum_scenarios.get(k, np.nan) * 100 for k in historical_keys]

    x = np.arange(len(historical_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, classical_vals, width, label="Classical", color="#4C72B0")
    ax.bar(x + width / 2, quantum_vals, width, label="Quantum (QAOA)", color="#DD8452")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(historical_keys, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Historical Scenario Stress Test — Portfolio Impact")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = RESULTS_DIR / "scenario_results.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(args.stocks) as f:
        config = yaml.safe_load(f)

    tickers = config["tickers"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    qaoa_cfg = config.get("qaoa", {})

    print(f"Fetching prices for {len(tickers)} tickers...")
    prices = fetch_prices(tickers, start_date, end_date)
    returns = compute_returns(prices)
    mean_returns = annualized_return(returns)
    cov_matrix = annualized_covariance(returns)

    print("Fetching Nifty 50 benchmark...")
    bench_returns = fetch_benchmark_returns(start_date, end_date)
    try:
        from src.data_loader import fetch_prices as _fp
        bench_prices = _fp(["^NSEI"], start_date, end_date).iloc[:, 0]
    except Exception:
        bench_prices = None

    print("Running classical optimization...")
    classical_result = max_sharpe_portfolio(mean_returns, cov_matrix)

    print("Running QAOA optimization...")
    quantum_result = qaoa_portfolio(
        mean_returns,
        cov_matrix,
        budget=qaoa_cfg.get("budget", 5),
        risk_factor=qaoa_cfg.get("risk_factor", 0.5),
        reps=qaoa_cfg.get("reps", 2),
        max_iter=qaoa_cfg.get("max_iter", 200),
        backend=args.backend,
    )

    classical_returns = portfolio_return_series(prices, classical_result.weights)
    quantum_returns = portfolio_return_series(prices, quantum_result.weights)

    print("\nGenerating charts...")
    saved_paths = []
    saved_paths.append(plot_price_history(prices, bench_prices))
    saved_paths.append(plot_correlation_heatmap(returns))
    saved_paths.append(
        plot_efficient_frontier(mean_returns, cov_matrix, classical_result.weights, quantum_result.weights)
    )
    saved_paths.append(plot_weight_comparison(classical_result.weights, quantum_result.weights))
    saved_paths.append(plot_cumulative_vs_benchmark(classical_returns, quantum_returns, bench_returns))

    print("Running scenario tests for both portfolios (this may take a while)...")
    classical_scenarios = run_monte_carlo_shock(mean_returns, cov_matrix, classical_result.weights)
    classical_scenarios.update(run_historical_scenarios(classical_result.weights))
    quantum_scenarios = run_monte_carlo_shock(mean_returns, cov_matrix, quantum_result.weights)
    quantum_scenarios.update(run_historical_scenarios(quantum_result.weights))
    saved_paths.append(plot_scenario_results(classical_scenarios, quantum_scenarios))

    print("\nSaved charts:")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
