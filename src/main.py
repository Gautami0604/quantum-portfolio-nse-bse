"""
main.py

CLI entry point: loads a stock universe from a YAML config, fetches price
data, runs both the classical and quantum (QAOA) optimizers, and writes a
comparison to results/.

Usage:
    python -m src.main --stocks config/stocks.yaml --backend simulator
    python -m src.main --stocks config/stocks.yaml --backend ibm_hardware
"""

from __future__ import annotations

import argparse

import pandas as pd
import yaml

from src.backtest import plot_comparison, save_comparison
from src.benchmark_metrics import compute_full_benchmark_report
from src.classical_opt import max_sharpe_portfolio
from src.data_loader import compute_returns, fetch_prices
from src.quantum_opt import qaoa_portfolio
from src.utils import annualized_covariance, annualized_return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantum vs classical portfolio optimization")
    parser.add_argument("--stocks", default="config/stocks.yaml", help="Path to stock universe YAML")
    parser.add_argument(
        "--backend",
        choices=["simulator", "ibm_hardware"],
        default="simulator",
        help="Quantum backend to use (default: simulator)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the benchmark comparison (IR, transfer coefficient, VaR, scenarios) vs Nifty 50",
    )
    parser.add_argument(
        "--benchmark-days",
        type=int,
        default=252,
        help="Trailing window (trading days) for the benchmark comparison (default: 252, ~1 year)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.stocks) as f:
        config = yaml.safe_load(f)

    tickers = config["tickers"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    qaoa_cfg = config.get("qaoa", {})

    print(f"Fetching prices for {len(tickers)} tickers ({start_date} to {end_date})...")
    prices = fetch_prices(tickers, start_date, end_date)
    returns = compute_returns(prices)

    mean_returns = annualized_return(returns)
    cov_matrix = annualized_covariance(returns)

    print("Running classical mean-variance optimization...")
    classical_result = max_sharpe_portfolio(mean_returns, cov_matrix)
    print(classical_result.summary())

    print(f"\nRunning QAOA optimization on backend='{args.backend}'...")
    quantum_result = qaoa_portfolio(
        mean_returns,
        cov_matrix,
        budget=qaoa_cfg.get("budget", 5),
        risk_factor=qaoa_cfg.get("risk_factor", 0.5),
        reps=qaoa_cfg.get("reps", 2),
        max_iter=qaoa_cfg.get("max_iter", 200),
        backend=args.backend,
    )
    print(quantum_result.summary())

    csv_path = save_comparison([classical_result, quantum_result])
    png_path = plot_comparison([classical_result, quantum_result])
    print(f"\nSaved comparison table to {csv_path}")
    print(f"Saved comparison chart to {png_path}")

    if args.benchmark:
        bench_start = (prices.index[-1] - pd.Timedelta(days=int(args.benchmark_days * 1.45))).strftime("%Y-%m-%d")
        bench_end = prices.index[-1].strftime("%Y-%m-%d")

        for label, result in [("Classical", classical_result), ("Quantum (QAOA)", quantum_result)]:
            print(f"\n--- Benchmark report: {label} portfolio vs Nifty 50 (last ~{args.benchmark_days} trading days) ---")
            report = compute_full_benchmark_report(
                weights=result.weights,
                mean_returns=mean_returns,
                cov_matrix=cov_matrix,
                prices=prices,
                start_date=bench_start,
                end_date=bench_end,
            )
            print(report.summary())


if __name__ == "__main__":
    main()
