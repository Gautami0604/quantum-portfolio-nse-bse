"""
backtest.py

Compares classical and quantum portfolio results side by side and produces
a simple bar-chart visualization + CSV summary saved to results/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import OptimizationResult

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def compare_results(results: list[OptimizationResult]) -> pd.DataFrame:
    """Builds a tidy comparison table from a list of OptimizationResult."""
    rows = [
        {
            "method": r.method,
            "backend": r.backend,
            "expected_return": r.expected_return,
            "volatility": r.volatility,
            "sharpe_ratio": r.sharpe_ratio,
            "runtime_seconds": r.runtime_seconds,
        }
        for r in results
    ]
    return pd.DataFrame(rows)


def save_comparison(results: list[OptimizationResult], filename: str = "comparison.csv") -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = compare_results(results)
    out_path = RESULTS_DIR / filename
    df.to_csv(out_path, index=False)
    return out_path


def plot_comparison(results: list[OptimizationResult], filename: str = "comparison.png") -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = compare_results(results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ["expected_return", "volatility", "sharpe_ratio"]
    titles = ["Expected Return", "Volatility", "Sharpe Ratio"]

    for ax, metric, title in zip(axes, metrics, titles):
        ax.bar(df["method"], df[metric], color=["#4C72B0", "#DD8452"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    out_path = RESULTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
