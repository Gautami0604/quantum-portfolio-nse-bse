"""
data_loader.py

Fetches and caches historical price data for NSE/BSE tickers using yfinance,
and computes daily returns.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of tickers.

    Parameters
    ----------
    tickers : list[str]
        yfinance-style tickers, e.g. ["RELIANCE.NS", "TCS.NS"].
    start_date, end_date : str
        Date range in "YYYY-MM-DD" format.
    use_cache : bool
        If True, reads/writes a local CSV cache in `data/` to avoid
        re-downloading on every run.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices, indexed by date, one column per ticker.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"prices_{start_date}_{end_date}_{len(tickers)}tickers.csv"
    cache_path = DATA_DIR / cache_key

    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns a MultiIndex column frame when multiple tickers are
    # requested; normalize down to a single "Close" price table.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all").ffill().dropna()

    if use_cache:
        prices.to_csv(cache_path)

    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily percentage returns from a price DataFrame."""
    return prices.pct_change().dropna()


if __name__ == "__main__":
    # Quick manual sanity check.
    sample = fetch_prices(
        ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        "2023-01-01",
        "2023-12-31",
    )
    print(sample.head())
    print(compute_returns(sample).head())
