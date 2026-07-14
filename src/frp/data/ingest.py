"""Pull and cache historical price data from Yahoo Finance."""

import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def fetch_prices(ticker: str, period: str = "3y", force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch historical daily prices for a ticker, caching to disk.

    Args:
        ticker: e.g. "AAPL"
        period: e.g. "3y", "1y", "6mo"
        force_refresh: if True, ignore cache and re-download

    Returns:
        DataFrame indexed by date with OHLCV columns.
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache_path = DATA_DIR / f"{ticker}_{period}.csv"

    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

    df.to_csv(cache_path)
    return df


if __name__ == "__main__":
    df = fetch_prices("AAPL")
    print(df.tail())
    print(f"\nRows fetched: {len(df)}")