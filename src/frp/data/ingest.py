"""Pull and cache historical price data from Yahoo Finance."""

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def _clean_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize price DataFrame with single-level columns and clean float datatypes."""
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        first_level = [str(c) for c in df.columns.get_level_values(0)]
        second_level = [str(c) for c in df.columns.get_level_values(1)]
        if any(c in ["Close", "High", "Low", "Open", "Volume", "Adj Close"] for c in first_level):
            df.columns = df.columns.get_level_values(0)
        elif any(c in ["Close", "High", "Low", "Open", "Volume", "Adj Close"] for c in second_level):
            df.columns = df.columns.get_level_values(1)
        else:
            df.columns = [f"{c[0]}_{c[1]}" for c in df.columns]

    # Clean index
    df = df[~df.index.astype(str).str.lower().isin(["date", "ticker", "price"])]
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]

    # Coerce numeric values
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    return df


def fetch_prices(ticker: str, period: str = "3y", force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch historical daily prices for a ticker, caching to disk.

    Args:
        ticker: e.g. "AAPL"
        period: e.g. "3y", "1y", "6mo"
        force_refresh: if True, ignore cache and re-download

    Returns:
        DataFrame indexed by date with OHLCV columns (all numeric floats).
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache_path = DATA_DIR / f"{ticker}_{period}.csv"

    if cache_path.exists() and not force_refresh:
        try:
            df = pd.read_csv(cache_path, index_col=0)
            df = _clean_price_dataframe(df)
            if not df.empty and ("Close" in df.columns or "Adj Close" in df.columns):
                return df
        except Exception:
            pass  # Re-download if cached file is unparseable

    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

    df = _clean_price_dataframe(df)
    if df.empty:
        raise ValueError(f"Failed to extract valid price series for ticker '{ticker}'.")

    # Cache clean CSV
    df.to_csv(cache_path)
    return df


if __name__ == "__main__":
    df = fetch_prices("AAPL")
    print(df.tail())
    print(f"\nRows fetched: {len(df)}")