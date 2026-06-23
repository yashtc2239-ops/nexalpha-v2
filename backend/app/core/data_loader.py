"""
Fetches OHLCV data from yfinance and normalizes it.

Fix vs legacy: legacy did manual string-splitting to "fix" dates (`d.split('-')`),
which breaks on any unexpected format. pandas.to_datetime() is the standardized,
battle-tested way to parse dates — never hand-roll date parsing.

Fix #2 (found via real-world testing): if the caller doesn't pass start/end,
yfinance defaults to period='1mo' (~20 trading days). Indicators like SMA-200
and ADX-14 need 200+ rows to produce a non-NaN value — with only ~20 rows,
ta's ADXIndicator crashes with an IndexError instead of a clean message.
Fix: always request a minimum lookback window, and fail with a clear,
actionable ValueError (not a cryptic IndexError) if there still isn't enough
data after the request.
"""
import datetime
import pandas as pd
import yfinance as yf
from app.logger import logger

MIN_LOOKBACK_DAYS = 730   # ~2 years — comfortably covers SMA-200 + buffer
MIN_ROWS_REQUIRED = 220   # SMA-200 needs 200; leave headroom for holidays/gaps


def load_price_data(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    if not ticker:
        raise ValueError("ticker is required")

    if not start:
        start = (datetime.date.today() - datetime.timedelta(days=MIN_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    else:
        start = pd.to_datetime(start).strftime("%Y-%m-%d")
    end = pd.to_datetime(end).strftime("%Y-%m-%d") if end else None

    df = yf.download(tickers=str(ticker), start=start, end=end,
                     auto_adjust=False, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker={ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index().rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df = df[["date"] + numeric_cols].copy()  # drop stray cols like Adj Close
    df[numeric_cols] = df[numeric_cols].astype(float)
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    if len(df) < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"Only {len(df)} trading days of data available for {ticker} "
            f"(need at least {MIN_ROWS_REQUIRED} for indicators like SMA-200). "
            "Try a ticker with a longer listing history, or an earlier start date."
        )

    logger.info(f"Loaded {len(df)} rows for {ticker} ({start} to {end})")
    return df
