import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def synthetic_ohlcv():
    """Deterministic synthetic price series — tests never hit the network (yfinance)."""
    n = 400
    rng = np.random.RandomState(42)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({
        "date": dates,
        "open": close + rng.normal(0, 0.5, n),
        "high": close + abs(rng.normal(1, 0.5, n)),
        "low": close - abs(rng.normal(1, 0.5, n)),
        "close": close,
        "volume": rng.randint(1000, 10000, n).astype(float),
    })
    return df
