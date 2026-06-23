from app.core.indicators import add_indicators, fibonacci_levels


def test_add_indicators_produces_expected_columns(synthetic_ohlcv):
    out = add_indicators(synthetic_ohlcv)
    for col in ["sma_50", "sma_200", "rsi_14", "macd", "atr_14", "obv"]:
        assert col in out.columns
    assert out.isna().sum().sum() == 0  # no leftover NaNs after dropna


def test_fibonacci_levels_ordering(synthetic_ohlcv):
    ind = add_indicators(synthetic_ohlcv)
    fib = fibonacci_levels(ind, lookback=120)
    assert fib["high"] >= fib["0.236"] >= fib["0.5"] >= fib["0.618"] >= fib["low"]
