"""
Technical indicators using the `ta` library (industry standard, used over
hand-rolled formulas to avoid subtle off-by-one bugs in rolling-window math).
"""
import ta


def add_indicators(df):
    df = df.copy()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    df["returns"] = close.pct_change()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()
    df["ema_12"] = ta.trend.EMAIndicator(close, window=12).ema_indicator()
    df["ema_26"] = ta.trend.EMAIndicator(close, window=26).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["adx_14"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    df["mfi_14"] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()
    df["momentum_10"] = ta.momentum.ROCIndicator(close, window=10).roc()
    df["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    return df.dropna().reset_index(drop=True)


def fibonacci_levels(df, lookback=120):
    sub = df.tail(lookback)
    high, low = float(sub["high"].max()), float(sub["low"].min())
    diff = high - low
    return {
        "high": high,
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff,
        "0.618": high - 0.618 * diff,
        "low": low,
    }
