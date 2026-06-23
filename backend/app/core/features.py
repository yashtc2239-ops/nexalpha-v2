"""
Feature engineering — this is where the legacy project's data-leakage bug lived.

LEAKAGE_COLUMNS are columns that encode the *answer* (future price/direction).
They must NEVER appear in the X matrix used by any model, at training, prediction,
OR backtesting time. The legacy backtest.py forgot this at backtest time, which is
exactly why that bug was so dangerous — it was correct in one place and wrong in
another, which is the most common real-world way leakage bugs survive code review.

Fix: this module exports get_feature_columns() — the SINGLE source of truth for
which columns are valid features. Every other module (train, predict, backtest)
must import and use this function instead of recomputing column lists themselves.
"""
import pandas as pd

LEAKAGE_COLUMNS = ["target_close", "future_ret", "future_dir", "date"]


def prepare_features(df, target_horizon=7):
    df = df.copy().reset_index(drop=True)

    if "date" not in df.columns:
        df.insert(0, "date", pd.date_range(start="2000-01-01", periods=len(df)))

    df["target_close"] = df["close"].shift(-target_horizon)
    df["future_ret"] = (df["target_close"] - df["close"]) / df["close"]
    df["future_dir"] = (df["future_ret"] > 0).astype(int)

    df = df.dropna().reset_index(drop=True)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].astype(float)
    y_reg = df["future_ret"].astype(float)
    y_clf = df["future_dir"].astype(int)

    return df, X, y_reg, y_clf, feature_cols


def get_feature_columns(df):
    """Single source of truth for valid feature columns. Sorted for determinism —
    column ORDER must be identical at train and inference time, or a RandomForest
    will silently apply the wrong weights to the wrong values."""
    return sorted([c for c in df.columns if c not in LEAKAGE_COLUMNS])


def select_features(df, feature_cols):
    """Used at predict/backtest time to guarantee the exact same columns,
    in the exact same order, that the model was trained on."""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    return df[feature_cols].astype(float)
