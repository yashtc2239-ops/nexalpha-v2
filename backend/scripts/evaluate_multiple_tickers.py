"""
Evaluates the SAME methodology across multiple tickers and horizons, to answer
a specific question: "is 52.5% accuracy a one-off bad result, or the genuine,
repeatable performance of this approach?"

Why this matters: a single number from a single (ticker, horizon) run tells you
almost nothing — it could be luck, an unusually trending/choppy period for that
specific stock, or a horizon that happens to suit/not suit that stock's volatility.
Averaging across multiple tickers and horizons gives a far more defensible number
to cite in a README/interview than one anecdote.

Usage:
    cd backend
    PYTHONPATH=. python scripts/evaluate_multiple_tickers.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from app.core.data_loader import load_price_data
from app.core.indicators import add_indicators
from app.core.features import prepare_features
from app.models.train import train_models, load_models

TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]   # same 4 as the paper
HORIZONS = [5, 7, 10]
MODEL_DIR = "/tmp/eval_models"   # scratch dir, not the real models_store


def evaluate_one(ticker, horizon):
    df = load_price_data(ticker)
    df_ind = add_indicators(df)
    df_feat, X, y_reg, y_clf, feature_cols = prepare_features(df_ind, target_horizon=horizon)

    train_models(ticker, df_feat, X, y_reg, y_clf, feature_cols, horizon,
                 model_dir=MODEL_DIR, force=True)
    _, _, meta = load_models(ticker, horizon, model_dir=MODEL_DIR)

    return {
        "ticker": ticker,
        "horizon": horizon,
        "n_rows": len(df_feat),
        "clf_accuracy": round(meta["metrics"]["clf_accuracy"], 4),
        "clf_f1": round(meta["metrics"]["clf_f1"], 4),
        "reg_mse": round(meta["metrics"]["reg_mse"], 6),
    }


def main():
    results = []
    for ticker in TICKERS:
        for horizon in HORIZONS:
            try:
                print(f"Evaluating {ticker} @ horizon={horizon}d ...")
                results.append(evaluate_one(ticker, horizon))
            except Exception as e:
                print(f"  SKIPPED ({ticker}, h={horizon}): {e}")

    df = pd.DataFrame(results)
    print("\n=== Full results ===")
    print(df.to_string(index=False))

    print("\n=== Summary ===")
    print(f"Mean accuracy across all runs: {df['clf_accuracy'].mean():.4f}")
    print(f"Std dev:                       {df['clf_accuracy'].std():.4f}")
    print(f"Min / Max accuracy:            {df['clf_accuracy'].min():.4f} / {df['clf_accuracy'].max():.4f}")
    print(f"Mean F1:                       {df['clf_f1'].mean():.4f}")

    out_path = os.path.join(os.path.dirname(__file__), "evaluation_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
