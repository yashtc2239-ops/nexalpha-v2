"""
Backtesting — this is where the legacy bug lived (feeding future_ret/future_dir
back into the model as if they were features).

Two fixes:
1. Feature selection ALWAYS goes through select_features(df, feature_cols) using
   the exact columns/order saved at training time. No more `select_dtypes(number)`
   grabbing leakage columns.
2. Out-of-sample only: we backtest on the held-out TEST split, not on rows the
   model was trained on. Backtesting on training data is "testing on the exam
   you already saw the answer key for" — it always looks great and means nothing.
   This is the standard fix for what's technically called look-ahead bias.
"""
import numpy as np
from app.core.features import select_features
from app.config import config
from app.logger import logger


def run_backtest(ticker, df_feat, clf, feature_cols, n_test_rows,
                 initial_capital=None, commission=None, slippage=None):
    initial_capital = initial_capital or config.DEFAULT_INITIAL_CAPITAL
    commission = commission if commission is not None else config.DEFAULT_COMMISSION
    slippage = slippage if slippage is not None else config.DEFAULT_SLIPPAGE

    df = df_feat.copy().reset_index(drop=True)
    test_df = df.tail(n_test_rows).reset_index(drop=True)

    X_test = select_features(test_df, feature_cols)
    preds = clf.predict(X_test.values)
    test_df = test_df.copy()
    test_df["pred_signal"] = np.where(preds == 1, 1, -1)

    cash = initial_capital
    position = 0.0
    trades, equity_curve = [], []

    for _, row in test_df.iterrows():
        price = float(row["close"])
        signal = int(row["pred_signal"])
        date = str(row["date"])

        if signal == 1 and cash >= price:
            qty = int((cash * config.POSITION_SIZE_PCT) // price)
            if qty > 0:
                cost = qty * price * (1 + slippage) + commission
                cash -= cost
                position += qty
                trades.append({"type": "buy", "date": date, "price": price, "qty": qty})
        elif signal == -1 and position > 0:
            revenue = position * price * (1 - slippage) - commission
            cash += revenue
            trades.append({"type": "sell", "date": date, "price": price, "qty": position})
            position = 0

        equity_curve.append({"date": date, "equity": cash + position * price})

    final_equity = equity_curve[-1]["equity"] if equity_curve else initial_capital
    total_return = final_equity / initial_capital - 1

    returns = np.diff([e["equity"] for e in equity_curve]) / np.array(
        [e["equity"] for e in equity_curve][:-1]) if len(equity_curve) > 1 else np.array([0])
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)) if len(returns) > 1 else 0.0

    logger.info(f"Backtest {ticker}: return={total_return:.4f} sharpe={sharpe:.2f} "
                f"trades={len(trades)} (out-of-sample rows={n_test_rows})")

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "total_return": round(total_return, 4),
        "sharpe_ratio": round(sharpe, 3),
        "n_trades": len(trades),
        "out_of_sample_rows": n_test_rows,
        "note": "Backtested on held-out test split only — not on training data.",
    }
