"""
Inference. Always loads feature_cols from model metadata (never recomputes them)
— this is the concrete fix for the column-mismatch bug class.
"""
from app.models.train import load_models
from app.core.features import select_features


def predict_price_and_signal(ticker, df_feat, horizon, model_dir=None):
    reg, clf, meta = load_models(ticker, horizon, model_dir)
    feature_cols = meta["feature_cols"]

    X = select_features(df_feat, feature_cols)
    last_X = X.iloc[[-1]].values

    pred_return = float(reg.predict(last_X)[0])
    current_price = float(df_feat.iloc[-1]["close"])
    pred_price = current_price * (1 + pred_return)

    proba = clf.predict_proba(last_X)[0]
    signal_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

    from app.config import config
    if signal_prob > config.SIGNAL_BUY_THRESHOLD:
        signal = 1
    elif signal_prob < config.SIGNAL_SELL_THRESHOLD:
        signal = -1
    else:
        signal = 0

    return {
        "predicted_return_pct": round(pred_return * 100, 4),
        "predicted_price": round(pred_price, 2),
        "signal": signal,
        "signal_prob": round(signal_prob, 4),
        "model_trained_at": meta["trained_at"],
        "model_metrics": meta["metrics"],
    }
