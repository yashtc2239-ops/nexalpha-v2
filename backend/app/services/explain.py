"""
Explainability.

explain_with_indicators(): rule-based, human-readable reasons from raw indicator
values — kept from legacy because it was genuinely correct logic.

shap_explain(): REPLACES the legacy hardcoded dummy dict with a real SHAP
TreeExplainer computation. RandomForest is a tree ensemble, so TreeExplainer
gives EXACT Shapley values efficiently (polynomial time, not the exponential-time
brute force SHAP would otherwise need) — this is why TreeExplainer specifically,
not the generic KernelExplainer.
"""
import numpy as np
from app.core.features import select_features


def explain_with_indicators(row):
    reasons = []

    def gv(name):
        v = row.get(name)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    rsi = gv("rsi_14")
    if rsi == rsi:  # not NaN
        if rsi > 70:
            reasons.append(f"RSI(14)={rsi:.1f} -> Overbought")
        elif rsi < 30:
            reasons.append(f"RSI(14)={rsi:.1f} -> Oversold")
        else:
            reasons.append(f"RSI(14)={rsi:.1f} -> Neutral")

    sma_50, sma_200 = gv("sma_50"), gv("sma_200")
    if sma_50 == sma_50 and sma_200 == sma_200:
        reasons.append("SMA50 > SMA200 -> Bullish crossover" if sma_50 > sma_200
                       else "SMA50 < SMA200 -> Bearish crossover")

    macd, macd_signal = gv("macd"), gv("macd_signal")
    if macd == macd and macd_signal == macd_signal:
        reasons.append("MACD > Signal -> Bullish momentum" if macd > macd_signal
                       else "MACD < Signal -> Bearish momentum")

    return reasons


def shap_explain(clf, df_feat, feature_cols, top_n=5):
    try:
        import shap
    except ImportError:
        return {"error": "shap not installed", "top_features": []}

    X = select_features(df_feat, feature_cols)
    sample = X.tail(min(200, len(X)))  # cap sample size for latency

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(sample)

    # shap's return shape varies by version: a list [class_0, class_1] (older),
    # or a single (n_samples, n_features, n_classes) array (newer). Normalize both
    # to a 2D (n_samples, n_features) array for the positive class.
    if isinstance(shap_values, list):
        vals = shap_values[1]
    else:
        vals = np.asarray(shap_values)
        if vals.ndim == 3:
            vals = vals[:, :, 1]

    mean_abs_shap = np.abs(vals).mean(axis=0)

    ranked = sorted(zip(feature_cols, mean_abs_shap), key=lambda x: -x[1])[:top_n]
    return {
        "top_features": [
            {"feature": f, "mean_abs_shap": round(float(v), 5)} for f, v in ranked
        ],
        "sample_size": len(sample),
        "note": "Real SHAP TreeExplainer values, computed on the last "
                f"{len(sample)} rows of feature data.",
    }
