"""
PSI (Population Stability Index) — measures how much a feature's distribution has
shifted between two periods (e.g. training data vs last 30 days of live data).

Why PSI specifically (vs just eyeballing a chart): it gives ONE number per feature
that is comparable across features and across time, so you can set a threshold and
alert automatically. This is what makes it a 'production' technique vs an ad-hoc
notebook check.

Formula: PSI = sum( (actual_pct - expected_pct) * ln(actual_pct / expected_pct) )
over N buckets. Rule of thumb: <0.1 = no significant shift, 0.1-0.25 = moderate
shift (watch it), >0.25 = significant shift (retrain).

Bug found via end-to-end testing (not just unit tests): raw price-level features
(close, sma_200, ema_*, bollinger bands, OBV) are non-stationary for any trending
stock — they drift over time by definition, independent of whether the MODEL is
stale. Running PSI on them fires "critical" on almost every call, which makes the
alert useless (a monitor that's always red carries no information). Fix: restrict
default drift monitoring to bounded/stationary indicators (oscillators, returns,
MACD histogram) where a distribution shift is actually informative about a real
regime change, not just normal price drift.
"""
import numpy as np
from app.config import config

# Features safe for PSI: bounded oscillators, returns, and momentum — these have a
# roughly stable distribution in normal market conditions, so a shift in THEM means
# something changed (volatility regime, momentum regime), not just "price went up".
STATIONARY_FEATURES = {
    "rsi_14", "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d",
    "adx_14", "mfi_14", "momentum_10", "returns", "atr_14",
}


def calculate_psi(expected, actual, buckets=None):
    buckets = buckets or config.PSI_BUCKETS
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(breakpoints) < 3:
        return 0.0  # not enough distinct values to bucket meaningfully

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # avoid log(0) / div-by-0 on empty buckets
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def detect_drift(training_df, live_df, feature_cols, restrict_to_stationary=True):
    """Returns per-feature PSI + an overall verdict. Call this periodically
    (e.g. weekly cron) comparing training-time feature distributions to
    the most recent live data.

    restrict_to_stationary=True (default): only monitor features in
    STATIONARY_FEATURES. This is the actual fix — STATIONARY_FEATURES was
    defined above but never consulted here, so every call still ran PSI on
    raw price levels (close, sma_200, ema_*, bollinger bands, obv) and fired
    "critical" on every single feature, every single call — a monitor that's
    always red carries no information. Pass restrict_to_stationary=False only
    if you explicitly want the full (noisier) feature-level view for debugging.
    """
    cols_to_check = (
        [c for c in feature_cols if c in STATIONARY_FEATURES]
        if restrict_to_stationary else feature_cols
    )

    results = {}
    for col in cols_to_check:
        if col in training_df.columns and col in live_df.columns:
            psi = calculate_psi(training_df[col].dropna(), live_df[col].dropna())
            if psi >= config.PSI_CRITICAL_THRESHOLD:
                status = "critical"
            elif psi >= config.PSI_WARNING_THRESHOLD:
                status = "warning"
            else:
                status = "stable"
            results[col] = {"psi": round(psi, 4), "status": status}

    max_psi = max((v["psi"] for v in results.values()), default=0.0)
    overall = "critical" if max_psi >= config.PSI_CRITICAL_THRESHOLD else (
        "warning" if max_psi >= config.PSI_WARNING_THRESHOLD else "stable")

    return {"per_feature": results, "overall_status": overall, "max_psi": round(max_psi, 4)}
