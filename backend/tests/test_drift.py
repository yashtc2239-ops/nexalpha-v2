"""
Regression tests for two real bugs found via end-to-end testing (not caught by
unit tests alone — both only showed up when hitting the actual /api/drift route
with realistic data):

1. STATIONARY_FEATURES was defined but never wired into detect_drift(), so PSI
   ran on non-stationary price levels (close, sma_200, ema_*) and fired
   "critical" on every feature, every call — a useless, always-red monitor.
2. The live-data comparison window (30 rows) was too small for a reliable
   10-bucket PSI estimate — even two samples from the IDENTICAL distribution
   produced wildly different PSI values purely from sampling noise.
"""
import numpy as np
from app.models.drift import detect_drift, calculate_psi, STATIONARY_FEATURES


def test_detect_drift_excludes_non_stationary_features_by_default():
    rng = np.random.RandomState(0)
    n = 200
    training_df = {
        "close": rng.normal(1000, 50, n).cumsum(),   # non-stationary price level
        "rsi_14": rng.normal(50, 10, n),              # stationary oscillator
    }
    import pandas as pd
    training_df = pd.DataFrame(training_df)
    live_df = training_df.copy()

    result = detect_drift(training_df, live_df, ["close", "rsi_14"])

    assert "close" not in result["per_feature"], (
        "non-stationary price feature must be excluded from PSI by default"
    )
    assert "rsi_14" in result["per_feature"]


def test_psi_is_stable_with_adequate_sample_size():
    """Same distribution, no real drift -> PSI should stay well below the
    critical threshold once the live sample is large enough (~90 rows)."""
    rng = np.random.RandomState(0)
    training = rng.normal(50, 10, 380)

    psi_values = [calculate_psi(training, rng.normal(50, 10, 90)) for _ in range(5)]
    assert all(p < 0.25 for p in psi_values), (
        f"PSI should not falsely flag 'critical' on identical distributions: {psi_values}"
    )
