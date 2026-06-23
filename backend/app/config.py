"""
Centralized configuration. NO hardcoded magic numbers anywhere else in the codebase —
every tunable constant lives here, overridable via environment variables.
This is the #1 fix vs the legacy version (which scattered constants across files).
"""
import os


class Config:
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models_store"))
    DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
    DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "db", "results.db"))

    # --- Model hyperparameters ---
    RF_REG_N_ESTIMATORS = int(os.environ.get("RF_REG_N_ESTIMATORS", 300))
    RF_REG_MAX_DEPTH = int(os.environ.get("RF_REG_MAX_DEPTH", 8))
    RF_CLF_N_ESTIMATORS = int(os.environ.get("RF_CLF_N_ESTIMATORS", 300))
    RF_CLF_MAX_DEPTH = int(os.environ.get("RF_CLF_MAX_DEPTH", 8))
    RANDOM_STATE = int(os.environ.get("RANDOM_STATE", 42))
    TEST_SIZE = float(os.environ.get("TEST_SIZE", 0.2))

    # --- Signal thresholds ---
    SIGNAL_BUY_THRESHOLD = float(os.environ.get("SIGNAL_BUY_THRESHOLD", 0.55))
    SIGNAL_SELL_THRESHOLD = float(os.environ.get("SIGNAL_SELL_THRESHOLD", 0.45))

    # --- Backtest economics ---
    DEFAULT_INITIAL_CAPITAL = float(os.environ.get("DEFAULT_INITIAL_CAPITAL", 100000))
    DEFAULT_COMMISSION = float(os.environ.get("DEFAULT_COMMISSION", 20))
    DEFAULT_SLIPPAGE = float(os.environ.get("DEFAULT_SLIPPAGE", 0.0005))
    POSITION_SIZE_PCT = float(os.environ.get("POSITION_SIZE_PCT", 0.2))

    # --- Drift detection ---
    PSI_WARNING_THRESHOLD = float(os.environ.get("PSI_WARNING_THRESHOLD", 0.1))
    PSI_CRITICAL_THRESHOLD = float(os.environ.get("PSI_CRITICAL_THRESHOLD", 0.25))
    PSI_BUCKETS = int(os.environ.get("PSI_BUCKETS", 10))
    DRIFT_LIVE_WINDOW = int(os.environ.get("DRIFT_LIVE_WINDOW", 90))

    # --- Cache ---
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", 3600))

    # --- Model staleness ---
    MODEL_MAX_AGE_DAYS = int(os.environ.get("MODEL_MAX_AGE_DAYS", 7))


config = Config()
