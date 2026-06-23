"""
Model training + persistence.

Fixes vs legacy:
1. Model filename keys on (ticker, horizon) — legacy keyed only on ticker, so a
   model trained for horizon=7 was silently reused for horizon=30 requests.
2. Feature column list + order is persisted as metadata next to the model
   (`*_meta.json`), and loaded back at predict/backtest time — this is what
   actually prevents the column-mismatch bug, not just "being careful".
3. Staleness check: metadata stores trained_at; predict.py refuses to silently
   serve a model older than MODEL_MAX_AGE_DAYS without retraining.
4. Metrics are logged via MLflow for experiment tracking — every training run
   is reproducible and comparable, not just a print() statement that vanishes.
"""
import os
import json
import time
import joblib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

from app.config import config
from app.logger import logger

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def _model_paths(ticker, horizon, model_dir):
    key = f"{ticker}_h{horizon}"
    return {
        "reg": os.path.join(model_dir, f"{key}_reg.pkl"),
        "clf": os.path.join(model_dir, f"{key}_clf.pkl"),
        "meta": os.path.join(model_dir, f"{key}_meta.json"),
    }


def is_model_fresh(meta_path):
    if not os.path.exists(meta_path):
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    age_days = (time.time() - meta["trained_at"]) / 86400
    return age_days < config.MODEL_MAX_AGE_DAYS


def train_models(ticker, df_feat, X, y_reg, y_clf, feature_cols, horizon,
                 model_dir=None, force=False):
    model_dir = model_dir or config.MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    paths = _model_paths(ticker, horizon, model_dir)

    if not force and is_model_fresh(paths["meta"]):
        logger.info(f"Skipping training for {ticker} h={horizon} — fresh model exists")
        return paths

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=config.TEST_SIZE, shuffle=False)
    _, _, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=config.TEST_SIZE, shuffle=False)

    reg = RandomForestRegressor(
        n_estimators=config.RF_REG_N_ESTIMATORS,
        max_depth=config.RF_REG_MAX_DEPTH,
        random_state=config.RANDOM_STATE, n_jobs=-1)
    reg.fit(X_train, y_reg_train)
    reg_mse = mean_squared_error(y_reg_test, reg.predict(X_test))

    clf = RandomForestClassifier(
        n_estimators=config.RF_CLF_N_ESTIMATORS,
        max_depth=config.RF_CLF_MAX_DEPTH,
        class_weight="balanced",  # fix: without this, the classifier degenerates to
        # always predicting the majority class on imbalanced tickers (found via
        # multi-ticker testing — TCS.NS gave F1=0.000 with 0 backtest trades, i.e.
        # the model never predicted "up" even once, despite "good-looking" accuracy)
        random_state=config.RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_clf_train)
    clf_preds = clf.predict(X_test)
    clf_acc = accuracy_score(y_clf_test, clf_preds)

    class_balance = float(y_clf_train.mean())  # fraction of "up" days in training set
    if class_balance < 0.35 or class_balance > 0.65:
        logger.warning(
            f"{ticker} h={horizon}: training classes are imbalanced "
            f"({class_balance:.1%} 'up' days) — classifier may favor the majority class"
        )
    clf_f1 = f1_score(y_clf_test, clf_preds, zero_division=0)

    joblib.dump(reg, paths["reg"])
    joblib.dump(clf, paths["clf"])

    meta = {
        "ticker": ticker,
        "horizon": horizon,
        "feature_cols": feature_cols,
        "trained_at": time.time(),
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "metrics": {
            "reg_mse": reg_mse, "clf_accuracy": clf_acc, "clf_f1": clf_f1,
            "train_class_balance_pct_up": round(class_balance * 100, 1),
        },
    }
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)

    if MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run(run_name=f"{ticker}_h{horizon}"):
                mlflow.log_params({
                    "ticker": ticker, "horizon": horizon,
                    "rf_reg_n_estimators": config.RF_REG_N_ESTIMATORS,
                    "rf_clf_n_estimators": config.RF_CLF_N_ESTIMATORS,
                })
                mlflow.log_metrics({
                    "reg_mse": reg_mse, "clf_accuracy": clf_acc, "clf_f1": clf_f1,
                })
        except Exception as e:
            logger.warning(f"MLflow logging skipped: {e}")

    logger.info(f"Trained {ticker} h={horizon} | MSE={reg_mse:.6f} ACC={clf_acc:.3f} F1={clf_f1:.3f}")
    return paths


def load_models(ticker, horizon, model_dir=None):
    model_dir = model_dir or config.MODEL_DIR
    paths = _model_paths(ticker, horizon, model_dir)
    if not all(os.path.exists(p) for p in paths.values()):
        raise FileNotFoundError(f"No trained model found for {ticker} h={horizon}")
    reg = joblib.load(paths["reg"])
    clf = joblib.load(paths["clf"])
    with open(paths["meta"]) as f:
        meta = json.load(f)
    return reg, clf, meta
