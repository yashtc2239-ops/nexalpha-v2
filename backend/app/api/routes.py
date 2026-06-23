"""Flask routes — kept thin. All business logic lives in core/models/services;
routes only parse requests, call the pipeline, and serialize responses."""
import time
from flask import Blueprint, request, jsonify

from app.config import config
from app.logger import logger
from app.cache import cache
from app.core.data_loader import load_price_data
from app.core.indicators import add_indicators, fibonacci_levels
from app.core.features import prepare_features
from app.models.train import train_models
from app.models.predict import predict_price_and_signal
from app.models.drift import detect_drift
from app.services.backtest import run_backtest
from app.services.explain import explain_with_indicators, shap_explain
from app.db import save_run

bp = Blueprint("api", __name__)


@bp.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "nexalpha-v2", "cache_backend": cache.backend})


@bp.route("/api/analyze", methods=["POST"])
def analyze():
    payload = request.json or {}
    ticker = payload.get("ticker")
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    start, end = payload.get("start"), payload.get("end")
    horizon = int(payload.get("horizon_days", 7))

    cache_key = f"analyze:{ticker}:{start}:{end}:{horizon}"
    cached = cache.get(cache_key)
    if cached:
        cached["from_cache"] = True
        return jsonify(cached)

    try:
        df = load_price_data(ticker, start, end)
        df_ind = add_indicators(df)
        fib = fibonacci_levels(df_ind)
        df_feat, X, y_reg, y_clf, feature_cols = prepare_features(df_ind, target_horizon=horizon)

        train_models(ticker, df_feat, X, y_reg, y_clf, feature_cols, horizon)
        pred = predict_price_and_signal(ticker, df_feat, horizon)

        from app.models.train import load_models
        _, clf, meta = load_models(ticker, horizon)
        reasons = explain_with_indicators(df_feat.iloc[-1])
        shap_summary = shap_explain(clf, df_feat, feature_cols)

        result = {
            "ticker": ticker,
            "as_of": str(df_feat.iloc[-1]["date"]),
            "generated_at": time.time(),
            "current_price": float(df_feat.iloc[-1]["close"]),
            "prediction": pred,
            "fibonacci_levels": fib,
            "indicator_reasons": reasons,
            "shap_explanation": shap_summary,
            "from_cache": False,
        }

        cache.set(cache_key, result)
        try:
            save_run(result)
        except Exception as e:
            logger.warning(f"DB save failed (non-fatal): {e}")

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception(f"analyze failed for {ticker}")
        return jsonify({"error": f"internal error: {e}"}), 500


@bp.route("/api/backtest", methods=["POST"])
def api_backtest():
    payload = request.json or {}
    ticker = payload.get("ticker")
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    horizon = int(payload.get("horizon_days", 7))
    initial_cap = float(payload.get("initial_cap", config.DEFAULT_INITIAL_CAPITAL))
    commission = float(payload.get("commission", config.DEFAULT_COMMISSION))
    slippage = float(payload.get("slippage", config.DEFAULT_SLIPPAGE))

    try:
        df = load_price_data(ticker, payload.get("start"), payload.get("end"))
        df_ind = add_indicators(df)
        df_feat, X, y_reg, y_clf, feature_cols = prepare_features(df_ind, target_horizon=horizon)

        train_models(ticker, df_feat, X, y_reg, y_clf, feature_cols, horizon)
        from app.models.train import load_models
        _, clf, meta = load_models(ticker, horizon)

        result = run_backtest(ticker, df_feat, clf, feature_cols, meta["n_test_rows"],
                              initial_cap, commission, slippage)
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception(f"backtest failed for {ticker}")
        return jsonify({"error": f"internal error: {e}"}), 500


@bp.route("/api/drift/<ticker>")
def drift_check(ticker):
    horizon = int(request.args.get("horizon_days", 7))
    try:
        from app.models.train import load_models
        _, _, meta = load_models(ticker, horizon)

        df = load_price_data(ticker)
        df_ind = add_indicators(df)
        df_feat, _, _, _, feature_cols = prepare_features(df_ind, target_horizon=horizon)

        n_test = meta["n_test_rows"]
        training_slice = df_feat.iloc[:-n_test] if n_test < len(df_feat) else df_feat
        live_window = min(config.DRIFT_LIVE_WINDOW, n_test if n_test else config.DRIFT_LIVE_WINDOW)
        live_slice = df_feat.tail(live_window)

        drift_result = detect_drift(training_slice, live_slice, feature_cols)
        return jsonify(drift_result)
    except FileNotFoundError:
        return jsonify({"error": f"no trained model for {ticker} h={horizon}"}), 404
