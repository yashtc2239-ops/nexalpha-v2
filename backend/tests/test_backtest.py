from app.core.indicators import add_indicators
from app.core.features import prepare_features
from app.models.train import train_models
from app.models.train import load_models
from app.services.backtest import run_backtest


def test_backtest_runs_only_on_out_of_sample_rows(synthetic_ohlcv, tmp_path):
    ind = add_indicators(synthetic_ohlcv)
    df_feat, X, y_reg, y_clf, feature_cols = prepare_features(ind, target_horizon=7)

    train_models("TEST.NS", df_feat, X, y_reg, y_clf, feature_cols, horizon=7,
                 model_dir=str(tmp_path), force=True)
    _, clf, meta = load_models("TEST.NS", horizon=7, model_dir=str(tmp_path))

    result = run_backtest("TEST.NS", df_feat, clf, feature_cols, meta["n_test_rows"])

    assert result["out_of_sample_rows"] == meta["n_test_rows"]
    assert len(result["equity_curve"]) == meta["n_test_rows"]
    assert "sharpe_ratio" in result
