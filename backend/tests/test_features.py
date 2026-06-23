from app.core.indicators import add_indicators
from app.core.features import prepare_features, LEAKAGE_COLUMNS, get_feature_columns, select_features


def test_no_leakage_columns_in_feature_matrix(synthetic_ohlcv):
    """The critical regression test for the bug class found in the legacy project:
    none of target_close/future_ret/future_dir/date may ever appear in X."""
    ind = add_indicators(synthetic_ohlcv)
    df_feat, X, y_reg, y_clf, feature_cols = prepare_features(ind, target_horizon=7)

    for leaky_col in LEAKAGE_COLUMNS:
        assert leaky_col not in feature_cols
        assert leaky_col not in X.columns


def test_select_features_matches_training_columns(synthetic_ohlcv):
    ind = add_indicators(synthetic_ohlcv)
    df_feat, X, y_reg, y_clf, feature_cols = prepare_features(ind, target_horizon=7)

    reselected = select_features(df_feat, feature_cols)
    assert list(reselected.columns) == feature_cols


def test_select_features_raises_on_missing_column(synthetic_ohlcv):
    ind = add_indicators(synthetic_ohlcv)
    df_feat, *_ = prepare_features(ind, target_horizon=7)
    try:
        select_features(df_feat, ["nonexistent_col"])
        assert False, "expected ValueError"
    except ValueError:
        pass
