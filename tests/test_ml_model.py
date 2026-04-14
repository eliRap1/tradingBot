from ml_model import MLMetaModel


def test_feature_vectors_same_length():
    model = MLMetaModel()
    trade = {
        "strategies": ["momentum", "supertrend"],
        "pnl": 50.0,
        "strategy_scores": {"momentum": 0.6, "supertrend": 0.4},
        "num_agreeing": 2,
        "composite_score": 0.5,
    }
    live_features = {
        "strategy_scores": {"momentum": 0.6, "supertrend": 0.4},
        "num_agreeing": 2,
        "composite_score": 0.5,
    }

    train_vec = model._trade_to_features(trade)
    pred_vec = model._build_feature_vector(live_features)

    assert train_vec is not None
    assert pred_vec is not None
    assert len(train_vec) == len(pred_vec)


def test_feature_vectors_use_float_scores():
    model = MLMetaModel()
    trade = {
        "strategies": ["momentum"],
        "strategy_scores": {"momentum": 0.7},
        "num_agreeing": 1,
        "composite_score": 0.7,
        "pnl": 10.0,
    }
    vec = model._trade_to_features(trade)
    assert vec[0] == 0.7


def test_no_r_multiple_in_training_features():
    model = MLMetaModel()
    trade = {
        "strategies": ["momentum"],
        "strategy_scores": {"momentum": 0.7},
        "num_agreeing": 1,
        "composite_score": 0.7,
        "pnl": 100.0,
        "r_multiple": 5.0,
    }
    vec1 = model._trade_to_features(trade)
    trade["r_multiple"] = 0.5
    vec2 = model._trade_to_features(trade)
    assert vec1 == vec2
