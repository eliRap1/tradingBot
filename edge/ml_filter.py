from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np

from state_db import StateDB

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "research", "ml_filter.pkl")

REGIME_MAP = {"bull_trending": 3, "bull_choppy": 2, "bear_trending": 1, "bear_choppy": 0}
SESSION_MAP = {"open": 0, "mid": 1, "close": 2}
STRATEGY_NAMES = [
    "momentum", "mean_reversion", "breakout", "supertrend",
    "stoch_rsi", "vwap_reclaim", "gap", "liquidity_sweep",
]


@dataclass
class MLMetadata:
    n_samples: int = 0
    auc: float = 0.0
    train_date: str = ""


class MLSignalFilter:
    """Chronological LightGBM classifier with passthrough fallback."""

    def __init__(self, min_trades: int = 100):
        self.min_trades = min_trades
        self.model = None
        self.meta = MLMetadata()
        self._last_train_n: int = 0  # trade count at last training run
        self._retrain_interval: int = 50  # retrain every N new trades
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_PATH):
            return
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self.model = data.get("model")
            self.meta = data.get("meta", MLMetadata())
        except Exception:
            self.model = None

    def maybe_train(self, trades: list[dict] | None = None) -> bool:
        trades = trades or StateDB().get_trades()
        n = len(trades)
        if n < self.min_trades:
            return False
        # Only retrain when enough new trades have accumulated since last run
        if n < self._last_train_n + self._retrain_interval:
            return False
        try:
            import lightgbm as lgb
        except Exception:
            return False

        X, y = [], []
        for trade in trades:
            vec = self._trade_to_features(trade)
            if vec is None:
                continue
            X.append(vec)
            y.append(1 if (trade.get("r_multiple") or 0) >= 1.0 or trade.get("pnl", 0) > 0 else 0)
        if len(X) < self.min_trades:
            return False

        split = max(int(len(X) * 0.7), 1)
        X_train = np.array(X[:split])
        y_train = np.array(y[:split])
        X_test = np.array(X[split:])
        y_test = np.array(y[split:])
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        if len(X_test):
            valid_sets.append(lgb.Dataset(X_test, label=y_test, reference=train_data))
        self.model = lgb.train(
            {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.05,
                "num_leaves": 15,
                "verbose": -1,
            },
            train_data,
            num_boost_round=100,
            valid_sets=valid_sets,
        )
        auc = 0.0
        if len(X_test):
            try:
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(y_test, self.model.predict(X_test)))
            except Exception:
                auc = 0.0
        self.meta = MLMetadata(n_samples=len(X), auc=auc, train_date=os.path.basename(__file__))
        self._last_train_n = len(X)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": self.model, "meta": self.meta}, f)
        return True

    def predict_quality(self, features: dict | None = None) -> float:
        if self.model is None or not features:
            return 1.0
        vec = self._build_feature_vector(features)
        try:
            prob = float(self.model.predict(np.array([vec]))[0])
            return max(0.1, min(1.0, prob))
        except Exception:
            return 1.0

    def _trade_to_features(self, trade: dict) -> list[float] | None:
        scores = trade.get("strategy_scores", {})
        strategies = set(trade.get("strategies", []))
        if not scores and not strategies:
            return None
        regime = trade.get("regime_4state", "bull_choppy")
        return [
            *[float(scores.get(name, 1.0 if name in strategies else 0.0)) for name in STRATEGY_NAMES],
            float(trade.get("composite_score", 0.0)),
            float(trade.get("num_agreeing", len(strategies))),
            float(REGIME_MAP.get(regime, 2)),
            float(trade.get("high_vol", False)),
            float(trade.get("size_multiplier", 1.0)),
            float(trade.get("spread_pct", 0.0)),
            float(trade.get("ofi_score", 0.0)),
            float(trade.get("spy_corr", 0.5)),
            float(trade.get("hour_of_day", 12)),
            float(trade.get("day_of_week", 2)),
            float(SESSION_MAP.get(trade.get("session_bucket", "mid"), 1)),
            float(trade.get("days_to_earnings", 99)),
            float(trade.get("nq_overnight_move", 0.0)),
        ]

    def _build_feature_vector(self, features: dict) -> list[float]:
        strategy_scores = features.get("strategy_scores", {})
        return [
            *[float(strategy_scores.get(name, 0.0)) for name in STRATEGY_NAMES],
            float(features.get("composite_score", 0.0)),
            float(features.get("num_agreeing", 0)),
            float(REGIME_MAP.get(features.get("regime", "bull_choppy"), 2)),
            float(features.get("high_vol", False)),
            float(features.get("size_multiplier", 1.0)),
            float(features.get("spread_pct", 0.0)),
            float(features.get("ofi_score", 0.0)),
            float(features.get("spy_corr", 0.5)),
            float(features.get("hour_of_day", 12)),
            float(features.get("day_of_week", 2)),
            float(SESSION_MAP.get(features.get("session_bucket", "mid"), 1)),
            float(features.get("days_to_earnings", 99)),
            float(features.get("nq_overnight_move", 0.0)),
        ]
