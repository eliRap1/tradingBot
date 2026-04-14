"""
ML meta-model — LightGBM signal stacker.

Takes raw strategy scores + market features as input, predicts probability
of a profitable trade. Trained on closed trade history with strategy attribution.

Features:
  - Per-strategy score (8 features, one per strategy)
  - Composite score
  - Number of agreeing strategies
  - Market regime (encoded)
  - ADX, RSI of entry bar
  - Order flow imbalance
  - Hourly bias (encoded)

Target: 1 if trade was profitable, 0 if loss.

Usage:
  model = MLMetaModel()
  model.train(tracker.trades)         # periodic retraining
  p = model.predict(features_dict)    # returns probability 0-1
"""

import os
import json
import numpy as np
from utils import setup_logger

log = setup_logger("ml_model")

MODEL_FILE = os.path.join(os.path.dirname(__file__), "ml_model.bin")
FEATURE_FILE = os.path.join(os.path.dirname(__file__), "ml_features.json")

STRATEGY_NAMES = [
    "momentum", "mean_reversion", "breakout", "supertrend",
    "stoch_rsi", "vwap_reclaim", "gap", "liquidity_sweep",
]

REGIME_MAP = {"bull": 2, "chop": 1, "bear": 0}
BIAS_MAP = {"bullish": 2, "neutral": 1, "bearish": 0}


class MLMetaModel:
    def __init__(self, min_trades: int = 20):
        self.min_trades = min_trades
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load saved model if available."""
        try:
            import lightgbm as lgb
            if os.path.exists(MODEL_FILE):
                self.model = lgb.Booster(model_file=MODEL_FILE)
                log.info("ML meta-model loaded from disk")
        except Exception as e:
            log.warning(f"Failed to load ML model: {e}")
            self.model = None

    def train(self, trades: list[dict]) -> bool:
        """
        Train the meta-model on historical trade data.

        Each trade needs: pnl, strategies (list), and optionally
        strategy_scores, regime, adx, etc. stored in the trade record.

        Returns True if training succeeded.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            log.warning("lightgbm not installed — ML meta-model disabled")
            return False

        # Filter trades with strategy attribution
        valid_trades = [t for t in trades if t.get("strategies")]
        if len(valid_trades) < self.min_trades:
            log.info(f"ML model needs {self.min_trades} trades with strategy data, "
                     f"have {len(valid_trades)} — skipping training")
            return False

        X, y = [], []
        for t in valid_trades:
            features = self._trade_to_features(t)
            if features is not None:
                X.append(features)
                y.append(1 if t["pnl"] >= 0 else 0)

        if len(X) < self.min_trades:
            return False

        X = np.array(X)
        y = np.array(y)

        # Train/val split (last 20% as validation)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 15,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 5,
            "verbose": -1,
        }

        callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]
        self.model = lgb.train(
            params, train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        # Save model
        self.model.save_model(MODEL_FILE)

        # Log performance
        val_pred = self.model.predict(X_val)
        val_acc = np.mean((val_pred > 0.5) == y_val)
        log.info(f"ML meta-model trained on {len(X_train)} trades, "
                 f"val accuracy={val_acc:.1%} ({len(X_val)} val trades)")

        return True

    def predict(self, features: dict) -> float | None:
        """
        Predict probability of a profitable trade.

        Args:
            features: dict with keys matching _build_feature_vector

        Returns:
            float 0-1 (probability of profit) or None if model unavailable
        """
        if self.model is None:
            return None

        vec = self._build_feature_vector(features)
        if vec is None:
            return None

        try:
            prob = float(self.model.predict(np.array([vec]))[0])
            return prob
        except Exception as e:
            log.warning(f"ML prediction failed: {e}")
            return None

    def _trade_to_features(self, trade: dict) -> list | None:
        """Convert a closed trade record to the live prediction feature space."""
        scores = trade.get("strategy_scores", {})
        strats = set(trade.get("strategies", []))
        if not strats and not scores:
            return None

        features = []
        for s in STRATEGY_NAMES:
            if s in scores:
                features.append(float(scores[s]))
            elif s in strats:
                features.append(1.0)
            else:
                features.append(0.0)
        features.append(float(trade.get("num_agreeing", len(strats))))
        features.append(float(trade.get("composite_score", 0.0)))

        return features

    def _build_feature_vector(self, features: dict) -> list | None:
        """Build feature vector from live signal data."""
        vec = []

        # Per-strategy scores
        scores = features.get("strategy_scores", {})
        for s in STRATEGY_NAMES:
            vec.append(float(scores.get(s, 0.0)))

        # Number of agreeing
        vec.append(float(features.get("num_agreeing", 0)))

        # Composite score as proxy for R-multiple at entry
        vec.append(float(features.get("composite_score", 0.0)))

        return vec

    @property
    def is_ready(self) -> bool:
        return self.model is not None
