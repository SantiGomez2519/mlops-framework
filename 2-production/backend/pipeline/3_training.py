from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from backend.pipeline.config import Config


class Training:
    def run(self, featured: pd.DataFrame, scaler: StandardScaler) -> dict:
        X = featured[Config.FEATURE_COLS]
        y = featured[Config.TARGET]
        y_original = featured["price"]

        X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = train_test_split(
            X,
            y,
            y_original,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
        )

        model = GradientBoostingRegressor(
            **Config.BEST_PARAMS, random_state=Config.RANDOM_STATE
        )
        model.fit(X_train, y_train)

        train_metrics = self._evaluate(model, X_train, y_train, y_train_orig)
        test_metrics = self._evaluate(model, X_test, y_test, y_test_orig)

        bundle = {
            "model": model,
            "scaler": scaler,
            "feature_cols": Config.FEATURE_COLS,
            "target": Config.TARGET,
            "model_name": Config.MODEL_NAME,
            "params": {**Config.BEST_PARAMS, "random_state": Config.RANDOM_STATE},
            "metrics": {"train": train_metrics, "test": test_metrics},
            "current_year": Config.CURRENT_YEAR,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

        Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, Config.MODEL_PATH)

        print(f"Saved model bundle to {Config.MODEL_PATH}")
        print(
            f"Train R2: {train_metrics['r2']:.4f}  RMSE: ${train_metrics['rmse']:,.2f}  "
            f"MAE: ${train_metrics['mae']:,.2f}  MAPE: {train_metrics['mape']:.4f}"
        )
        print(
            f"Test  R2: {test_metrics['r2']:.4f}  RMSE: ${test_metrics['rmse']:,.2f}  "
            f"MAE: ${test_metrics['mae']:,.2f}  MAPE: {test_metrics['mape']:.4f}"
        )
        return bundle

    @staticmethod
    def _evaluate(model, X, y_log, y_orig) -> dict:
        pred_log = model.predict(X)
        pred = np.expm1(pred_log)
        y = np.expm1(y_log)
        return {
            "r2": float(r2_score(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "mape": float(mean_absolute_percentage_error(y, pred)),
        }
