from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Model:
    def __init__(self, bundle: dict):
        self._bundle = bundle

    @classmethod
    def load(cls, path: Path) -> "Model":
        return cls(joblib.load(path))

    @property
    def scaler(self) -> StandardScaler:
        return self._bundle["scaler"]

    @property
    def feature_cols(self) -> list[str]:
        return self._bundle["feature_cols"]

    def predict(self, features: list) -> float:
        X = pd.DataFrame([features], columns=self.feature_cols)
        log_price = self._bundle["model"].predict(X)[0]
        return float(np.expm1(log_price))

    def info(self) -> dict:
        test = self._bundle["metrics"]["test"]
        return {
            "model_name": self._bundle["model_name"],
            "model_type": type(self._bundle["model"]).__name__,
            "params": self._bundle["params"],
            "test_r2": test["r2"],
            "test_rmse": test["rmse"],
            "test_mae": test["mae"],
            "test_mape": test["mape"],
        }
