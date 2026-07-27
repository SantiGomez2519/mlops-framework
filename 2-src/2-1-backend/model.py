import json
import numpy as np
import pandas as pd
import joblib

MODEL_PATH = r"C:\Users\Usuario\Desktop\EAFIT\Exoesqueleto\mlops-framework\shared\models\best_model.joblib"
RESULTS_PATH = r"C:\Users\Usuario\Desktop\EAFIT\Exoesqueleto\mlops-framework\shared\models\experiment_results.json"
FEATURE_LIST_PATH = r"C:\Users\Usuario\Desktop\EAFIT\Exoesqueleto\mlops-framework\shared\data\features\feature_list.json"

with open(FEATURE_LIST_PATH) as f:
    FEATURE_COLS = json.load(f)

model = joblib.load(MODEL_PATH)

with open(RESULTS_PATH) as f:
    experiment_results = json.load(f)

MODEL_NAME = experiment_results["best_model"]
MODEL_METRICS = experiment_results["final_metrics"]


def get_model_info() -> dict:
    return {
        "model_name": MODEL_NAME,
        "model_type": type(model).__name__,
        "test_r2": MODEL_METRICS["Test_R2"],
        "test_rmse": MODEL_METRICS["Test_RMSE"],
        "test_mae": MODEL_METRICS["Test_MAE"],
        "test_mape": MODEL_METRICS["Test_MAPE"],
    }


def predict_price(features: list) -> float:
    X = pd.DataFrame([features], columns=FEATURE_COLS)
    log_price = model.predict(X)[0]
    return float(np.expm1(log_price))
