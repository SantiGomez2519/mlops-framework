import json
from pathlib import Path

import numpy as np

_STAGE_DIR = Path(__file__).resolve().parent.parent

SCALER_PATH = _STAGE_DIR / "shared" / "data" / "featured" / "scaler_params.json"
FEATURE_LIST_PATH = _STAGE_DIR / "shared" / "data" / "featured" / "feature_list.json"

CONDITION_MAP = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}
LOCATIONS = ["Downtown", "Mountain", "Rural", "Suburb", "Urban", "Waterfront"]
CURRENT_YEAR = 2026

with open(SCALER_PATH) as f:
    _scaler_params = json.load(f)

with open(FEATURE_LIST_PATH) as f:
    FEATURE_COLS = json.load(f)


def preprocess_input(data: dict) -> list:
    sqft = float(data["sqft"])
    bedrooms = int(data["bedrooms"])
    bathrooms = float(data["bathrooms"])
    location = data["location"]
    year_built = int(data["year_built"])
    condition = data["condition"]

    house_age = CURRENT_YEAR - year_built
    total_rooms = bedrooms + bathrooms
    bath_bed_ratio = bathrooms / max(bedrooms, 1)
    is_luxury = 1 if (CONDITION_MAP[condition] >= 3 and sqft >= 2500) else 0
    is_new = 1 if house_age <= 15 else 0
    sqft_per_bedroom = sqft / max(bedrooms, 1)

    raw = {
        "sqft": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "house_age": house_age,
        "total_rooms": total_rooms,
        "bath_bed_ratio": bath_bed_ratio,
        "sqft_per_bedroom": sqft_per_bedroom,
    }

    scaled = {}
    for key in _scaler_params["features"]:
        mean = _scaler_params["mean"][key]
        std = _scaler_params["std"][key]
        scaled[key] = (raw[key] - mean) / std if std != 0 else 0.0

    row = {}
    row["sqft"] = scaled["sqft"]
    row["bedrooms"] = scaled["bedrooms"]
    row["bathrooms"] = scaled["bathrooms"]
    row["year_built"] = float(year_built)
    row["house_age"] = scaled["house_age"]
    row["condition_encoded"] = float(CONDITION_MAP[condition])

    for loc in LOCATIONS:
        row[f"loc_{loc}"] = 1.0 if location == loc else 0.0

    row["total_rooms"] = scaled["total_rooms"]
    row["bath_bed_ratio"] = scaled["bath_bed_ratio"]
    row["is_luxury"] = float(is_luxury)
    row["is_new"] = float(is_new)
    row["sqft_per_bedroom"] = scaled["sqft_per_bedroom"]

    return [row[col] for col in FEATURE_COLS]
