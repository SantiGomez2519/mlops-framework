from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]


class Config:
    BACKEND_DIR = _BACKEND_DIR

    DATA_DIR = BACKEND_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    PREPROCESSED_DIR = DATA_DIR / "preprocessed"
    FEATURED_DIR = DATA_DIR / "featured"
    MODELS_DIR = BACKEND_DIR / "models"

    RAW_PATH = RAW_DIR / "house_data.csv"
    PREPROCESSED_PATH = PREPROCESSED_DIR / "house_data_preprocessed.csv"
    FEATURED_PATH = FEATURED_DIR / "house_data_features.csv"
    MODEL_PATH = MODELS_DIR / "house_price_model.joblib"

    CURRENT_YEAR = 2026
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    MODEL_NAME = "Gradient Boosting"
    BEST_PARAMS = {
        "learning_rate": 0.1,
        "max_depth": 7,
        "min_samples_split": 2,
        "n_estimators": 200,
    }

    CONDITION_MAP = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}
    LOCATIONS = ["Downtown", "Mountain", "Rural", "Suburb", "Urban", "Waterfront"]

    FEATURES_TO_SCALE = [
        "sqft",
        "bedrooms",
        "bathrooms",
        "house_age",
        "total_rooms",
        "bath_bed_ratio",
        "sqft_per_bedroom",
    ]

    FEATURE_COLS = [
        "sqft",
        "bedrooms",
        "bathrooms",
        "year_built",
        "house_age",
        "loc_Downtown",
        "loc_Mountain",
        "loc_Rural",
        "loc_Suburb",
        "loc_Urban",
        "loc_Waterfront",
        "condition_encoded",
        "total_rooms",
        "bath_bed_ratio",
        "is_luxury",
        "is_new",
        "sqft_per_bedroom",
    ]

    TARGET = "log_price"
