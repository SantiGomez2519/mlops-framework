import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from backend.pipeline.config import Config


class FeatureEngineering:
    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
        df_encoded = pd.get_dummies(df, columns=["location"], prefix="loc", drop_first=False)

        df_encoded["condition_encoded"] = df_encoded["condition"].map(Config.CONDITION_MAP)
        df_encoded.drop(columns=["condition"], inplace=True)

        df_encoded["total_rooms"] = df_encoded["bedrooms"] + df_encoded["bathrooms"]
        df_encoded["bath_bed_ratio"] = (
            df_encoded["bathrooms"] / df_encoded["bedrooms"].replace(0, 1)
        )
        df_encoded["is_luxury"] = (
            (df_encoded["condition_encoded"] >= 3) & (df_encoded["sqft"] >= 2500)
        ).astype(int)
        df_encoded["is_new"] = (df_encoded["house_age"] <= 15).astype(int)
        df_encoded["sqft_per_bedroom"] = (
            df_encoded["sqft"] / df_encoded["bedrooms"].replace(0, 1)
        )
        df_encoded["log_price"] = np.log1p(df_encoded["price"])

        scaler = StandardScaler()
        df_encoded[Config.FEATURES_TO_SCALE] = scaler.fit_transform(
            df_encoded[Config.FEATURES_TO_SCALE]
        )

        columns = Config.FEATURE_COLS + ["price", "log_price"]
        return df_encoded[columns], scaler

    @staticmethod
    def transform_one(house: dict, scaler: StandardScaler) -> list[float]:
        sqft = float(house["sqft"])
        bedrooms = int(house["bedrooms"])
        bathrooms = float(house["bathrooms"])
        location = house["location"]
        year_built = int(house["year_built"])
        condition = house["condition"]

        house_age = Config.CURRENT_YEAR - year_built
        total_rooms = bedrooms + bathrooms
        bath_bed_ratio = bathrooms / max(bedrooms, 1)
        is_luxury = 1 if (Config.CONDITION_MAP[condition] >= 3 and sqft >= 2500) else 0
        is_new = 1 if house_age <= 15 else 0
        sqft_per_bedroom = sqft / max(bedrooms, 1)

        row = {
            "sqft": sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "year_built": year_built,
            "house_age": house_age,
            **{f"loc_{loc}": 1.0 if location == loc else 0.0 for loc in Config.LOCATIONS},
            "condition_encoded": float(Config.CONDITION_MAP[condition]),
            "total_rooms": total_rooms,
            "bath_bed_ratio": bath_bed_ratio,
            "is_luxury": float(is_luxury),
            "is_new": float(is_new),
            "sqft_per_bedroom": sqft_per_bedroom,
        }

        df = pd.DataFrame([row])[Config.FEATURE_COLS]
        df[Config.FEATURES_TO_SCALE] = scaler.transform(df[Config.FEATURES_TO_SCALE])
        return df.iloc[0].tolist()

    def run(self) -> tuple[pd.DataFrame, StandardScaler]:
        df = pd.read_csv(Config.PREPROCESSED_PATH)
        featured, scaler = self.fit_transform(df)
        Config.FEATURED_DIR.mkdir(parents=True, exist_ok=True)
        featured.to_csv(Config.FEATURED_PATH, index=False)
        return featured, scaler
