import numpy as np
import pandas as pd

from backend.pipeline.config import Config


class Preprocessing:
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["price"])

        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])

        df["price"] = df["price"].astype(float)
        df["sqft"] = df["sqft"].astype(int)
        df["bedrooms"] = df["bedrooms"].astype(int)
        df["bathrooms"] = df["bathrooms"].astype(float)
        df["year_built"] = df["year_built"].astype(int)

        for col in ["price", "sqft"]:
            df[col] = self._cap_outliers(df[col])

        df["house_age"] = Config.CURRENT_YEAR - df["year_built"]

        assert df["price"].min() > 0, "Price has non-positive values!"
        assert df["sqft"].min() > 0, "Sqft has non-positive values!"
        assert df["bedrooms"].min() >= 0, "Bedrooms has negative values!"
        assert df["bathrooms"].min() >= 0, "Bathrooms has negative values!"
        assert df["year_built"].min() >= 1900, "year_built has suspicious values!"
        assert df["year_built"].max() <= Config.CURRENT_YEAR, "year_built has future dates!"
        assert df["location"].isin(Config.LOCATIONS).all(), "Unexpected location values!"
        assert df["condition"].isin(Config.CONDITION_MAP).all(), "Unexpected condition values!"

        return df

    @staticmethod
    def _cap_outliers(series: pd.Series, factor: float = 1.5) -> pd.Series:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return series.clip(q1 - factor * iqr, q3 + factor * iqr)

    def run(self) -> pd.DataFrame:
        df = pd.read_csv(Config.RAW_PATH)
        processed = self.process(df)
        Config.PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        processed.to_csv(Config.PREPROCESSED_PATH, index=False)
        return processed
