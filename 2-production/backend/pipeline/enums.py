from enum import Enum

from backend.pipeline.config import Config


class DataStage(str, Enum):
    RAW = "raw"
    PREPROCESSED = "preprocessed"
    FEATURED = "featured"


class Location(str, Enum):
    DOWNTOWN = "Downtown"
    MOUNTAIN = "Mountain"
    RURAL = "Rural"
    SUBURB = "Suburb"
    URBAN = "Urban"
    WATERFRONT = "Waterfront"


class Condition(str, Enum):
    POOR = "Poor"
    FAIR = "Fair"
    GOOD = "Good"
    EXCELLENT = "Excellent"

    @property
    def encoded(self) -> int:
        return Config.CONDITION_MAP[self.value]
