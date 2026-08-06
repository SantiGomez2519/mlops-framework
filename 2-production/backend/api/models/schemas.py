from pydantic import BaseModel, Field

from backend.pipeline.enums import Condition, Location


class HouseInput(BaseModel):
    sqft: float = Field(gt=0)
    bedrooms: int = Field(ge=0)
    bathrooms: float = Field(ge=0)
    location: Location
    year_built: int = Field(ge=1900, le=2026)
    condition: Condition


class PredictionResponse(BaseModel):
    predicted_price: float
    model_name: str
    r2_score: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    params: dict
    test_r2: float
    test_rmse: float
    test_mae: float
    test_mape: float


class LocationsResponse(BaseModel):
    locations: list[str]


class HealthResponse(BaseModel):
    status: str
    message: str
