from fastapi import APIRouter, Depends

from backend.api.api.deps import get_prediction_service
from backend.api.models.schemas import HouseInput, PredictionResponse
from backend.api.services.prediction_service import PredictionService

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(
    data: HouseInput,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    return service.predict(data)
