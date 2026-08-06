from fastapi import APIRouter, Depends

from backend.api.api.deps import get_model_service
from backend.api.models.schemas import LocationsResponse, ModelInfoResponse
from backend.api.services.model_service import ModelService

router = APIRouter(prefix="/model")


@router.get("/info", response_model=ModelInfoResponse)
def model_info(
    service: ModelService = Depends(get_model_service),
) -> ModelInfoResponse:
    return ModelInfoResponse(**service.info())


@router.get("/locations", response_model=LocationsResponse)
def model_locations(
    service: ModelService = Depends(get_model_service),
) -> LocationsResponse:
    return LocationsResponse(locations=service.locations())
