from fastapi import APIRouter

from backend.api.models.schemas import HealthResponse

router = APIRouter()


@router.get("/", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok", message="House Price Prediction API is running"
    )
