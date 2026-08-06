from fastapi import Depends

from backend.api.core.exceptions import ModelNotLoadedError
from backend.api.services.model_service import ModelService
from backend.api.services.prediction_service import PredictionService
from backend.pipeline.config import Config
from backend.pipeline.model import Model

_model: Model | None = None


def get_model() -> Model:
    global _model
    if _model is None:
        try:
            _model = Model.load(Config.MODEL_PATH)
        except (OSError, ValueError) as exc:
            raise ModelNotLoadedError from exc
    return _model


def get_prediction_service(model: Model = Depends(get_model)) -> PredictionService:
    return PredictionService(model)


def get_model_service(model: Model = Depends(get_model)) -> ModelService:
    return ModelService(model)
