from backend.api.models.schemas import HouseInput, PredictionResponse
from backend.pipeline import FeatureEngineering
from backend.pipeline.model import Model


class PredictionService:
    def __init__(self, model: Model):
        self._model = model

    def predict(self, data: HouseInput) -> PredictionResponse:
        features = FeatureEngineering.transform_one(
            data.model_dump(mode="json"), self._model.scaler
        )
        price = self._model.predict(features)
        info = self._model.info()
        return PredictionResponse(
            predicted_price=round(price, 2),
            model_name=info["model_name"],
            r2_score=info["test_r2"],
        )
