from backend.pipeline.config import Config
from backend.pipeline.model import Model


class ModelService:
    def __init__(self, model: Model):
        self._model = model

    def info(self) -> dict:
        return self._model.info()

    def locations(self) -> list[str]:
        return list(Config.LOCATIONS)
