from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.api.routes import health, model, predict
from backend.api.core.exceptions import ModelNotLoadedError


def create_app() -> FastAPI:
    app = FastAPI(title="House Price Prediction API", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={"detail": "Model bundle is not available. Run `python -m backend.pipeline` first."},
        )

    app.include_router(health.router)
    app.include_router(model.router)
    app.include_router(predict.router)

    return app


app = create_app()
