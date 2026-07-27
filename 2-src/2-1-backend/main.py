from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import get_model_info, predict_price
from preprocessing import preprocess_input, LOCATIONS

app = FastAPI(title="House Price Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HouseInput(BaseModel):
    sqft: float
    bedrooms: int
    bathrooms: float
    location: str
    year_built: int
    condition: str


class PredictionResponse(BaseModel):
    predicted_price: float
    model_name: str
    r2_score: float


@app.get("/")
def health_check():
    return {"status": "ok", "message": "House Price Prediction API is running"}


@app.get("/model/info")
def model_info():
    return get_model_info()


@app.get("/model/locations")
def model_locations():
    return {"locations": LOCATIONS}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: HouseInput):
    features = preprocess_input(data.model_dump())
    price = predict_price(features)
    info = get_model_info()
    return PredictionResponse(
        predicted_price=round(price, 2),
        model_name=info["model_name"],
        r2_score=info["test_r2"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

