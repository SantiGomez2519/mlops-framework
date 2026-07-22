# MLOps Framework — House Price Predictor

End-to-end MLOps project that trains a Gradient Boosting model to predict house prices and serves predictions through a FastAPI backend with a Vue.js frontend.

## Project Structure

```
mlops-framework/
├── notebooks/                          # Experimentation notebooks (run in order)
│   ├── 1_preprocessing.ipynb           # Data cleaning, type fixing, outlier handling
│   ├── 2_exploratory_data_analysis.ipynb  # Statistical summaries, visualizations, correlations
│   ├── 3_feature_engineering.ipynb     # Encoding, scaling, new features, model-ready data
│   ├── 4_experimentation.ipynb         # Model training, evaluation, hyperparameter tuning
│   └── requirements.txt
├── 2-src-backend/                      # FastAPI prediction service
│   ├── main.py                         # API endpoints (health, info, predict)
│   ├── model.py                        # Model loading and inference
│   ├── preprocessing.py                # Feature engineering pipeline for single predictions
│   └── requirements.txt
├── 3-src-frontend/                     # Vue.js web interface
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.js
│       └── App.vue                     # Prediction form + result display
└── shared/                             # Shared data and model artifacts
    ├── data/
    │   ├── raw/                        # Original CSV (85 rows, 7 columns)
    │   ├── processed/                  # Cleaned data after preprocessing
    │   └── features/                   # Engineered features, scaler params, feature list
    └── models/
        ├── best_model.joblib           # Serialized Gradient Boosting model
        └── experiment_results.json     # All model metrics and parameters
```

## Dataset

`shared/data/raw/house_data.csv` — 85 houses with 7 columns:

| Column | Type | Description |
|--------|------|-------------|
| `price` | float | Sale price (target) |
| `sqft` | int | Square footage |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | float | Number of bathrooms |
| `location` | string | Suburb, Downtown, Rural, Waterfront, Urban, Mountain |
| `year_built` | int | Year the house was built |
| `condition` | string | Poor, Fair, Good, Excellent |

## Model Performance

Best model: **Gradient Boosting** (tuned via GridSearchCV, 5-fold CV R² = 0.998)

| Metric | Train | Test |
|--------|-------|------|
| R² | 0.9999 | 0.9489 |
| RMSE | $2,501 | $62,093 |
| MAE | $806 | $38,911 |
| MAPE | 0.16% | 5.81% |

## Running the Notebooks

1. Install dependencies:

```bash
cd notebooks
pip install -r requirements.txt
```

2. Run notebooks in order (1 through 4) in Jupyter or VS Code.

## Running the Backend API

```bash
cd 2-src-backend
pip install -r requirements.txt
python main.py
```

The API starts at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/model/info` | Model metadata and test metrics |
| GET | `/model/locations` | Available location values |
| POST | `/predict` | Predict house price |

### Predict Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sqft": 2000,
    "bedrooms": 3,
    "bathrooms": 2.0,
    "location": "Suburb",
    "year_built": 1990,
    "condition": "Good"
  }'
```

Response:

```json
{
  "predicted_price": 547331.07,
  "model_name": "Gradient Boosting",
  "r2_score": 0.9489
}
```

## Running the Frontend

```bash
cd 3-src-frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`. The Vite dev server proxies API requests to the backend at port 8000.

## Tech Stack

- **ML**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **Backend**: FastAPI, uvicorn, Pydantic
- **Frontend**: Vue.js 3, Vite
- **Serialization**: joblib
