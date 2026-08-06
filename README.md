# MLOps Framework — House Price Predictor

End-to-end MLOps project that trains a Gradient Boosting model to predict house prices and serves predictions through a FastAPI backend with a Vue.js frontend. Experiment tracking via **MLflow** in Docker. Split into two independent stages: `1-experimentation` (research) and `2-production` (serving).

## Project Structure

```
mlops-framework/
├── 1-experimentation/                # Stage 1: research (self-contained)
│   ├── 1_1_raw_data_profiling.ipynb
│   ├── 1_2_raw_data_preprocessing.ipynb
│   ├── 1_3_preprocessed_data_exploratory_data_analysis.ipynb
│   ├── 1_4_preprocessed_data_feature_engineer.ipynb
│   ├── 1_5_featured_data_model_training.ipynb   # Trains models + logs to MLflow
│   ├── 1_6_featured_data_model_evaluation.ipynb
│   ├── data/                         # raw/ → processed/ → features/
│   ├── models/                       # best_model.joblib + experiment_results.json
│   ├── docker-compose.yml            # MLflow Tracking Server
│   └── requirements.txt
└── 2-production/                     # Stage 2: serving (self-contained)
    ├── backend/                      # FastAPI prediction service
    │   ├── main.py                   # API endpoints (health, info, predict)
    │   ├── model.py                  # Model loading and inference
    │   ├── preprocessing.py          # Feature engineering pipeline
    │   └── requirements.txt
    ├── frontend/                     # Vue.js web interface
    │   ├── index.html
    │   ├── package.json
    │   ├── vite.config.js
    │   └── src/
    │       ├── main.js
    │       └── App.vue               # Prediction form + result display
    └── shared/                       # Frozen contract deployed from stage 1
        ├── data/features/            #   scaler params + feature list
        └── models/                   #   deployed model + metrics
```

## Dataset

`1-experimentation/data/raw/house_data.csv` — 85 houses with 7 columns:

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

## MLflow Tracking (Docker)

Experiment tracking runs on a Dockerized MLflow server (defined in `1-experimentation/docker-compose.yml`). It uses a SQLite backend store with a Docker volume for persistence.

### Start MLflow Server

```bash
cd 1-experimentation
docker compose up -d
```

### Access MLflow UI

Open [http://localhost:5555](http://localhost:5555) to view experiments, compare runs, and download logged models.

### How It Works

- Notebook `1_5` sends params, metrics, and models to the MLflow server via HTTP
- Each of the 7 models is logged as a separate run
- The best tuned model is logged with `mlflow.sklearn.log_model()` in a dedicated run
- Data persists in a Docker volume (survives container restarts)

## Running the Notebooks

1. Install dependencies:

```bash
cd 1-experimentation
pip install -r requirements.txt
```

2. Run notebooks in order (1_1 through 1_6) in Jupyter or VS Code. The working directory must be `1-experimentation/` (all paths are relative to it).
3. Ensure the MLflow server is running (`docker compose up -d`) before running `1_5`.

## Deploying to Production

After training, copy the artifacts forward from stage 1 to stage 2:

```bash
cp 1-experimentation/data/features/*  2-production/shared/data/features/
cp 1-experimentation/models/*          2-production/shared/models/
```

## Running the Backend API

```bash
cd 2-production/backend
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
cd 2-production/frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`. The Vite dev server proxies API requests to the backend at port 8000.

## Tech Stack

- **ML**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **Tracking**: MLflow (Docker)
- **Backend**: FastAPI, uvicorn, Pydantic
- **Frontend**: Vue.js 3, Vite
- **Serialization**: joblib
