# Notebook Pipeline — How the Data Changes at Each Stage

The house-price pipeline in `1-experimentation/` is a chain of 6 notebooks. Each stage reads the file the previous stage produced, and writes the next one. Nothing is transformed in place: the table you see at stage *n+1* is the table from stage *n* plus whatever that stage added, changed, or removed.

```
data/data_raw.csv
   └─▶ 1.1 Profiling + Split (reads data_raw.csv → writes data_raw_train.csv / data_raw_test.csv)
   └─▶ 1.2 Preprocessing     (reads data_raw_train.csv → writes data_preprocessed_train.csv)
   └─▶ 1.3 EDA               (reads data_preprocessed_train.csv, writes nothing)
   └─▶ 1.4 Feature Eng.      (reads data_preprocessed_train.csv → writes data_feature_train.csv + artifacts/preprocessing/)
   └─▶ 1.5 Model Training    (reads data_feature_train.csv → writes artifacts/models/ + artifacts/experiments/; CV-based selection)
   └─▶ 1.6 Evaluation        (reads data_raw_test.csv + artifacts/preprocessing/ + artifacts/models/ → held-out metrics)
```

Datasets live flat in `data/` (no subfolders). Everything that describes the trained model lives under `artifacts/`: `artifacts/preprocessing/data_features.json` (feature order) + `data_scaler.json` (scaler mean/std) are the **configuration** that 1.6 re-applies to the held-out test split, `artifacts/models/` holds the trained model, and `artifacts/experiments/` holds the run report.

All paths in this document refer to artifacts under `1-experimentation/` (e.g. `data/`, `artifacts/`). The notebooks live in `1-experimentation/notebooks/`; when running them, the notebook working directory must be `1-experimentation/notebooks/` so their `../data/...` / `../artifacts/...` paths resolve.

## Running Example

To make the changes concrete, every stage below shows the **same two houses**, taken from rows 0 and 1 of `data/data_raw.csv`:

| House | price | sqft | bedrooms | bathrooms | location | year_built | condition |
|-------|------:|-----:|---------:|----------:|----------|-----------:|-----------|
| **A** | $495,000 | 1,527 | 2 | 1.5 | Suburb | 1956 | Good |
| **B** | $752,000 | 2,526 | 3 | 2.5 | Downtown | 1998 | Excellent |

---

## Stage 1.1 — Raw Data Profiling (`1_1_raw_data_profiling.ipynb`)

**Objective:** understand the raw data before touching it.

**What it does:**
- Loads `data/data_raw.csv`
- Prints shape, data types, and memory usage
- Checks for missing values (count + percentage)
- Lists unique values per column and the categorical distributions (`location`, `condition`)
- **Splits the raw table** 80/20 (`random_state=42`) → 67 train / 17 test rows, saved as `data/data_raw_train.csv` and `data/data_raw_test.csv`. The test split is kept **raw** — no later stage touches it until 1.6.

**Output:** `data/data_raw_train.csv` (67 rows), `data/data_raw_test.csv` (17 rows).

**Example (the table is unchanged):**

| price | sqft | bedrooms | bathrooms | location | year_built | condition |
|------:|-----:|---------:|----------:|----------|-----------:|-----------|
| $495,000 | 1,527 | 2 | 1.5 | Suburb | 1956 | Good |
| $752,000 | 2,526 | 3 | 2.5 | Downtown | 1998 | Excellent |

---

## Stage 1.2 — Raw Data Preprocessing (`1_2_raw_data_preprocessing.ipynb`)

**Objective:** clean the data and make it consistent.

**What it does (in order):**
1. Drops rows with missing `price`
2. Imputes numerical columns with the median, categorical columns with the mode
3. Fixes data types (`price`, `bathrooms` → float; `sqft`, `bedrooms`, `year_built` → int)
4. Caps outliers in `price` and `sqft` using the IQR method (`clip` at Q1 − 1.5·IQR / Q3 + 1.5·IQR)
5. Creates the new feature **`house_age = 2026 − year_built`** (`CURRENT_YEAR` is hardcoded to 2026)
6. Runs validation asserts (no non-positive prices/sqft, no future `year_built`, valid `location`/`condition` values)

**Note:** only the **train split** (`data_raw_train.csv`) is preprocessed here. The held-out test split stays raw on disk and gets the exact same treatment in 1.6 as part of the config process.

**Output:** `data/data_preprocessed_train.csv` (67 rows, 8 columns).

**Example — the same two rows, now with `house_age` added** (neither of these rows was imputed or capped, so their values are unchanged):

| price | sqft | bedrooms | bathrooms | location | year_built | condition | house_age |
|------:|-----:|---------:|----------:|----------|-----------:|-----------|----------:|
| $495,000 | 1,527 | 2 | 1.5 | Suburb | 1956 | Good | **70** |
| $752,000 | 2,526 | 3 | 2.5 | Downtown | 1998 | Excellent | **28** |

---

## Stage 1.3 — EDA (`1_3_preprocessed_data_exploratory_data_analysis.ipynb`)

**Objective:** explore patterns before building features.

**What it does:**
- Loads `data/data_preprocessed_train.csv`
- Analyzes the target `price` (distribution, skewness 1.013) and motivates the **log transform** used later
- Plots distributions of `sqft`, `bedrooms`, `bathrooms`, `house_age`
- Analyzes categorical features (`location`, `condition`) and price by group
- Correlation heatmap + pairwise relationships
- Prints key insights (e.g., `sqft` is the strongest predictor of price)

**Output:** none — analysis only. **The table does not change**; it is still the preprocessed table.

---

## Stage 1.4 — Feature Engineering (`1_4_preprocessed_data_feature_engineer.ipynb`)

**Objective:** turn the preprocessed table into model-ready features.

**What it does (in order):**
1. **Encodes `location`** with one-hot encoding → 6 columns `loc_Downtown`, `loc_Mountain`, `loc_Rural`, `loc_Suburb`, `loc_Urban`, `loc_Waterfront`
2. **Encodes `condition`** ordinally → `condition_encoded` (`Poor`→0, `Fair`→1, `Good`→2, `Excellent`→3) and drops the original column
3. **Creates new features**:
   - `total_rooms = bedrooms + bathrooms`
   - `bath_bed_ratio = bathrooms / bedrooms`
   - `is_luxury = (condition_encoded ≥ 3) & (sqft ≥ 2500)` → 0/1
   - `is_new = (house_age ≤ 15)` → 0/1
   - `sqft_per_bedroom = sqft / bedrooms`
   - `log_price = log1p(price)` — the **actual training target**
4. **Standard-scales** 7 numerics with `StandardScaler`: `sqft`, `bedrooms`, `bathrooms`, `house_age`, `total_rooms`, `bath_bed_ratio`, `sqft_per_bedroom`. `year_built` stays raw.
5. Defines `FEATURE_COLS` (17 features) and saves:
   - `data/data_feature_train.csv`
   - `artifacts/preprocessing/data_features.json` (feature order — part of the config process)
   - `artifacts/preprocessing/data_scaler.json` (mean/std of the scaler — required by 1.6 and, in spirit, by the backend later)

The scaler is fitted on the **train split only**; 1.6 will re-use its mean/std on the test split without refitting.

**Example — the two rows after encoding, feature creation, and scaling** (scaled values are `(x − mean) / std`; this is the table the model will see):

| Feature | House A | House B | Notes |
|---------|--------:|--------:|-------|
| `sqft` (scaled) | −1.028 | 0.518 | (1527−2191.5)/646.1 · (2526−2191.5)/646.1 |
| `bedrooms` (scaled) | −1.012 | 0.169 | |
| `bathrooms` (scaled) | −0.839 | 0.376 | |
| `year_built` | 1956 | 1998 | **not scaled** |
| `house_age` (scaled) | 1.344 | −0.823 | |
| `loc_Downtown` | 0 | **1** | only the active location is 1 |
| `loc_Mountain` / `loc_Rural` / `loc_Urban` / `loc_Waterfront` | 0 | 0 | |
| `loc_Suburb` | **1** | 0 | |
| `condition_encoded` | 2 (Good) | 3 (Excellent) | |
| `total_rooms` (scaled) | −0.946 | 0.277 | 3.5 → 5.5 |
| `bath_bed_ratio` (scaled) | −0.064 | 0.564 | 0.75 → 0.83 |
| `is_luxury` | 0 | **1** | B: Excellent + ≥2500 sqft |
| `is_new` | 0 | 0 | neither is ≤15 years old |
| `sqft_per_bedroom` (scaled) | −0.113 | 0.855 | 763.5 → 842.0 |
| `price` | $495,000 | $752,000 | kept for reference |
| `log_price` | 13.112 | 13.530 | `log1p(price)` — the target |

---

## Stage 1.5 — Model Training (`1_5_featured_data_model_training.ipynb`)

**Objective:** train and select the best regression model.

**What it does:**
1. Loads `data/data_feature_train.csv` + `artifacts/preprocessing/data_features.json`; `X` = 17 features, `y` = `log_price`
2. **No internal train/test split** — the model trains on all 67 train rows; model selection uses **5-fold cross-validation** (the split already happened in 1.1, and the held-out test stays raw for 1.6)
3. Configures MLflow (`http://localhost:5555`, experiment `house_price_prediction` — **the Docker MLflow server must be running**)
4. Defines `evaluate_model`, which **inverts the log transform with `np.expm1`** so all metrics are in dollars, and reports train metrics + 5-fold CV R² per model
5. Trains and logs 7 models (Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, SVR) as individual MLflow runs
6. Ranks by CV R², cross-validates the top 3, then **GridSearchCV** on the best (Gradient Boosting → `learning_rate=0.1, max_depth=5, min_samples_split=5, n_estimators=300`)
7. Logs the tuned best model to MLflow and saves:
   - `artifacts/models/best_model.joblib`
   - `artifacts/experiments/experiment_results.json` (train + CV metrics; held-out test metrics are appended by 1.6)

**Example — where the two houses landed in the split** (made by 1.1 with `random_state=42`):

| House | Split | Note |
|-------|-------|------|
| **A** | **TEST** | used for final evaluation in 1.6 |
| **B** | **TRAIN** | the model saw it during fitting |

> The split is random; these two specific rows are shown only because they are the running example. 1.5 never changes the table here — it only reads it.

---

## Stage 1.6 — Model Evaluation (`1_6_featured_data_model_evaluation.ipynb`)

**Objective:** judge how well the tuned model generalizes.

**What it does (the config process):**
1. Loads the **raw** held-out test split `data/data_raw_test.csv` plus the saved configuration `artifacts/preprocessing/data_features.json`, `artifacts/preprocessing/data_scaler.json`, and `artifacts/models/best_model.joblib` + `artifacts/experiments/experiment_results.json`
2. Re-applies the exact training transforms to the test split in its own cells: preprocessing from 1.2 (dtypes, IQR capping, `house_age = 2026 − year_built`), encoding + derived features from 1.4 (6 location one-hots, `condition_encoded`, the 6 new features + `log_price`), then scales the 7 numerics with the **train-fitted** scaler (mean/std from `data_scaler.json` — no refit) and selects columns in `FEATURE_COLS` order
3. Predicts on the test set and **inverts the log** (`np.expm1`) → final held-out metrics on the 17 test houses (test R² = 0.904, RMSE ≈ $72,710, MAE ≈ $44,665, MAPE ≈ 6.66%)
4. Predicted-vs-actual scatter, residual analysis, residual histogram
5. Feature importance (Gradient Boosting exposes `feature_importances_`)
6. Prints an experiment summary and appends `test_metrics` to `artifacts/experiments/experiment_results.json`

**Example — predicted vs actual price** (real predictions from `best_model.joblib`):

| House | Split | Actual price | Predicted price | Residual (actual − predicted) |
|-------|-------|-------------:|----------------:|------------------------------:|
| **A** | TEST | $495,000 | $377,835 | +$117,165 |
| **B** | TRAIN | $752,000 | $752,001 | −$1 |

House **A** is a genuine held-out test prediction (about $117K off — typical error at this dataset size). House **B** is a training row shown for reference: the model reproduces it exactly, which is expected for data it has already seen, not a sign of generalization.

---

## Summary — How the Two Rows Evolve

| | Stage 1.1 / 1.2 (raw → preprocessed) | Stage 1.4 (featured) | Stage 1.1 (split) | Stage 1.6 (evaluated) |
|---|---|---|---|---|
| **House A** | `Suburb · 1956 · Good · house_age=70` | `loc_Suburb=1 · cond=2 · is_luxury=0 · is_new=0 · log_price=13.112` | TEST | predicted **$377,835** vs actual $495,000 |
| **House B** | `Downtown · 1998 · Excellent · house_age=28` | `loc_Downtown=1 · cond=3 · is_luxury=1 · is_new=0 · log_price=13.530` | TRAIN | predicted **$752,001** vs actual $752,000 |
