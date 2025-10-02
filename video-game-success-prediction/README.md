# Video Game Success Prediction

End-to-end ML project to predict whether a video game will be a Hit (total_sales ≥ 1.0) using tabular features and to explore sales trends via a Streamlit app.

## Project structure
- data/
  - raw/: original dataset from Kaggle
  - processed/: cleaned/engineered datasets
- notebooks/
  - 01_exploration.ipynb — dataset understanding + EDA
  - 03_modeling.ipynb — training + evaluation + feature importance + SHAP
  - (Preprocessing and explainability are implemented in src/ and the app.)
- src/
  - preprocessing.py — cleaning, labeling, encoding helpers
  - train_model.py — trains and saves a RandomForest pipeline (model.pkl)
  - evaluate.py — evaluation helpers (placeholder)
  - utils.py — shared utilities (placeholder)
- app/
  - app.py — Streamlit application for prediction + visualizations
- docs/
  - report.md — notes for your final report
- tests/
  - test_preprocessing.py — unit tests for preprocessing
- requirements.txt — pinned dependencies
- model.pkl — trained model pipeline (created after training)

## Dataset
Place `vg_sales_2024.csv` in `data/raw/` with at least the following columns:
- total_sales (float)
- genre (str), platform (str), publisher (str)
- critic_score (numeric)
- release_date (date-like, e.g., 2015-10-27) or release_year (int)
- Optional: na_sales, eu_sales, jp_sales, other_sales for region visuals

Notes:
- `src/preprocessing.clean_dataset` converts `release_date` to year (nullable Int64) in the same column. The training script additionally ensures a `release_year` column exists for modeling.

## Setup (Windows PowerShell)
```powershell
# From project root
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you don’t have Python 3.11 installed, install it (e.g., via Microsoft Store or `winget install Python.Python.3.11`). Using 3.11/3.12 avoids building heavy packages from source.

## Train a model (produces model.pkl)
```powershell
# Assumes data/raw/vg_sales_2024.csv exists
python -m src.train_model
```
This script:
- Cleans the dataset and adds the binary `Hit` label (total_sales ≥ 1.0)
- Builds a Pipeline: OneHotEncoder for categoricals + RandomForestClassifier
- Uses a stratified train/test split
- Prints metrics and 5-fold CV summary
- Saves the pipeline to `model.pkl`

## Run the Streamlit app
```powershell
# After training
streamlit run app\app.py
```
App features:
- Sidebar prediction form: genre, platform, publisher, critic_score, release_year
- Predicts Hit vs Not Hit and shows P(Hit)
- Visualizations: sales by genre, platform, and region

## Notebooks
Open and run:
1. `01_exploration.ipynb` — load, inspect, and visualize top-10 categories and correlations
2. `03_modeling.ipynb` — modeling experiments (overlaps with src/train_model.ipynb)

Note: Preprocessing is implemented in `src/preprocessing.ipynb` and used by training/app. Explainability visuals are provided in the Streamlit app (Insights). You may also add deeper explainability to `03_modeling.ipynb` if needed.

## Preprocessing functions (src/preprocessing.py)
- `clean_dataset(df, date_col='release_date', fill_unknown='Unknown', bool_fill=False, drop_all_nan_rows=True)`
  - Drops duplicates, optional all-NaN rows
  - Strips whitespace in object columns
  - Converts `release_date` to `year` (nullable Int64) in the same column
  - Fills NaNs: numeric→median, categorical→Unknown, boolean→False
- `add_hit_label(df, sales_col='total_sales', threshold=1.0, label_col='Hit', dtype='Int8')`
  - Adds a binary `Hit` column (1 if total_sales ≥ threshold else 0; NaN treated as 0)
- `encode_categoricals(df, columns=("genre","platform","publisher"), drop_first=False, dummy_na=False)`
  - One-hot encodes selected categorical columns using `pandas.get_dummies`

## Testing
Install pytest and run tests:
```powershell
pip install pytest
pytest -q
```

## Troubleshooting
- Pandas build errors on Windows usually mean the Python version is too new or a compiler is missing. Prefer Python 3.11/3.12 so wheels are used.
- If the Streamlit app can’t find `model.pkl`, run the training step first.
- If your CSV column names differ, update `src/train_model.py` and the Streamlit inputs accordingly.
