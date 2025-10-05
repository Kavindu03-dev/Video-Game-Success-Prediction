# Video Game Success Prediction

A Fundamentals of Data Mining project: predict whether a video game will be a commercial success (≥ 1M units) using metadata from `vg_sales_2024.csv` and evaluate multiple models.

## Target definition
- success = 1 if `total_sales` ≥ 1.0 (million units), else 0.
- Rationale: aligns with a tangible real-world goal — forecasting if a new title can reach 1M units.

## Pipeline overview
1. Load CSV
2. Engineer target and features
   - Numeric: `critic_score`, `release_year`
   - Categorical: `console`, `genre`, `publisher`, `developer`
3. Preprocessing
   - Impute missing values (median/most_frequent)
   - Scale numeric features; one-hot encode categoricals
4. Train/test split (80/20, stratified)
5. Models (4)
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - SVC (RBF)
6. Metrics: Accuracy and F1; choose best by F1 then accuracy
7. Persist best model (joblib)

## Setup (Windows PowerShell)

```powershell
# Create venv (optional)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Train (both classification + regression with 5-fold CV)
python .\src\train.py --task both --cv-folds 5

# Quick train without cross-validation (faster)
python .\src\train.py --task both --no-cv

# Only classification 
python .\src\train.py --task classification

# Run app (Streamlit)
streamlit run .\src\app.py
```

## Files
- `data/vg_sales_2024.csv` — dataset
- `src/train.py` — unified training (classification + regression, optional cross-validation)
- `src/app.py` — Streamlit prediction app
- `models/` — saved models + metrics:
   - `best_model.joblib`, `metrics.json` (classification)
   - `best_regressor.joblib`, `regressor_metrics.json` (regression)

## Notes
- Feel free to adjust `SUCCESS_THRESHOLD` in `src/train.py` if your rubric defines success differently.
- If class imbalance is high, consider using class_weight='balanced' for some models or tune thresholds.
- Extend features (e.g., region dummy variables, franchise detection) for better accuracy.
