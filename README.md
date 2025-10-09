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

## Large model files (Git LFS)
The classification/regression artifacts (e.g., `best_model.joblib`) can exceed GitHub's 100 MB limit. This repo is configured to use **Git LFS** for any `*.joblib` files via the `.gitattributes` file.

### One-time setup (per machine)
```powershell
# Install Git LFS (choose one method)
winget install Git.GitLFS
# or: choco install git-lfs

git lfs install
```

### Adding / updating a model artifact
After training creates/updates a `*.joblib` file:
```powershell
# Ensure LFS is tracking (already in .gitattributes, but safe to confirm)
git lfs track "*.joblib"

# Stage LFS attributes if first time
git add .gitattributes

# Re-stage the model if it was previously added without LFS
git rm --cached models/best_model.joblib 2>$null

# Add model + other changes
git add models/best_model.joblib models/best_regressor.joblib models/*.json
git commit -m "Update trained models"
git push origin main
```

### Verifying LFS handling
```powershell
git lfs ls-files
```
You should see the Joblib artifacts listed. Collaborators must also have Git LFS installed before cloning/pulling.

### Alternative: skip committing large binaries
If you prefer not to store models, add this to `.gitignore` and use a release asset / DVC / cloud storage:
```
models/*.joblib
```
Then provide a script or instructions to regenerate the model (`python .\src\train.py`).
