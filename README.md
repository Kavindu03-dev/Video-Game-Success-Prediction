# Video Game Success Prediction

End-to-end ML project to predict whether a video game will be a Hit (total_sales ≥ 1.0) and explore sales trends via a Streamlit app.

This repository contains the working project under `video-game-success-prediction/`.

## Project structure

```
video-game-success-prediction/
	app/                # Streamlit app (app.py)
	data/               # Place dataset here
		raw/              # Original CSV(s)
		processed/        # Any cleaned/edited exports
		vg_sales_2024.csv # Preferred location (or put in data/raw/)
	docs/
	notebooks/          # 01..04 notebooks for EDA, preprocessing, modeling
	src/                # Training & preprocessing code
		preprocessing.py
		train_model.py
		evaluate.py
		utils.py
	tests/
	requirements.txt    # Pinned deps
	run_project.bat     # Helper launcher (path may need update, see notes)
```

## Dataset

Place `vg_sales_2024.csv` in `video-game-success-prediction/data/` (preferred) or `video-game-success-prediction/data/raw/`.

Required/expected columns (case-insensitive):
- total_sales (float)
- genre (str), platform (str), publisher (str)
- critic_score (numeric)
- release_date (date-like, e.g., 2015-10-27) or release_year (int)
- Optional: na_sales, eu_sales, jp_sales, other_sales (used for region visuals)

Notes:
- `src/preprocessing.clean_dataset` converts `release_date` to a year in the same column and imputes missing values.
- `src/train_model.py` ensures a `release_year` column exists for modeling.

## Quickstart (Windows PowerShell)

Open PowerShell in the repository root, then:

```powershell
cd "video-game-success-prediction"

# 1) Create & activate a virtual environment
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1

# 2) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Train the model (produces model.pkl at project root)
python -m src.train_model

# 4) Run the Streamlit app (default port 8501, or choose one)
streamlit run app\app.py --server.port 8504
```

Then open the URL shown (e.g., http://localhost:8504).

## What you get

- Trained pipeline saved as `model.pkl` (OneHotEncoder + RandomForestClassifier).
- Streamlit app to:
	- Predict Hit vs Not Hit from sidebar inputs (genre, platform, publisher, critic_score, release_year)
	- Explore sales by genre/platform/publisher and by regions
	- Build a small dashboard of charts interactively

## Notebooks

In `video-game-success-prediction/notebooks/`:
1. `01_exploration.ipynb` — EDA and dataset understanding
2. `02_preprocessing.ipynb` — cleaning & feature engineering
3. `03_modeling.ipynb` — model training, evaluation, importances, SHAP
4. `04_explainability.ipynb` — optional extra explainability

## Testing

Run unit tests (optional):

```powershell
cd "video-game-success-prediction"
pip install pytest
pytest -q
```

## Troubleshooting

- If the app complains that `model.pkl` is missing, run the training step first: `python -m src.train_model`.
- If cross-validation in training raises memory warnings, it’s due to high-cardinality one-hot features on a large dataset.
	- You can temporarily comment out the `cross_validate` block in `src/train_model.py` or reduce one-hot cardinality using `OneHotEncoder` parameters such as `min_frequency` or `max_categories`.
- If `run_project.bat` fails to `cd` into a wrong path, it has a hard-coded path from a different machine. Launch the app using the commands above instead, or update that file to use the current repository path.
- Prefer Python 3.11+ on Windows so wheels are available for scientific packages.

## License

This project is for educational purposes. Please ensure you comply with the dataset’s license and terms of use.
