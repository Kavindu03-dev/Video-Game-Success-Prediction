from __future__ import annotations

"""
Evaluation script for the trained model.
This file was converted from evaluate.ipynb.

You can extend this with functions to load the saved model and compute
holdout or cross-validation metrics, or generate plots.
"""

from pathlib import Path
import pickle
import pandas as pd

try:
	from src.preprocessing import clean_dataset, add_hit_label
except Exception:
	# Fallback if run from within src/ directly
	from preprocessing import clean_dataset, add_hit_label  # type: ignore


def load_model(model_path: Path):
	with open(model_path, "rb") as f:
		return pickle.load(f)


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	df = clean_dataset(df)
	if 'release_year' not in df.columns and 'release_date' in df.columns:
		df['release_year'] = df['release_date']
	df = add_hit_label(df, sales_col='total_sales', threshold=1.0, label_col='Hit')
	return df


def main() -> None:
	project_root = Path(__file__).resolve().parents[1]
	data_path = project_root / 'data' / 'vg_sales_2024.csv'
	if not data_path.exists():
		data_path = project_root / 'data' / 'raw' / 'vg_sales_2024.csv'
	model_path = project_root / 'model.pkl'

	if not data_path.exists():
		raise FileNotFoundError(f"Dataset not found at {data_path}.")
	if not model_path.exists():
		raise FileNotFoundError(f"Model not found at {model_path}. Train it first.")

	print("Loading data and model...")
	df = load_and_prepare(data_path)
	model = load_model(model_path)

	# Simple evaluation on full data (not ideal; replace with proper split)
	features = ['genre', 'platform', 'publisher', 'critic_score', 'release_year']
	missing = [c for c in features if c not in df.columns]
	for c in missing:
		df[c] = pd.NA
	X = df[features]
	preds = model.predict_proba(X)[:, 1]
	print("Sample probabilities:", preds[:10])
	if 'Hit' in df.columns:
		from sklearn.metrics import roc_auc_score
		try:
			auc = roc_auc_score(df['Hit'].astype(int), preds)
			print(f"AUC (on full data): {auc:.3f}")
		except Exception:
			pass


if __name__ == "__main__":
	main()
