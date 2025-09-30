from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	f1_score,
	precision_score,
	recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.preprocessing import clean_dataset, add_hit_label


def load_data(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	df = clean_dataset(df)  # converts release_date -> year in same column if present
	# Standardize a 'release_year' feature
	if 'release_year' not in df.columns:
		if 'release_date' in df.columns:
			df['release_year'] = df['release_date']
		else:
			df['release_year'] = pd.NA

	# Create target label
	df = add_hit_label(df, sales_col='total_sales', threshold=1.0, label_col='Hit')
	return df


def build_pipeline(categorical: list[str], numeric: list[str]) -> Pipeline:
	pre = ColumnTransformer(
		transformers=[
			(
				'cat',
				OneHotEncoder(handle_unknown='ignore', sparse_output=False),
				categorical,
			),
			# numeric passthrough (RF doesn't require scaling)
			('num', 'passthrough', numeric),
		],
		remainder='drop',
		verbose_feature_names_out=False,
	)

	clf = RandomForestClassifier(
		n_estimators=300,
		random_state=42,
		n_jobs=-1,
	)

	pipe = Pipeline(steps=[('pre', pre), ('clf', clf)])
	return pipe


def train_and_evaluate(df: pd.DataFrame) -> Tuple[Pipeline, dict]:
	feature_cols_cat = ['genre', 'platform', 'publisher']
	feature_cols_num = ['critic_score', 'release_year']

	# Ensure these columns exist
	for col in feature_cols_cat + feature_cols_num + ['Hit']:
		if col not in df.columns:
			df[col] = pd.NA

	X = df[feature_cols_cat + feature_cols_num].copy()
	y = df['Hit'].astype('Int64').fillna(0).astype(int)

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)

	# Build pipeline and fit
	pipe = build_pipeline(feature_cols_cat, feature_cols_num)
	pipe.fit(X_train, y_train)

	# Evaluate RF
	y_pred = pipe.predict(X_test)
	metrics = {
		'accuracy': accuracy_score(y_test, y_pred),
		'precision': precision_score(y_test, y_pred, zero_division=0),
		'recall': recall_score(y_test, y_pred, zero_division=0),
		'f1': f1_score(y_test, y_pred, zero_division=0),
		'report': classification_report(y_test, y_pred, digits=3),
	}

	# Optional cross-validation summary
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1'}
	cv_res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
	metrics['cv_mean'] = {k: float(cv_res[f'test_{k}'].mean()) for k in scoring}
	metrics['cv_std'] = {k: float(cv_res[f'test_{k}'].std()) for k in scoring}

	return pipe, metrics


def save_model(model: Pipeline, path: Path) -> None:
	with open(path, 'wb') as f:
		pickle.dump(model, f)


def main() -> None:
	project_root = Path(__file__).resolve().parents[1]
	# Prefer data/vg_sales_2024.csv, fallback to data/raw/vg_sales_2024.csv
	data_path = project_root / 'data' / 'vg_sales_2024.csv'
	if not data_path.exists():
		data_path = project_root / 'data' / 'raw' / 'vg_sales_2024.csv'
	model_path = project_root / 'model.pkl'

	if not data_path.exists():
		raise FileNotFoundError(f"Dataset not found at {data_path}. Please place vg_sales_2024.csv there.")

	print("Loading and cleaning data...")
	df = load_data(data_path)
	print(f"Samples: {len(df)}, Hit rate: {df['Hit'].mean():.3f}")

	print("Training model...")
	model, metrics = train_and_evaluate(df)
	print("\nEvaluation (holdout):")
	print(metrics['report'])
	print("CV means:", metrics['cv_mean'])
	print("CV stds:", metrics['cv_std'])

	print(f"Saving model to {model_path} ...")
	save_model(model, model_path)
	print("Done.")


if __name__ == "__main__":
	main()

