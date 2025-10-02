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
	roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
	from xgboost import XGBClassifier  # type: ignore
	HAS_XGB = True
except Exception:
	XGBClassifier = None  # type: ignore
	HAS_XGB = False

try:
	from src.preprocessing import clean_dataset, add_hit_label
except Exception:
	# Fallback if running from src/ directly
	from preprocessing import clean_dataset, add_hit_label  # type: ignore


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


def build_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
	"""Build a shared preprocessor for all models."""
	return ColumnTransformer(
		transformers=[
			(
				'cat',
				OneHotEncoder(handle_unknown='ignore', sparse_output=False),
				categorical,
			),
			('num', 'passthrough', numeric),
		],
		remainder='drop',
		verbose_feature_names_out=False,
	)


def _build_model_candidates() -> dict:
	"""Return a dict of candidate estimators to try."""
	models = {
		'RandomForest': RandomForestClassifier(
			n_estimators=300,
			random_state=42,
			n_jobs=-1,
		),
		'LogisticRegression': LogisticRegression(
			solver='lbfgs',
			max_iter=1000,
			n_jobs=-1,
		),
	}
	if HAS_XGB and XGBClassifier is not None:
		models['XGBoost'] = XGBClassifier(
			n_estimators=300,
			learning_rate=0.1,
			max_depth=6,
			subsample=0.8,
			colsample_bytree=0.8,
			random_state=42,
			tree_method='hist',
			eval_metric='logloss',
			n_jobs=-1,
		)
	return models


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

	pre = build_preprocessor(feature_cols_cat, feature_cols_num)

	candidates = _build_model_candidates()
	results: dict[str, dict] = {}
	best_name = None
	best_score = (-1.0, -1.0)  # (f1, roc_auc)
	best_pipe: Pipeline | None = None

	for name, est in candidates.items():
		pipe = Pipeline(steps=[('pre', pre), ('clf', est)])
		pipe.fit(X_train, y_train)

		# Predict and score
		y_pred = pipe.predict(X_test)
		# Probabilities for AUC if available
		try:
			y_scores = pipe.predict_proba(X_test)[:, 1]
		except Exception:
			try:
				# Fallback to decision function
				y_scores = pipe.decision_function(X_test)
			except Exception:
				y_scores = None

		f1 = f1_score(y_test, y_pred, zero_division=0)
		roc = roc_auc_score(y_test, y_scores) if y_scores is not None else float('nan')
		model_metrics = {
			'accuracy': accuracy_score(y_test, y_pred),
			'precision': precision_score(y_test, y_pred, zero_division=0),
			'recall': recall_score(y_test, y_pred, zero_division=0),
			'f1': f1,
			'roc_auc': roc,
			'report': classification_report(y_test, y_pred, digits=3),
		}

		# Cross-validation
		cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
		scoring = {
			'accuracy': 'accuracy',
			'precision': 'precision',
			'recall': 'recall',
			'f1': 'f1',
			'roc_auc': 'roc_auc',
		}
		try:
			cv_res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
			model_metrics['cv_mean'] = {k: float(cv_res[f'test_{k}'].mean()) for k in scoring}
			model_metrics['cv_std'] = {k: float(cv_res[f'test_{k}'].std()) for k in scoring}
		except Exception as e:
			# If some scorers fail (e.g., roc_auc due to no proba), degrade gracefully
			cv_scoring = {k: v for k, v in scoring.items() if k != 'roc_auc'}
			cv_res = cross_validate(pipe, X, y, cv=cv, scoring=cv_scoring, n_jobs=-1, return_train_score=False)
			model_metrics['cv_mean'] = {k: float(cv_res[f'test_{k}'].mean()) for k in cv_scoring}
			model_metrics['cv_std'] = {k: float(cv_res[f'test_{k}'].std()) for k in cv_scoring}

		results[name] = model_metrics

		# Select best by (f1, roc_auc)
		comp_roc = roc if not pd.isna(roc) else -1.0
		key = (f1, comp_roc)
		if key > best_score:
			best_score = key
			best_name = name
			best_pipe = pipe

	assert best_pipe is not None and best_name is not None
	# Return best pipeline and its metrics (plus a summary table)
	best_metrics = results[best_name]
	best_metrics['all_models'] = results
	best_metrics['best_model'] = best_name
	return best_pipe, best_metrics


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

	print("Training candidate models (RandomForest, LogisticRegression" + (", XGBoost" if HAS_XGB else "") + ")...")
	model, metrics = train_and_evaluate(df)
	print("\nBest model:", metrics.get('best_model'))
	print("Holdout report (best):")
	print(metrics['report'])
	if 'roc_auc' in metrics and not pd.isna(metrics['roc_auc']):
		print(f"ROC AUC: {metrics['roc_auc']:.3f}")
	if 'cv_mean' in metrics:
		print("CV means:", metrics['cv_mean'])
	if 'cv_std' in metrics:
		print("CV stds:", metrics['cv_std'])

	# Print compact summary of all models
	all_models = metrics.get('all_models', {})
	if all_models:
		print("\nModel comparison (holdout f1 / roc_auc):")
		for name, m in all_models.items():
			f1 = m.get('f1', float('nan'))
			roc = m.get('roc_auc', float('nan'))
			try:
				print(f" - {name:18s} f1={f1:.3f}  roc_auc={(roc if not pd.isna(roc) else float('nan')):.3f}")
			except Exception:
				print(f" - {name:18s} f1={f1}  roc_auc={roc}")

	print(f"Saving model to {model_path} ...")
	save_model(model, model_path)
	print("Done.")


if __name__ == "__main__":
	main()
