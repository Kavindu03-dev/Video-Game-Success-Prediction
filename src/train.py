import os
import re
import joblib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from preprocess import engineer_target, build_preprocessor, build_features

warnings.filterwarnings("ignore")

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH = os.environ.get(
    'VG_DATA_PATH',
    os.path.join(WORKSPACE_ROOT, 'data', 'vg_sales_2024.csv')
)
ARTIFACT_DIR = os.path.join(WORKSPACE_ROOT, 'models')

# Define success label based on total sales threshold
SUCCESS_THRESHOLD = 1.0  # million units


def load_data(path: str) -> pd.DataFrame:
    # Ensure path exists; suggest using VG_DATA_PATH to override
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            f"- Ensure the file exists (default expected: {os.path.join(WORKSPACE_ROOT, 'data', 'vg_sales_2024.csv')})\n"
            f"- Or set environment variable VG_DATA_PATH to the CSV path before running train.py"
        )
    df = pd.read_csv(path)
    return df


# preprocessing functions now live in preprocess.py


def get_models():
    models = {
        'log_regression': LogisticRegression(max_iter=200),
        'random_forest': RandomForestClassifier(n_estimators=300, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'svc': SVC(kernel='rbf', probability=True, random_state=42)
    }
    return models


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Model: {name}\n  Accuracy: {acc:.4f}\n  F1: {f1:.4f}\n")
    return acc, f1


def _print_preprocessing_summary(df_raw: pd.DataFrame, df_target: pd.DataFrame):
    print("\n=== Preprocessing Summary ===")
    print(f"Rows before: {len(df_raw):,}")
    print(f"Rows after dropping missing total_sales: {len(df_target):,}")
    dropped = len(df_raw) - len(df_target)
    if dropped:
        print(f"Dropped rows (no total_sales): {dropped:,}")

    # Build feature view for summary (reflects normalization/bucketing)
    df_feat = build_features(df_target)
    cat_cols = ['console', 'genre', 'publisher', 'developer']
    num_cols = ['critic_score', 'release_year']

    # Class distribution
    vc = df_target['success'].value_counts(dropna=False)
    print("Class distribution (success=1):", {int(k): int(v) for k, v in vc.items()})

    # Missing counts
    miss = df_feat[cat_cols + num_cols].isna().sum().to_dict()
    print("Missing per feature:", {k: int(v) for k, v in miss.items()})

    # Cardinality of categoricals
    card = {c: int(df_feat[c].nunique(dropna=True)) for c in cat_cols if c in df_feat}
    print("Unique categories:", card)

    # Numeric ranges
    num_stats = {}
    for c in num_cols:
        if c in df_feat:
            s = df_feat[c]
            num_stats[c] = {
                'min': float(s.min(skipna=True)) if len(s) else None,
                'max': float(s.max(skipna=True)) if len(s) else None,
                'median': float(s.median(skipna=True)) if len(s) else None,
            }
    print("Numeric stats:", num_stats)
    print("============================\n")


def train_and_select_best(df: pd.DataFrame):
    df_raw = df.copy()
    df = engineer_target(df)

    # Drop obvious leakage columns (also handled inside build_features)
    df = df.drop(columns=['img', 'title'], errors='ignore')

    _print_preprocessing_summary(df_raw, df)

    preprocessor, X, y = build_preprocessor(df)

    # Sanity check for class balance (post feature selection)
    vc = y.value_counts(dropna=False)
    if len(vc) < 2 or int(vc.min()) < 10:
        print("Warning: Highly imbalanced or single-class target detected. Results may be unreliable.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = get_models()
    results = []
    fitted_models = {}

    for name, clf in models.items():
        pipe = Pipeline(steps=[('prep', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)
        acc, f1 = evaluate_model(name, pipe, X_test, y_test)
        results.append((name, acc, f1, pipe))
        fitted_models[name] = pipe

    # Choose best by F1 first, then accuracy
    results.sort(key=lambda x: (x[2], x[1]), reverse=True)
    best_name, best_acc, best_f1, best_model = results[0]

    print("Best model:", best_name)
    print(f"Accuracy: {best_acc:.4f}, F1: {best_f1:.4f}")

    # Persist
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(ARTIFACT_DIR, 'best_model.joblib'))
    meta = {
        'selected_model': best_name,
        'accuracy': best_acc,
        'f1': best_f1,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'success_threshold_million_units': SUCCESS_THRESHOLD
    }
    pd.Series(meta).to_json(os.path.join(ARTIFACT_DIR, 'metrics.json'))

    return best_name, best_acc, best_f1


def main():
    print("Workspace root:", WORKSPACE_ROOT)
    print("Loading data from:", DATA_PATH)
    df = load_data(DATA_PATH)
    print("Rows:", len(df))
    best_name, best_acc, best_f1 = train_and_select_best(df)


if __name__ == '__main__':
    main()
