import os
import joblib
import warnings
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC
from preprocess import (
    engineer_target,
    build_preprocessor,
    build_features,
    build_preprocessor_regression,
)

warnings.filterwarnings("ignore")

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH = os.environ.get(
    'VG_DATA_PATH',
    os.path.join(WORKSPACE_ROOT, 'data', 'vg_sales_2024.csv')
)
ARTIFACT_DIR = os.path.join(WORKSPACE_ROOT, 'models')

# Define success label based on total sales threshold (also defined in preprocess for target engineering)
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


def get_classification_models():
    return {
        'log_regression': LogisticRegression(max_iter=200),
        'random_forest': RandomForestClassifier(n_estimators=300, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'svc': SVC(kernel='rbf', probability=True, random_state=42),
    }


def get_regression_models():
    models = {
        'random_forest_reg': RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42),
        'gradient_boosting_reg': GradientBoostingRegressor(n_estimators=200, max_depth=7, random_state=42),
    }
    # Optional XGBoost (ignore if unavailable)
    try:
        from xgboost import XGBRegressor  # type: ignore
        models['xgboost_reg'] = XGBRegressor(
            n_estimators=200, max_depth=7, learning_rate=0.1, random_state=42
        )
    except Exception:
        pass
    return models


def evaluate_classification_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Model: {name}\n  Test Accuracy: {acc:.4f}\n  Test F1: {f1:.4f}\n")
    return acc, f1


def evaluate_regression_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Model: {name}\n  Test MAE: {mae:.4f}\n  Test RMSE: {rmse:.4f}\n  Test R2: {r2:.4f}\n")
    return mae, rmse, r2


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


def train_classification(df: pd.DataFrame, cv_folds: int = 5, use_cv: bool = True):
    """Train classification models to predict success (>= threshold) with optional cross-validation."""
    print("\n===== CLASSIFICATION (Success Prediction) =====")
    df_raw = df.copy()
    df = engineer_target(df)
    df = df.drop(columns=['img', 'title'], errors='ignore')
    _print_preprocessing_summary(df_raw, df)

    preprocessor, X, y = build_preprocessor(df)

    # Split first, then perform CV on training partition only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = get_classification_models()
    results = []  # (name, cv_mean_f1, cv_mean_acc, test_acc, test_f1, pipeline)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if use_cv else None

    for name, clf in models.items():
        pipe = Pipeline(steps=[('prep', preprocessor), ('clf', clf)])
        if use_cv:
            print(f"CV evaluating {name} (f1, accuracy)...")
            f1_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1')
            acc_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
            cv_mean_f1, cv_mean_acc = f1_scores.mean(), acc_scores.mean()
            print(f"  CV Mean F1: {cv_mean_f1:.4f} | CV Mean Acc: {cv_mean_acc:.4f}")
        else:
            cv_mean_f1 = cv_mean_acc = float('nan')

        # Fit on full training split and evaluate holdout
        pipe.fit(X_train, y_train)
        test_acc, test_f1 = evaluate_classification_model(name, pipe, X_test, y_test)
        results.append((name, cv_mean_f1, cv_mean_acc, test_acc, test_f1, pipe))

    # Select best by CV F1 (if available) else test F1, tie-breaker accuracy
    if use_cv:
        results.sort(key=lambda x: (x[1], x[2]), reverse=True)
    else:
        results.sort(key=lambda x: (x[4], x[3]), reverse=True)
    best = results[0]
    best_name, cv_mean_f1, cv_mean_acc, test_acc, test_f1, best_model = best

    print("Best classification model:", best_name)
    print(f"  Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")
    if use_cv:
        print(f"  CV Mean F1: {cv_mean_f1:.4f} | CV Mean Acc: {cv_mean_acc:.4f}")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(ARTIFACT_DIR, 'best_model.joblib'))
    metrics = {
        'selected_model': best_name,
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'cv_mean_f1': float(cv_mean_f1) if use_cv else None,
        'cv_mean_accuracy': float(cv_mean_acc) if use_cv else None,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'task': 'classification',
        'success_threshold_million_units': SUCCESS_THRESHOLD,
    }
    pd.Series(metrics).to_json(os.path.join(ARTIFACT_DIR, 'metrics.json'))
    return metrics


def train_regression(df: pd.DataFrame, cv_folds: int = 5, use_cv: bool = True):
    """Train regression models to predict total_sales directly with optional cross-validation."""
    print("\n===== REGRESSION (Total Sales Prediction) =====")
    df_raw = df.copy()
    # Coerce numeric columns of interest
    for col in ['total_sales', 'critic_score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['total_sales'])
    df = df.drop(columns=['img', 'title'], errors='ignore')

    preprocessor, X, y = build_preprocessor_regression(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_regression_models()
    results = []  # (name, cv_mean_r2, cv_mean_mae, cv_mean_rmse, test_mae, test_rmse, test_r2, pipe)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42) if use_cv else None

    for name, reg in models.items():
        pipe = Pipeline(steps=[('prep', preprocessor), ('reg', reg)])
        if use_cv:
            print(f"CV evaluating {name} (r2, mae, rmse)...")
            r2_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='r2')
            mae_scores = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
            rmse_scores = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
            cv_mean_r2 = r2_scores.mean()
            cv_mean_mae = mae_scores.mean()
            cv_mean_rmse = rmse_scores.mean()
            print(f"  CV Mean R2: {cv_mean_r2:.4f} | CV Mean MAE: {cv_mean_mae:.4f} | CV Mean RMSE: {cv_mean_rmse:.4f}")
        else:
            cv_mean_r2 = cv_mean_mae = cv_mean_rmse = float('nan')

        pipe.fit(X_train, y_train)
        test_mae, test_rmse, test_r2 = evaluate_regression_model(name, pipe, X_test, y_test)
        results.append((name, cv_mean_r2, cv_mean_mae, cv_mean_rmse, test_mae, test_rmse, test_r2, pipe))

    # Select best by CV R2 if available else test R2
    if use_cv:
        results.sort(key=lambda x: x[1], reverse=True)
    else:
        results.sort(key=lambda x: x[6], reverse=True)
    best = results[0]
    best_name, cv_mean_r2, cv_mean_mae, cv_mean_rmse, test_mae, test_rmse, test_r2, best_model = best

    print("Best regression model:", best_name)
    print(f"  Test MAE: {test_mae:.4f} | Test RMSE: {test_rmse:.4f} | Test R2: {test_r2:.4f}")
    if use_cv:
        print(f"  CV Mean R2: {cv_mean_r2:.4f} | CV Mean MAE: {cv_mean_mae:.4f} | CV Mean RMSE: {cv_mean_rmse:.4f}")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(ARTIFACT_DIR, 'best_regressor.joblib'))
    metrics = {
        'selected_model': best_name,
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2),
        'cv_mean_r2': float(cv_mean_r2) if use_cv else None,
        'cv_mean_mae': float(cv_mean_mae) if use_cv else None,
        'cv_mean_rmse': float(cv_mean_rmse) if use_cv else None,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'task': 'regression',
        'target': 'total_sales',
    }
    pd.Series(metrics).to_json(os.path.join(ARTIFACT_DIR, 'regressor_metrics.json'))
    return metrics
def parse_args():
    p = argparse.ArgumentParser(description="Train video game success (classification) and total sales (regression) models.")
    p.add_argument('--task', choices=['classification', 'regression', 'both'], default='both', help='Which task(s) to train.')
    p.add_argument('--cv-folds', type=int, default=5, help='Number of cross-validation folds (default: 5).')
    p.add_argument('--no-cv', action='store_true', help='Disable cross-validation (quick run).')
    return p.parse_args()


def main(task: str = 'both', cv_folds: int = 5, use_cv: bool = True):
    print("Workspace root:", WORKSPACE_ROOT)
    print("Loading data from:", DATA_PATH)
    df = load_data(DATA_PATH)
    print("Rows:", len(df))

    results = {}
    if task in ('classification', 'both'):
        results['classification'] = train_classification(df, cv_folds=cv_folds, use_cv=use_cv)
    if task in ('regression', 'both'):
        results['regression'] = train_regression(df, cv_folds=cv_folds, use_cv=use_cv)
    print("\nTraining complete. Artifacts saved in 'models/'.")
    return results


if __name__ == '__main__':
    args = parse_args()
    main(task=args.task, cv_folds=args.cv_folds, use_cv=not args.no_cv)
