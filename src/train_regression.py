import os
import joblib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from preprocess import build_features, build_preprocessor_regression

warnings.filterwarnings("ignore")

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH = os.environ.get(
    'VG_DATA_PATH',
    os.path.join(WORKSPACE_ROOT, 'data', 'vg_sales_2024.csv')
)
ARTIFACT_DIR = os.path.join(WORKSPACE_ROOT, 'models')


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            f"- Ensure the file exists (default expected: {os.path.join(WORKSPACE_ROOT, 'data', 'vg_sales_2024.csv')})\n"
            f"- Or set environment variable VG_DATA_PATH to the CSV path"
        )
    df = pd.read_csv(path)
    return df


def get_regression_models():
    """Define regression models for total_sales prediction."""
    models = {
        'random_forest_reg': RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42),
        'gradient_boosting_reg': GradientBoostingRegressor(n_estimators=200, max_depth=7, random_state=42)
    }
    
    # Try to include XGBoost if available
    try:
        from xgboost import XGBRegressor
        models['xgboost_reg'] = XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.1, random_state=42)
    except ImportError:
        print("XGBoost not available, skipping XGBRegressor")
    
    return models


def evaluate_regression_model(name, model, X_test, y_test):
    """Evaluate regression model with standard metrics."""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel: {name}")
    print(f"  MAE (Mean Absolute Error): {mae:.4f} million units")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f} million units")
    print(f"  R² Score: {r2:.4f}")
    
    return mae, rmse, r2


def _print_preprocessing_summary(df_raw: pd.DataFrame, df_processed: pd.DataFrame):
    print("\n=== Preprocessing Summary (Regression) ===")
    print(f"Rows before: {len(df_raw):,}")
    print(f"Rows after dropping missing total_sales: {len(df_processed):,}")
    dropped = len(df_raw) - len(df_processed)
    if dropped:
        print(f"Dropped rows (no total_sales): {dropped:,}")

    # Build feature view for summary
    df_feat = build_features(df_processed)
    cat_cols = ['console', 'genre', 'publisher', 'developer']
    num_cols = ['critic_score', 'release_year']

    # Target distribution
    if 'total_sales' in df_processed.columns:
        sales = df_processed['total_sales']
        print(f"Total Sales stats:")
        print(f"  Min: {sales.min():.4f}, Max: {sales.max():.4f}")
        print(f"  Mean: {sales.mean():.4f}, Median: {sales.median():.4f}")
        print(f"  Std: {sales.std():.4f}")

    # Missing counts
    miss = df_feat[cat_cols + num_cols].isna().sum().to_dict()
    print("Missing per feature:", {k: int(v) for k, v in miss.items()})

    # Cardinality of categoricals
    card = {c: int(df_feat[c].nunique(dropna=True)) for c in cat_cols if c in df_feat}
    print("Unique categories:", card)
    print("==========================================\n")


def train_and_select_best_regressor(df: pd.DataFrame):
    """Train multiple regression models and select the best based on R² score."""
    df_raw = df.copy()
    
    # Coerce numeric columns
    for col in ['total_sales', 'critic_score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing total_sales
    df = df.dropna(subset=['total_sales'])
    
    # Drop obvious leakage columns
    df = df.drop(columns=['img', 'title'], errors='ignore')
    
    _print_preprocessing_summary(df_raw, df)
    
    preprocessor, X, y = build_preprocessor_regression(df)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = get_regression_models()
    results = []
    fitted_models = {}
    
    for name, reg in models.items():
        print(f"\nTraining {name}...")
        pipe = Pipeline(steps=[('prep', preprocessor), ('reg', reg)])
        pipe.fit(X_train, y_train)
        mae, rmse, r2 = evaluate_regression_model(name, pipe, X_test, y_test)
        results.append((name, mae, rmse, r2, pipe))
        fitted_models[name] = pipe
    
    # Choose best by R² score (higher is better)
    results.sort(key=lambda x: x[3], reverse=True)
    best_name, best_mae, best_rmse, best_r2, best_model = results[0]
    
    print("\n" + "="*50)
    print("BEST REGRESSION MODEL:", best_name)
    print(f"  MAE: {best_mae:.4f} million units")
    print(f"  RMSE: {best_rmse:.4f} million units")
    print(f"  R² Score: {best_r2:.4f}")
    print("="*50)
    
    # Persist best model
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    model_path = os.path.join(ARTIFACT_DIR, 'best_regressor.joblib')
    joblib.dump(best_model, model_path)
    print(f"\nSaved best regressor to: {model_path}")
    
    # Save metrics
    meta = {
        'selected_model': best_name,
        'mae': float(best_mae),
        'rmse': float(best_rmse),
        'r2_score': float(best_r2),
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'task': 'regression',
        'target': 'total_sales'
    }
    metrics_path = os.path.join(ARTIFACT_DIR, 'regressor_metrics.json')
    pd.Series(meta).to_json(metrics_path)
    print(f"Saved metrics to: {metrics_path}")
    
    return best_name, best_mae, best_rmse, best_r2


def main():
    print("="*50)
    print("TRAINING REGRESSION MODEL FOR TOTAL SALES PREDICTION")
    print("="*50)
    print("Workspace root:", WORKSPACE_ROOT)
    print("Loading data from:", DATA_PATH)
    df = load_data(DATA_PATH)
    print("Rows:", len(df))
    
    best_name, best_mae, best_rmse, best_r2 = train_and_select_best_regressor(df)
    
    print("\n✓ Regression model training complete!")
    print("  You can now use 'best_regressor.joblib' to predict total_sales")


if __name__ == '__main__':
    main()
