from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

SUCCESS_THRESHOLD = 1.0  # million units


def _bucket_rare_categories(series: pd.Series, min_count: int = 20, other_label: str = 'other') -> pd.Series:
    """Replace infrequent categories with a common 'other' label to reduce high cardinality.
    Applies only to object dtype series.
    """
    if series.dtype != 'object':
        return series
    vc = series.value_counts(dropna=True)
    rare = vc[vc < min_count].index
    return series.where(~series.isin(rare), other_label)


def engineer_target(df: pd.DataFrame, threshold: float = SUCCESS_THRESHOLD) -> pd.DataFrame:
    df = df.copy()
    # Coerce numeric columns
    for col in ['total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'critic_score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing total_sales (cannot derive target); avoids mislabeling as 0
    if 'total_sales' in df.columns:
        df = df.dropna(subset=['total_sales'])

    # Clip critic_score to sensible range [0, 10]
    if 'critic_score' in df.columns:
        df['critic_score'] = df['critic_score'].clip(lower=0, upper=10)

    # Create target
    df['success'] = (df['total_sales'] >= threshold).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Extract release year
    if 'release_date' in df.columns:
        df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

    # Normalize categorical text: strip whitespace and lowercase for consistency
    for col in ['console', 'genre', 'publisher', 'developer']:
        if col in df.columns:
            if df[col].dtype != 'object':
                df[col] = df[col].astype('object')
            df[col] = df[col].str.strip().str.lower()

    # Bucket rare categories to reduce one-hot dimensionality
    for col in ['publisher', 'developer']:
        if col in df.columns:
            df[col] = _bucket_rare_categories(df[col], min_count=20, other_label='other')
            # Ensure object dtype and use np.nan for missing
            df[col] = df[col].astype('object').replace({pd.NA: np.nan})

    # Drop leakage/unnecessary columns
    df = df.drop(columns=['img', 'title'], errors='ignore')
    return df


def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, pd.DataFrame, pd.Series]:
    df = build_features(df)
    # Explicit feature sets to avoid dtype drift issues
    categorical_features = ['console', 'genre', 'publisher', 'developer']
    numeric_features = ['critic_score', 'release_year']

    # Coerce expected dtypes
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype('object').replace({pd.NA: np.nan})
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    feature_cols = categorical_features + numeric_features

    X = df[feature_cols]
    y = df['success']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, X, y
