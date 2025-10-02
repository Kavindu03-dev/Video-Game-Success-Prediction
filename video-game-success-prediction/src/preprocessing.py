

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:

	out = df.copy()

	# Numeric median imputation
	num_cols = out.select_dtypes(include=["number"]).columns.tolist()
	for c in num_cols:
		if c in out.columns:
			median_val = out[c].median()
			if pd.notna(median_val):
				out[c] = out[c].fillna(median_val)

	# Categorical/boolean mode imputation
	cat_cols = out.select_dtypes(include=["object", "category", "bool", "boolean"]).columns.tolist()
	for c in cat_cols:
		if c in out.columns:
			mode_vals = out[c].mode(dropna=True)
			if not mode_vals.empty:
				out[c] = out[c].fillna(mode_vals.iloc[0])

	return out


def clean_dataset(
	df: pd.DataFrame,
	*,
	date_col: str = "release_date",
	fill_unknown: str = "Unknown",
	bool_fill: bool = False,
	drop_all_nan_rows: bool = True,
) -> pd.DataFrame:
	
	cleaned = df.copy()

	# 1) Remove exact duplicate rows
	cleaned = cleaned.drop_duplicates()

	# 2) Optionally drop rows that are entirely NaN
	if drop_all_nan_rows:
		cleaned = cleaned.dropna(how="all")

	# 3) Strip whitespace in object columns without converting NaN to strings
	obj_cols = cleaned.select_dtypes(include=["object"]).columns
	if len(obj_cols) > 0:
		for c in obj_cols:
			cleaned[c] = cleaned[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

	# 4) Convert release date to year (nullable Int64) if present
	if date_col in cleaned.columns:
		parsed = pd.to_datetime(cleaned[date_col], errors="coerce", utc=False)
		cleaned[date_col] = parsed.dt.year.astype("Int64")

	# 5) Drop rows with missing total_sales
	if "total_sales" in cleaned.columns:
		cleaned = cleaned[~cleaned["total_sales"].isna()].reset_index(drop=True)

	# 6) Drop non-predictive columns
	for col in ["img"]:
		if col in cleaned.columns:
			cleaned = cleaned.drop(columns=[col])

	# 7) Impute missing values by dtype (retain behavior)
	# Numeric columns including pandas nullable ints
	num_cols = list(cleaned.select_dtypes(include=["number"]).columns)
	nullable_int_cols = list(cleaned.select_dtypes(include=["Int64", "Int32", "Int16"]).columns)
	for c in set(nullable_int_cols) - set(num_cols):
		num_cols.append(c)

	# Boolean columns (numpy bool and pandas BooleanDtype)
	bool_cols = list(cleaned.select_dtypes(include=["bool", "boolean"]).columns)

	# Categorical/object/category columns
	cat_cols = list(cleaned.select_dtypes(include=["object", "category"]).columns)

	# Remove overlaps
	cat_cols = [c for c in cat_cols if c not in num_cols and c not in bool_cols]

	# Fill numeric with median
	for c in num_cols:
		if c in cleaned.columns:
			median_val = cleaned[c].median()
			cleaned[c] = cleaned[c].fillna(median_val)

	# Fill boolean with specified bool_fill
	for c in bool_cols:
		if c in cleaned.columns:
			cleaned[c] = cleaned[c].fillna(bool_fill)

	# Fill categorical with fill_unknown token
	for c in cat_cols:
		if c in cleaned.columns:
			cleaned[c] = cleaned[c].fillna(fill_unknown)

	return cleaned


def add_hit_label(
	df: pd.DataFrame,
	*,
	sales_col: str = "total_sales",
	threshold: float = 1.0,
	label_col: str = "Hit",
	dtype: str = "Int8",
) -> pd.DataFrame:
	
	out = df.copy()
	if sales_col not in out.columns:
		raise KeyError(f"Column '{sales_col}' not found in DataFrame")
	is_hit = out[sales_col].fillna(0).ge(threshold)
	out[label_col] = is_hit.astype(dtype)
	return out


def encode_categoricals(
	df: pd.DataFrame,
	*,
	columns: Optional[Sequence[str]] = ("genre", "platform", "publisher"),
	drop_first: bool = False,
	dummy_na: bool = False,
	prefix_sep: str = "=",
) -> pd.DataFrame:
	
	if columns is None:
		columns = []
	existing = [c for c in columns if c in df.columns]
	if len(existing) == 0:
		# Nothing to encode; return a copy to avoid side effects
		return df.copy()
	encoded = pd.get_dummies(
		df,
		columns=list(existing),
		drop_first=drop_first,
		dummy_na=dummy_na,
		prefix=existing,
		prefix_sep=prefix_sep,
	)
	return encoded


__all__ = [
	"clean_dataset",
	"add_hit_label",
	"encode_categoricals",
	"handle_missing_values",
]
