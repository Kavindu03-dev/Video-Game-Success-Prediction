"""
Preprocessing utilities for the Video Game Success Prediction project.

This module provides functions for cleaning the dataset, including:
- Removing duplicate rows
- Handling missing values by dtype
- Converting a release date column into a year
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd


def clean_dataset(
	df: pd.DataFrame,
	*,
	date_col: str = "release_date",
	fill_unknown: str = "Unknown",
	bool_fill: bool = False,
	drop_all_nan_rows: bool = True,
) -> pd.DataFrame:
	"""
	Clean a video game sales dataframe.

	Operations performed (in order):
	1) Drop exact duplicate rows.
	2) Optionally drop rows that are entirely NaN.
	3) Strip leading/trailing whitespace in object columns.
	4) Convert `date_col` to year (Int64) if present, coercing invalids to <NA>.
	5) Impute missing values by dtype:
	   - Numeric (including pandas nullable ints): median
	   - Categorical/object/category: fill with `fill_unknown`
	   - Boolean/bool/boolean: fill with `bool_fill`

	Parameters
	----------
	df : pd.DataFrame
		Input dataframe.
	date_col : str, default "release_date"
		Column name to parse as dates and convert to year.
	fill_unknown : str, default "Unknown"
		Token used to fill missing values for categorical columns.
	bool_fill : bool, default False
		Value used to fill missing values for boolean columns.
	drop_all_nan_rows : bool, default True
		Whether to drop rows that are entirely NaN before imputation.

	Returns
	-------
	pd.DataFrame
		A cleaned copy of the input dataframe.

	Notes
	-----
	- The `date_col` is replaced in-place with the extracted year as pandas Int64
	  (nullable integer) to preserve missing years.
	- If `date_col` is not present, the step is skipped.
	"""

	# Work on a copy to avoid mutating callers' data
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
		# Extract year; keep as nullable integer to preserve missing
		cleaned[date_col] = parsed.dt.year.astype("Int64")

	# 5) Impute missing values by dtype
	# Identify numeric columns including pandas nullable ints
	num_cols = list(cleaned.select_dtypes(include=["number"]).columns)
	# Include pandas nullable integer dtypes explicitly (Int64, Int32, Int16)
	nullable_int_cols = list(
		cleaned.select_dtypes(include=["Int64", "Int32", "Int16"]).columns
	)
	for c in set(nullable_int_cols) - set(num_cols):
		num_cols.append(c)

	# Boolean columns (both numpy bool and pandas BooleanDtype)
	bool_cols = list(cleaned.select_dtypes(include=["bool", "boolean"]).columns)

	# Categorical/object/category columns
	cat_cols = list(
		cleaned.select_dtypes(include=["object", "category"]).columns
	)

	# Remove overlaps to avoid double-filling
	cat_cols = [c for c in cat_cols if c not in num_cols and c not in bool_cols]

	# Fill numeric columns with median
	for c in num_cols:
		if c in cleaned.columns:
			median_val = cleaned[c].median()
			cleaned[c] = cleaned[c].fillna(median_val)

	# Fill boolean columns with specified bool_fill
	for c in bool_cols:
		if c in cleaned.columns:
			cleaned[c] = cleaned[c].fillna(bool_fill)

	# Fill categorical columns with fill_unknown token
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
	"""
	Add a binary hit label column based on a sales threshold.

	A value of 1 indicates `sales_col >= threshold`, else 0. Missing sales are
	treated as 0.

	Parameters
	----------
	df : pd.DataFrame
		Input dataframe.
	sales_col : str, default "total_sales"
		Column containing total sales values (numeric).
	threshold : float, default 1.0
		Threshold in the same units as `sales_col` to define a hit.
	label_col : str, default "Hit"
		Name of the output binary column.
	dtype : str, default "Int8"
		The dtype used for the binary column.

	Returns
	-------
	pd.DataFrame
		A copy of df with the new label column appended.
	"""
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
	"""
	One-hot encode selected categorical columns using pandas.get_dummies.

	Parameters
	----------
	df : pd.DataFrame
		Input dataframe.
	columns : sequence of str, optional
		Categorical column names to encode. Defaults to ("genre","platform","publisher").
	drop_first : bool, default False
		Whether to drop the first category for each encoded variable.
	dummy_na : bool, default False
		Whether to add a column for NaNs.
	prefix_sep : str, default "="
		Separator between the column name and category in the dummy column names.

	Returns
	-------
	pd.DataFrame
		Dataframe with specified categorical columns one-hot encoded.
	"""
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
]

