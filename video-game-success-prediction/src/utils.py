"""
Utility functions (converted from utils.ipynb)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
	"""Save DataFrame to CSV creating parent dirs."""
	path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(path, index=index)


__all__ = [
	"save_dataframe",
]
