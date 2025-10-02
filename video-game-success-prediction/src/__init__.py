"""src package exports."""
try:
	from .preprocessing import clean_dataset, add_hit_label, encode_categoricals, handle_missing_values  # noqa: F401
except Exception:
	# In environments where relative import resolution differs, avoid hard failures
	clean_dataset = None  # type: ignore
	add_hit_label = None  # type: ignore
	encode_categoricals = None  # type: ignore
	handle_missing_values = None  # type: ignore

__all__ = [
	"clean_dataset",
	"add_hit_label",
	"encode_categoricals",
	"handle_missing_values",
]
