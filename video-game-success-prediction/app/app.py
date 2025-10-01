from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

st.set_page_config(page_title="Video Game Success Predictor", layout="wide")

st.title("🎮 Video Game Success Prediction")
st.caption("Predict whether a game is a Hit (total_sales ≥ 1.0), explore trends, and edit data inline.")


@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
	with open(model_path, 'rb') as f:
		return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_data(csv_path: Path) -> pd.DataFrame:
	return pd.read_csv(csv_path)


project_root = Path(__file__).resolve().parents[1]
model_path = project_root / 'model.pkl'
# Prefer data/vg_sales_2024.csv, fallback to data/raw/vg_sales_2024.csv
data_path = project_root / 'data' / 'vg_sales_2024.csv'
if not data_path.exists():
	data_path = project_root / 'data' / 'raw' / 'vg_sales_2024.csv'

model = None
if model_path.exists():
	try:
		model = load_model(model_path)
	except Exception as e:
		st.error(f"Failed to load model: {e}")
else:
	st.warning("Model file not found. Train and save a model to 'model.pkl' first (run src/train_model.py).")

df = None
if data_path.exists():
	try:
		df = load_data(data_path)
		# Normalize column names: strip spaces
		df = df.copy()
		df.columns = df.columns.str.strip()
	except Exception as e:
		st.error(f"Failed to load data: {e}")
else:
	st.warning("Dataset not found in data/raw/vg_sales_2024.csv.")


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
	"""Find a column name in df by trying multiple candidates (case-insensitive)."""
	cols = {str(c).strip().lower(): c for c in df.columns}
	for name in candidates:
		key = str(name).strip().lower()
		if key in cols:
			return cols[key]
	return None


# Helpers for Developer Dashboard and preprocessing
def _ensure_release_year(df: pd.DataFrame) -> pd.DataFrame:
	"""Ensure a 'release_year' column exists by extracting year from release_date or using synonyms."""
	if df is None or df.empty:
		return df
	if 'release_year' in df.columns:
		return df
	# Try common date/year columns
	year_col = _resolve_column(df, ['release_year', 'year'])
	if year_col and year_col in df.columns:
		if year_col != 'release_year':
			df = df.rename(columns={year_col: 'release_year'})
		return df
	date_col = _resolve_column(df, ['release_date', 'date'])
	if date_col and date_col in df.columns:
		tmp = pd.to_datetime(df[date_col], errors='coerce')
		df = df.copy()
		df['release_year'] = tmp.dt.year
	return df


def _categorical_candidates(df: pd.DataFrame) -> list[str]:
	cats = df.select_dtypes(include=['object', 'category']).columns.tolist()
	# Add known categoricals if typed as numeric by mistake
	for col in ['genre', 'platform', 'publisher']:
		if col in df.columns and col not in cats:
			cats.append(col)
	if 'release_year' in df.columns and df['release_year'].nunique() < 200:
		cats.append('release_year')
	return sorted(list(dict.fromkeys(cats)))


def _numeric_candidates(df: pd.DataFrame) -> list[str]:
	nums = df.select_dtypes(include=['number']).columns.tolist()
	return sorted(nums)


def _ensure_total_sales(df: pd.DataFrame) -> pd.DataFrame:
	"""Ensure df has a 'total_sales' column; compute from regions if necessary."""
	if df is None or df.empty:
		return df
	cols_lower = {str(c).strip().lower(): c for c in df.columns}
	# Direct mapping if total_sales exists (any case)
	if 'total_sales' in cols_lower:
		# Rename to canonical casing if needed
		if cols_lower['total_sales'] != 'total_sales':
			df = df.rename(columns={cols_lower['total_sales']: 'total_sales'})
		return df
	# Map common synonyms like Global_Sales
	for alias in ['global_sales', 'globalsales', 'global']:
		if alias in cols_lower:
			df = df.rename(columns={cols_lower[alias]: 'total_sales'})
			return df
	# Try compute from regions
	regions = ['na_sales', 'eu_sales', 'jp_sales', 'other_sales']
	present = [cols_lower[r] for r in regions if r in cols_lower]
	if present:
		df = df.copy()
		df['total_sales'] = df[present].sum(axis=1, skipna=True)
	return df


with st.sidebar:
	st.header("Prediction Inputs")
	st.write("Fill in the fields and click Predict.")

	if st.button("🔄 Refresh data", help="Clear cache and reload the dataset"):
		try:
			st.cache_data.clear()
			st.cache_resource.clear()
			st.success("Caches cleared. Use the menu 'Rerun' to reload.")
		except Exception as e:
			st.warning(f"Could not clear cache: {e}")

	# Normalize and ensure total_sales
	if df is not None:
		df = _ensure_total_sales(df)

	# Resolve columns case-insensitively
	genre_col = _resolve_column(df, ["genre"]) if df is not None else None
	platform_col = _resolve_column(df, ["platform"]) if df is not None else None
	publisher_col = _resolve_column(df, ["publisher"]) if df is not None else None

	# Safely populate select boxes from data when available
	genre_options = sorted(df[genre_col].dropna().unique().tolist()) if df is not None and genre_col in (df.columns if df is not None else []) else []
	platform_options = sorted(df[platform_col].dropna().unique().tolist()) if df is not None and platform_col in (df.columns if df is not None else []) else []
	publisher_options = sorted(df[publisher_col].dropna().unique().tolist()) if df is not None and publisher_col in (df.columns if df is not None else []) else []

	# Fallback to some common values if dataset not available
	if not genre_options:
		genre_options = ["Action", "Adventure", "Sports", "RPG", "Shooter", "Racing", "Platform", "Puzzle"]
	if not platform_options:
		platform_options = ["PS4", "XOne", "Switch", "PC", "PS3", "Xbox360", "Wii"]
	if not publisher_options:
		publisher_options = ["Nintendo", "EA", "Activision", "Ubisoft", "Sony", "Microsoft"]

	genre = st.selectbox("Genre", options=genre_options, index=0)
	platform = st.selectbox("Platform", options=platform_options, index=0)
	publisher = st.selectbox("Publisher", options=publisher_options, index=0)
	critic_score = st.number_input("Critic Score", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
	release_year = st.number_input("Release Year", min_value=1980, max_value=2030, value=2015, step=1)

	do_predict = st.button("Predict Hit")


def predict_hit(model, genre: str, platform: str, publisher: str, critic_score: float, release_year: int) -> tuple[int, float]:
	if model is None:
		raise RuntimeError("Model is not loaded.")
	# Build single-row DataFrame following training feature schema
	X = pd.DataFrame([
		{
			'genre': genre if genre != 'Unknown' else None,
			'platform': platform if platform != 'Unknown' else None,
			'publisher': publisher if publisher != 'Unknown' else None,
			'critic_score': critic_score,
			'release_year': int(release_year),
		}
	])
	proba = model.predict_proba(X)[:, 1][0]
	pred = int(proba >= 0.5)
	return pred, float(proba)


# ------------- Navigation Tabs -------------
tabs = st.tabs(["Overview", "Explore", "Predict", "Insights", "Developer Dashboard"])

# Shared: ensure total_sales and resolved column names
if df is not None and not df.empty:
	df = _ensure_total_sales(df)
	genre_col = _resolve_column(df, ["genre", "category", "type"]) or "genre"
	platform_col = _resolve_column(df, ["platform", "console", "system", "platform_name"]) or "platform"
	publisher_col = _resolve_column(df, ["publisher"]) or "publisher"
	total_col = _resolve_column(df, ["total_sales", "global_sales", "globalsales", "global"]) or "total_sales"
else:
	genre_col = platform_col = publisher_col = total_col = None


# Overview Tab
with tabs[0]:
	st.subheader("Overview")
	if df is None or df.empty:
		st.info("Dataset not loaded. Place vg_sales_2024.csv under data/ or data/raw/.")
	else:
		c1, c2, c3 = st.columns(3)
		with c1:
			st.metric("Rows", f"{len(df):,}")
		with c2:
			n_cols = len(df.columns)
			st.metric("Columns", f"{n_cols}")
		with c3:
			# Estimate hit rate if Hit exists or computable
			hit_col = _resolve_column(df, ["hit"]) or "Hit"
			if hit_col in df.columns:
				st.metric("Hit Rate", f"{pd.to_numeric(df[hit_col], errors='coerce').mean():.2%}")
			else:
				st.metric("Hit Rate", "—")

		st.write("Preview")
		st.dataframe(df.head(20), use_container_width=True)

		st.write("Edit and Save (optional)")
		edited = st.data_editor(df.head(200), use_container_width=True, num_rows="dynamic")
		if st.button("Save edited sample to data/processed/edited.csv"):
			out_path = project_root / 'data' / 'processed' / 'edited.csv'
			try:
				out_path.parent.mkdir(parents=True, exist_ok=True)
				edited.to_csv(out_path, index=False)
				st.success(f"Saved edited sample to {out_path}")
			except Exception as e:
				st.error(f"Failed to save: {e}")


# Explore Tab
with tabs[1]:
	st.subheader("Explore Sales")
	if df is None or df.empty:
		st.info("Dataset not loaded.")
	else:
		dim = st.selectbox("Group by", options=[("Genre", genre_col), ("Platform", platform_col), ("Publisher", publisher_col)], index=0, format_func=lambda x: x[0])
		topn = st.slider("Top N", min_value=5, max_value=30, value=10, step=1)
		dim_col = dim[1]
		if dim_col in df.columns and total_col in df.columns:
			s = df.groupby(dim_col, dropna=False)[total_col].sum().sort_values(ascending=False).head(topn)
			fig, ax = plt.subplots(figsize=(10, 5))
			sns.barplot(x=s.values, y=s.index, ax=ax, palette="viridis", legend=False)
			ax.set_xlabel("Total Sales")
			ax.set_ylabel(dim[0])
			st.pyplot(fig)
		else:
			st.warning("Required columns for this plot were not found.")


# Predict Tab
with tabs[2]:
	st.subheader("Predict Hit / Not Hit")
	c1, c2 = st.columns([1, 2])
	with c1:
		threshold = st.slider("Decision threshold (P[Hit])", min_value=0.05, max_value=0.95, value=0.5, step=0.05)
		if do_predict:
			try:
				pred, proba = predict_hit(model, genre, platform, publisher, critic_score, release_year)
				pred = int(proba >= threshold)
				label = "Hit" if pred == 1 else "Not Hit"
				st.metric("Prediction", label, delta=f"P(Hit) = {proba:.2%}")
				st.progress(min(max(proba, 0.0), 1.0), text=f"Probability of Hit: {proba:.2%}")
			except Exception as e:
				st.error(str(e))
		else:
			st.info("Set inputs in the sidebar and click Predict.")

	with c2:
		st.write("Batch Prediction (edit rows and click Predict)")
		template = pd.DataFrame([
			{"genre": genre_options[0] if genre_options else "Action", "platform": platform_options[0] if platform_options else "PS4", "publisher": publisher_options[0] if publisher_options else "Nintendo", "critic_score": 75.0, "release_year": 2015}
		])
		batch_df = st.data_editor(template, use_container_width=True, num_rows="dynamic")
		if st.button("Predict for all rows"):
			try:
				if model is None:
					raise RuntimeError("Model is not loaded. Train model first.")
				preds = model.predict_proba(batch_df)[:, 1]
				labels = (preds >= threshold).astype(int)
				out = batch_df.copy()
				out["P(Hit)"] = preds
				out["Pred"] = labels
				st.dataframe(out, use_container_width=True)
			except Exception as e:
				st.error(f"Batch prediction failed: {e}")


# Insights Tab
with tabs[3]:
	st.subheader("Insights")
	if df is None or df.empty:
		st.info("Dataset not loaded.")
	else:
		sub = st.selectbox("Insight", ["Sales by Region", "Correlation Heatmap", "Feature Importance (model)"])

		if sub == "Sales by Region":
			# Detect all numeric '*sales*' columns except total/global
			cols_lower = {str(c).strip().lower(): c for c in df.columns}
			exclude = {"total_sales", "global_sales", "globalsales", "global"}
			region_cols = []
			for c in df.columns:
				key = str(c).strip().lower()
				if "sales" in key and key not in exclude and pd.api.types.is_numeric_dtype(df[c]):
					region_cols.append(c)
			if len(region_cols) >= 1:
				agg = df[region_cols].sum(numeric_only=True).sort_values(ascending=False)
				if not agg.empty:
					labels = [str(col).replace("_", " ").title() for col in agg.index]
					fig, ax = plt.subplots(figsize=(8, 5))
					sns.barplot(x=agg.values, y=labels, ax=ax, palette="crest", legend=False)
					ax.set_xlabel("Total Sales")
					ax.set_ylabel("Region")
					st.pyplot(fig)
					st.caption(f"Using region columns: {', '.join(map(str, agg.index))}")
				else:
					st.info("No regional sales data to aggregate.")
			else:
				st.info("No region-specific sales columns found.")

		elif sub == "Correlation Heatmap":
			num_df = df.select_dtypes(include="number")
			if num_df.shape[1] >= 2:
				corr = num_df.corr(numeric_only=True)
				plt.figure(figsize=(10, 8))
				sns.heatmap(corr, cmap="coolwarm", annot=False)
				st.pyplot(plt.gcf())
			else:
				st.info("Not enough numeric columns.")

		elif sub == "Feature Importance (model)":
			if model is None:
				st.info("Model not loaded. Train model to see feature importances.")
			else:
				try:
					# Attempt to get feature importances from pipeline
					clf = getattr(model, 'named_steps', {}).get('clf', None)
					pre = getattr(model, 'named_steps', {}).get('pre', None)
					if clf is None or pre is None or not hasattr(clf, 'feature_importances_'):
						st.info("Current model does not expose feature importances.")
					else:
						feat_names = pre.get_feature_names_out()
						importances = clf.feature_importances_
						order = importances.argsort()[::-1]
						topn = st.slider("Top N features", 5, 40, 20)
						sel = order[:topn]
						fig, ax = plt.subplots(figsize=(8, min(10, 0.4 * topn + 2)))
						sns.barplot(x=importances[sel], y=feat_names[sel], ax=ax, palette="mako", legend=False)
						ax.set_xlabel("Importance")
						ax.set_ylabel("Feature")
						st.pyplot(fig)
				except Exception as e:
					st.error(f"Failed to compute feature importances: {e}")


# Developer Dashboard Tab
with tabs[4]:
	st.subheader("Developer Dashboard")
	if df is None or df.empty:
		st.info("Dataset not loaded.")
	else:
		# Prepare data
		df = _ensure_release_year(df)
		cats = _categorical_candidates(df)
		nums = _numeric_candidates(df)

		# Chart builder controls (Top N moved below filters)
		c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
		with c1:
			x_axis = st.selectbox("X Axis", options=cats + nums, index=(cats + nums).index('platform') if 'platform' in (cats + nums) else 0)
		with c2:
			y_axis = st.selectbox("Y Axis (metric)", options=nums, index=nums.index('total_sales') if 'total_sales' in nums else 0)
		with c3:
			agg_fn = st.selectbox("Aggregation", options=['sum', 'mean', 'median', 'count'])
		with c4:
			chart_type = st.selectbox("Chart Type", options=['bar', 'line', 'scatter', 'area'], index=0)

		# Optional filters
		f1, f2, f3, f4 = st.columns(4)
		with f1:
			genre_f = st.multiselect("Filter Genre", options=sorted(df.get(genre_col, pd.Series(dtype=str)).dropna().unique().tolist()) if genre_col in df.columns else [])
		with f2:
			plat_f = st.multiselect("Filter Platform", options=sorted(df.get(platform_col, pd.Series(dtype=str)).dropna().unique().tolist()) if platform_col in df.columns else [])
		with f3:
			pub_f = st.multiselect("Filter Publisher", options=sorted(df.get(publisher_col, pd.Series(dtype=str)).dropna().unique().tolist()) if publisher_col in df.columns else [])
		with f4:
			# Release Year range filter (dual-handle slider)
			# Ensure release_year exists (already ensured above)
			years_series = pd.to_numeric(df.get('release_year', pd.Series(dtype=int)), errors='coerce')
			if years_series.dropna().empty:
				# Fallback: attempt to derive from release_date if present
				date_col = _resolve_column(df, ['release_date', 'date'])
				if date_col in df.columns:
					years_series = pd.to_datetime(df[date_col], errors='coerce').dt.year
			min_y = int(years_series.dropna().min()) if not years_series.dropna().empty else 1980
			max_y = int(years_series.dropna().max()) if not years_series.dropna().empty else 2030
			year_range = st.slider("Release Year", min_value=min_y, max_value=max_y, value=(min_y, max_y), step=1, help="Select start and end year")

		dfx = df.copy()
		if genre_col in dfx.columns and genre_f:
			dfx = dfx[dfx[genre_col].isin(genre_f)]
		if platform_col in dfx.columns and plat_f:
			dfx = dfx[dfx[platform_col].isin(plat_f)]
		if publisher_col in dfx.columns and pub_f:
			dfx = dfx[dfx[publisher_col].isin(pub_f)]
		# Apply year range filter
		if 'year_range' in locals() and isinstance(year_range, (list, tuple)) and len(year_range) == 2:
			start_y, end_y = int(year_range[0]), int(year_range[1])
			if 'release_year' in dfx.columns:
				dfx = dfx[(pd.to_numeric(dfx['release_year'], errors='coerce') >= start_y) & (pd.to_numeric(dfx['release_year'], errors='coerce') <= end_y)]

		# Top N after filters
		topn = st.slider("Top N", 5, 50, 10)

		# Aggregate
		if x_axis in dfx.columns and y_axis in dfx.columns:
			grouped = dfx.groupby(x_axis)[y_axis]
			if agg_fn == 'sum':
				s = grouped.sum()
			elif agg_fn == 'mean':
				s = grouped.mean()
			elif agg_fn == 'median':
				s = grouped.median()
			elif agg_fn == 'count':
				s = grouped.count()
			s = s.sort_values(ascending=False).head(topn).reset_index(name=y_axis) if hasattr(s, 'reset_index') else s.sort_values(ascending=False).head(topn)
			if isinstance(s, pd.Series):
				s = s.reset_index()

			# Build Altair chart with selected type
			if chart_type == 'bar':
				base = alt.Chart(s).mark_bar().encode(
					x=alt.X(f"{y_axis}:Q", title=y_axis.replace('_', ' ').title()),
					y=alt.Y(f"{x_axis}:N", sort='-x', title=x_axis.replace('_', ' ').title()),
					tooltip=list(s.columns)
				).properties(height=400)
			elif chart_type == 'line':
				base = alt.Chart(s).mark_line(point=True).encode(
					x=alt.X(f"{x_axis}:O", title=x_axis.replace('_', ' ').title()),
					y=alt.Y(f"{y_axis}:Q", title=y_axis.replace('_', ' ').title()),
					tooltip=list(s.columns)
				).properties(height=400)
			elif chart_type == 'scatter':
				# For scatter, we need raw data points, not aggregated
				if len(dfx) > 1000:  # Limit points for performance
					dfx_sample = dfx.sample(1000)
				else:
					dfx_sample = dfx
				base = alt.Chart(dfx_sample).mark_circle(size=60).encode(
					x=alt.X(f"{x_axis}:O" if x_axis in cats else f"{x_axis}:Q", title=x_axis.replace('_', ' ').title()),
					y=alt.Y(f"{y_axis}:Q", title=y_axis.replace('_', ' ').title()),
					tooltip=[x_axis, y_axis]
				).properties(height=400)
			elif chart_type == 'area':
				base = alt.Chart(s).mark_area().encode(
					x=alt.X(f"{x_axis}:O", title=x_axis.replace('_', ' ').title()),
					y=alt.Y(f"{y_axis}:Q", title=y_axis.replace('_', ' ').title()),
					tooltip=list(s.columns)
				).properties(height=400)
			else:
				# Default to bar
				base = alt.Chart(s).mark_bar().encode(
					x=alt.X(f"{y_axis}:Q", title=y_axis.replace('_', ' ').title()),
					y=alt.Y(f"{x_axis}:N", sort='-x', title=x_axis.replace('_', ' ').title()),
					tooltip=list(s.columns)
				).properties(height=400)

			# Chart collection in session state
			if 'chart_specs' not in st.session_state:
				st.session_state.chart_specs = []

			add_col1, add_col2 = st.columns([1, 3])
			with add_col1:
				if st.button("Add Chart"):
					# Store raw data for scatter plots, aggregated for others
					data_to_store = dfx[[x_axis, y_axis]].to_dict(orient='list') if chart_type == 'scatter' else s.to_dict(orient='list')
					# Prepare release year filter snapshot
					release_year_filter = [int(year_range[0]), int(year_range[1])] if ('year_range' in locals() and isinstance(year_range, (list, tuple))) else None
					st.session_state.chart_specs.append({
						'x_axis': x_axis,
						'y_axis': y_axis,
						'agg_fn': agg_fn,
						'chart_type': chart_type,
						'topn': topn,
						'filters': {
							'genre': genre_f,
							'platform': plat_f,
							'publisher': pub_f,
							'release_year': release_year_filter,
						},
						'data': data_to_store,
					})
			with add_col2:
				st.altair_chart(base, use_container_width=True)

			st.markdown("---")
			st.subheader("Your Dashboard")
			if st.session_state.chart_specs:
				# Render charts in a 2-column grid
				cols = st.columns(2)
				for i, spec in enumerate(st.session_state.chart_specs):
					df_spec = pd.DataFrame(spec['data'])
					chart_type_spec = spec.get('chart_type', 'bar')
					
					if chart_type_spec == 'bar':
						chart = alt.Chart(df_spec).mark_bar().encode(
							x=alt.X(f"{spec['y_axis']}:Q", title=spec['y_axis'].replace('_',' ').title()),
							y=alt.Y(f"{spec['x_axis']}:N", sort='-x', title=spec['x_axis'].replace('_',' ').title()),
							tooltip=list(df_spec.columns)
						).properties(height=300)
					elif chart_type_spec == 'line':
						chart = alt.Chart(df_spec).mark_line(point=True).encode(
							x=alt.X(f"{spec['x_axis']}:O", title=spec['x_axis'].replace('_',' ').title()),
							y=alt.Y(f"{spec['y_axis']}:Q", title=spec['y_axis'].replace('_',' ').title()),
							tooltip=list(df_spec.columns)
						).properties(height=300)
					elif chart_type_spec == 'scatter':
						chart = alt.Chart(df_spec).mark_circle(size=60).encode(
							x=alt.X(f"{spec['x_axis']}:Q", title=spec['x_axis'].replace('_',' ').title()),
							y=alt.Y(f"{spec['y_axis']}:Q", title=spec['y_axis'].replace('_',' ').title()),
							tooltip=list(df_spec.columns)
						).properties(height=300)
					elif chart_type_spec == 'area':
						chart = alt.Chart(df_spec).mark_area().encode(
							x=alt.X(f"{spec['x_axis']}:O", title=spec['x_axis'].replace('_',' ').title()),
							y=alt.Y(f"{spec['y_axis']}:Q", title=spec['y_axis'].replace('_',' ').title()),
							tooltip=list(df_spec.columns)
						).properties(height=300)
					else:
						# Default to bar
						chart = alt.Chart(df_spec).mark_bar().encode(
							x=alt.X(f"{spec['y_axis']}:Q", title=spec['y_axis'].replace('_',' ').title()),
							y=alt.Y(f"{spec['x_axis']}:N", sort='-x', title=spec['x_axis'].replace('_',' ').title()),
							tooltip=list(df_spec.columns)
						).properties(height=300)
					
					cols[i % 2].altair_chart(chart, use_container_width=True)
				if st.button("Clear Dashboard"):
					st.session_state.chart_specs = []
			else:
				st.info("Use 'Add Chart' to collect charts here for side-by-side comparison.")
		else:
			st.warning("Select valid X and Y columns available in the dataset.")
