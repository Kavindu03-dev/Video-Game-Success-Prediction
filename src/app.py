from __future__ import annotations

from pathlib import Path
import math
import joblib
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
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


project_root = Path(__file__).resolve().parents[1]
model_path = project_root / 'models' / 'best_model.joblib'
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
    st.warning("Model file not found. Train and save a model to 'models/best_model.joblib' first (run src/train.py).")

if model is not None and not hasattr(model, 'predict_proba'):
    st.info("Loaded model does not expose predict_proba; probability shown may be based on decision function or class label.")

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
    st.warning("Dataset not found under data/vg_sales_2024.csv or data/raw/vg_sales_2024.csv.")


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column name in df by trying multiple candidates (case-insensitive)."""
    cols = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        key = str(name).strip().lower()
        if key in cols:
            return cols[key]
    return None


# Helpers for Dashboard and preprocessing
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
    for col in ['genre', 'console', 'publisher', 'developer']:
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
        if cols_lower['total_sales'] != 'total_sales':
            df = df.rename(columns={cols_lower['total_sales']: 'total_sales'})
        return df
    # Map common synonyms like Global_Sales
    for alias in ['global_sales', 'globalsales', 'global']:
        if alias in cols_lower:
            df = df.rename(columns={cols_lower[alias]: 'total_sales'})
            return df
    # Try compute from regions (support PAL/EU)
    region_aliases = [('na_sales',), ('eu_sales', 'pal_sales'), ('jp_sales',), ('other_sales',)]
    present = []
    for aliases in region_aliases:
        for al in aliases:
            if al in cols_lower:
                present.append(cols_lower[al])
                break
    if present:
        df = df.copy()
        df['total_sales'] = df[present].sum(axis=1, skipna=True)
    return df


with st.sidebar:
    st.header("Navigation")
    if 'page' not in st.session_state:
        st.session_state.page = "Explore"
    for label in ["Explore", "Predict", "Insights", "Developer Dashboard"]:
        st.button(label, use_container_width=True, key=f"nav_{label.replace(' ', '_')}", disabled=(st.session_state.page == label), help=f"Go to {label}")
        if st.session_state.get(f"nav_{label.replace(' ', '_')}"):
            st.session_state.page = label
        st.write("")
    page = st.session_state.page


def _normalize_text(val: str | None) -> str | None:
    if val is None:
        return None
    s = str(val).strip().lower()
    return s if s else None


def predict_hit(model, genre: str, console: str, publisher: str, developer: str, critic_score: float, release_year: int) -> tuple[int, float]:
    if model is None:
        raise RuntimeError("Model is not loaded.")
    # Build single-row DataFrame following training feature schema
    X = pd.DataFrame([
        {
            'genre': _normalize_text(genre if genre != 'Unknown' else None),
            'console': _normalize_text(console if console != 'Unknown' else None),
            'publisher': _normalize_text(publisher if publisher != 'Unknown' else None),
            'developer': _normalize_text(developer if developer != 'Unknown' else None),
            'critic_score': float(max(0.0, min(10.0, critic_score))),
            'release_year': int(release_year),
        }
    ])
    # Some models may not implement predict_proba
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[:, 1][0]
    else:
        pred_raw = model.predict(X)[0]
        proba = float(pred_raw)
    pred = int(proba >= 0.5)
    return pred, float(proba)


# Shared: ensure total_sales and resolved column names
if df is not None and not df.empty:
    df = _ensure_total_sales(df)
    genre_col = _resolve_column(df, ["genre", "category", "type"]) or "genre"
    console_col = _resolve_column(df, ["console", "platform", "system", "platform_name"]) or "console"
    publisher_col = _resolve_column(df, ["publisher"]) or "publisher"
    developer_col = _resolve_column(df, ["developer", "dev"]) or "developer"
    total_col = _resolve_column(df, ["total_sales", "global_sales", "globalsales", "global"]) or "total_sales"
else:
    genre_col = console_col = publisher_col = developer_col = total_col = None


if page == "Explore":
    st.subheader("Explore Sales")
    if df is None or df.empty:
        st.info("Dataset not loaded.")
    else:
        dim = st.selectbox("Group by", options=[("Genre", genre_col), ("Console", console_col), ("Publisher", publisher_col)], index=0, format_func=lambda x: x[0])
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

elif page == "Predict":
    st.subheader("Predict Hit / Not Hit")
    if df is not None:
        df = _ensure_total_sales(df)
    genre_col_sb = _resolve_column(df, ["genre"]) if df is not None else None
    console_col_sb = _resolve_column(df, ["console", "platform"]) if df is not None else None
    publisher_col_sb = _resolve_column(df, ["publisher"]) if df is not None else None
    developer_col_sb = _resolve_column(df, ["developer", "dev"]) if df is not None else None

    genre_options = sorted(df.get(genre_col_sb, pd.Series(dtype=str)).dropna().unique().tolist()) if df is not None and genre_col_sb in df.columns else []
    console_options = sorted(df.get(console_col_sb, pd.Series(dtype=str)).dropna().unique().tolist()) if df is not None and console_col_sb in df.columns else []
    publisher_options = sorted(df.get(publisher_col_sb, pd.Series(dtype=str)).dropna().unique().tolist()) if df is not None and publisher_col_sb in df.columns else []
    developer_options = sorted(df.get(developer_col_sb, pd.Series(dtype=str)).dropna().unique().tolist()) if df is not None and developer_col_sb in df.columns else []

    if not genre_options:
        genre_options = ["action", "adventure", "sports", "role-playing", "shooter", "racing", "platform", "puzzle"]
    if not console_options:
        console_options = ["ps4", "xone", "switch", "pc", "ps3", "x360", "wii"]
    if not publisher_options:
        publisher_options = ["nintendo", "ea", "activision", "ubisoft", "sony", "microsoft"]
    if not developer_options:
        developer_options = ["nintendo", "ea", "ubisoft", "fromsoftware", "capcom", "square enix"]

    # Determine a default release year (median from dataset) to use internally
    if df is not None and 'release_year' in df.columns:
        _ry = pd.to_numeric(df['release_year'], errors='coerce')
        default_release_year = int(_ry.median()) if _ry.notna().any() else 2015
    else:
        default_release_year = 2015

    ci1, ci2 = st.columns([1, 2])
    with ci1:
        st.markdown("### Prediction Inputs")
        genre = st.selectbox("Genre", options=genre_options, index=0)
        console = st.selectbox("Console/Platform", options=console_options, index=0)
        publisher = st.selectbox("Publisher", options=publisher_options, index=0)
        developer = st.selectbox("Developer", options=developer_options, index=0)
        critic_score = st.number_input("Critic Score (0-10)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        if st.button("Predict Hit"):
            try:
                pred, proba = predict_hit(model, genre, console, publisher, developer, critic_score, default_release_year)
                label = "Hit" if pred == 1 else "Not Hit"
                st.metric("Prediction", label, delta=f"P(Hit) = {proba:.2%}")
                st.progress(min(max(proba, 0.0), 1.0), text=f"Probability of Hit: {proba:.2%}")
            except Exception as e:
                st.error(str(e))

    with ci2:
        st.markdown("### Batch Prediction")
        template = pd.DataFrame([
            {"genre": genre_options[0], "console": console_options[0], "publisher": publisher_options[0], "developer": developer_options[0], "critic_score": 7.5}
        ])
        batch_df = st.data_editor(template, use_container_width=True, num_rows="dynamic")
        if st.button("Predict for all rows"):
            try:
                if model is None:
                    raise RuntimeError("Model is not loaded. Train model first.")
                # Apply same normalization as training
                for col in ['genre', 'console', 'publisher', 'developer']:
                    if col in batch_df.columns:
                        batch_df[col] = batch_df[col].astype('string').str.strip().str.lower()
                if 'critic_score' in batch_df.columns:
                    batch_df['critic_score'] = pd.to_numeric(batch_df['critic_score'], errors='coerce').clip(0, 10)
                # Ensure release_year exists (use dataset median if missing)
                if 'release_year' not in batch_df.columns:
                    batch_df['release_year'] = default_release_year
                else:
                    batch_df['release_year'] = pd.to_numeric(batch_df['release_year'], errors='coerce').fillna(default_release_year).astype(int)

                preds = model.predict_proba(batch_df)[:, 1] if hasattr(model, 'predict_proba') else model.predict(batch_df).astype(float)
                labels = (preds >= 0.5).astype(int)
                out = batch_df.copy()
                out["P(Hit)"] = preds
                out["Pred"] = labels
                st.dataframe(out, use_container_width=True)
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

    st.divider()
    st.markdown("### Predict Years to Hit (avg yearly sales)")
    st.caption("Estimate years needed to reach Hit based on the average yearly sales of similar games. Assumes Hit threshold = 1.0 and starting sales = 0.")

    genre_sel = st.selectbox("Genre (filter)", options=["Any"] + genre_options, index=0)
    console_sel = st.selectbox("Console (filter)", options=["Any"] + console_options, index=0)
    publisher_sel = st.selectbox("Publisher (filter)", options=["Any"] + publisher_options, index=0)
    developer_sel = st.selectbox("Developer (filter)", options=["Any"] + developer_options, index=0)

    if st.button("Estimate years to hit (avg yearly sales)"):
        try:
            if df is None or df.empty:
                st.info("Dataset not loaded.")
            else:
                df_rate = _ensure_release_year(_ensure_total_sales(df.copy()))
                if 'release_year' not in df_rate.columns or total_col not in df_rate.columns:
                    st.warning("Required columns not available to compute average yearly sales.")
                else:
                    sub = df_rate.copy()
                    if genre_sel != "Any" and genre_col in sub.columns:
                        sub = sub[sub[genre_col] == genre_sel]
                    if console_sel != "Any" and console_col in sub.columns:
                        sub = sub[sub[console_col] == console_sel]
                    if publisher_sel != "Any" and publisher_col in sub.columns:
                        sub = sub[sub[publisher_col] == publisher_sel]
                    if developer_sel != "Any" and developer_col in sub.columns:
                        sub = sub[sub[developer_col] == developer_sel]

                    if sub.empty:
                        st.info("No matching rows for the selected filters.")
                    else:
                        CURRENT_YEAR = 2025
                        vals_year = pd.to_numeric(sub['release_year'], errors='coerce')
                        years_elapsed = (CURRENT_YEAR - vals_year)
                        years_elapsed = years_elapsed.where(years_elapsed >= 1, 1)
                        vals_sales = pd.to_numeric(sub[total_col], errors='coerce')
                        rate = vals_sales / years_elapsed
                        avg_rate = float(rate.mean(skipna=True)) if rate.notna().any() else float('nan')

                        if not pd.notna(avg_rate) or avg_rate <= 0:
                            st.info("Cannot estimate: average yearly sales is not available or non-positive for the selected filters.")
                        else:
                            hit_threshold = 1.0
                            current_sales = 0.0
                            years_needed = math.ceil(max(0.0, hit_threshold - current_sales) / avg_rate)
                            cya, cyb, cyc = st.columns(3)
                            with cya: st.metric("Avg yearly sales", f"{avg_rate:.3f}")
                            with cyb: st.metric("Assumed Hit threshold", f"{hit_threshold:.1f}")
                            with cyc: st.metric("Estimated years to Hit", f"{years_needed}")
                            st.caption(f"Based on {len(sub)} matching games.")
        except Exception as e:
            st.error(f"Failed to estimate years to hit: {e}")

elif page == "Insights":
    st.subheader("Insights")
    if df is None or df.empty:
        st.info("Dataset not loaded.")
    else:
        sub = st.selectbox("Insight", ["Sales by Region", "Correlation Heatmap", "Feature Importance (model)"])

        if sub == "Sales by Region":
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
                    clf = getattr(model, 'named_steps', {}).get('clf', None)
                    pre = getattr(model, 'named_steps', {}).get('prep', None)
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

elif page == "Developer Dashboard":
    st.subheader("Developer Dashboard")
    if df is None or df.empty:
        st.info("Dataset not loaded.")
    else:
        df = _ensure_release_year(df)
        cats = _categorical_candidates(df)
        nums = _numeric_candidates(df)

        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            default_x = 'console' if 'console' in (cats + nums) else (cats + nums)[0]
            x_axis = st.selectbox("X Axis", options=cats + nums, index=(cats + nums).index(default_x) if default_x in (cats + nums) else 0)
        with c2:
            y_axis = st.selectbox("Y Axis (metric)", options=nums, index=nums.index('total_sales') if 'total_sales' in nums else 0)
        with c3:
            agg_fn = st.selectbox("Aggregation", options=['sum', 'mean', 'median', 'count'])
        with c4:
            chart_type = st.selectbox("Chart Type", options=['bar', 'line', 'scatter', 'area'], index=0)

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            genre_f = st.multiselect("Filter Genre", options=sorted(df.get(genre_col, pd.Series(dtype=str)).dropna().unique().tolist()) if genre_col in df.columns else [])
        with f2:
            cons_f = st.multiselect("Filter Console", options=sorted(df.get(console_col, pd.Series(dtype=str)).dropna().unique().tolist()) if console_col in df.columns else [])
        with f3:
            pub_f = st.multiselect("Filter Publisher", options=sorted(df.get(publisher_col, pd.Series(dtype=str)).dropna().unique().tolist()) if publisher_col in df.columns else [])
        with f4:
            years_series = pd.to_numeric(df.get('release_year', pd.Series(dtype=int)), errors='coerce')
            if years_series.dropna().empty:
                date_col = _resolve_column(df, ['release_date', 'date'])
                if date_col in df.columns:
                    years_series = pd.to_datetime(df[date_col], errors='coerce').dt.year
            min_y = int(years_series.dropna().min()) if not years_series.dropna().empty else 1980
            max_y = int(years_series.dropna().max()) if not years_series.dropna().empty else 2030
            year_range = st.slider("Release Year", min_value=min_y, max_value=max_y, value=(min_y, max_y), step=1, help="Select start and end year")

        dfx = df.copy()
        if genre_col in dfx.columns and genre_f:
            dfx = dfx[dfx[genre_col].isin(genre_f)]
        if console_col in dfx.columns and cons_f:
            dfx = dfx[dfx[console_col].isin(cons_f)]
        if publisher_col in dfx.columns and pub_f:
            dfx = dfx[dfx[publisher_col].isin(pub_f)]
        if 'year_range' in locals() and isinstance(year_range, (list, tuple)) and len(year_range) == 2:
            start_y, end_y = int(year_range[0]), int(year_range[1])
            if 'release_year' in dfx.columns:
                dfx = dfx[(pd.to_numeric(dfx['release_year'], errors='coerce') >= start_y) & (pd.to_numeric(dfx['release_year'], errors='coerce') <= end_y)]

        topn = st.slider("Top N", 5, 50, 10)

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
                if len(dfx) > 1000:
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
                base = alt.Chart(s).mark_bar().encode(
                    x=alt.X(f"{y_axis}:Q", title=y_axis.replace('_', ' ').title()),
                    y=alt.Y(f"{x_axis}:N", sort='-x', title=x_axis.replace('_', ' ').title()),
                    tooltip=list(s.columns)
                ).properties(height=400)

            if 'chart_specs' not in st.session_state:
                st.session_state.chart_specs = []

            add_col1, add_col2 = st.columns([1, 3])
            with add_col1:
                if st.button("Add Chart"):
                    data_to_store = dfx[[x_axis, y_axis]].to_dict(orient='list') if chart_type == 'scatter' else s.to_dict(orient='list')
                    release_year_filter = [int(year_range[0]), int(year_range[1])] if ('year_range' in locals() and isinstance(year_range, (list, tuple))) else None
                    st.session_state.chart_specs.append({
                        'x_axis': x_axis,
                        'y_axis': y_axis,
                        'agg_fn': agg_fn,
                        'chart_type': chart_type,
                        'topn': topn,
                        'filters': {
                            'genre': genre_f,
                            'console': cons_f,
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
