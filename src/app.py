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

# Centered title & caption (replaces default st.title / st.caption)
st.markdown(
    """
    <div style='text-align:center; padding: 0.75rem 0 0.25rem;'>
        <h1 style='margin-bottom:0.4rem; font-size:2.55rem;'>🎮 Video Game Success Prediction</h1>
            <p style='font-size:1.05rem; color:#5f6368; margin:0;'>Effortlessly predict which video games will become hits, explore market trends with interactive analytics, and forecast sales for multiple titles at once—all in a streamlined, user-friendly app.</p>
    </div>
    <hr style='margin-top:1.1rem; margin-bottom:0.8rem; border: none; border-top: 1px solid #e0e0e0;' />
    """,
    unsafe_allow_html=True
)


@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


project_root = Path(__file__).resolve().parents[1]
model_path = project_root / 'models' / 'best_model.joblib'
regressor_path = project_root / 'models' / 'best_regressor.joblib'

# Prefer data/vg_sales_2024.csv, fallback to data/raw/vg_sales_2024.csv
data_path = project_root / 'data' / 'vg_sales_2024.csv'
if not data_path.exists():
    data_path = project_root / 'data' / 'raw' / 'vg_sales_2024.csv'

model = None
if model_path.exists():
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load classification model: {e}")
else:
    st.warning("Classification model file not found. Train and save a model to 'models/best_model.joblib' first (run src/train.py).")

if model is not None and not hasattr(model, 'predict_proba'):
    st.info("Loaded model does not expose predict_proba; probability shown may be based on decision function or class label.")

# Load regression model (optional)
regressor = None
if regressor_path.exists():
    try:
        regressor = load_model(regressor_path)
    except Exception as e:
        st.error(f"Failed to load regression model: {e}")
else:
    st.info("Regression model not found. Run src/train_regression.py to predict total sales.")

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
    # Inject custom CSS for gradient buttons
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] button {
            background: linear-gradient(90deg, #000000 0%, #1a1a2e 60%, #232526 100%) !important;
            color: #fff !important;
            border: none !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        }
        section[data-testid="stSidebar"] button:disabled {
            background: #222 !important;
            color: #666 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Add logo at the top of sidebar
    logo_path = project_root / 'logo.png'
    if logo_path.exists():
        st.image(str(logo_path), width=200)
        st.markdown("---")
    
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


def predict_sales(regressor, genre: str, console: str, publisher: str, developer: str, critic_score: float, release_year: int) -> float:
    """Predict total_sales using regression model.
    
    Returns:
        float: predicted total sales in million units
    """
    if regressor is None:
        raise RuntimeError("Regressor is not loaded.")
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
    predicted_sales = regressor.predict(X)[0]
    predicted_sales = max(0.0, float(predicted_sales))  # Ensure non-negative
    
    return predicted_sales


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
    st.subheader("📊 Sales Analytics Dashboard")
    st.markdown("Explore video game sales data with interactive visualizations and insights")
    
    if df is None or df.empty:
        st.info("Dataset not loaded.")
    else:
        # Top control bar (removed redundant 'Sales Analysis' heading for cleaner UI)
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            chart_type = st.selectbox(
                "Chart Type", 
                options=["Bar Chart", "Horizontal Bar", "Pie Chart", "Donut Chart", "Line Chart"],
                index=0,
                help="Choose your preferred visualization style"
            )
        with col2:
            metric_type = st.selectbox(
                "Metric", 
                options=["Total Sales", "Average Sales", "Median Sales", "Game Count"],
                index=0,
                help="Select the metric to analyze"
            )
        with col3:
            st.markdown("<div style='text-align:right; padding-top:0.4rem; font-size:0.9rem; color:#666;'>Configure chart & metric</div>", unsafe_allow_html=True)

        st.markdown("<hr style='margin:0.4rem 0 0.9rem; border:none; border-top:1px solid #e5e5e5;' />", unsafe_allow_html=True)
        
        # Main controls
        col4, col5, col6 = st.columns([2, 1, 1])
        
        with col4:
            dim = st.selectbox(
                "Group by", 
                options=[("Genre", genre_col), ("Console", console_col), ("Publisher", publisher_col), ("Developer", developer_col)], 
                index=0, 
                format_func=lambda x: x[0],
                help="Select the category to analyze"
            )
        
        with col5:
            topn = st.slider("Top N", min_value=5, max_value=50, value=15, step=1, help="Number of top items to display")
        
        with col6:
            show_percentages = st.checkbox("Show Percentages", value=True, help="Display percentage values on charts")
        
        # Additional filters
        with st.expander("🔍 Advanced Filters", expanded=False):
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                year_range = st.slider(
                    "Release Year Range", 
                    min_value=1980, 
                    max_value=2025, 
                    value=(1990, 2025),
                    step=1,
                    help="Filter games by release year"
                )
            
            with filter_col2:
                min_sales = st.number_input(
                    "Minimum Sales (M)", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=0.0, 
                    step=0.1,
                    help="Filter out games with sales below this threshold"
                )
            
            with filter_col3:
                sort_order = st.selectbox(
                    "Sort Order", 
                    options=["Descending", "Ascending"], 
                    index=0,
                    help="Sort results in ascending or descending order"
                )
        
        dim_col = dim[1]
        if dim_col in df.columns and total_col in df.columns:
            # Apply filters
            df_filtered = df.copy()
            
            # Year filter
            if 'release_year' in df_filtered.columns:
                df_filtered = df_filtered[
                    (pd.to_numeric(df_filtered['release_year'], errors='coerce') >= year_range[0]) &
                    (pd.to_numeric(df_filtered['release_year'], errors='coerce') <= year_range[1])
                ]
            
            # Sales filter
            df_filtered = df_filtered[pd.to_numeric(df_filtered[total_col], errors='coerce') >= min_sales]
            
            if df_filtered.empty:
                st.warning("No data matches the selected filters. Try adjusting your criteria.")
            else:
                # Calculate metrics based on selection
                if metric_type == "Total Sales":
                    s = df_filtered.groupby(dim_col, dropna=False)[total_col].sum()
                elif metric_type == "Average Sales":
                    s = df_filtered.groupby(dim_col, dropna=False)[total_col].mean()
                elif metric_type == "Median Sales":
                    s = df_filtered.groupby(dim_col, dropna=False)[total_col].median()
                elif metric_type == "Game Count":
                    s = df_filtered.groupby(dim_col, dropna=False).size()
                
                # Sort data
                ascending = sort_order == "Ascending"
                s = s.sort_values(ascending=ascending).head(topn)
                
                if s.empty:
                    st.warning("No data available for the selected criteria.")
                else:
                    # Create visualizations based on chart type
                    if chart_type in ["Bar Chart", "Horizontal Bar"]:
                        # Use Altair for better interactivity
                        chart_data = pd.DataFrame({
                            dim[0]: s.index,
                            metric_type: s.values
                        })
                        
                        if chart_type == "Bar Chart":
                            chart = alt.Chart(chart_data).mark_bar(
                                cornerRadius=4,
                                stroke='white',
                                strokeWidth=1
                            ).encode(
                                x=alt.X(f'{metric_type}:Q', 
                                       title=metric_type,
                                       axis=alt.Axis(format='.2f')),
                                y=alt.Y(f'{dim[0]}:N', 
                                       sort=alt.EncodingSortField(field=metric_type, order='descending'),
                                       title=dim[0]),
                                color=alt.Color(f'{metric_type}:Q', 
                                              scale=alt.Scale(scheme='viridis'),
                                              legend=None),
                                tooltip=[
                                    alt.Tooltip(f'{dim[0]}:N', title=dim[0]),
                                    alt.Tooltip(f'{metric_type}:Q', title=metric_type, format='.2f')
                                ]
                            ).properties(
                                height=400,
                                width=600
                            )
                        else:  # Horizontal Bar
                            chart = alt.Chart(chart_data).mark_bar(
                                cornerRadius=4,
                                stroke='white',
                                strokeWidth=1
                            ).encode(
                                x=alt.X(f'{metric_type}:Q', 
                                       title=metric_type,
                                       axis=alt.Axis(format='.2f')),
                                y=alt.Y(f'{dim[0]}:N', 
                                       sort=alt.EncodingSortField(field=metric_type, order='descending'),
                                       title=dim[0]),
                                color=alt.Color(f'{metric_type}:Q', 
                                              scale=alt.Scale(scheme='plasma'),
                                              legend=None),
                                tooltip=[
                                    alt.Tooltip(f'{dim[0]}:N', title=dim[0]),
                                    alt.Tooltip(f'{metric_type}:Q', title=metric_type, format='.2f')
                                ]
                            ).properties(
                                height=max(300, len(s) * 25),
                                width=600
                            )
                        
                        st.altair_chart(chart, use_container_width=True)
                    
                    elif chart_type in ["Pie Chart", "Donut Chart"]:
                        # Create pie/donut chart
                        chart_data = pd.DataFrame({
                            dim[0]: s.index,
                            metric_type: s.values
                        })
                        
                        # Calculate percentages if requested
                        if show_percentages:
                            total = s.sum()
                            chart_data['Percentage'] = (chart_data[metric_type] / total * 100).round(1)
                            chart_data['Label'] = chart_data[dim[0]] + ' (' + chart_data['Percentage'].astype(str) + '%)'
                        else:
                            chart_data['Label'] = chart_data[dim[0]]
                        
                        if chart_type == "Pie Chart":
                            chart = alt.Chart(chart_data).mark_arc(
                                innerRadius=0,
                                outerRadius=150,
                                stroke='white',
                                strokeWidth=2
                            ).encode(
                                theta=alt.Theta(f'{metric_type}:Q'),
                                color=alt.Color(f'{dim[0]}:N', 
                                              scale=alt.Scale(scheme='set3')),
                                tooltip=[
                                    alt.Tooltip(f'{dim[0]}:N', title=dim[0]),
                                    alt.Tooltip(f'{metric_type}:Q', title=metric_type, format='.2f')
                                ]
                            ).properties(
                                width=400,
                                height=400
                            )
                        else:  # Donut Chart
                            chart = alt.Chart(chart_data).mark_arc(
                                innerRadius=60,
                                outerRadius=150,
                                stroke='white',
                                strokeWidth=2
                            ).encode(
                                theta=alt.Theta(f'{metric_type}:Q'),
                                color=alt.Color(f'{dim[0]}:N', 
                                              scale=alt.Scale(scheme='set3')),
                                tooltip=[
                                    alt.Tooltip(f'{dim[0]}:N', title=dim[0]),
                                    alt.Tooltip(f'{metric_type}:Q', title=metric_type, format='.2f')
                                ]
                            ).properties(
                                width=400,
                                height=400
                            )
                        
                        # Add text labels
                        text = alt.Chart(chart_data).mark_text(
                            align='center',
                            baseline='middle',
                            fontSize=12,
                            fontWeight='bold'
                        ).encode(
                            text=alt.Text('Label:N'),
                            color=alt.value('white')
                        )
                        
                        final_chart = (chart + text).resolve_scale(color='independent')
                        st.altair_chart(final_chart, use_container_width=True)
                    
                    elif chart_type == "Line Chart":
                        # Create line chart for trend analysis
                        chart_data = pd.DataFrame({
                            dim[0]: s.index,
                            metric_type: s.values
                        })
                        
                        chart = alt.Chart(chart_data).mark_line(
                            point=True,
                            strokeWidth=3,
                            stroke='#1f77b4'
                        ).encode(
                            x=alt.X(f'{dim[0]}:N', 
                                   title=dim[0],
                                   axis=alt.Axis(angle=-45)),
                            y=alt.Y(f'{metric_type}:Q', 
                                   title=metric_type,
                                   axis=alt.Axis(format='.2f')),
                            tooltip=[
                                alt.Tooltip(f'{dim[0]}:N', title=dim[0]),
                                alt.Tooltip(f'{metric_type}:Q', title=metric_type, format='.2f')
                            ]
                        ).properties(
                            height=400,
                            width=600
                        )
                        
                        st.altair_chart(chart, use_container_width=True)
                    
                    # Display summary statistics
                    st.markdown("---")
                    col7, col8, col9, col10 = st.columns(4)
                    
                    with col7:
                        st.metric(
                            "Total Items", 
                            f"{len(s):,}",
                            help="Number of categories shown"
                        )
                    
                    with col8:
                        total_value = s.sum() if metric_type != "Game Count" else s.sum()
                        st.metric(
                            "Total Value", 
                            f"{total_value:,.2f}",
                            help=f"Sum of all {metric_type.lower()}"
                        )
                    
                    with col9:
                        avg_value = s.mean()
                        st.metric(
                            "Average", 
                            f"{avg_value:,.2f}",
                            help=f"Average {metric_type.lower()}"
                        )
                    
                    with col10:
                        max_value = s.max()
                        st.metric(
                            "Highest", 
                            f"{max_value:,.2f}",
                            help=f"Highest {metric_type.lower()}"
                        )
                    
                    # Display data table
                    st.markdown("### 📋 Detailed Data")
                    display_data = pd.DataFrame({
                        'Rank': range(1, len(s) + 1),
                        dim[0]: s.index,
                        metric_type: s.values
                    })
                    
                    if show_percentages and metric_type != "Game Count":
                        total = s.sum()
                        display_data['Percentage'] = (display_data[metric_type] / total * 100).round(2)
                    
                    st.dataframe(
                        display_data, 
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download option
                    csv = display_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Data as CSV",
                        data=csv,
                        file_name=f"{dim[0].lower()}_{metric_type.lower().replace(' ', '_')}_analysis.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("Required columns for this analysis were not found in the dataset.")

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
        # Removed explicit 'Prediction Inputs' title for a cleaner minimal UI
        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
        genre = st.selectbox("Genre", options=genre_options, index=0)
        console = st.selectbox("Console/Platform", options=console_options, index=0)
        publisher = st.selectbox("Publisher", options=publisher_options, index=0)
        developer = st.selectbox("Developer", options=developer_options, index=0)
        critic_score = st.number_input("Critic Score (0-10)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        
        if st.button("Predict", type="primary"):
            try:
                # Classification prediction
                classification_pred = None
                classification_label = None
                classification_proba = None
                
                if model is not None:
                    pred, proba = predict_hit(model, genre, console, publisher, developer, critic_score, default_release_year)
                    classification_label = "Hit" if pred == 1 else "Not Hit"
                    classification_proba = proba
                    
                    st.markdown("#### 🎯 Hit Classification Model")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prediction", classification_label)
                    with col2:
                        st.metric("Probability", f"{proba:.2%}")
                    st.progress(min(max(proba, 0.0), 1.0), text=f"P(Hit) = {proba:.2%}")
                    st.caption("Classification model predicts if total sales ≥ 1.0M units")
                
                # Regression prediction
                if regressor is not None:
                    predicted_sales = predict_sales(regressor, genre, console, publisher, developer, critic_score, default_release_year)
                    
                    st.markdown("#### 📊 Regression Sales Prediction")
                    st.metric("Predicted Sales", f"{predicted_sales:.2f}M units")
                    st.caption("Regression model predicts the total sales in millions of units")
                    
                    # Show insights based on both predictions
                    if classification_label:
                        st.markdown("---")
                        st.markdown("#### � Combined Insights")
                        
                        if classification_label == "Hit":
                            st.success(f"✅ **Classification predicts Hit** with {classification_proba:.1%} probability")
                            st.write(f"📊 Expected sales: **{predicted_sales:.2f}M units**")
                            if predicted_sales >= 1.5:
                                st.write("💪 Strong sales potential - well above threshold")
                            elif predicted_sales >= 1.0:
                                st.write("✓ Moderate sales potential - near threshold")
                            else:
                                st.write("⚠️ Sales estimate below typical Hit threshold (1.0M)")
                        else:
                            st.info(f"📊 **Classification predicts Not Hit** ({classification_proba:.1%} probability)")
                            st.write(f"📊 Expected sales: **{predicted_sales:.2f}M units**")
                            if predicted_sales >= 0.8:
                                st.write("💡 Close to threshold - niche success possible")
                            else:
                                st.write("📉 Lower sales expected")
                else:
                    st.info("💡 Run `src/train_regression.py` to get total sales predictions")
                    
                # Append this prediction into batch table automatically
                if 'batch_rows' not in st.session_state:
                    st.session_state.batch_rows = pd.DataFrame(columns=[
                        'genre', 'console', 'publisher', 'developer', 'critic_score', 'release_year',
                        'p_hit', 'label', 'predicted_sales'
                    ])
                new_entry = {
                    'genre': genre,
                    'console': console,
                    'publisher': publisher,
                    'developer': developer,
                    'critic_score': critic_score,
                    'release_year': default_release_year,
                    'p_hit': classification_proba if classification_proba is not None else float('nan'),
                    'label': classification_label if classification_label else None,
                    'predicted_sales': predicted_sales if 'predicted_sales' in locals() else float('nan')
                }
                st.session_state.batch_rows = pd.concat([st.session_state.batch_rows, pd.DataFrame([new_entry])], ignore_index=True)

            except Exception as e:
                st.error(str(e))

    with ci2:
        st.markdown("### Batch Prediction")
        # Initialize batch storage in session state
        if 'batch_rows' not in st.session_state:
            st.session_state.batch_rows = pd.DataFrame(columns=[
                'genre', 'console', 'publisher', 'developer', 'critic_score', 'release_year',
                'p_hit', 'label', 'predicted_sales'
            ])

        def _recompute_predictions(df_in: pd.DataFrame) -> pd.DataFrame:
            df_proc = df_in.copy()
            if df_proc.empty:
                return df_proc
            # Normalize inputs
            for col in ['genre', 'console', 'publisher', 'developer']:
                if col in df_proc.columns:
                    df_proc[col] = df_proc[col].astype('string').str.strip().str.lower()
            if 'critic_score' in df_proc.columns:
                df_proc['critic_score'] = pd.to_numeric(df_proc['critic_score'], errors='coerce').clip(0, 10)
            if 'release_year' not in df_proc.columns:
                df_proc['release_year'] = default_release_year
            else:
                df_proc['release_year'] = pd.to_numeric(df_proc['release_year'], errors='coerce').fillna(default_release_year).astype(int)

            if model is not None:
                try:
                    preds = model.predict_proba(df_proc)[:, 1] if hasattr(model, 'predict_proba') else model.predict(df_proc).astype(float)
                except Exception:
                    preds = [float('nan')] * len(df_proc)
            else:
                preds = [float('nan')] * len(df_proc)
            labels = [(p >= 0.5) if pd.notna(p) else False for p in preds]

            # Regression predictions
            if regressor is not None:
                try:
                    sales_preds = regressor.predict(df_proc)
                except Exception:
                    sales_preds = [float('nan')] * len(df_proc)
            else:
                sales_preds = [float('nan')] * len(df_proc)

            df_proc['p_hit'] = preds
            df_proc['label'] = ["Hit" if l else "Not Hit" for l in labels]
            df_proc['predicted_sales'] = sales_preds
            return df_proc

        # Button to add current single-input selection as a new row
        # Editable view (exclude prediction columns from editing)
        editable_cols = ['genre', 'console', 'publisher', 'developer', 'critic_score', 'release_year']
        col_config = {
            'p_hit': st.column_config.NumberColumn("P(Hit)", format="%.3f", disabled=True),
            'label': st.column_config.TextColumn("Label", disabled=True),
            'predicted_sales': st.column_config.NumberColumn("Predicted Sales (M)", format="%.2f", disabled=True)
        }
        batch_editor_df = st.data_editor(
            st.session_state.batch_rows,
            use_container_width=True,
            num_rows="dynamic",
            column_config=col_config,
            disabled=[c for c in st.session_state.batch_rows.columns if c not in editable_cols]
        )

        # Detect manual edits and recompute predictions
        if not batch_editor_df.equals(st.session_state.batch_rows):
            # Keep only editable columns from edited df, then recompute
            merged = batch_editor_df[editable_cols].copy()
            st.session_state.batch_rows = _recompute_predictions(merged)
            # Force immediate refresh so predictions update without extra click
            st.experimental_rerun()
        # Download capability (always reflects current predictions in editor)
        if not st.session_state.batch_rows.empty:
            dl_csv = st.session_state.batch_rows.to_csv(index=False)
            st.download_button(
                label="Download Batch Predictions CSV",
                data=dl_csv,
                file_name="batch_predictions.csv",
                mime="text/csv"
            )

    st.divider()
    st.markdown("### Predict Years to Hit")
    st.caption("Estimate years needed to reach Hit based on the average yearly sales of similar games. Assumes if game Hit 1 Million sells it will be Hit and starting sales = 0.")

    genre_sel = st.selectbox("Genre ", options=["Any"] + genre_options, index=0)
    console_sel = st.selectbox("Console ", options=["Any"] + console_options, index=0)
    publisher_sel = st.selectbox("Publisher ", options=["Any"] + publisher_options, index=0)
    developer_sel = st.selectbox("Developer ", options=["Any"] + developer_options, index=0)

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
                            # Show only average yearly sales and estimated years (hide assumed threshold per request)
                            cya, cyc = st.columns(2)
                            with cya:
                                st.metric("Avg yearly sales", f"{avg_rate:.3f}")
                            with cyc:
                                st.metric("Estimated years to Hit", f"{years_needed}")
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
            st.markdown("### 🌍 Regional Sales Analysis")
            st.markdown("Compare sales performance across different regions")
            
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
                    # Create interactive chart with Altair
                    chart_data = pd.DataFrame({
                        'Region': [str(col).replace("_", " ").title() for col in agg.index],
                        'Sales': agg.values,
                        'Percentage': (agg.values / agg.sum() * 100).round(1)
                    })
                    
                    # Chart type selection
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        region_chart_type = st.selectbox(
                            "Chart Type",
                            options=["Bar Chart", "Pie Chart", "Donut Chart", "Horizontal Bar"],
                            index=0,
                            key="region_chart_type"
                        )
                    
                    with col2:
                        show_region_percentages = st.checkbox("Show Percentages", value=True, key="region_percentages")
                    
                    if region_chart_type == "Bar Chart":
                        chart = alt.Chart(chart_data).mark_bar(
                            cornerRadius=6,
                            stroke='white',
                            strokeWidth=2
                        ).encode(
                            x=alt.X('Sales:Q', 
                                   title="Total Sales (Millions)",
                                   axis=alt.Axis(format='.1f')),
                            y=alt.Y('Region:N', 
                                   sort='-x',
                                   title="Region"),
                            color=alt.Color('Sales:Q', 
                                          scale=alt.Scale(scheme='viridis'),
                                          legend=None),
                            tooltip=[
                                alt.Tooltip('Region:N', title="Region"),
                                alt.Tooltip('Sales:Q', title="Sales (M)", format='.2f'),
                                alt.Tooltip('Percentage:Q', title="Percentage", format='.1f')
                            ]
                        ).properties(
                            height=400,
                            width=600
                        )
                        st.altair_chart(chart, use_container_width=True)
                    
                    elif region_chart_type == "Horizontal Bar":
                        chart = alt.Chart(chart_data).mark_bar(
                            cornerRadius=6,
                            stroke='white',
                            strokeWidth=2
                        ).encode(
                            x=alt.X('Sales:Q', 
                                   title="Total Sales (Millions)",
                                   axis=alt.Axis(format='.1f')),
                            y=alt.Y('Region:N', 
                                   sort='-x',
                                   title="Region"),
                            color=alt.Color('Sales:Q', 
                                          scale=alt.Scale(scheme='plasma'),
                                          legend=None),
                            tooltip=[
                                alt.Tooltip('Region:N', title="Region"),
                                alt.Tooltip('Sales:Q', title="Sales (M)", format='.2f'),
                                alt.Tooltip('Percentage:Q', title="Percentage", format='.1f')
                            ]
                        ).properties(
                            height=max(300, len(chart_data) * 30),
                            width=600
                        )
                        st.altair_chart(chart, use_container_width=True)
                    
                    elif region_chart_type in ["Pie Chart", "Donut Chart"]:
                        # Prepare labels with percentages
                        if show_region_percentages:
                            chart_data['Label'] = chart_data['Region'] + ' (' + chart_data['Percentage'].astype(str) + '%)'
                        else:
                            chart_data['Label'] = chart_data['Region']
                        
                        if region_chart_type == "Pie Chart":
                            chart = alt.Chart(chart_data).mark_arc(
                                innerRadius=0,
                                outerRadius=150,
                                stroke='white',
                                strokeWidth=3
                            ).encode(
                                theta=alt.Theta('Sales:Q'),
                                color=alt.Color('Region:N', 
                                              scale=alt.Scale(scheme='set3')),
                                tooltip=[
                                    alt.Tooltip('Region:N', title="Region"),
                                    alt.Tooltip('Sales:Q', title="Sales (M)", format='.2f'),
                                    alt.Tooltip('Percentage:Q', title="Percentage", format='.1f')
                                ]
                            ).properties(
                                width=500,
                                height=500
                            )
                        else:  # Donut Chart
                            chart = alt.Chart(chart_data).mark_arc(
                                innerRadius=80,
                                outerRadius=150,
                                stroke='white',
                                strokeWidth=3
                            ).encode(
                                theta=alt.Theta('Sales:Q'),
                                color=alt.Color('Region:N', 
                                              scale=alt.Scale(scheme='set3')),
                                tooltip=[
                                    alt.Tooltip('Region:N', title="Region"),
                                    alt.Tooltip('Sales:Q', title="Sales (M)", format='.2f'),
                                    alt.Tooltip('Percentage:Q', title="Percentage", format='.1f')
                                ]
                            ).properties(
                                width=500,
                                height=500
                            )
                        
                        # Add text labels
                        text = alt.Chart(chart_data).mark_text(
                            align='center',
                            baseline='middle',
                            fontSize=11,
                            fontWeight='bold'
                        ).encode(
                            text=alt.Text('Label:N'),
                            color=alt.value('white')
                        )
                        
                        final_chart = (chart + text).resolve_scale(color='independent')
                        st.altair_chart(final_chart, use_container_width=True)
                    
                    # Summary metrics
                    st.markdown("---")
                    col3, col4, col5, col6 = st.columns(4)
                    
                    with col3:
                        st.metric("Total Regions", f"{len(agg)}")
                    
                    with col4:
                        st.metric("Total Sales", f"{agg.sum():.1f}M")
                    
                    with col5:
                        st.metric("Top Region", f"{chart_data.iloc[0]['Region']}")
                    
                    with col6:
                        st.metric("Top Region %", f"{chart_data.iloc[0]['Percentage']:.1f}%")
                    
                    # Data table
                    st.markdown("### 📊 Regional Sales Breakdown")
                    st.dataframe(
                        chart_data[['Region', 'Sales', 'Percentage']], 
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download option
                    csv = chart_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Regional Data as CSV",
                        data=csv,
                        file_name="regional_sales_analysis.csv",
                        mime="text/csv"
                    )
                    
                    st.caption(f"Using region columns: {', '.join(map(str, agg.index))}")
                else:
                    st.info("No regional sales data to aggregate.")
            else:
                st.info("No region-specific sales columns found.")

        elif sub == "Correlation Heatmap":
            st.markdown("### 🔥 Correlation Analysis")
            st.markdown("Explore relationships between numeric variables")
            
            num_df = df.select_dtypes(include="number")
            if num_df.shape[1] >= 2:
                corr = num_df.corr(numeric_only=True)
                
                # Chart options
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    show_values = st.checkbox("Show Values", value=True, key="corr_values")
                with col2:
                    show_annotations = st.checkbox("Show Annotations", value=False, key="corr_annotations")
                
                # Create interactive heatmap with Altair
                corr_data = corr.reset_index().melt(id_vars='index', var_name='Variable 2', value_name='Correlation')
                corr_data.columns = ['Variable 1', 'Variable 2', 'Correlation']
                
                # Create heatmap
                heatmap = alt.Chart(corr_data).mark_rect(
                    stroke='white',
                    strokeWidth=1
                ).encode(
                    x=alt.X('Variable 2:N', title="Variable 2"),
                    y=alt.Y('Variable 1:N', title="Variable 1"),
                    color=alt.Color('Correlation:Q', 
                                  scale=alt.Scale(domain=[-1, 1], scheme='redblue'),
                                  legend=alt.Legend(title="Correlation")),
                    tooltip=[
                        alt.Tooltip('Variable 1:N', title="Variable 1"),
                        alt.Tooltip('Variable 2:N', title="Variable 2"),
                        alt.Tooltip('Correlation:Q', title="Correlation", format='.3f')
                    ]
                ).properties(
                    width=500,
                    height=500
                )
                
                # Add text annotations if requested
                if show_annotations:
                    text = alt.Chart(corr_data).mark_text(
                        align='center',
                        baseline='middle',
                        fontSize=10,
                        fontWeight='bold'
                    ).encode(
                        x=alt.X('Variable 2:N'),
                        y=alt.Y('Variable 1:N'),
                        text=alt.Text('Correlation:Q', format='.2f'),
                        color=alt.condition(
                            alt.datum.Correlation > 0.5,
                            alt.value('white'),
                            alt.value('black')
                        )
                    )
                    final_heatmap = heatmap + text
                else:
                    final_heatmap = heatmap
                
                st.altair_chart(final_heatmap, use_container_width=True)
                
                # Summary statistics
                st.markdown("---")
                col4, col5, col6 = st.columns(3)
                
                with col4:
                    strong_corr = corr_data[(corr_data['Correlation'].abs() > 0.7) & (corr_data['Correlation'] != 1.0)]
                    st.metric("Strong Correlations (|r| > 0.7)", f"{len(strong_corr)}")
                
                with col5:
                    positive_corr = corr_data[(corr_data['Correlation'] > 0.5) & (corr_data['Correlation'] != 1.0)]
                    st.metric("Positive Correlations (r > 0.5)", f"{len(positive_corr)}")
                
                with col6:
                    negative_corr = corr_data[(corr_data['Correlation'] < -0.5) & (corr_data['Correlation'] != 1.0)]
                    st.metric("Negative Correlations (r < -0.5)", f"{len(negative_corr)}")
                
                # Display correlation table
                if show_values:
                    st.markdown("### 📊 Correlation Matrix")
                    st.dataframe(
                        corr.round(3), 
                        use_container_width=True
                    )
                
                # Download option
                csv = corr.to_csv()
                st.download_button(
                    label="📥 Download Correlation Matrix as CSV",
                    data=csv,
                    file_name="correlation_matrix.csv",
                    mime="text/csv"
                )
            else:
                st.info("Not enough numeric columns for correlation analysis.")

        elif sub == "Feature Importance (model)":
            st.markdown("### 🎯 Feature Importance Analysis")
            st.markdown("Understand which features are most important for predicting game success")
            
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
                        
                        # Controls
                        col1, col2, col3 = st.columns([1, 1, 2])
                        with col1:
                            topn = st.slider("Top N features", 5, 40, 20, key="feature_topn")
                        with col2:
                            chart_type = st.selectbox(
                                "Chart Type",
                                options=["Horizontal Bar", "Vertical Bar", "Pie Chart"],
                                index=0,
                                key="feature_chart_type"
                            )
                        with col3:
                            show_values = st.checkbox("Show Values", value=True, key="feature_values")
                        
                        sel = order[:topn]
                        
                        # Prepare data
                        feature_data = pd.DataFrame({
                            'Feature': feat_names[sel],
                            'Importance': importances[sel],
                            'Percentage': (importances[sel] / importances[sel].sum() * 100).round(2)
                        })
                        
                        if chart_type == "Horizontal Bar":
                            chart = alt.Chart(feature_data).mark_bar(
                                cornerRadius=4,
                                stroke='white',
                                strokeWidth=1
                            ).encode(
                                x=alt.X('Importance:Q', 
                                       title="Feature Importance",
                                       axis=alt.Axis(format='.3f')),
                                y=alt.Y('Feature:N', 
                                       sort='-x',
                                       title="Feature"),
                                color=alt.Color('Importance:Q', 
                                              scale=alt.Scale(scheme='viridis'),
                                              legend=None),
                                tooltip=[
                                    alt.Tooltip('Feature:N', title="Feature"),
                                    alt.Tooltip('Importance:Q', title="Importance", format='.4f'),
                                    alt.Tooltip('Percentage:Q', title="Percentage", format='.1f')
                                ]
                            ).properties(
                                height=max(300, len(feature_data) * 25),
                                width=600
                            )
                            st.altair_chart(chart, use_container_width=True)
                        
                        elif chart_type == "Vertical Bar":
                            chart = alt.Chart(feature_data).mark_bar(
                                cornerRadius=4,
                                stroke='white',
                                strokeWidth=1
                            ).encode(
                                x=alt.X('Feature:N', 
                                       sort='-y',
                                       title="Feature",
                                       axis=alt.Axis(angle=-45)),
                                y=alt.Y('Importance:Q', 
                                       title="Feature Importance",
                                       axis=alt.Axis(format='.3f')),
                                color=alt.Color('Importance:Q', 
                                              scale=alt.Scale(scheme='plasma'),
                                              legend=None),
                                tooltip=[
                                    alt.Tooltip('Feature:N', title="Feature"),
                                    alt.Tooltip('Importance:Q', title="Importance", format='.4f'),
                                    alt.Tooltip('Percentage:Q', title="Percentage", format='.1f')
                                ]
                            ).properties(
                                height=400,
                                width=600
                            )
                            st.altair_chart(chart, use_container_width=True)
                        
                        elif chart_type == "Pie Chart":
                            # Prepare labels
                            if show_values:
                                feature_data['Label'] = feature_data['Feature'] + ' (' + feature_data['Percentage'].astype(str) + '%)'
                            else:
                                feature_data['Label'] = feature_data['Feature']
                            
                            chart = alt.Chart(feature_data).mark_arc(
                                innerRadius=0,
                                outerRadius=150,
                                stroke='white',
                                strokeWidth=2
                            ).encode(
                                theta=alt.Theta('Importance:Q'),
                                color=alt.Color('Feature:N', 
                                              scale=alt.Scale(scheme='set3')),
                                tooltip=[
                                    alt.Tooltip('Feature:N', title="Feature"),
                                    alt.Tooltip('Importance:Q', title="Importance", format='.4f'),
                                    alt.Tooltip('Percentage:Q', title="Percentage", format='.1f')
                                ]
                            ).properties(
                                width=500,
                                height=500
                            )
                            
                            # Add text labels
                            text = alt.Chart(feature_data).mark_text(
                                align='center',
                                baseline='middle',
                                fontSize=10,
                                fontWeight='bold'
                            ).encode(
                                text=alt.Text('Label:N'),
                                color=alt.value('white')
                            )
                            
                            final_chart = (chart + text).resolve_scale(color='independent')
                            st.altair_chart(final_chart, use_container_width=True)
                        
                        # Summary statistics
                        st.markdown("---")
                        col4, col5, col6 = st.columns(3)
                        
                        with col4:
                            st.metric("Total Features", f"{len(feature_data)}")
                        
                        with col5:
                            top_feature = feature_data.iloc[0]
                            st.metric("Top Feature", f"{top_feature['Feature']}")
                        
                        with col6:
                            st.metric("Top Feature %", f"{top_feature['Percentage']:.1f}%")
                        
                        # Feature importance table
                        if show_values:
                            st.markdown("### 📊 Feature Importance Details")
                            st.dataframe(
                                feature_data[['Feature', 'Importance', 'Percentage']], 
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        # Download option
                        csv = feature_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Feature Importance as CSV",
                            data=csv,
                            file_name="feature_importance_analysis.csv",
                            mime="text/csv"
                        )
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
