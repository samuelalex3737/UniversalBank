import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")


# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(
    page_title="Bank Personal Loan Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A clean, consistent Plotly theme
PLOTLY_TEMPLATE = "plotly_white"


# -----------------------------
# Utilities
# -----------------------------
def _find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first candidate column that exists in df (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce columns to numeric when possible."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _clean_zipcode(series: pd.Series) -> pd.Series:
    """Zipcode sometimes comes as int/float; keep as string for grouping."""
    if series is None:
        return series
    s = series.copy()
    # Convert to string safely; preserve missing
    s = s.apply(lambda x: np.nan if pd.isna(x) else str(int(x)) if str(x).replace(".", "", 1).isdigit() else str(x))
    return s


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardize common column name variants without hard-coding one dataset format
    # We'll map known variants if present
    rename_map = {}
    # Typical UCI/Bank dataset has: "Personal Loan", "Securities Account", "CD Account", "Online", "CreditCard"
    # Some variants may use underscores. We'll normalize by stripping spaces for internal logic later.
    for col in df.columns:
        # Trim whitespace in column names
        new_col = col.strip()
        if new_col != col:
            rename_map[col] = new_col
    if rename_map:
        df = df.rename(columns=rename_map)

    # Coerce typical numeric columns if present
    numeric_candidates = [
        "Age", "Experience", "Income", "CCAvg", "Mortgage",
        "Family", "Education", "ZIP Code", "Zip Code", "Zipcode",
        "Personal Loan", "Securities Account", "CD Account", "Online", "CreditCard"
    ]
    existing_numeric = [c for c in numeric_candidates if c in df.columns]
    df = _safe_numeric(df, existing_numeric)

    # Handle zipcode-like column if present
    zip_col = _find_first_existing_column(df, ["ZIP Code", "Zip Code", "Zipcode", "ZIPCODE", "Zip"])
    if zip_col:
        df[zip_col] = _clean_zipcode(df[zip_col])

    # Basic missing value handling
    # - numeric: fill with median
    # - categorical/object: fill with mode (or "Unknown" if no mode)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].isna().any():
                mode = df[col].mode()
                df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "Unknown")

    return df


def infer_columns(df: pd.DataFrame) -> dict:
    """
    Infer key columns robustly. Returns a dict with expected keys.
    Raises a helpful error if critical columns are missing.
    """
    col = {}

    col["personal_loan"] = _find_first_existing_column(df, ["Personal Loan", "PersonalLoan", "Loan", "Personal_Loan"])
    col["income"] = _find_first_existing_column(df, ["Income", "Annual Income", "income"])
    col["age"] = _find_first_existing_column(df, ["Age", "age"])
    col["ccavg"] = _find_first_existing_column(df, ["CCAvg", "CC Avg", "CreditCardAvg", "CC_Avg"])
    col["education"] = _find_first_existing_column(df, ["Education", "education"])
    col["family"] = _find_first_existing_column(df, ["Family", "family", "Family Size", "FamilySize"])
    col["mortgage"] = _find_first_existing_column(df, ["Mortgage", "mortgage"])
    col["zipcode"] = _find_first_existing_column(df, ["ZIP Code", "Zip Code", "Zipcode", "ZIPCODE", "Zip"])

    col["securities"] = _find_first_existing_column(df, ["Securities Account", "SecuritiesAccount", "Securities_Account"])
    col["cd"] = _find_first_existing_column(df, ["CD Account", "CDAccount", "CD_Account"])
    col["creditcard"] = _find_first_existing_column(df, ["CreditCard", "Credit Card", "Credit_Card"])

    # Validate required columns for the assignment visualizations
    required = ["personal_loan", "income", "age", "ccavg"]
    missing = [k for k in required if col.get(k) is None]
    if missing:
        raise ValueError(
            "Dataset missing required columns for the dashboard: "
            + ", ".join(missing)
            + "\nFound columns: "
            + ", ".join(df.columns)
        )

    return col


def apply_filters(df: pd.DataFrame, cols: dict) -> Tuple[pd.DataFrame, dict]:
    """Sidebar filters: personal loan, education, family size."""
    with st.sidebar:
        st.title("Filters")

        # Personal Loan status filter
        loan_col = cols["personal_loan"]
        loan_vals = sorted(df[loan_col].dropna().unique().tolist())
        # Force to 0/1 style display if applicable
        loan_display = loan_vals
        default_loan = loan_vals  # all

        selected_loan = st.multiselect(
            "Personal Loan Status",
            options=loan_display,
            default=default_loan,
            help="Filter customers by whether they accepted the Personal Loan.",
        )

        # Education filter (if present)
        edu_col = cols.get("education")
        selected_edu = None
        if edu_col:
            edu_vals = sorted(df[edu_col].dropna().unique().tolist())
            selected_edu = st.multiselect(
                "Education",
                options=edu_vals,
                default=edu_vals,
                help="Filter by education level.",
            )

        # Family size filter (if present)
        fam_col = cols.get("family")
        selected_fam = None
        if fam_col:
            fam_vals = sorted(df[fam_col].dropna().unique().tolist())
            selected_fam = st.multiselect(
                "Family Size",
                options=fam_vals,
                default=fam_vals,
                help="Filter by family size.",
            )

    fdf = df.copy()
    if selected_loan is not None:
        fdf = fdf[fdf[loan_col].isin(selected_loan)]
    if cols.get("education") and selected_edu is not None:
        fdf = fdf[fdf[cols["education"]].isin(selected_edu)]
    if cols.get("family") and selected_fam is not None:
        fdf = fdf[fdf[cols["family"]].isin(selected_fam)]

    selections = {
        "loan": selected_loan,
        "education": selected_edu,
        "family": selected_fam,
    }
    return fdf, selections


def kpi_row(df: pd.DataFrame, cols: dict):
    loan_col = cols["personal_loan"]
    income_col = cols["income"]

    total = len(df)
    accept_rate = (df[loan_col].mean() * 100) if total else 0.0
    avg_income = df[income_col].mean() if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers (filtered)", f"{total:,}")
    c2.metric("Personal Loan Acceptance Rate", f"{accept_rate:.1f}%")
    c3.metric("Average Income", f"{avg_income:,.2f}")
    # Optional: mortgage avg if present
    if cols.get("mortgage"):
        c4.metric("Average Mortgage", f"{df[cols['mortgage']].mean():,.2f}")
    else:
        c4.metric("Numeric Columns", f"{df.select_dtypes(include=np.number).shape[1]}")


# -----------------------------
# Plot Builders
# -----------------------------
def plot_income_age_histograms(df: pd.DataFrame, cols: dict):
    loan_col, income_col, age_col = cols["personal_loan"], cols["income"], cols["age"]

    # Ensure we have both classes for meaningful comparison
    # Use overlay with transparency
    fig_income = px.histogram(
        df,
        x=income_col,
        color=loan_col,
        barmode="overlay",
        nbins=30,
        opacity=0.65,
        template=PLOTLY_TEMPLATE,
        title="Income Distribution (Count) by Personal Loan Status",
        labels={income_col: "Income", loan_col: "Personal Loan"},
    )
    fig_income.update_layout(legend_title_text="Personal Loan", bargap=0.05)

    fig_age = px.histogram(
        df,
        x=age_col,
        color=loan_col,
        barmode="overlay",
        nbins=30,
        opacity=0.65,
        template=PLOTLY_TEMPLATE,
        title="Age Distribution (Count) by Personal Loan Status",
        labels={age_col: "Age", loan_col: "Personal Loan"},
    )
    fig_age.update_layout(legend_title_text="Personal Loan", bargap=0.05)

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_income, use_container_width=True)
    c2.plotly_chart(fig_age, use_container_width=True)


def plot_scatter_ccavg_income(df: pd.DataFrame, cols: dict):
    loan_col, income_col, ccavg_col = cols["personal_loan"], cols["income"], cols["ccavg"]

    fig = px.scatter(
        df,
        x=income_col,
        y=ccavg_col,
        color=loan_col,
        template=PLOTLY_TEMPLATE,
        title="CCAvg vs Income (Colored by Personal Loan Status)",
        labels={income_col: "Income", ccavg_col: "CCAvg (Avg Credit Card Spending)", loan_col: "Personal Loan"},
        opacity=0.75,
        hover_data=[c for c in [cols.get("education"), cols.get("family"), cols.get("mortgage")] if c],
    )
    fig.update_traces(marker=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)


def plot_zip_income_loan_multivariate(df: pd.DataFrame, cols: dict):
    zip_col = cols.get("zipcode")
    income_col = cols["income"]
    loan_col = cols["personal_loan"]

    if not zip_col:
        st.info("Zipcode column not found in dataset; skipping Zipcode multivariate analysis.")
        return

    # Aggregate by zipcode for clarity
    agg = (
        df.groupby(zip_col)
        .agg(
            avg_income=(income_col, "mean"),
            customers=(income_col, "size"),
            loan_rate=(loan_col, "mean"),
        )
        .reset_index()
        .rename(columns={zip_col: "Zipcode"})
    )
    # Keep top zipcodes by customers to avoid overly dense plot
    agg = agg.sort_values("customers", ascending=False).head(60)

    fig = px.scatter(
        agg,
        x="Zipcode",
        y="avg_income",
        size="customers",
        color="loan_rate",
        color_continuous_scale="RdYlGn",
        template=PLOTLY_TEMPLATE,
        title="Zipcode vs Avg Income vs Personal Loan (size=customers, color=loan rate)",
        labels={"avg_income": "Average Income", "loan_rate": "Loan Acceptance Rate"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        st.info("Not enough numeric columns for correlation heatmap.")
        return

    corr = num_df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        template=PLOTLY_TEMPLATE,
        title="Correlation Heatmap (Numeric Columns)",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Correlation"))
    st.plotly_chart(fig, use_container_width=True)


def plot_family_analysis(df: pd.DataFrame, cols: dict):
    fam_col = cols.get("family")
    income_col = cols["income"]
    mortgage_col = cols.get("mortgage")
    ccavg_col = cols["ccavg"]
    loan_col = cols["personal_loan"]

    if not fam_col:
        st.info("Family column not found in dataset; skipping Family Analysis.")
        return

    # Family vs Income segmented by loan
    fig1 = px.box(
        df,
        x=fam_col,
        y=income_col,
        color=loan_col,
        template=PLOTLY_TEMPLATE,
        title="Family Size vs Income (Segmented by Personal Loan)",
        labels={fam_col: "Family Size", income_col: "Income", loan_col: "Personal Loan"},
    )

    # Bubble: Family vs Income with mortgage/ccavg encoding
    # Use whichever additional measure exists; prefer mortgage for size and ccavg for hover
    hover_cols = [c for c in [mortgage_col, ccavg_col, cols.get("education")] if c]
    size_col = mortgage_col if mortgage_col else ccavg_col

    if size_col:
        fig2 = px.scatter(
            df,
            x=fam_col,
            y=income_col,
            color=loan_col,
            size=size_col,
            template=PLOTLY_TEMPLATE,
            title="Family Size vs Income (size=Mortgage/CCAvg, colored by Personal Loan)",
            labels={fam_col: "Family Size", income_col: "Income", size_col: "Size Metric", loan_col: "Personal Loan"},
            hover_data=hover_cols,
            opacity=0.75,
        )
    else:
        fig2 = px.scatter(
            df,
            x=fam_col,
            y=income_col,
            color=loan_col,
            template=PLOTLY_TEMPLATE,
            title="Family Size vs Income (Colored by Personal Loan)",
            labels={fam_col: "Family Size", income_col: "Income", loan_col: "Personal Loan"},
            hover_data=hover_cols,
            opacity=0.75,
        )

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig1, use_container_width=True)
    c2.plotly_chart(fig2, use_container_width=True)


def plot_financial_products_comparison(df: pd.DataFrame, cols: dict):
    loan_col = cols["personal_loan"]
    products = {
        "Securities Account": cols.get("securities"),
        "CD Account": cols.get("cd"),
        "Credit Card": cols.get("creditcard"),
    }
    available = {k: v for k, v in products.items() if v is not None}
    if not available:
        st.info("No product indicator columns (Securities/CD/CreditCard) found; skipping products comparison.")
        return

    # Build a tidy table of acceptance rates by product ownership
    rows = []
    for product_name, colname in available.items():
        # Expect binary 0/1; if not, still compute for unique values
        for val in sorted(df[colname].dropna().unique().tolist()):
            subset = df[df[colname] == val]
            rate = subset[loan_col].mean() if len(subset) else np.nan
            rows.append(
                {
                    "Product": product_name,
                    "Has Product (value)": str(val),
                    "Loan Acceptance Rate": rate * 100 if pd.notna(rate) else np.nan,
                    "Customers": len(subset),
                }
            )
    tidy = pd.DataFrame(rows)

    fig = px.bar(
        tidy,
        x="Product",
        y="Loan Acceptance Rate",
        color="Has Product (value)",
        barmode="group",
        text=tidy["Loan Acceptance Rate"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else ""),
        template=PLOTLY_TEMPLATE,
        title="Impact of Financial Products on Personal Loan Acceptance",
        labels={"Loan Acceptance Rate": "Acceptance Rate (%)"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis=dict(range=[0, min(100, max(5, tidy["Loan Acceptance Rate"].max() + 10))]))

    c1, c2 = st.columns([2, 1])
    c1.plotly_chart(fig, use_container_width=True)
    c2.subheader("Summary Table")
    c2.dataframe(
        tidy.sort_values(["Product", "Has Product (value)"]),
        use_container_width=True,
        hide_index=True,
    )


def plot_box_whiskers(df: pd.DataFrame, cols: dict):
    loan_col, income_col, ccavg_col = cols["personal_loan"], cols["income"], cols["ccavg"]

    # Credit card spending: prefer CCAvg (commonly avg monthly spend). If a different column exists, use it.
    # We'll interpret "Credit Card spending" as CCAvg.
    fig1 = px.box(
        df,
        x=loan_col,
        y=ccavg_col,
        template=PLOTLY_TEMPLATE,
        title="Box Plot: Credit Card Spending (CCAvg) by Personal Loan Status",
        labels={loan_col: "Personal Loan", ccavg_col: "CCAvg"},
        points="outliers",
    )

    fig2 = px.box(
        df,
        x=loan_col,
        y=ccavg_col,
        color=loan_col,
        template=PLOTLY_TEMPLATE,
        title="Box Plot: CCAvg Distribution (Segmented)",
        labels={loan_col: "Personal Loan", ccavg_col: "CCAvg"},
        points="outliers",
    )

    fig3 = px.box(
        df,
        x=loan_col,
        y=income_col,
        color=loan_col,
        template=PLOTLY_TEMPLATE,
        title="Box Plot: Income Distribution by Personal Loan Status",
        labels={loan_col: "Personal Loan", income_col: "Income"},
        points="outliers",
    )

    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(fig1, use_container_width=True)
    c2.plotly_chart(fig2, use_container_width=True)
    c3.plotly_chart(fig3, use_container_width=True)


def plot_education_analysis(df: pd.DataFrame, cols: dict):
    edu_col = cols.get("education")
    if not edu_col:
        st.info("Education column not found in dataset; skipping Education Analysis.")
        return

    income_col = cols["income"]
    loan_col = cols["personal_loan"]

    fig = px.box(
        df,
        x=edu_col,
        y=income_col,
        color=loan_col,
        template=PLOTLY_TEMPLATE,
        title="Education Level vs Income (Segmented by Personal Loan Status)",
        labels={edu_col: "Education Level", income_col: "Income", loan_col: "Personal Loan"},
        points="outliers",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Add a rate chart by education
    rate = (
        df.groupby(edu_col)
        .agg(loan_rate=(loan_col, "mean"), customers=(loan_col, "size"))
        .reset_index()
    )
    rate["loan_rate_pct"] = rate["loan_rate"] * 100

    fig2 = px.bar(
        rate,
        x=edu_col,
        y="loan_rate_pct",
        text=rate["loan_rate_pct"].map(lambda x: f"{x:.1f}%"),
        template=PLOTLY_TEMPLATE,
        title="Personal Loan Acceptance Rate by Education Level",
        labels={edu_col: "Education Level", "loan_rate_pct": "Acceptance Rate (%)"},
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(yaxis=dict(range=[0, min(100, max(5, rate["loan_rate_pct"].max() + 10))]))
    st.plotly_chart(fig2, use_container_width=True)


def plot_mortgage_analysis(df: pd.DataFrame, cols: dict):
    mort_col = cols.get("mortgage")
    fam_col = cols.get("family")
    if not mort_col or not fam_col:
        st.info("Mortgage and/or Family columns not found; skipping Mortgage Analysis.")
        return

    income_col = cols["income"]
    loan_col = cols["personal_loan"]

    # 3-variable scatter: Mortgage vs Income; Family as size; Loan as color
    fig = px.scatter(
        df,
        x=mort_col,
        y=income_col,
        size=fam_col,
        color=loan_col,
        template=PLOTLY_TEMPLATE,
        title="Mortgage vs Income (size=Family Size, color=Personal Loan)",
        labels={mort_col: "Mortgage", income_col: "Income", fam_col: "Family Size", loan_col: "Personal Loan"},
        opacity=0.75,
        hover_data=[c for c in [cols.get("education"), cols.get("ccavg")] if c],
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Main App
# -----------------------------
def main():
    st.title("Bank Customer Personal Loan Acceptance Dashboard")
    st.caption(
        "Interactive analysis of customer demographics, financial behavior, and product ownership to understand "
        "drivers of **Personal Loan** acceptance."
    )

    with st.expander("Data Source & Instructions", expanded=False):
        st.markdown(
            """
- Upload/Place your dataset file as **`bank.csv`** in the same folder as `app.py`, or change the path in the sidebar.
- Expected columns (common banking dataset): `Age`, `Income`, `CCAvg`, `Family`, `Education`, `Mortgage`,
  `Personal Loan`, `Securities Account`, `CD Account`, `CreditCard`, `ZIP Code` (variations supported).
            """.strip()
        )

    with st.sidebar:
        st.subheader("Dataset")
        data_path = st.text_input("CSV file path", value="bank.xls", help="Path to the banking CSV file.")

    try:
        df = load_data(data_path)
        cols = infer_columns(df)
    except Exception as e:
        st.error(f"Failed to load/parse dataset: {e}")
        st.stop()

    # Filters
    fdf, selections = apply_filters(df, cols)

    # KPIs
    st.subheader("Key Metrics")
    kpi_row(fdf, cols)
    st.divider()

    # Layout sections
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Distributions", "Relationships", "Products & Segments", "Correlation & Deep Dives"]
    )

    with tab1:
        st.subheader("1) Histograms: Income & Age Distribution")
        plot_income_age_histograms(fdf, cols)

        st.subheader("7) Box & Whisker Plots")
        plot_box_whiskers(fdf, cols)

    with tab2:
        st.subheader("2) Scatter: CCAvg vs Income")
        plot_scatter_ccavg_income(fdf, cols)

        st.subheader("3) Multivariate: Zipcode vs Income vs Personal Loan")
        plot_zip_income_loan_multivariate(fdf, cols)

        st.subheader("9) Mortgage Analysis: Mortgage vs Income vs Family Size")
        plot_mortgage_analysis(fdf, cols)

    with tab3:
        st.subheader("6) Financial Products Comparison")
        plot_financial_products_comparison(fdf, cols)

        st.subheader("8) Education Analysis")
        plot_education_analysis(fdf, cols)

    with tab4:
        st.subheader("4) Correlation Heatmap")
        plot_correlation_heatmap(fdf)

        st.subheader("5) Family Analysis")
        plot_family_analysis(fdf, cols)

        with st.expander("Show filtered data table", expanded=False):
            st.dataframe(fdf, use_container_width=True)

    st.divider()
    st.markdown(
        "Deployed-ready Streamlit app structure: **cache-enabled loading**, **sidebar filters**, and **interactive Plotly charts**."
    )


if __name__ == "__main__":

    main()
