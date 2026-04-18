from __future__ import annotations

import io
import re
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


DATA_FILE_CANDIDATES = [
    Path("MagicBricksProject_Cleaned_Data.csv"),
    Path("Cleaned_Data.csv"),
]
DATA_FILE = next((path for path in DATA_FILE_CANDIDATES if path.exists()), DATA_FILE_CANDIDATES[-1])
MIN_VALID_AREA_SQFT = 100
MAX_REASONABLE_PRICE_PER_SQFT = 500
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
REQUIRED_CSV_COLUMNS = [
    "City",
    "BHK",
    "Location",
    "Property Type",
    "Furnishing",
    "Property Facing",
    "Bathroom",
    "Balcony",
    "Tenant Preferred",
    "Availability",
]

CITY_COLORS = {
    "Hyderabad": "#008C8C",
    "Bangalore": "#E85D75",
    "Mumbai": "#DCA51F",
    "Pune": "#2E8B57",
    "Chennai": "#FF8A5B",
}

COLOR_SEQUENCE = [
    "#008C8C",
    "#E85D75",
    "#DCA51F",
    "#2E8B57",
    "#FF8A5B",
    "#7A7F35",
    "#3C9D7D",
]

GALLERY_IMAGES = [
    ("Average Rental Price by City", "1_rental_price_by_city.png"),
    ("Price Distribution Comparison", "2_price_distribution_comparison.png"),
    ("BHK Distribution", "3_bhk_distribution.png"),
    ("Price vs Area", "4_price_vs_area.png"),
    ("Furnishing Analysis", "5_furnishing_analysis.png"),
    ("Property Type Distribution", "6_property_type_distribution.png"),
    ("Tenant Preference", "7_tenant_preference.png"),
    ("Property Facing", "8_property_facing.png"),
    ("Price Heatmap", "9_price_heatmap.png"),
    ("Bathroom and Balcony Analysis", "10_bathroom_balcony_analysis.png"),
    ("Top Locations", "11_top_locations.png"),
    ("Least Expensive Locations", "11_least_locations.png"),
    ("Correlation Heatmap", "12_correlation_heatmap.png"),
    ("Availability Status", "13_availability_status.png"),
    ("Price per Sqft", "14_price_per_sqft.png"),
    (
        "City Wise Top 5 Least and Most Expensive Locations",
        "dashboard_assets/15_city_location_extremes_preview.png",
    ),
    ("Pairplot", "16 Pairplot.png"),
]


st.set_page_config(
    page_title="MagicBricks Rental EDA Dashboard",
    page_icon="MB",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #202322;
        --muted: #66706a;
        --paper: #f6f8f7;
        --panel: #ffffff;
        --line: #dfe7e2;
        --teal: #008c8c;
        --coral: #e85d75;
        --gold: #dca51f;
        --green: #2e8b57;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }

    .dashboard-title {
        font-size: 2.45rem;
        line-height: 1.08;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
    }

    .dashboard-subtitle {
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.55;
        margin-bottom: 1rem;
        max-width: 980px;
    }

    .accent-line {
        height: 5px;
        width: 100%;
        background: linear-gradient(90deg, var(--teal), var(--coral), var(--gold), var(--green));
        border-radius: 4px;
        margin: 0.4rem 0 1rem 0;
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        min-height: 112px;
        box-shadow: 0 8px 20px rgba(32, 35, 34, 0.05);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 800;
    }

    div[data-testid="stPlotlyChart"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.4rem;
    }

    .insight-panel {
        background: #ffffff;
        border: 1px solid var(--line);
        border-left: 5px solid var(--teal);
        border-radius: 8px;
        padding: 0.95rem 1rem;
        margin: 0.5rem 0 1rem 0;
        color: var(--ink);
        line-height: 1.55;
    }

    .small-note {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.45;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid var(--teal);
        color: #ffffff;
        background: var(--teal);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_money(value: object) -> float:
    text = str(value).strip().replace(",", "").replace("INR", "").replace("Rs.", "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return np.nan

    amount = float(match.group(1))
    lower_text = text.lower()

    if "cr" in lower_text or "crore" in lower_text:
        amount *= 10_000_000
    elif "lac" in lower_text or "lakh" in lower_text:
        amount *= 100_000

    return amount


def parse_number(value: object) -> float:
    text = str(value).replace(",", "").strip().lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return np.nan

    number = float(match.group(1))
    if "sqyrd" in text or "sq yard" in text or "sq. yard" in text:
        return number * 9
    if "cent" in text:
        return number * 435.6
    if "sqm" in text or "sq m" in text or "sq. m" in text:
        return number * 10.7639
    return number


def format_inr(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"

    value = float(value)
    sign = "-" if value < 0 else ""
    absolute_value = abs(value)

    if absolute_value >= 10_000_000:
        return f"{sign}INR {absolute_value / 10_000_000:.2f} Cr"
    if absolute_value >= 100_000:
        return f"{sign}INR {absolute_value / 100_000:.2f} Lac"
    return f"{sign}INR {absolute_value:,.0f}"


def format_count(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "0"
    return f"{int(round(float(value))):,}"


def format_price_per_sqft(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"INR {float(value):.2f}/sqft"


def data_source_label(source_name: str | None = None) -> str:
    return source_name or DATA_FILE.name


def with_common_layout(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 21, "color": "#202322"},
        },
        height=height,
        margin={"l": 32, "r": 28, "t": 78, "b": 45},
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=COLOR_SEQUENCE,
        font={"family": "Arial, sans-serif", "color": "#202322"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        hoverlabel={"bgcolor": "white", "font_size": 13, "font_family": "Arial, sans-serif"},
        uniformtext={"minsize": 10, "mode": "hide"},
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title_standoff=12, automargin=True)
    fig.update_yaxes(gridcolor="#edf2ef", zeroline=False, title_standoff=12, automargin=True)
    return fig


def make_bar(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    text: str | None = None,
    height: int = 430,
) -> go.Figure:
    fig = px.bar(
        data,
        x=x,
        y=y,
        color=color,
        text=text,
        color_discrete_map=CITY_COLORS,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(
        marker_line_color="#202322",
        marker_line_width=0.8,
        textposition="outside",
        cliponaxis=False,
    )
    return with_common_layout(fig, title, height)


def prepare_dashboard_dataframe(raw_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    raw_df = raw_df.drop(columns=[c for c in raw_df.columns if c.startswith("Unnamed")], errors="ignore")

    for column in raw_df.select_dtypes(include=["object", "string"]).columns:
        raw_df[column] = raw_df[column].astype(str).str.strip()

    price_source = next((c for c in raw_df.columns if c.lower().startswith("price")), None)
    area_source = next((c for c in raw_df.columns if c.lower().startswith("area")), None)
    if price_source is None or area_source is None:
        raise ValueError("The data file must include price and area columns.")

    missing_columns = [column for column in REQUIRED_CSV_COLUMNS if column not in raw_df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"The CSV is missing required dashboard columns: {missing}.")

    raw_df["Price"] = raw_df[price_source].apply(parse_money)
    raw_df["Area"] = raw_df[area_source].apply(parse_number)
    raw_df["Balcony Count"] = pd.to_numeric(raw_df.get("Balcony"), errors="coerce")
    raw_df["Price per sqft"] = raw_df["Price"] / raw_df["Area"].replace({0: np.nan})
    raw_df["Price Label"] = raw_df["Price"].apply(format_inr)
    raw_df["Area Label"] = raw_df["Area"].apply(lambda x: f"{x:,.0f} sqft" if pd.notna(x) else "NA")
    raw_df["Source File"] = source_name

    numeric_columns = ["BHK", "Bathroom"]
    for column in numeric_columns:
        if column in raw_df.columns:
            raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")

    clean_df = raw_df.dropna(subset=["Price", "Area", "BHK", "Bathroom"]).query("Price > 0 and Area > 0").copy()
    if clean_df.empty:
        raise ValueError("No valid rows were found after parsing price, area, BHK, and bathroom values.")
    return clean_df


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    raw_df = pd.read_csv(DATA_FILE)
    return prepare_dashboard_dataframe(raw_df, DATA_FILE.name)


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_bytes: bytes, source_name: str) -> pd.DataFrame:
    raw_df = pd.read_csv(io.BytesIO(file_bytes))
    return prepare_dashboard_dataframe(raw_df, source_name)


def filter_dataframe(df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, bool]:
    with st.sidebar:
        st.header("Filters")

        include_extremes = st.toggle(
            "Show data-quality outliers",
            value=False,
            help="Off removes records with unrealistic area or price-per-sqft values so the charts stay accurate and readable.",
        )

        working_df = df.copy()
        if not include_extremes:
            working_df = working_df[
                working_df["Area"].ge(MIN_VALID_AREA_SQFT)
                & working_df["Price per sqft"].le(MAX_REASONABLE_PRICE_PER_SQFT)
            ]

        selected_cities = st.multiselect(
            "City",
            options=sorted(working_df["City"].dropna().unique()),
            default=sorted(working_df["City"].dropna().unique()),
        )

        selected_bhk = st.multiselect(
            "BHK",
            options=sorted(working_df["BHK"].dropna().astype(int).unique()),
            default=sorted(working_df["BHK"].dropna().astype(int).unique()),
        )

        selected_property_type = st.multiselect(
            "Property type",
            options=sorted(working_df["Property Type"].dropna().unique()),
            default=sorted(working_df["Property Type"].dropna().unique()),
        )

        selected_furnishing = st.multiselect(
            "Furnishing",
            options=sorted(working_df["Furnishing"].dropna().unique()),
            default=sorted(working_df["Furnishing"].dropna().unique()),
        )

        selected_tenant = st.multiselect(
            "Tenant preferred",
            options=sorted(working_df["Tenant Preferred"].dropna().unique()),
            default=sorted(working_df["Tenant Preferred"].dropna().unique()),
        )

        min_price = int(np.floor(working_df["Price"].min() / 5_000) * 5_000)
        max_price = int(np.ceil(working_df["Price"].max() / 5_000) * 5_000)
        if min_price == max_price:
            max_price = min_price + 5_000

        price_range = st.slider(
            "Monthly rent range",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            step=5_000,
        )

        min_area = int(max(0, np.floor(working_df["Area"].min() / 50) * 50))
        max_area = int(np.ceil(working_df["Area"].max() / 50) * 50)
        if min_area == max_area:
            max_area = min_area + 50

        area_range = st.slider(
            "Area range in sqft",
            min_value=min_area,
            max_value=max_area,
            value=(min_area, max_area),
            step=50,
        )

        filtered = working_df[
            working_df["City"].isin(selected_cities)
            & working_df["BHK"].astype(int).isin(selected_bhk)
            & working_df["Property Type"].isin(selected_property_type)
            & working_df["Furnishing"].isin(selected_furnishing)
            & working_df["Tenant Preferred"].isin(selected_tenant)
            & working_df["Price"].between(price_range[0], price_range[1])
            & working_df["Area"].between(area_range[0], area_range[1])
        ].copy()

        st.markdown(
            f"""
            <div class="small-note">
            Active sample: <strong>{format_count(len(filtered))}</strong> listings from
            <strong>{format_count(len(df))}</strong> cleaned records.<br>
            Source: <strong>{data_source_label(source_name)}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return filtered, include_extremes


def show_metric_cards(df: pd.DataFrame) -> None:
    top_city = "NA"
    top_city_value = np.nan
    if not df.empty:
        city_prices = df.groupby("City")["Price"].mean().sort_values(ascending=False)
        if not city_prices.empty:
            top_city = city_prices.index[0]
            top_city_value = city_prices.iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Filtered listings", format_count(len(df)))
    c2.metric("Average rent", format_inr(df["Price"].mean()))
    c3.metric("Median rent", format_inr(df["Price"].median()))
    c4.metric("Average area", f"{df['Area'].mean():,.0f} sqft" if not df.empty else "NA")
    c5.metric("Costliest city", top_city, format_inr(top_city_value))


def average_price_by_city(df: pd.DataFrame) -> go.Figure:
    city_avg = (
        df.groupby("City", as_index=False)["Price"]
        .mean()
        .sort_values("Price", ascending=False)
        .assign(Label=lambda x: x["Price"].apply(format_inr))
    )
    fig = make_bar(city_avg, "City", "Price", "Average Monthly Rent by City", color="City", text="Label")
    fig.update_yaxes(title="Average monthly rent", tickprefix="INR ")
    fig.update_xaxes(title="")
    return fig


def price_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        df,
        x="City",
        y="Price",
        color="City",
        box=True,
        points="outliers",
        color_discrete_map=CITY_COLORS,
    )
    fig.update_yaxes(title="Monthly rent", tickprefix="INR ")
    fig.update_xaxes(title="")
    return with_common_layout(fig, "Monthly Rent Distribution by City", 460)


def price_vs_area(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="Area",
        y="Price",
        color="City",
        size="BHK",
        hover_name="Location",
        hover_data={
            "Property Type": True,
            "Furnishing": True,
            "BHK": True,
            "Bathroom": True,
            "Price Label": True,
            "Area Label": True,
            "Price": False,
            "Area": False,
        },
        color_discrete_map=CITY_COLORS,
        opacity=0.72,
    )
    fig.update_traces(marker={"line": {"width": 0.6, "color": "#202322"}})
    fig.update_yaxes(title="Monthly rent", tickprefix="INR ")
    fig.update_xaxes(title="Area (sqft)")
    return with_common_layout(fig, "Monthly Rent vs Property Area", 520)


def bhk_distribution(df: pd.DataFrame) -> go.Figure:
    bhk_city = df.groupby(["City", "BHK"], as_index=False).size().rename(columns={"size": "Count"})
    bhk_city["BHK"] = bhk_city["BHK"].astype(int).astype(str) + " BHK"
    fig = px.bar(
        bhk_city,
        x="City",
        y="Count",
        color="BHK",
        barmode="group",
        color_discrete_sequence=COLOR_SEQUENCE,
        category_orders={"BHK": sorted(bhk_city["BHK"].unique(), key=lambda value: int(value.split()[0]))},
    )
    fig.update_yaxes(title="Listings")
    fig.update_xaxes(title="")
    return with_common_layout(fig, "BHK Configuration Mix by City", 430)


def bhk_share(df: pd.DataFrame) -> go.Figure:
    counts = df["BHK"].value_counts().sort_index().reset_index()
    counts.columns = ["BHK", "Count"]
    counts["BHK"] = counts["BHK"].astype(int).astype(str) + " BHK"
    fig = px.pie(
        counts,
        names="BHK",
        values="Count",
        hole=0.48,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return with_common_layout(fig, "Overall BHK Share", 430)


def furnishing_analysis(df: pd.DataFrame) -> go.Figure:
    furnishing_city = (
        df.groupby(["City", "Furnishing"], as_index=False)
        .size()
        .rename(columns={"size": "Count"})
    )
    furnishing_price = (
        df.groupby("Furnishing", as_index=False)["Price"]
        .mean()
        .sort_values("Price", ascending=False)
    )
    furnishing_price["Label"] = furnishing_price["Price"].apply(format_inr)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Listings by city", "Average rent by furnishing"),
    )
    for furnishing in furnishing_city["Furnishing"].unique():
        subset = furnishing_city[furnishing_city["Furnishing"] == furnishing]
        fig.add_trace(
            go.Bar(name=furnishing, x=subset["City"], y=subset["Count"]),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Bar(
            x=furnishing_price["Furnishing"],
            y=furnishing_price["Price"],
            text=furnishing_price["Label"],
            textposition="outside",
            marker_color=COLOR_SEQUENCE[: len(furnishing_price)],
            marker_line={"color": "#202322", "width": 0.8},
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="Listings", row=1, col=1)
    fig.update_yaxes(title_text="Average rent", tickprefix="INR ", row=1, col=2)
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    return with_common_layout(fig, "Furnishing Mix and Rent Premium", 460)


def property_type_distribution(df: pd.DataFrame) -> go.Figure:
    prop_city = (
        df.groupby(["City", "Property Type"], as_index=False)
        .size()
        .rename(columns={"size": "Count"})
    )
    fig = px.bar(
        prop_city,
        x="City",
        y="Count",
        color="Property Type",
        barmode="group",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_yaxes(title="Listings")
    fig.update_xaxes(title="")
    return with_common_layout(fig, "Property Type Mix by City", 430)


def tenant_preference(df: pd.DataFrame) -> go.Figure:
    tenant_city = (
        df.groupby(["City", "Tenant Preferred"], as_index=False)
        .size()
        .rename(columns={"size": "Count"})
    )
    fig = px.bar(
        tenant_city,
        x="City",
        y="Count",
        color="Tenant Preferred",
        barmode="group",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_yaxes(title="Listings")
    fig.update_xaxes(title="")
    return with_common_layout(fig, "Tenant Preference by City", 430)


def property_facing(df: pd.DataFrame) -> go.Figure:
    facing_counts = df["Property Facing"].value_counts().head(10).reset_index()
    facing_counts.columns = ["Property Facing", "Count"]
    fig = make_bar(
        facing_counts,
        "Property Facing",
        "Count",
        "Property Facing Direction Mix",
        text="Count",
    )
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(title="Listings")
    return fig


def availability_status(df: pd.DataFrame) -> go.Figure:
    availability_counts = df["Availability"].value_counts().reset_index()
    availability_counts.columns = ["Availability", "Count"]
    fig = make_bar(
        availability_counts,
        "Availability",
        "Count",
        "Property Availability Status",
        text="Count",
    )
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(title="Listings")
    return fig


def price_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(values="Price", index="BHK", columns="City", aggfunc="mean").sort_index()
    text = pivot.apply(lambda column: column.map(lambda x: format_inr(x).replace("INR ", "") if pd.notna(x) else ""))

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index.astype(int),
            colorscale=[
                [0.0, "#f4fbf8"],
                [0.35, "#66c2a5"],
                [0.7, "#f1c453"],
                [1.0, "#e85d75"],
            ],
            colorbar={"title": "Avg rent"},
            text=text.values,
            hovertemplate="City: %{x}<br>BHK: %{y}<br>Average rent: INR %{z:,.0f}<extra></extra>",
        )
    )
    for row_index, bhk in enumerate(pivot.index):
        for col_index, city in enumerate(pivot.columns):
            value = pivot.iloc[row_index, col_index]
            if pd.notna(value):
                fig.add_annotation(
                    x=city,
                    y=int(bhk),
                    text=text.iloc[row_index, col_index],
                    showarrow=False,
                    font={"size": 12, "color": "#202322"},
                )
    fig.update_yaxes(title="BHK")
    fig.update_xaxes(title="City")
    return with_common_layout(fig, "Average Monthly Rent by BHK and City", 480)


def location_rankings(df: pd.DataFrame, city: str) -> go.Figure:
    city_df = df[df["City"] == city]
    location_avg = (
        city_df.groupby("Location", as_index=False)
        .agg(Price=("Price", "mean"), Listings=("Location", "size"))
        .query("Listings >= 1")
    )

    least = location_avg.sort_values("Price", ascending=True).head(5)
    most = location_avg.sort_values("Price", ascending=False).head(5)

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.16,
        subplot_titles=("Most expensive localities", "Most affordable localities"),
    )
    fig.add_trace(
        go.Bar(
            x=most.sort_values("Price", ascending=True)["Price"],
            y=most.sort_values("Price", ascending=True)["Location"],
            orientation="h",
            text=most.sort_values("Price", ascending=True)["Price"].apply(format_inr),
            textposition="outside",
            marker_color="#E85D75",
            marker_line={"color": "#202322", "width": 0.8},
            hovertemplate="%{y}<br>Average rent: INR %{x:,.0f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=least.sort_values("Price", ascending=False)["Price"],
            y=least.sort_values("Price", ascending=False)["Location"],
            orientation="h",
            text=least.sort_values("Price", ascending=False)["Price"].apply(format_inr),
            textposition="outside",
            marker_color="#008C8C",
            marker_line={"color": "#202322", "width": 0.8},
            hovertemplate="%{y}<br>Average rent: INR %{x:,.0f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Average monthly rent", tickprefix="INR ", row=1, col=1)
    fig.update_xaxes(title_text="Average monthly rent", tickprefix="INR ", row=2, col=1)
    fig.update_yaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="", row=2, col=1)
    return with_common_layout(fig, f"{city} Locality Rent Rankings", 640)


def bathroom_balcony_analysis(df: pd.DataFrame) -> go.Figure:
    bathroom_counts = df["Bathroom"].value_counts().sort_index()
    balcony_counts = df["Balcony Count"].dropna().value_counts().sort_index()
    bathroom_price = df.groupby("Bathroom")["Price"].mean().sort_index()
    balcony_price = df.dropna(subset=["Balcony Count"]).groupby("Balcony Count")["Price"].mean().sort_index()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Bathroom distribution",
            "Balcony distribution",
            "Average rent by bathrooms",
            "Average rent by balconies",
        ),
    )
    fig.add_trace(
        go.Bar(x=bathroom_counts.index, y=bathroom_counts.values, marker_color="#008C8C", name="Bathrooms"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=balcony_counts.index, y=balcony_counts.values, marker_color="#E85D75", name="Balconies"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=bathroom_price.index,
            y=bathroom_price.values,
            mode="lines+markers",
            marker={"size": 10, "color": "#008C8C"},
            line={"width": 3, "color": "#008C8C"},
            name="Avg rent by bathrooms",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=balcony_price.index,
            y=balcony_price.values,
            mode="lines+markers",
            marker={"size": 10, "color": "#E85D75"},
            line={"width": 3, "color": "#E85D75"},
            name="Avg rent by balconies",
        ),
        row=2,
        col=2,
    )
    fig.update_yaxes(title_text="Listings", row=1, col=1)
    fig.update_yaxes(title_text="Listings", row=1, col=2)
    fig.update_yaxes(title_text="Average rent", tickprefix="INR ", row=2, col=1)
    fig.update_yaxes(title_text="Average rent", tickprefix="INR ", row=2, col=2)
    fig.update_xaxes(title_text="Bathrooms", row=1, col=1)
    fig.update_xaxes(title_text="Balconies", row=1, col=2)
    fig.update_xaxes(title_text="Bathrooms", row=2, col=1)
    fig.update_xaxes(title_text="Balconies", row=2, col=2)
    return with_common_layout(fig, "Bathroom, Balcony, and Rent Patterns", 680)


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_columns = ["BHK", "Price", "Area", "Bathroom", "Balcony Count", "Price per sqft"]
    available = [column for column in numeric_columns if column in df.columns]
    corr = df[available].corr(numeric_only=True)
    label_map = {
        "BHK": "BHK",
        "Price": "Monthly rent",
        "Area": "Area",
        "Bathroom": "Bathrooms",
        "Balcony Count": "Balconies",
        "Price per sqft": "Rent/sqft",
    }
    labels = [label_map.get(column, column) for column in corr.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0.0, "#e85d75"],
                [0.5, "#ffffff"],
                [1.0, "#008c8c"],
            ],
            text=np.round(corr.values, 2),
            hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
        )
    )
    for row_index, row_label in enumerate(labels):
        for col_index, col_label in enumerate(labels):
            fig.add_annotation(
                x=col_label,
                y=row_label,
                text=f"{corr.iloc[row_index, col_index]:.2f}",
                showarrow=False,
                font={"size": 12, "color": "#202322"},
            )
    return with_common_layout(fig, "Numeric Feature Correlation", 540)


def price_per_sqft_analysis(df: pd.DataFrame) -> go.Figure:
    price_sqft_city = (
        df.groupby("City", as_index=False)["Price per sqft"]
        .median()
        .sort_values("Price per sqft", ascending=False)
    )
    price_sqft_city["Label"] = price_sqft_city["Price per sqft"].map(format_price_per_sqft)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Median rent per sqft by city", "Rent per sqft distribution"),
    )
    fig.add_trace(
        go.Bar(
            x=price_sqft_city["City"],
            y=price_sqft_city["Price per sqft"],
            text=price_sqft_city["Label"],
            textposition="outside",
            marker_color=[CITY_COLORS.get(city, "#008C8C") for city in price_sqft_city["City"]],
            marker_line={"color": "#202322", "width": 0.8},
            name="Median rent per sqft",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=df["Price per sqft"].replace([np.inf, -np.inf], np.nan).dropna(),
            nbinsx=42,
            marker_color="#E85D75",
            marker_line={"color": "#202322", "width": 0.3},
            name="Price per sqft",
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="Median rent per sqft", row=1, col=1)
    fig.update_yaxes(title_text="Listings", row=1, col=2)
    fig.update_xaxes(title_text="Rent per sqft", row=1, col=2)
    return with_common_layout(fig, "Space Value: Monthly Rent per Sqft", 500)


def scatter_matrix(df: pd.DataFrame) -> go.Figure:
    sample = df[["BHK", "Price", "Area", "Bathroom", "Price per sqft", "City"]].dropna()
    if len(sample) > 800:
        sample = sample.sample(800, random_state=42)

    fig = px.scatter_matrix(
        sample,
        dimensions=["BHK", "Price", "Area", "Bathroom", "Price per sqft"],
        color="City",
        color_discrete_map=CITY_COLORS,
        opacity=0.65,
        labels={
            "Price": "Monthly rent",
            "Area": "Area",
            "Bathroom": "Bathrooms",
            "Price per sqft": "Rent/sqft",
        },
    )
    fig.update_traces(diagonal_visible=False, marker={"size": 4})
    return with_common_layout(fig, "Numeric Relationship Matrix", 780)


def pdf_escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def truncate_text(value: object, max_chars: int) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


class SimplePDFReport:
    def __init__(self) -> None:
        self.width = 612
        self.height = 792
        self.margin = 44
        self.pages: list[list[str]] = []
        self.ops: list[str] = []
        self.y = 0
        self.add_page()

    def add_page(self) -> None:
        if self.ops:
            self.pages.append(self.ops)
        self.ops = []
        self.y = self.height - self.margin

    def finish_pages(self) -> None:
        if self.ops:
            self.pages.append(self.ops)
            self.ops = []

    def ensure_space(self, needed: float) -> None:
        if self.y - needed < self.margin:
            self.add_page()

    def rect(self, x: float, y: float, width: float, height: float, color: tuple[float, float, float]) -> None:
        r, g, b = color
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} {width:.2f} {height:.2f} re f")

    def line(self, x1: float, y1: float, x2: float, y2: float, color: tuple[float, float, float] = (0.87, 0.90, 0.88)) -> None:
        r, g, b = color
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} RG 0.8 w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def text(
        self,
        x: float,
        y: float,
        value: object,
        size: int = 10,
        bold: bool = False,
        color: tuple[float, float, float] = (0.12, 0.14, 0.13),
    ) -> None:
        r, g, b = color
        font = "F2" if bold else "F1"
        safe_text = pdf_escape(value)
        self.ops.append(f"BT {r:.3f} {g:.3f} {b:.3f} rg /{font} {size} Tf {x:.2f} {y:.2f} Td ({safe_text}) Tj ET")

    def paragraph(self, value: object, size: int = 10, width_chars: int = 98, bold: bool = False) -> None:
        for line in textwrap.wrap(str(value), width=width_chars) or [""]:
            self.ensure_space(size + 8)
            self.text(self.margin, self.y, line, size=size, bold=bold)
            self.y -= size + 5

    def heading(self, value: object) -> None:
        self.ensure_space(32)
        self.y -= 8
        self.text(self.margin, self.y, value, size=14, bold=True, color=(0.0, 0.45, 0.45))
        self.y -= 9
        self.line(self.margin, self.y, self.width - self.margin, self.y, color=(0.0, 0.55, 0.55))
        self.y -= 14

    def table(self, headers: list[str], rows: list[list[object]], widths: list[int]) -> None:
        row_height = 18
        total_width = sum(widths)
        self.ensure_space(row_height * 2)
        self.rect(self.margin, self.y - 4, total_width, row_height, (0.93, 0.96, 0.95))
        x = self.margin + 4
        for header, width in zip(headers, widths):
            self.text(x, self.y, truncate_text(header, max(8, int(width / 5.2))), size=9, bold=True)
            x += width
        self.y -= row_height
        self.line(self.margin, self.y + 4, self.margin + total_width, self.y + 4)

        for row in rows:
            self.ensure_space(row_height + 4)
            x = self.margin + 4
            for value, width in zip(row, widths):
                self.text(x, self.y, truncate_text(value, max(8, int(width / 5.2))), size=8)
                x += width
            self.y -= row_height
        self.y -= 8

    def build(self) -> bytes:
        self.finish_pages()
        objects: list[bytes | None] = [None]

        def reserve() -> int:
            objects.append(None)
            return len(objects) - 1

        def set_object(object_id: int, content: str | bytes) -> None:
            if isinstance(content, str):
                content = content.encode("latin-1", errors="replace")
            objects[object_id] = content

        catalog_id = reserve()
        pages_id = reserve()
        font_regular_id = reserve()
        font_bold_id = reserve()
        set_object(font_regular_id, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        set_object(font_bold_id, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

        page_ids: list[int] = []
        for page_ops in self.pages:
            stream = "\n".join(page_ops).encode("latin-1", errors="replace")
            content_id = reserve()
            page_id = reserve()
            set_object(content_id, b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
            set_object(
                page_id,
                (
                    f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {self.width} {self.height}] "
                    f"/Resources << /Font << /F1 {font_regular_id} 0 R /F2 {font_bold_id} 0 R >> >> "
                    f"/Contents {content_id} 0 R >>"
                ),
            )
            page_ids.append(page_id)

        kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
        set_object(pages_id, f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>")
        set_object(catalog_id, f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

        output = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for object_id, content in enumerate(objects[1:], start=1):
            offsets.append(len(output))
            output.extend(f"{object_id} 0 obj\n".encode("ascii"))
            output.extend(content or b"")
            output.extend(b"\nendobj\n")

        xref_offset = len(output)
        output.extend(f"xref\n0 {len(objects)}\n".encode("ascii"))
        output.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        output.extend(
            (
                f"trailer\n<< /Size {len(objects)} /Root {catalog_id} 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        return bytes(output)


def build_pdf_report(df: pd.DataFrame, source_name: str, include_extremes: bool) -> bytes:
    report = SimplePDFReport()
    report.rect(0, 0, report.width, report.height, (0.98, 0.99, 0.98))
    report.text(report.margin, report.y, "MagicBricks Rental Market Dashboard", size=20, bold=True, color=(0.0, 0.45, 0.45))
    report.y -= 24
    report.text(report.margin, report.y, f"EDA PDF Report | Source: {source_name}", size=11, bold=True)
    report.y -= 16
    report.text(report.margin, report.y, f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}", size=9, color=(0.38, 0.43, 0.40))
    report.y -= 20
    report.rect(report.margin, report.y, report.width - (report.margin * 2), 4, (0.0, 0.55, 0.55))
    report.y -= 24

    city_prices = df.groupby("City")["Price"].mean().sort_values(ascending=False)
    best_value_city = df.groupby("City")["Price per sqft"].median().sort_values(ascending=True)
    most_common_bhk = df["BHK"].mode().iloc[0] if not df["BHK"].mode().empty else np.nan
    most_common_property = df["Property Type"].mode().iloc[0] if not df["Property Type"].mode().empty else "NA"
    outlier_text = (
        "Data-quality outliers included."
        if include_extremes
        else f"Hidden outliers: area below {MIN_VALID_AREA_SQFT} sqft or rent above {format_price_per_sqft(MAX_REASONABLE_PRICE_PER_SQFT)}."
    )

    report.heading("Executive Summary")
    summary_rows = [
        ["Listings", format_count(len(df)), "Average rent", format_inr(df["Price"].mean())],
        ["Median rent", format_inr(df["Price"].median()), "Average area", f"{df['Area'].mean():,.0f} sqft"],
        ["Costliest city", city_prices.index[0], "Avg rent", format_inr(city_prices.iloc[0])],
        ["Best value city", best_value_city.index[0], "Median rent/sqft", format_price_per_sqft(best_value_city.iloc[0])],
        ["Common BHK", f"{int(most_common_bhk)} BHK" if pd.notna(most_common_bhk) else "NA", "Common type", most_common_property],
    ]
    report.table(["Metric", "Value", "Metric", "Value"], summary_rows, [112, 140, 112, 140])
    report.paragraph(outlier_text, size=9)

    report.heading("City Rent Summary")
    city_summary = (
        df.groupby("City", as_index=False)
        .agg(
            Listings=("City", "size"),
            Average_Rent=("Price", "mean"),
            Median_Rent=("Price", "median"),
            Median_Area=("Area", "median"),
            Median_Rent_Per_Sqft=("Price per sqft", "median"),
        )
        .sort_values("Average_Rent", ascending=False)
    )
    report.table(
        ["City", "Listings", "Avg rent", "Median rent", "Median area", "Median rent/sqft"],
        [
            [
                row.City,
                format_count(row.Listings),
                format_inr(row.Average_Rent),
                format_inr(row.Median_Rent),
                f"{row.Median_Area:,.0f} sqft",
                format_price_per_sqft(row.Median_Rent_Per_Sqft),
            ]
            for row in city_summary.itertuples(index=False)
        ],
        [80, 62, 92, 92, 88, 92],
    )

    report.heading("Property Mix")
    mix_rows: list[list[object]] = []
    bhk_counts = df["BHK"].value_counts().sort_index()
    furnishing_counts = df["Furnishing"].value_counts()
    property_counts = df["Property Type"].value_counts()
    tenant_counts = df["Tenant Preferred"].value_counts()
    for index in range(max(len(bhk_counts), len(furnishing_counts), len(property_counts), len(tenant_counts), 5)):
        mix_rows.append(
            [
                f"{int(bhk_counts.index[index])} BHK: {format_count(bhk_counts.iloc[index])}" if index < len(bhk_counts) else "",
                f"{furnishing_counts.index[index]}: {format_count(furnishing_counts.iloc[index])}" if index < len(furnishing_counts) else "",
                f"{property_counts.index[index]}: {format_count(property_counts.iloc[index])}" if index < len(property_counts) else "",
                f"{tenant_counts.index[index]}: {format_count(tenant_counts.iloc[index])}" if index < len(tenant_counts) else "",
            ]
        )
    report.table(["BHK", "Furnishing", "Property type", "Tenant preference"], mix_rows[:8], [120, 132, 120, 132])

    report.heading("Top Localities by Average Rent")
    top_locations = (
        df.groupby(["City", "Location"], as_index=False)
        .agg(Average_Rent=("Price", "mean"), Listings=("Location", "size"), Median_Area=("Area", "median"))
        .sort_values("Average_Rent", ascending=False)
        .head(10)
    )
    report.table(
        ["City", "Location", "Avg rent", "Listings", "Median area"],
        [
            [row.City, row.Location, format_inr(row.Average_Rent), format_count(row.Listings), f"{row.Median_Area:,.0f} sqft"]
            for row in top_locations.itertuples(index=False)
        ],
        [74, 178, 94, 70, 88],
    )

    report.heading("Most Affordable Localities")
    affordable_locations = (
        df.groupby(["City", "Location"], as_index=False)
        .agg(Average_Rent=("Price", "mean"), Listings=("Location", "size"), Median_Area=("Area", "median"))
        .sort_values("Average_Rent", ascending=True)
        .head(10)
    )
    report.table(
        ["City", "Location", "Avg rent", "Listings", "Median area"],
        [
            [row.City, row.Location, format_inr(row.Average_Rent), format_count(row.Listings), f"{row.Median_Area:,.0f} sqft"]
            for row in affordable_locations.itertuples(index=False)
        ],
        [74, 178, 94, 70, 88],
    )

    report.heading("Correlation Notes")
    corr = df[["Price", "Area", "BHK", "Bathroom", "Price per sqft"]].corr(numeric_only=True)["Price"].drop("Price")
    corr_rows = [[metric.replace("Price per sqft", "Rent/sqft"), f"{value:.2f}"] for metric, value in corr.sort_values(ascending=False).items()]
    report.table(["Feature", "Correlation with monthly rent"], corr_rows, [230, 190])
    report.paragraph("Use the interactive dashboard for full charts, filters, hover details, and exported notebook visuals.", size=9)

    return report.build()


def get_dashboard_data() -> tuple[pd.DataFrame, str]:
    with st.sidebar:
        st.header("Data")
        uploaded_csv = st.file_uploader(
            "Upload a new CSV",
            type=["csv"],
            help="Use a MagicBricks-style cleaned CSV with the same columns as the project dataset.",
        )
        st.caption(
            "Required columns: City, BHK, Location, Price, Area, Property Type, Furnishing, Property Facing, Bathroom, Balcony, Tenant Preferred, Availability."
        )

    if uploaded_csv is not None:
        return load_uploaded_data(uploaded_csv.getvalue(), uploaded_csv.name), uploaded_csv.name

    return load_data(), DATA_FILE.name


def show_pdf_download(df: pd.DataFrame, source_name: str, include_extremes: bool) -> None:
    pdf_bytes = build_pdf_report(df, source_name, include_extremes)
    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name="magicbricks_dashboard_report.pdf",
        mime="application/pdf",
    )


def show_overview(df: pd.DataFrame) -> None:
    st.plotly_chart(average_price_by_city(df), width="stretch", config=PLOT_CONFIG)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(price_distribution(df), width="stretch", config=PLOT_CONFIG)
    with c2:
        st.plotly_chart(price_heatmap(df), width="stretch", config=PLOT_CONFIG)


def show_property_mix(df: pd.DataFrame) -> None:
    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.plotly_chart(bhk_distribution(df), width="stretch", config=PLOT_CONFIG)
    with c2:
        st.plotly_chart(bhk_share(df), width="stretch", config=PLOT_CONFIG)

    st.plotly_chart(furnishing_analysis(df), width="stretch", config=PLOT_CONFIG)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(property_type_distribution(df), width="stretch", config=PLOT_CONFIG)
    with c4:
        st.plotly_chart(tenant_preference(df), width="stretch", config=PLOT_CONFIG)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(property_facing(df), width="stretch", config=PLOT_CONFIG)
    with c6:
        st.plotly_chart(availability_status(df), width="stretch", config=PLOT_CONFIG)


def show_price_drivers(df: pd.DataFrame) -> None:
    st.plotly_chart(price_vs_area(df), width="stretch", config=PLOT_CONFIG)
    st.plotly_chart(price_per_sqft_analysis(df), width="stretch", config=PLOT_CONFIG)
    st.plotly_chart(bathroom_balcony_analysis(df), width="stretch", config=PLOT_CONFIG)
    c1, c2 = st.columns([0.9, 1.1])
    with c1:
        st.plotly_chart(correlation_heatmap(df), width="stretch", config=PLOT_CONFIG)
    with c2:
        st.plotly_chart(scatter_matrix(df), width="stretch", config=PLOT_CONFIG)


def show_locations(df: pd.DataFrame) -> None:
    available_cities = sorted(df["City"].dropna().unique())
    selected_city = st.selectbox("Choose a city for locality ranking", available_cities)
    st.plotly_chart(location_rankings(df, selected_city), width="stretch", config=PLOT_CONFIG)

    top_locations = (
        df.groupby(["City", "Location"], as_index=False)
        .agg(Average_Rent=("Price", "mean"), Listings=("Location", "size"), Average_Area=("Area", "mean"))
        .sort_values("Average_Rent", ascending=False)
        .head(25)
    )
    top_locations["Average Rent"] = top_locations["Average_Rent"].apply(format_inr)
    top_locations["Average Area"] = top_locations["Average_Area"].map(lambda x: f"{x:,.0f} sqft")
    st.dataframe(
        top_locations[["City", "Location", "Average Rent", "Average Area", "Listings"]],
        width="stretch",
        hide_index=True,
    )


def show_gallery() -> None:
    existing_images = [(title, Path(filename)) for title, filename in GALLERY_IMAGES if Path(filename).exists()]
    if not existing_images:
        st.info("No exported notebook chart images were found in the project folder.")
        return

    st.markdown(
        """
        <div class="insight-panel">
        Notebook exports from the original Matplotlib and Seaborn EDA are included here for presentation use.
        </div>
        """,
        unsafe_allow_html=True,
    )

    for index in range(0, len(existing_images), 2):
        columns = st.columns(2)
        for column, item in zip(columns, existing_images[index : index + 2]):
            title, image_path = item
            with column:
                st.image(str(image_path), caption=title, width="stretch")


def show_data_table(df: pd.DataFrame) -> None:
    display_columns = [
        "City",
        "Location",
        "BHK",
        "Price Label",
        "Area Label",
        "Property Type",
        "Furnishing",
        "Property Facing",
        "Bathroom",
        "Balcony",
        "Tenant Preferred",
        "Availability",
        "Price per sqft",
    ]
    available_columns = [column for column in display_columns if column in df.columns]
    display_df = df[available_columns].copy()
    if "Price per sqft" in display_df.columns:
        display_df["Price per sqft"] = display_df["Price per sqft"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")

    st.dataframe(display_df, width="stretch", hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data",
        data=csv_bytes,
        file_name="magicbricks_filtered_dashboard_data.csv",
        mime="text/csv",
    )


def show_insights(df: pd.DataFrame, include_extremes: bool) -> None:
    most_expensive_city = df.groupby("City")["Price"].mean().sort_values(ascending=False)
    best_value_city = df.groupby("City")["Price per sqft"].median().sort_values(ascending=True)
    most_common_bhk = df["BHK"].mode().iloc[0] if not df["BHK"].mode().empty else "NA"
    most_common_property = df["Property Type"].mode().iloc[0] if not df["Property Type"].mode().empty else "NA"

    outlier_note = (
        "Data-quality outliers are included in the current view."
        if include_extremes
        else f"Rows below {MIN_VALID_AREA_SQFT} sqft or above {format_price_per_sqft(MAX_REASONABLE_PRICE_PER_SQFT)} are hidden."
    )

    st.markdown(
        f"""
        <div class="insight-panel">
        <strong>Quick market read:</strong>
        {most_expensive_city.index[0]} leads average rent at {format_inr(most_expensive_city.iloc[0])}.
        {best_value_city.index[0]} has the lowest median rent per sqft at {format_price_per_sqft(best_value_city.iloc[0])}.
        The most common configuration is {int(most_common_bhk)} BHK, and {most_common_property} listings dominate the sample.
        {outlier_note}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown('<div class="dashboard-title">MagicBricks Rental Market Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="dashboard-subtitle">
        Clean, interactive EDA for Bangalore, Chennai, Hyderabad, Mumbai, and Pune using
        <strong>{data_source_label()}</strong> from the MagicBricks web scraping project.
        </div>
        <div class="accent-line"></div>
        """,
        unsafe_allow_html=True,
    )

    try:
        df = load_data()
    except Exception as exc:
        st.error(f"Unable to load dashboard data: {exc}")
        st.stop()

    filtered_df, include_extremes = filter_dataframe(df, DATA_FILE.name)

    if filtered_df.empty:
        st.warning("No listings match the selected filters. Adjust the sidebar filters to continue.")
        st.stop()

    show_metric_cards(filtered_df)
    show_insights(filtered_df, include_extremes)

    tab_overview, tab_mix, tab_drivers, tab_locations, tab_gallery, tab_data = st.tabs(
        [
            "Market Overview",
            "Property Mix",
            "Price Drivers",
            "Locations",
            "EDA Gallery",
            "Data",
        ]
    )

    with tab_overview:
        show_overview(filtered_df)

    with tab_mix:
        show_property_mix(filtered_df)

    with tab_drivers:
        show_price_drivers(filtered_df)

    with tab_locations:
        show_locations(filtered_df)

    with tab_gallery:
        show_gallery()

    with tab_data:
        show_data_table(filtered_df)


if __name__ == "__main__":
    main()
