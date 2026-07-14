# ============================================================
# Global Land Temperature Dashboard  —  Streamlit app
# ------------------------------------------------------------
# An interactive web app to explore global temperature anomalies
# (°C relative to a historical baseline) from 1850 to today.
#
# Features:
#   - Title + description
#   - Sidebar controls: data-source selector + year-range slider
#   - Dynamic Plotly line chart that updates on the fly
#   - Expandable section showing the filtered raw data (+ CSV download)
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px

# ---- Page configuration ----
st.set_page_config(
    page_title="Global Temperature Dashboard",
    page_icon="🌍",
    layout="wide",
)

# Public, reliable mirror of the global temperature anomaly series
DATA_URL = "https://raw.githubusercontent.com/datasets/global-temp/master/data/annual.csv"


# ---- Data loading + cleaning (cached so it runs once) ----
@st.cache_data
def load_data():
    """Load and clean the global temperature dataset."""
    df = pd.read_csv(DATA_URL)
    df = df.dropna()                       # remove missing rows
    df = df.drop_duplicates()              # protect data integrity
    df["Year"] = df["Year"].astype(int)    # standardize the year type
    df["Mean"] = df["Mean"].round(4)       # standardize decimals
    df = df.rename(columns={"Mean": "TempAnomaly"})  # clearer name
    return df.sort_values("Year").reset_index(drop=True)


df = load_data()

# ---- Header ----
st.title("🌍 Global Land Temperature Dashboard")
st.markdown(
    "Explore **global temperature anomalies** (°C, relative to a historical "
    "baseline) from **1850 to today**. A positive value means warmer than the "
    "baseline. Use the **sidebar** to choose a data source and a range of years."
)

# ---- Sidebar controls ----
st.sidebar.header("⚙️ Controls")

sources = sorted(df["Source"].unique())
source = st.sidebar.radio(
    "Data source",
    sources,
    index=0,
    help="GCAG (NOAA) and GISTEMP (NASA) are two independent global series.",
)

min_year = int(df["Year"].min())
max_year = int(df["Year"].max())
year_range = st.sidebar.slider(
    "Select year range",
    min_value=min_year,
    max_value=max_year,
    value=(1900, max_year),
)

# ---- Filter the data based on the widgets ----
mask = (df["Source"] == source) & (df["Year"].between(year_range[0], year_range[1]))
filtered = df[mask].sort_values("Year").reset_index(drop=True)

# Guard against an empty selection
if filtered.empty:
    st.warning("No data for this selection. Try a wider year range.")
    st.stop()

# ---- Key metrics ----
col1, col2, col3 = st.columns(3)
warmest = filtered.loc[filtered["TempAnomaly"].idxmax()]
col1.metric("Years shown", f"{year_range[0]}–{year_range[1]}")
col2.metric("Warmest year", int(warmest["Year"]), f"{warmest['TempAnomaly']:.2f} °C")
col3.metric("Average anomaly", f"{filtered['TempAnomaly'].mean():.2f} °C")

# ---- Dynamic line chart ----
fig = px.line(
    filtered,
    x="Year",
    y="TempAnomaly",
    markers=True,
    title=f"Temperature Anomaly Over Time — {source}",
    labels={"TempAnomaly": "Anomaly (°C)", "Year": "Year"},
)
fig.add_hline(y=0, line_dash="dash", line_color="gray",
              annotation_text="baseline", annotation_position="bottom right")
fig.update_layout(hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# ---- Expandable raw data ----
with st.expander("🔎 Show filtered raw data"):
    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "⬇️ Download this data as CSV",
        filtered.to_csv(index=False),
        file_name="filtered_temperatures.csv",
        mime="text/csv",
    )

st.caption("Data source: NOAA GCAG & NASA GISTEMP via the datahub.io global-temp dataset.")
