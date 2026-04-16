import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Saxbys Purchase Dashboard", layout="wide")

st.title("☕ Saxbys Purchase Dashboard")
st.caption("Visualize food-item purchases from transaction-style 0/1 data.")

REQUIRED_BASE_COLUMNS = ["Year", "Weekday", "Customer"]


def validate_and_prepare(df: pd.DataFrame):
    # Keep original names for display, but strip surrounding spaces
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Food-item columns are everything after the first three required fields
    item_cols = [c for c in df.columns if c not in REQUIRED_BASE_COLUMNS]
    if not item_cols:
        raise ValueError("No food-item columns found. Expected item columns after Year, Weekday, Customer.")

    # Standardize types
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Weekday"] = df["Weekday"].astype(str).str.strip()
    df["Customer"] = df["Customer"].astype(str).str.strip()

    for col in item_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Optional cleanup: force binary-like values into 0/1 if needed
    for col in item_cols:
        df[col] = (df[col] > 0).astype(int)

    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if set(df["Weekday"]).intersection(weekday_order):
        df["Weekday"] = pd.Categorical(df["Weekday"], categories=weekday_order, ordered=True)

    return df, item_cols


@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        sample = pd.DataFrame(
            {
                "Year": [2024, 2024, 2024, 2025, 2025, 2025],
                "Weekday": ["Monday", "Tuesday", "Friday", "Monday", "Wednesday", "Saturday"],
                "Customer": ["C001", "C002", "C003", "C004", "C005", "C006"],
                "Coffee": [1, 0, 1, 1, 0, 1],
                "Tea": [0, 1, 0, 0, 1, 0],
                "Bagel": [1, 0, 0, 1, 1, 0],
                "Cookie": [0, 1, 1, 0, 0, 1],
            }
        )
        return validate_and_prepare(sample)

    # Works for CSV. If your uploaded file is Excel, switch to pd.read_excel(uploaded_file)
    df = pd.read_csv(uploaded_file)
    return validate_and_prepare(df)


with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload transaction data", type=["csv"])
    st.markdown(
        "Expected structure: first columns are `Year`, `Weekday`, `Customer`; all remaining columns are food items coded as 0/1."
    )

try:
    df, item_cols = load_data(uploaded_file)
except Exception as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    year_options = sorted(df["Year"].dropna().unique().tolist())
    selected_years = st.multiselect("Select year", options=year_options, default=year_options)

filtered = df[df["Year"].isin(selected_years)].copy()

if filtered.empty:
    st.warning("No data matches the selected year filter.")
    st.stop()

# KPI cards for classroom clarity
col1, col2, col3 = st.columns(3)
col1.metric("Rows / Visits", f"{len(filtered):,}")
col2.metric("Unique Customers", f"{filtered['Customer'].nunique():,}")
col3.metric("Available Food Items", f"{len(item_cols):,}")

st.divider()

# -----------------------------
# Chart 1: Sales per food item
# -----------------------------
st.subheader("Sales per Food Item")
st.caption("Because each item column is coded 1 = purchased, 0 = not purchased, sales here means purchase counts.")

item_sales = (
    filtered[item_cols]
    .sum()
    .reset_index()
)
item_sales.columns = ["Food Item", "Purchases"]
item_sales = item_sales.sort_values("Purchases", ascending=False)

bar = (
    alt.Chart(item_sales)
    .mark_bar()
    .encode(
        x=alt.X("Food Item:N", sort="-y", title="Food Item"),
        y=alt.Y("Purchases:Q", title="Number of Purchases"),
        tooltip=["Food Item", "Purchases"],
    )
    .properties(height=420)
)

st.altair_chart(bar, use_container_width=True)

st.subheader("Sales Table")
st.dataframe(item_sales, use_container_width=True, hide_index=True)

# -----------------------------
# Load category mapping files
# -----------------------------
st.divider()
st.subheader("Category Mapping")

with st.sidebar:
    bakery_file = st.file_uploader("Upload bakery items (bakery.csv)", type=["csv"], key="bakery")
    drink_file = st.file_uploader("Upload drink items (drink.csv)", type=["csv"], key="drink")
    food_file = st.file_uploader("Upload food items (food.csv)", type=["csv"], key="food")


def load_category_list(file):
    if file is None:
        return []
    df_cat = pd.read_csv(file)
    return df_cat.iloc[:, 0].dropna().astype(str).str.strip().tolist()

bakery_items = load_category_list(bakery_file)
drink_items = load_category_list(drink_file)
food_items = load_category_list(food_file)

# Build mapping dict
category_map = {}
for item in bakery_items:
    category_map[item] = "Bakery"
for item in drink_items:
    category_map[item] = "Drink"
for item in food_items:
    category_map[item] = "Food"

# -----------------------------
# Chart 2: Sales per category per weekday per year
# -----------------------------
st.divider()
st.subheader("Sales per Category by Weekday and Year")

# Convert wide -> long
long_df = filtered.melt(
    id_vars=["Year", "Weekday", "Customer"],
    value_vars=item_cols,
    var_name="Food Item",
    value_name="Purchase"
)

# Keep only purchased rows
long_df = long_df[long_df["Purchase"] == 1].copy()

# Map category
long_df["Category"] = long_df["Food Item"].map(category_map)
long_df = long_df.dropna(subset=["Category"])

# Aggregate
cat_summary = (
    long_df.groupby(["Year", "Weekday", "Category"], as_index=False)["Purchase"]
    .sum()
)

# Create combined x-axis label
cat_summary["Year-Weekday"] = cat_summary["Year"].astype(str) + " - " + cat_summary["Weekday"].astype(str)

chart2 = (
    alt.Chart(cat_summary)
    .mark_bar()
    .encode(
        x=alt.X("Year-Weekday:N", title="Year & Weekday", sort=None),
        y=alt.Y("Purchase:Q", title="Purchases"),
        color=alt.Color("Category:N", title="Food Category"),
        tooltip=["Year", "Weekday", "Category", "Purchase"]
    )
    .properties(height=420)
)

st.altair_chart(chart2, use_container_width=True)

st.subheader("Category Summary Table")
st.dataframe(cat_summary, use_container_width=True, hide_index=True)

with st.expander("How this chart works"):
    st.markdown(
        """
        Steps:
        1. Convert data from wide format (many item columns) to long format.
        2. Keep only rows where Purchase = 1.
        3. Map each item to a category (Drink, Food, Bakery).
        4. Group by Year, Weekday, and Category.
        5. Plot using combined Year-Weekday on x-axis.
        """
    )
