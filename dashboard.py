import streamlit as st
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Saxbys Purchase Dashboard", layout="wide")

st.title("☕ Saxbys Purchase Dashboard")
st.caption("Visualize item purchases, category performance, customer purchases, and regression-based predicted sales.")

REQUIRED_BASE_COLUMNS = ["Year", "Weekday", "Customer"]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def normalize_name(x):
    return str(x).strip().lower()


def validate_and_prepare(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    item_cols = [c for c in df.columns if c not in REQUIRED_BASE_COLUMNS]
    if not item_cols:
        raise ValueError("No item columns found after Year, Weekday, Customer.")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Weekday"] = df["Weekday"].astype(str).str.strip()
    df["Customer"] = df["Customer"].astype(str).str.strip()

    for col in item_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[col] = (df[col] > 0).astype(int)

    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    return df, item_cols


@st.cache_data
def load_main_data(uploaded_file):
    if uploaded_file is None:
        sample = pd.DataFrame(
            {
                "Year": [2024, 2024, 2024, 2025, 2025, 2025],
                "Weekday": ["Monday", "Tuesday", "Friday", "Monday", "Wednesday", "Saturday"],
                "Customer": ["Alice", "Bob", "Carol", "Alice", "David", "Eve"],
                "Coffee": [1, 0, 1, 1, 0, 1],
                "Tea": [0, 1, 0, 0, 1, 0],
                "Bagel": [1, 0, 0, 1, 1, 0],
                "Cookie": [0, 1, 1, 0, 0, 1],
            }
        )
        return validate_and_prepare(sample)

    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    return validate_and_prepare(df)


@st.cache_data
def load_category_list(file):
    if file is None:
        return []
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    if df.empty:
        return []

    first_col = df.columns[0]
    return df[first_col].dropna().astype(str).str.strip().tolist()


def build_category_map(bakery_items, drink_items, food_items):
    category_map = {}
    for item in bakery_items:
        category_map[normalize_name(item)] = "Bakery"
    for item in drink_items:
        category_map[normalize_name(item)] = "Drink"
    for item in food_items:
        category_map[normalize_name(item)] = "Food"
    return category_map


def get_item_category(item_name, category_map):
    return category_map.get(normalize_name(item_name), "Other")


def make_long_data(filtered_df, item_cols, category_map):
    long_df = filtered_df.melt(
        id_vars=["Year", "Weekday", "Customer"],
        value_vars=item_cols,
        var_name="Item",
        value_name="Purchase",
    )

    long_df = long_df[long_df["Purchase"] == 1].copy()
    long_df["Category"] = long_df["Item"].apply(lambda x: get_item_category(x, category_map))
    return long_df


# -----------------------------
# Sidebar: uploads
# -----------------------------
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload main transaction data", type=["csv", "xlsx"])
    st.markdown("Expected columns: `Year`, `Weekday`, `Customer`, then one 0/1 column per item.")

    st.markdown("---")
    st.subheader("Upload category lists")
    bakery_file = st.file_uploader("Upload bakery items", type=["csv", "xlsx"], key="bakery")
    drink_file = st.file_uploader("Upload drink items", type=["csv", "xlsx"], key="drink")
    food_file = st.file_uploader("Upload food items", type=["csv", "xlsx"], key="food")


# -----------------------------
# Load data
# -----------------------------
try:
    df, item_cols = load_main_data(uploaded_file)
except Exception as e:
    st.error(f"Error loading main data: {e}")
    st.stop()

bakery_items = load_category_list(bakery_file)
drink_items = load_category_list(drink_file)
food_items = load_category_list(food_file)
category_map = build_category_map(bakery_items, drink_items, food_items)


# -----------------------------
# Sidebar: filters
# -----------------------------
with st.sidebar:
    st.markdown("---")
    st.header("Filters")
    year_options = sorted(df["Year"].dropna().unique().tolist())
    selected_years = st.multiselect("Select year", options=year_options, default=year_options)

filtered = df[df["Year"].isin(selected_years)].copy()

if filtered.empty:
    st.warning("No data matches the selected year filter.")
    st.stop()


# -----------------------------
# KPI cards
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Rows / Visits", f"{len(filtered):,}")
col2.metric("Unique Customers", f"{filtered['Customer'].nunique():,}")
col3.metric("Available Items", f"{len(item_cols):,}")

st.divider()


# -----------------------------
# Chart 1: Sales per Item
# -----------------------------
st.subheader("Sales per Item")
st.caption("Because each item column is coded 1 = purchased and 0 = not purchased, sales here means purchase counts.")

item_sales = filtered[item_cols].sum().reset_index()
item_sales.columns = ["Item", "Purchases"]
item_sales["Category"] = item_sales["Item"].apply(lambda x: get_item_category(x, category_map))
item_sales = item_sales.sort_values("Purchases", ascending=False)

chart1 = (
    alt.Chart(item_sales)
    .mark_bar()
    .encode(
        x=alt.X("Item:N", sort="-y", title="Item"),
        y=alt.Y("Purchases:Q", title="Number of Purchases"),
        color=alt.Color("Category:N", title="Food Category"),
        tooltip=["Item", "Category", "Purchases"],
    )
    .properties(height=420)
)

st.altair_chart(chart1, use_container_width=True)
st.dataframe(item_sales, use_container_width=True, hide_index=True)


# -----------------------------
# Chart 2: Sales per Category by Weekday and Year
# -----------------------------
st.divider()
st.subheader("Sales per Category by Weekday and Year")

long_df = make_long_data(filtered, item_cols, category_map)
long_df = long_df[long_df["Category"].isin(["Drink", "Food", "Bakery"])].copy()

if long_df.empty:
    st.warning("No mapped category data found. Check your category files.")
else:
    cat_summary = (
        long_df.groupby(["Year", "Weekday", "Category"], as_index=False)["Purchase"]
        .sum()
        .rename(columns={"Purchase": "Purchases"})
    )

    cat_summary["Weekday"] = pd.Categorical(
        cat_summary["Weekday"],
        categories=WEEKDAY_ORDER,
        ordered=True
    )
    cat_summary = cat_summary.sort_values(["Year", "Weekday", "Category"])
    cat_summary["Year-Weekday"] = cat_summary["Year"].astype(str) + " - " + cat_summary["Weekday"].astype(str)

    chart2 = (
        alt.Chart(cat_summary)
        .mark_bar()
        .encode(
            x=alt.X("Year-Weekday:N", title="Year and Weekday"),
            y=alt.Y("Purchases:Q", title="Purchases"),
            color=alt.Color("Category:N", title="Food Category"),
            tooltip=["Year", "Weekday", "Category", "Purchases"],
        )
        .properties(height=420)
    )

    st.altair_chart(chart2, use_container_width=True)
    st.dataframe(cat_summary, use_container_width=True, hide_index=True)

    # -----------------------------
    # Customer search table
    # -----------------------------
    st.divider()
    st.subheader("Customer Purchases")
    st.caption("Search for a customer and view each visit with weekday, year, purchased items, and total sales (purchase count).")

    search_name = st.text_input("Type a customer name to search", placeholder="Enter customer name")

    customer_view = filtered.copy()
    if search_name.strip():
        customer_view = customer_view[
            customer_view["Customer"].astype(str).str.contains(search_name.strip(), case=False, na=False)
        ].copy()

    if customer_view.empty:
        st.info("No customer entries match the search.")
    else:
        def extract_purchased_items(row):
            return [item for item in item_cols if row[item] == 1]

        customer_view["Purchased Items"] = customer_view.apply(extract_purchased_items, axis=1)
        customer_view["Purchased Items"] = customer_view["Purchased Items"].apply(lambda x: ", ".join(x) if x else "")
        customer_view["Total Sales"] = customer_view[item_cols].sum(axis=1)

        customer_display = customer_view[["Customer", "Year", "Weekday", "Purchased Items", "Total Sales"]].copy()
        st.dataframe(customer_display, use_container_width=True, hide_index=True)

    # -----------------------------
    # Regression model
    # Purchases = a + b*Weekday + c*Food Category
    # -----------------------------
    st.divider()
    st.subheader("Predicted Sales per Category by Weekday and Year")
    st.caption("Regression model: Purchases = a + b·Weekday + c·Food Category")

    reg_df = cat_summary.copy()
    reg_df["Weekday"] = reg_df["Weekday"].astype(str)
    reg_df["Category"] = reg_df["Category"].astype(str)

    X = pd.get_dummies(
        reg_df[["Weekday", "Category"]],
        columns=["Weekday", "Category"],
        drop_first=True
    )
    y = reg_df["Purchases"]

    if len(reg_df) >= 2 and X.shape[1] > 0:
        model = LinearRegression()
        model.fit(X, y)
        reg_df["Predicted"] = model.predict(X)
        reg_df["Predicted"] = reg_df["Predicted"].clip(lower=0)
    else:
        reg_df["Predicted"] = reg_df["Purchases"]

    reg_df["Year-Weekday"] = reg_df["Year"].astype(str) + " - " + reg_df["Weekday"].astype(str)

    bars = (
        alt.Chart(reg_df)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("Year-Weekday:N", title="Year and Weekday"),
            y=alt.Y("Purchases:Q", title="Actual Purchases"),
            color=alt.Color("Category:N", title="Food Category"),
            tooltip=["Year", "Weekday", "Category", "Purchases", "Predicted"],
        )
        .properties(height=420)
    )

    pred_points = (
        alt.Chart(reg_df)
        .mark_point(filled=True, size=90, color="black")
        .encode(
            x="Year-Weekday:N",
            y=alt.Y("Predicted:Q", title="Predicted Purchases"),
            tooltip=["Year", "Weekday", "Category", "Purchases", "Predicted"],
        )
    )

    pred_line = (
        alt.Chart(reg_df)
        .mark_line(color="black")
        .encode(
            x="Year-Weekday:N",
            y="Predicted:Q",
            detail="Category:N",
        )
    )

    st.altair_chart(bars + pred_line + pred_points, use_container_width=True)

    st.subheader("Regression Output Table")
    st.dataframe(
        reg_df[["Year", "Weekday", "Category", "Purchases", "Predicted"]],
        use_container_width=True,
        hide_index=True
    )

    with st.expander("How the regression works"):
        st.markdown(
            """
            The fitted model is:

            **Purchases = a + b·Weekday + c·Food Category**

            Since `Weekday` and `Food Category` are categorical variables,
            the app converts them into dummy variables before fitting the regression.
            """
        )