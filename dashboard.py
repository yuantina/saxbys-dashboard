import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Saxbys Performance Dashboard", layout="wide")

st.title("☕ Saxbys Coffee Shop Performance Dashboard")
st.caption("Upload sales data and explore key KPIs.")

def make_sample_data():
    data = [
        ["2026-01-05", "Monday", "Coffee", 820],
        ["2026-01-05", "Monday", "Tea", 260],
        ["2026-01-05", "Monday", "Food", 540],
        ["2026-01-06", "Tuesday", "Coffee", 900],
        ["2026-01-06", "Tuesday", "Tea", 240],
        ["2026-01-06", "Tuesday", "Food", 575],
        ["2026-01-07", "Wednesday", "Coffee", 980],
        ["2026-01-07", "Wednesday", "Tea", 230],
        ["2026-01-07", "Wednesday", "Food", 610],
        ["2026-01-08", "Thursday", "Coffee", 1010],
        ["2026-01-08", "Thursday", "Tea", 255],
        ["2026-01-08", "Thursday", "Food", 640],
        ["2026-01-09", "Friday", "Coffee", 1150],
        ["2026-01-09", "Friday", "Tea", 290],
        ["2026-01-09", "Friday", "Food", 720],
        ["2026-01-10", "Saturday", "Coffee", 1240],
        ["2026-01-10", "Saturday", "Tea", 305],
        ["2026-01-10", "Saturday", "Food", 790],
        ["2026-01-11", "Sunday", "Coffee", 980],
        ["2026-01-11", "Sunday", "Tea", 275],
        ["2026-01-11", "Sunday", "Food", 680],
    ]
    return pd.DataFrame(data, columns=["date", "weekday", "category", "sales"])

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = make_sample_data()

df["date"] = pd.to_datetime(df["date"], errors="coerce")

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["weekday"] = pd.Categorical(df["weekday"], categories=weekday_order, ordered=True)

categories = st.multiselect(
    "Select category",
    options=sorted(df["category"].unique()),
    default=sorted(df["category"].unique())
)

filtered = df[df["category"].isin(categories)].copy()

total_sales = filtered["sales"].sum()
st.metric("Total Sales", f"${total_sales:,.0f}")

summary = (
    filtered.groupby(["weekday", "category"], as_index=False)["sales"]
    .sum()
    .sort_values(["weekday", "category"])
)

chart = (
    alt.Chart(summary)
    .mark_bar()
    .encode(
        x=alt.X("weekday:N", sort=weekday_order, title="Weekday"),
        y=alt.Y("sales:Q", title="Sales ($)"),
        color="category:N",
        tooltip=["weekday", "category", "sales"]
    )
    .properties(height=400)
)

st.altair_chart(chart, use_container_width=True)

st.subheader("Summary Table")
st.dataframe(summary, use_container_width=True)