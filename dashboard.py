import streamlit as st
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression


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
# Load category mapping files BEFORE charts
# -----------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("Upload category lists")
    bakery_file = st.file_uploader("Upload bakery items (bakery.csv)", type=["csv"], key="bakery")
    drink_file = st.file_uploader("Upload drink items (drink.csv)", type=["csv"], key="drink")
    food_file = st.file_uploader("Upload food items (food.csv)", type=["csv"], key="food")


def load_category_list(file):
    if file is None:
        return []
    df_cat = pd.read_csv(file)
    first_col = df_cat.columns[0]
    return df_cat[first_col].dropna().astype(str).str.strip().tolist()

bakery_items = load_category_list(bakery_file)
drink_items = load_category_list(drink_file)
food_items = load_category_list(food_file)

category_map = {}
for item in bakery_items:
    category_map[item] = "Bakery"
for item in drink_items:
    category_map[item] = "Drink"
for item in food_items:
    category_map[item] = "Food"

# -----------------------------
# Chart 1: Sales per item (with category color)
# -----------------------------
st.subheader("Sales per Item")
st.caption("Because each item column is coded 1 = purchased, 0 = not purchased, sales here means purchase counts.")

item_sales = (
    filtered[item_cols]
    .sum()
    .reset_index()
)
item_sales.columns = ["Food Item", "Purchases"]

# Ensure category_map exists
if 'category_map' in locals():
    item_sales["Category"] = item_sales["Food Item"].map(category_map).fillna("Other")
else:
    item_sales["Category"] = "Unknown"

item_sales = item_sales.sort_values("Purchases", ascending=False)

bar = (
    alt.Chart(item_sales)
    .mark_bar()
    .encode(
        x=alt.X("Food Item:N", sort="-y", title="Item"),
        y=alt.Y("Purchases:Q", title="Number of Purchases"),
        color=alt.Color("Category:N", title="Food Category"),
        tooltip=["Food Item", "Category", "Purchases"],
    )
    .properties(height=420)
)

st.altair_chart(bar, use_container_width=True)

st.subheader("Sales Table")
st.dataframe(item_sales, use_container_width=True, hide_index=True)

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

    display_cols = ["Customer", "Year", "Weekday", "Purchased Items", "Total Sales"]
    customer_display = customer_view[display_cols].copy()

    st.dataframe(customer_display, use_container_width=True, hide_index=True)

    customer_csv = customer_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download customer search results",
        data=customer_csv,
        file_name="customer_purchases.csv",
        mime="text/csv",
    )

# -----------------------------
# Regression model:
# Purchases = a + b*Weekday + c*Category
# One-hot encode both Weekday and Category
# -----------------------------
st.divider()
st.subheader("Predicted Sales per Category by Weekday and Year")
st.caption("Linear regression model: Purchases = a + b·Weekday + c·Category")

if cat_summary.empty:
    st.info("No category summary available for regression.")
else:
    reg_df = cat_summary.copy()

    # Use Weekday and Category directly
    reg_df["Weekday"] = reg_df["Weekday"].astype(str)
    reg_df["Category"] = reg_df["Category"].astype(str)

    y = reg_df["Purchase"]

    X = pd.get_dummies(
        reg_df[["Weekday", "Category"]],
        columns=["Weekday", "Category"],
        drop_first=True
    )

    if len(reg_df) >= 2 and X.shape[1] > 0:
        model = LinearRegression()
        model.fit(X, y)
        reg_df["Predicted"] = pd.Series(model.predict(X), index=reg_df.index).clip(lower=0)
    else:
        reg_df["Predicted"] = y

    reg_df["Year-Weekday"] = (
        reg_df["Year"].astype(str) + " - " + reg_df["Weekday"].astype(str)
    )

    actual_bars = (
        alt.Chart(reg_df)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("Year-Weekday:N", title="Year & Weekday", sort=None),
            y=alt.Y("Purchase:Q", title="Actual Purchases", stack=None),
            xOffset="Category:N",
            color=alt.Color("Category:N", title="Food Category"),
            tooltip=["Year", "Weekday", "Category", "Purchase", "Predicted"]
        )
        .properties(height=420)
    )

    predicted_points = (
        alt.Chart(reg_df)
        .mark_point(filled=True, size=100, color="black")
        .encode(
            x=alt.X("Year-Weekday:N", sort=None),
            xOffset=alt.XOffset("Category:N"),
            y=alt.Y("Predicted:Q", title="Predicted Purchases"),
            tooltip=["Year", "Weekday", "Category", "Purchase", "Predicted"]
        )
    )

    st.altair_chart(actual_bars + predicted_points, use_container_width=True)

    st.subheader("Regression Output Table")
    regression_output = reg_df[
        ["Year", "Weekday", "Category", "Purchase", "Predicted"]
    ].rename(columns={"Purchase": "Actual Purchases"})

    st.dataframe(regression_output, use_container_width=True, hide_index=True)

    with st.expander("How the regression works"):
        st.markdown(
            """
            The model fitted is:

            **Purchases = a + b·Weekday + c·Category**

            Both **Weekday** and **Category** are treated as categorical variables,
            so the app uses **one-hot encoding** for both before fitting the regression.
            """
        )


# -----------------------------
# Association Rule Mining
# -----------------------------
st.divider()
st.subheader("Top 20 Association Rules")
st.caption("Association rules based on item co-purchases (Apriori algorithm)")

from mlxtend.frequent_patterns import apriori, association_rules

try:
    # Use only item columns (0/1 matrix)
    basket = filtered[item_cols].copy()

    # Ensure boolean (mlxtend prefers True/False)
    basket = basket.astype(bool)

    if basket.shape[0] < 2:
        st.info("Not enough data for association rule mining.")
    else:
        # Frequent itemsets
        frequent_itemsets = apriori(
            basket,
            min_support=0.01,   # adjust if needed
            use_colnames=True
        )

        if frequent_itemsets.empty:
            st.warning("No frequent itemsets found. Try lowering support.")
        else:
            # Generate rules
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=0.01   # adjust threshold
            )

            if rules.empty:
                st.warning("No association rules found. Try lowering thresholds.")
            else:
                # Clean formatting
                rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
                rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

                rules_display = rules[
                    ["antecedents", "consequents", "support", "confidence", "lift"]
                ].sort_values(by="lift", ascending=False)

                st.dataframe(
                    rules_display.head(20),
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download all rules",
                    data=rules_display.to_csv(index=False).encode("utf-8"),
                    file_name="association_rules.csv",
                    mime="text/csv"
                )

                with st.expander("How to interpret association rules"):
                    st.markdown(
                        """
                        - **Support**: how often the itemset appears in all transactions  
                        - **Confidence**: probability of buying RHS given LHS  
                        - **Lift**: how much stronger the rule is vs random chance  

                        Example:
                        - Rule: *Coffee → Bagel*
                        - Confidence = 0.6 → 60% of coffee buyers also buy bagel  
                        - Lift > 1 → meaningful positive association  
                        """
                    )

except Exception as e:
    st.error(f"Error in association rule mining: {e}")


# -----------------------------
# Customer Segmentation (KMeans)
# -----------------------------
st.divider()
st.subheader("Customer Segmentation")
st.caption("Cluster customers based on purchase patterns (0 = no purchase, 1 = purchase)")

from sklearn.cluster import KMeans

try:
    # Step 1: aggregate purchases per customer
    customer_matrix = (
        filtered.groupby("Customer")[item_cols]
        .sum()
    )

    if customer_matrix.shape[0] < 2:
        st.info("Not enough customers for clustering.")
    else:
        # Optional normalization (recommended)
        # Prevent heavy buyers from dominating clusters
        X = customer_matrix.div(customer_matrix.sum(axis=1), axis=0).fillna(0)

        # Step 2: choose number of clusters
        n_clusters = st.slider("Number of clusters", 2, 10, 3)

        # Step 3: fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        customer_matrix["Cluster"] = labels

        # -----------------------------
        # Cluster assignments table
        # -----------------------------
        st.subheader("Customer Clusters")
        st.dataframe(customer_matrix.reset_index(), use_container_width=True)

        # -----------------------------
        # Cluster sizes
        # -----------------------------
        cluster_counts = customer_matrix["Cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]

        chart = (
            alt.Chart(cluster_counts)
            .mark_bar()
            .encode(
                x="Cluster:O",
                y="Count:Q",
                tooltip=["Cluster", "Count"]
            )
        )

        st.subheader("Cluster Sizes")
        st.altair_chart(chart, use_container_width=True)

        # -----------------------------
        # Cluster profiles
        # -----------------------------
        profile = (
            customer_matrix
            .groupby("Cluster")[item_cols]
            .mean()
        )

        st.subheader("Cluster Profiles (average purchases)")
        st.dataframe(profile, use_container_width=True)

        # -----------------------------
    # View selected cluster (Customer-colored bars)
    # -----------------------------
    st.subheader("Sales by Item for Selected Cluster")

    cluster_options = sorted(customer_matrix["Cluster"].unique().tolist())
    selected_cluster = st.selectbox("Choose a cluster", options=cluster_options)

    cluster_customers = customer_matrix[
        customer_matrix["Cluster"] == selected_cluster
    ]

    if cluster_customers.empty:
        st.info("No customers in this cluster.")
    else:
        # Convert to long format: item x customer
        long_cluster = (
            cluster_customers[item_cols]
            .reset_index()
            .melt(
                id_vars="Customer",
                var_name="Item",
                value_name="Purchases"
            )
        )

        # Keep only positive purchases
        long_cluster = long_cluster[long_cluster["Purchases"] > 0]

        if long_cluster.empty:
            st.info("No purchases in this cluster.")
        else:
            chart = (
                alt.Chart(long_cluster)
                .mark_bar()
                .encode(
                    x=alt.X("Item:N", sort="-y", title="Item"),
                    y=alt.Y("Purchases:Q", title="Purchases"),
                    color=alt.Color("Customer:N", title="Customer"),  # 👈 key change
                    tooltip=["Customer", "Item", "Purchases"]
                )
                .properties(height=420)
            )

            st.altair_chart(chart, use_container_width=True)

            with st.expander("Show data table"):
                st.dataframe(long_cluster, use_container_width=True, hide_index=True)

        # -----------------------------
        # Explanation
        # -----------------------------
        with st.expander("How clustering works"):
            st.markdown(
                """
                - Each customer is represented by a vector of item purchases (0/1)
                - KMeans groups customers with similar purchase patterns
                - Normalization ensures fairness across heavy vs light buyers
                - Cluster profile shows average purchase rate per item
                """
            )

except Exception as e:
    st.error(f"Error in customer segmentation: {e}")


# -----------------------------
# Recommendation System
# -----------------------------
st.divider()
st.subheader("Item Recommendation")
st.caption("Recommend 5 items based on the customer's cluster. Items are sampled randomly with probabilities proportional to cluster sales, with zero-sales items set to 0.1 for exploration.")

import numpy as np

try:
    if "Cluster" not in customer_matrix.columns:
        st.info("Run customer segmentation first to generate clusters.")
    else:
        customer_options = sorted(customer_matrix.index.tolist())
        selected_customer = st.selectbox("Choose a customer for recommendation", options=customer_options)

        # Find customer's cluster
        customer_cluster = customer_matrix.loc[selected_customer, "Cluster"]
        st.write(f"**Customer:** {selected_customer}")
        st.write(f"**Assigned Cluster:** {customer_cluster}")

        # Cluster-level sales
        cluster_data = customer_matrix[customer_matrix["Cluster"] == customer_cluster].copy()
        cluster_item_sales = cluster_data[item_cols].sum()

        # Encourage exploration
        adjusted_sales = cluster_item_sales.replace(0, 0.1)

        recommendation_df = adjusted_sales.reset_index()
        recommendation_df.columns = ["Item", "Adjusted Cluster Sales"]

        recommendation_df["Probability"] = (
            recommendation_df["Adjusted Cluster Sales"] / recommendation_df["Adjusted Cluster Sales"].sum()
        )

        # -----------------------------
        # Sampling with session state
        # -----------------------------
        if "sampled_items" not in st.session_state:
            st.session_state.sampled_items = None

        def sample_items():
            n_recommend = min(5, len(recommendation_df))
            sampled = np.random.choice(
                recommendation_df["Item"],
                size=n_recommend,
                replace=False,
                p=recommendation_df["Probability"]
            )
            st.session_state.sampled_items = list(sampled)

        # First time OR new customer → sample automatically
        if st.session_state.sampled_items is None:
            sample_items()

        # Button to resample
        if st.button("Sample Again"):
            sample_items()

        sampled_items = st.session_state.sampled_items

        sampled_df = recommendation_df[
            recommendation_df["Item"].isin(sampled_items)
        ].copy()

        # Keep sampled order
        sampled_df["Order"] = sampled_df["Item"].apply(lambda x: sampled_items.index(x))
        sampled_df = sampled_df.sort_values("Order").drop(columns="Order")

        st.subheader("Recommended Items")
        st.dataframe(
            sampled_df[["Item", "Adjusted Cluster Sales", "Probability"]],
            use_container_width=True,
            hide_index=True,
        )

        rec_chart = (
            alt.Chart(sampled_df)
            .mark_bar()
            .encode(
                x=alt.X("Item:N", title="Recommended Item"),
                y=alt.Y("Probability:Q", title="Recommendation Probability"),
                tooltip=[
                    "Item",
                    "Adjusted Cluster Sales",
                    alt.Tooltip("Probability:Q", format=".3f")
                ],
            )
            .properties(height=350)
        )

        st.altair_chart(rec_chart, use_container_width=True)

        with st.expander("How recommendations are generated"):
            st.markdown(
                """
                - Find the selected customer's cluster
                - Compute item sales within that cluster
                - Replace zero-sales items with 0.1 to encourage exploration
                - Convert adjusted sales into probabilities
                - Randomly sample 5 items
                - Click **Sample Again** to generate new recommendations
                """
            )

except Exception as e:
    st.error(f"Error in recommendation section: {e}")