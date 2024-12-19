import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from babel.numbers import format_currency

# Set Seaborn style
sns.set(style="dark")

# Caching data loading
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    datetime_columns = ["order_approved_at", "order_estimated_delivery_date", "order_purchase_timestamp", "order_delivered_customer_date"]
    for column in datetime_columns:
        df[column] = pd.to_datetime(df[column])
    return df

# Load data
all_df = load_data("dashboard/all_df.csv")

# Define data processing functions
def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule="D", on="order_approved_at").agg({
        "order_id": "nunique",
        "payment_value": "sum"
    }).reset_index().rename(columns={
        "order_id": "order_count",
        "payment_value": "revenue"
    })
    return daily_orders_df

def create_sum_order_items_df(df):
    sum_order_items_df = df.groupby("product_category_name_english").order_item_id.sum().sort_values(ascending=False).reset_index()
    return sum_order_items_df

def create_by_city(df):
    by_city_df = df.groupby(by="customer_city").customer_unique_id.nunique().sort_values(ascending=False)
    by_city_df.rename("customer_count", inplace=True)
    top_7_cities = by_city_df.head(7)
    top_7_cities_df = top_7_cities.reset_index().rename(columns={"customer_city": "city"})
    return top_7_cities_df

def create_by_state(df):
    by_state_df = df.groupby(by="customer_state").customer_unique_id.nunique().sort_values(ascending=False)
    by_state_df.rename("customer_count", inplace=True)
    top_7_states = by_state_df.head(7)
    top_7_states_df = top_7_states.reset_index().rename(columns={"customer_state": "state"})
    return top_7_states_df

def create_products_orders(df):
    products_orders_df = df.groupby(by="product_category_name_english").agg({
        "order_item_id": "sum",
    }).sort_values(by="order_item_id", ascending=False).rename(columns={"order_item_id": "quantity"})
    return products_orders_df

def create_product_reviews(df):
    product_reviews_df = df.groupby(by="product_category_name_english").agg({
        "review_score": "mean"
    }).sort_values(by="review_score", ascending=False).reset_index()
    return product_reviews_df

def create_rfm(df):
    reference_date = df['order_approved_at'].max()
    rfm = df.groupby('customer_unique_id').agg({
        'order_approved_at': lambda x: (reference_date - x.max()).days,  # Recency
        'order_id': 'nunique',                                            # Frequency
        'payment_value': 'sum'                                            # Monetary
    }).reset_index().rename(columns={
        'order_approved_at': 'Recency',
        'order_id': 'Frequency',
        'payment_value': 'Monetary'
    })
    return rfm

# Extract min and max dates
min_date = all_df["order_approved_at"].min().date()
max_date = all_df["order_approved_at"].max().date()

# Sidebar
st.sidebar.title("Dashboard Navigation")

# Date range selection
start_date, end_date = st.sidebar.date_input(
    label="Time Span",
    min_value=min_date,
    max_value=max_date,
    value=[min_date, max_date]
)

# Static Outline
st.sidebar.header("List of Contents")
st.sidebar.markdown("""
- Daily Orders
- Customers Demographics
- Top Performing Products
- Product Reviews
- RFM Analysis
- Product Recommendations
""")

# Filter data based on selected dates
main_df = all_df[(all_df["order_approved_at"].dt.date >= start_date) & 
                (all_df["order_approved_at"].dt.date <= end_date)]

# Process data
daily_orders_df = create_daily_orders_df(main_df)
sum_order_items = create_sum_order_items_df(main_df)
bycity_df = create_by_city(main_df)
bystate_df = create_by_state(main_df)
products_orders_df = create_products_orders(main_df)
product_reviews_df = create_product_reviews(main_df)
rfm_df = create_rfm(main_df)

# Streamlit App Layout
st.header("E-Commerce Public Dashboard :money_with_wings: :chart:")

# Daily Orders Section
st.subheader("Daily Orders")
col1, col2 = st.columns(2)

with col1:
    total_orders = daily_orders_df.order_count.sum()
    st.metric("Total Orders", value=total_orders)

with col2:
    total_revenue = format_currency(daily_orders_df.revenue.sum(), "BRL ", locale="pt_BR")
    st.metric("Total Revenue", value=total_revenue)

fig, ax = plt.subplots(figsize=(16,8))
ax.plot(
    daily_orders_df["order_approved_at"],
    daily_orders_df["order_count"],
    marker='o', 
    linewidth=2,
    color="#87A2FF"
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
plt.tight_layout()
st.pyplot(fig)

# Customers Demographics Section
st.subheader("Customers Demographics")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = ["#87A2FF"] + ["#D3D3D3"]*6
    sns.barplot(
        x="customer_count",
        y="city",
        data=bycity_df,
        palette=colors
    )
    ax.set_title("Top 7 Cities by Customer Count", fontsize=30)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis="both", labelsize=20)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = ["#87A2FF"] + ["#D3D3D3"]*6
    sns.barplot(
        x="customer_count",
        y="state",
        data=bystate_df,
        palette=colors
    )
    ax.set_title("Top 7 States by Customer Count", fontsize=30)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis="both", labelsize=20)
    plt.tight_layout()
    st.pyplot(fig)

with st.expander("See Explanation", expanded=False):
    st.write("""
    Based on the chart, the customer base for the analyzed company is heavily concentrated in Sao Paulo, with Rio de Janeiro and Belo Horizonte also representing significant markets. While the other cities have a more even distribution of customers, Sao Paulo's dominance suggests a potential need for further market penetration or targeted marketing strategies in other regions to achieve a more balanced customer distribution.
    
    In conclusion, the chart reveals a clear customer concentration in Sao Paulo and highlights the need for further market expansion and diversification in other regions. By considering additional factors such as data source, customer demographics, and competitive landscape, the company can develop more effective strategies to optimize its customer base and achieve sustainable growth.
    """)

# Top Performing Products Section
st.subheader("Top 5 Best & Worst Performing Product Categories by Number of Sales")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = ["#87A2FF"] + ["#D3D3D3"]*4
    sns.barplot(
        x="quantity",
        y="product_category_name_english",
        data=products_orders_df.head(5),
        palette=colors
    )
    ax.set_title("Top 5 Best Performing Products", fontsize=30)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis="both", labelsize=20)
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("See Explanation", expanded=False):
        st.write("""
        Based on the chart, the "bed_bath_table" category emerges as the clear leader in terms of sales performance, while the "furniture_decor" category also demonstrates strong sales. The remaining categories exhibit more moderate sales levels, suggesting a diversified product portfolio with potential for further growth.
        """)

with col2:
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = ["#C7253E"] + ["#D3D3D3"]*4
    sns.barplot(
        x="quantity",
        y="product_category_name_english",
        data=products_orders_df.sort_values(by="quantity").head(5),
        palette=colors
    )
    ax.set_title("Top 5 Worst Performing Products", fontsize=30)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.invert_xaxis()
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(axis="both", labelsize=20)
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("See Explanation", expanded=False):
        st.write("""
        Based on the chart, the "security_and_services" category is identified as the most underperforming product, while "fashion_childrens_clothes" and "cds_dvds_musicals" also require attention. The remaining categories, "la_cuisine" and "arts_and_craftsmanship," show relatively better performance but may still benefit from targeted improvements.
        """)

# Product Reviews Section
st.subheader("Top 5 Product Categories by Highest and Lowest Average Reviews")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = ["#87A2FF"] + ["#D3D3D3"]*4
    sns.barplot(
        x="review_score",
        y="product_category_name_english",
        data=product_reviews_df.head(5),
        palette=colors
    )
    ax.set_title("Top 5 Highest Average Reviews", fontsize=30)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis="both", labelsize=20)
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("See Explanation", expanded=False):
        st.write("""
        Based on the chart, the "fashion_childrens_clothes" category emerges as the clear leader in terms of customer satisfaction, while the other categories also demonstrate strong customer ratings. This suggests that the company is offering products that meet or exceed customer expectations in these areas.
        """)

with col2:
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = ["#C7253E"] + ["#D3D3D3"]*4
    sns.barplot(
        x="review_score",
        y="product_category_name_english",
        data=product_reviews_df.sort_values(by="review_score").head(5),
        palette=colors
    )
    ax.set_title("Top 5 Lowest Average Reviews", fontsize=30)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.invert_xaxis()
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(axis="both", labelsize=20)
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("See Explanation", expanded=False):
        st.write("""
        Based on the chart, the "security_and_services" category is identified as the most underperforming product in terms of customer satisfaction, while the other categories also require attention. This suggests that the company may need to address specific issues or concerns within these product areas to improve customer ratings and overall satisfaction.
        """)

# Customer Segmentation (RFM Analysis)
st.subheader("Customer Segmentation")

# Define RFM segments with corresponding customer counts
rfm_segments = {
    'Loyal Customers': 50167,
    'Best Customers': 29980,
    'At Risk': 13737,
    'Lost Customers': 1536
}

# Create a plot
fig, ax = plt.subplots()

# Create a bar plot for the RFM segments
sns.barplot(x=list(rfm_segments.keys()), y=list(rfm_segments.values()), palette='Set2', ax=ax)

# Set plot title and labels
ax.set_title("Customer Segments Distribution")
ax.set_xlabel("Segment")
ax.set_ylabel("Number of Customers")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot in the Streamlit app
st.pyplot(fig)
with st.expander("See Explanation", expanded=False):
        st.write("""
        The RFM analysis reveals valuable insights into customer segmentation. The largest group, "Loyal Customers," represents over 50,000 customers who purchase frequently and are highly engaged, making them a key segment for retention strategies. "Best Customers," totaling around 30,000, are the most valuable due to their high spending and recent activity, and they should be prioritized for upselling and reward programs. The "At Risk" segment highlights customers who have not purchased recently, warranting re-engagement campaigns to prevent churn. Finally, the "Lost Customers" segment, though the smallest, represents customers who are inactive and may require strategic win-back efforts or can be deprioritized. These insights allow businesses to tailor their marketing strategies effectively, focusing on high-value customers while addressing potential churn.
        """)

st.caption('© Naufal Hadi Darmawan')
