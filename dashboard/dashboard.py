import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
sns.set(style="dark")


def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule="D", on="order_approved_at").agg({
        "order_id": "nunique",
        "payment_value": "sum"
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
        "order_id": "order_count",
        "payment_value": "revenue"
    }, inplace=True)
    
    return daily_orders_df

def create_sum_order_items_df(df):
    sum_order_items_df = df.groupby("product_category_name_english").order_item_id.sum().sort_values(ascending=False).reset_index()
    
    return sum_order_items_df

def create_by_city(df):
    by_city_df = all_df.groupby(by="customer_city").customer_id.nunique().sort_values(ascending=False)
    by_city_df.rename("customer_count", inplace=True)
    top_7_cities = by_city_df.head(7)
    top_7_cities_df = top_7_cities.reset_index()
    # Mengubah nama kolom agar sesuai dengan pengaturan barplot
    top_7_cities_df.rename(columns={"customer_city": "city", "customer_count": "customer_count"}, inplace=True)
    
    return top_7_cities_df

def create_by_state(df):
    by_state_df = all_df.groupby(by="customer_state").customer_id.nunique().sort_values(ascending=False)
    by_state_df.rename("customer_count", inplace=True)
    top_7_states = by_state_df.head(7)
    top_7_states_df = top_7_states.reset_index()
    # Mengubah nama kolom agar sesuai dengan pengaturan barplot
    top_7_states_df.rename(columns={"customer_state": "state", "customer_count": "customer_count"}, inplace=True)
    
    return top_7_states_df

def create_products_orders(df):
    products_orders_df = all_df.groupby(by="product_category_name_english").agg({
    "order_item_id": "sum",
    }).sort_values(by="order_item_id", ascending=False)
    products_orders_df = products_orders_df.rename(
        columns={"order_item_id": "quantity"}
    )
    
    return products_orders_df

def create_product_reviews(df):
    product_reviews_df = all_df.groupby(by="product_category_name_english").agg({
    "review_score": "mean"
    }).sort_values(by="review_score", ascending=False)

    return product_reviews_df

def create_rfm(df):
    # Tentukan tanggal referensi untuk menghitung Recency (misalnya, tanggal analisis)
    reference_date = all_df['order_approved_at'].max()

    # Grup data berdasarkan 'customer_id' dan hitung Recency, Frequency, Monetary
    rfm = all_df.groupby('customer_id').agg({
        'order_approved_at': lambda x: (reference_date - x.max()).days,  # Recency
        'order_id': 'nunique',                                   # Frequency (jumlah unik order)
        'payment_value': 'sum'                                     # Monetary (total pendapatan)
    }).reset_index()

    # Ubah nama kolom untuk memudahkan interpretasi
    rfm.columns = ['customer_id', 'Recency', 'Frequency', 'Monetary']
    
    return rfm
    
all_df = pd.read_csv("all_df.csv")
# Mengkonversi kolom datetime dengan format yang benar
datetime_columns = ["order_approved_at", "order_estimated_delivery_date"]
for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

# Mengambil rentang minimum dan maksimum tanggal dari data
min_date = all_df["order_approved_at"].min().date()  # Konversi ke format 'date'
max_date = all_df["order_approved_at"].max().date()  # Konversi ke format 'date'

# Sidebar untuk input rentang waktu
with st.sidebar:    
    start_date, end_date = st.date_input(
        label="Time Span", 
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]  # Set nilai default untuk seluruh data
    )

# Filter dataframe berdasarkan rentang waktu
main_df = all_df[(all_df["order_approved_at"].dt.date >= start_date) & 
                 (all_df["order_approved_at"].dt.date <= end_date)]


daily_orders_df = create_daily_orders_df(main_df)
sum_order_items = create_sum_order_items_df(main_df)
bycity_df = create_by_city(main_df)
bystate_df = create_by_state(main_df)
products_orders_df = create_products_orders(main_df)
product_reviews_df = create_product_reviews(main_df)
rfm_df = create_rfm(main_df)

st.header("E-Commerce Public Dashboard :money_with_wings: :chart:")

st.subheader("Daily Orders")

col1, col2 = st.columns([1,1])

with col1:
    total_orders = daily_orders_df.order_count.sum()
    st.metric("Total orders", value=total_orders)

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
st.pyplot(fig)

st.subheader("Customers Demographics")
col1, col2 = st.columns([1,1])

with col1:
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = ["#87A2FF", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    sns.barplot(
        x="customer_count",
        y="city",
        data=bycity_df,
        palette=colors
    )
    ax.set_title("Top 7 Cities by Customer Count", loc="center", fontsize=30)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis="both", labelsize=20)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = ["#87A2FF", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    sns.barplot(
        x="customer_count",
        y="state",
        data=bystate_df,
        palette=colors
    )
    ax.set_title("Top 7 States by Customer Count", loc="center", fontsize=30)
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
        
st.subheader("Top 5 Best Worst Performing Product Categories by Number of Sales")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = ["#87A2FF", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    sns.barplot(
        x="quantity",
        y="product_category_name_english",
        data=products_orders_df.head(5),
        palette=colors
    )
    ax.set_title("Top 5 Best Performing Products", loc="center", fontsize=30)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis="both", labelsize=20)
    st.pyplot(fig)
    
    with st.expander("See Explanation", expanded=False):
        st.write("""
        Based on the chart, the "bed_bath_table" category emerges as the clear leader in terms of sales performance, while the "furniture_decor" category also demonstrates strong sales. The remaining categories exhibit more moderate sales levels, suggesting a diversified product portfolio with potential for further growth.
        """)
with col2:
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = ["#C7253E", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    sns.barplot(
        x="quantity",
        y="product_category_name_english",
        data=products_orders_df.sort_values(ascending=True, by="quantity").head(5),
        palette=colors
    )
    ax.set_title("Top 5 Worst Performing Products", loc="center", fontsize=30)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.invert_xaxis()
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(axis="both", labelsize=20)
    st.pyplot(fig)
    with st.expander("See Explanation", expanded=False):
        st.write("""
        Based on the chart, the "security_and_services" category is identified as the most underperforming product, while "fashion_childrens_clothes" and "cds_dvds_musicals" also require attention. The remaining categories, "la_cuisine" and "arts_and_craftsmanship," show relatively better performance but may still benefit from targeted improvements.
        """)
    
st.subheader("Top 5 Product Categories by Highest and Lowest Average Reviews")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = ["#87A2FF", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    sns.barplot(
        x="review_score",
        y="product_category_name_english",
        data=product_reviews_df.head(5),
        palette=colors
    )
    ax.set_title("Top 5 Highest Average Reviews", loc="center", fontsize=30)
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
    colors = ["#C7253E", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    sns.barplot(
        x="review_score",
        y="product_category_name_english",
        data=product_reviews_df.sort_values(ascending=True, by="review_score").head(5),
        palette=colors
    )
    ax.set_title("Top 5 Lowest Average Reviews", loc="center", fontsize=30)
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
    
st.subheader("RFM Analysis")
col1, col2 = st.columns([2, 1])
with col1:
    # Set ukuran plot agar lebih besar
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))

    # Plot distribusi Recency
    sns.histplot(rfm_df['Recency'], kde=True, bins=20, color='blue', ax=axes[0])
    axes[0].set_title('Recency Distribution')

    # Plot distribusi Frequency
    sns.histplot(rfm_df['Frequency'], kde=True, bins=20, color='green', ax=axes[1])
    axes[1].set_title('Frequency Distribution')

    # Plot distribusi Monetary
    sns.histplot(rfm_df['Monetary'], kde=True, bins=20, color='red', ax=axes[2])
    axes[2].set_title('Monetary Distribution')

    # Mengatur layout plot
    plt.tight_layout()

    # Tampilkan plot ke dalam Streamlit
    st.pyplot(fig)

with col2:
    st.write("""
    RFM Analysis is a method used to segment customers based on three key metrics:
    
    - Recency: How long it has been since the customer last made a purchase.
    - Frequency: How often the customer makes purchases.
    - Monetary: How much money the customer has spent.
    
    The charts on the left show the distribution of each metric.
    """)
with st.expander("See Explanation", expanded=False):
        st.write("""
        - Recency Distribution: The Recency histogram shows that many customers made their last transaction within the last 100 days. This indicates that many customers are relatively new to making purchases. As the days increase, the number of customers who have made transactions decreases, which suggests that the longer the time since the last transaction, the less likely the customers are to remain active.

        - Frequency Distribution: The Frequency distribution shows that the majority of customers have only made one transaction. This means that most customers in the dataset have only transacted once. The lack of repeat transactions indicates low customer loyalty or a lack of recurring purchases.

        - Monetary Distribution: The Monetary distribution reveals that most customers generate relatively small revenues. There are a few customers who generate very high amounts of revenue, but they are outliers compared to the overall customer population. This indicates that most of the revenue comes from a small number of high-spending customers, while the majority of customers generate small revenues per transaction.
        """)

# Scatter plot untuk melihat hubungan antara Frequency dan Monetary
st.subheader("Scatter Plot: Frequency vs Monetary")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=rfm_df, x='Frequency', y='Monetary', hue='Recency', palette='cool', s=100, ax=ax)
ax.set_title('Scatter Plot Frequency vs Monetary')
ax.set_xlabel('Frequency')
ax.set_ylabel('Monetary')

# Tampilkan scatter plot ke dalam Streamlit
st.pyplot(fig)
with st.expander("See Explanation", expanded=False):
        st.write("""
        Scatter Plot (Frequency vs. Monetary): The scatter plot of Frequency vs. Monetary shows that customers who make more than one transaction are almost nonexistent. Most customers only transact once, but there is one customer who generates an exceptionally large amount of revenue. Customers with lower recency (more recent transactions) tend to have lower monetary values, while customers with higher recency exhibit more varied monetary values, ranging from very small to very large amounts.
        """)