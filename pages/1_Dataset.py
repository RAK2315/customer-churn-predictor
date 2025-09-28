import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Dataset Explorer", page_icon="📊") 
st.title("📁 Customer Churn Dataset Explorer")

# ----------------------
# Load dataset robustly
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "..", "Telco-Customer-Churn.csv")
df = pd.read_csv(csv_path)

# ----------------------
# Sidebar Filters
# ----------------------
st.sidebar.header("Filters")

# Categorical Filters
categorical_cols = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'
]

categorical_filters = {}
for col in categorical_cols:
    options = list(df[col].unique())
    categorical_filters[col] = st.sidebar.multiselect(f"{col}", options, options)

# Numeric Filters
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
numeric_filters = {}
for col in numeric_cols:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    numeric_filters[col] = st.sidebar.slider(
        f"{col} range", min_val, max_val, (min_val, max_val)
    )

# Apply filters
filtered_df = df.copy()
for col, selected in categorical_filters.items():
    if selected:
        filtered_df = filtered_df[filtered_df[col].isin(selected)]
for col, (min_val, max_val) in numeric_filters.items():
    filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]

st.sidebar.write(f"**Rows Selected:** {filtered_df.shape[0]}")

# ----------------------
# Dataset Preview
# ----------------------
st.subheader("Dataset Preview")
st.dataframe(filtered_df.head(10))

st.subheader("Summary Statistics")
st.write(filtered_df.describe())

# ----------------------
# Visualization Options
# ----------------------
st.subheader("Visualizations")

# 1. Churn distribution
st.write("### Churn Distribution")
churn_count = filtered_df['Churn'].value_counts().reset_index().rename(columns={'index':'Churn','Churn':'Count'})
chart1 = alt.Chart(churn_count).mark_bar().encode(
    x='Churn:N',
    y='Count:Q',
    color='Churn:N',
    tooltip=['Churn','Count']
)
st.altair_chart(chart1, use_container_width=True)

# 2. Tenure histogram by Churn
st.write("### Tenure Distribution by Churn")
chart2 = alt.Chart(filtered_df).mark_bar(opacity=0.6).encode(
    x=alt.X('tenure:Q', bin=alt.Bin(maxbins=30)),
    y='count()',
    color='Churn:N'
)
st.altair_chart(chart2, use_container_width=True)
st.write("") 
st.write("") 
st.write("") 
# ----------------------
# User-selected chart type for charges
# ----------------------
st.subheader("Chart Type Options for Charges")
charge_chart_type = st.radio(
    "Choose chart type for MonthlyCharges & TotalCharges:",
    options=["Boxplot", "Histogram", "Average Bar Chart"]
)

# MonthlyCharges chart
st.write("### Monthly Charges by Churn")
if charge_chart_type == "Boxplot":
    chart_mc = alt.Chart(filtered_df).mark_boxplot().encode(
        x='Churn:N',
        y='MonthlyCharges:Q',
        color='Churn:N'
    )
elif charge_chart_type == "Histogram":
    chart_mc = alt.Chart(filtered_df).mark_bar(opacity=0.6).encode(
        x=alt.X('MonthlyCharges:Q', bin=alt.Bin(maxbins=30)),
        y='count()',
        color='Churn:N'
    )
elif charge_chart_type == "Average Bar Chart":
    avg_mc = filtered_df.groupby('Churn')['MonthlyCharges'].mean().reset_index()
    chart_mc = alt.Chart(avg_mc).mark_bar().encode(
        x='Churn:N',
        y='MonthlyCharges:Q',
        color='Churn:N'
    )
st.altair_chart(chart_mc, use_container_width=True)

# TotalCharges chart
st.write("### Total Charges by Churn")
if charge_chart_type == "Boxplot":
    chart_tc = alt.Chart(filtered_df).mark_boxplot().encode(
        x='Churn:N',
        y='TotalCharges:Q',
        color='Churn:N'
    )
elif charge_chart_type == "Histogram":
    chart_tc = alt.Chart(filtered_df).mark_bar(opacity=0.6).encode(
        x=alt.X('TotalCharges:Q', bin=alt.Bin(maxbins=30)),
        y='count()',
        color='Churn:N'
    )
elif charge_chart_type == "Average Bar Chart":
    avg_tc = filtered_df.groupby('Churn')['TotalCharges'].mean().reset_index()
    chart_tc = alt.Chart(avg_tc).mark_bar().encode(
        x='Churn:N',
        y='TotalCharges:Q',
        color='Churn:N'
    )
st.altair_chart(chart_tc, use_container_width=True)

st.write("") 
st.write("") 
st.write("") 
# 5. Contract Type vs Churn
st.write("### Contract Type vs Churn")
contract_counts = filtered_df.groupby(['Contract','Churn']).size().reset_index(name='Count')
chart5 = alt.Chart(contract_counts).mark_bar().encode(
    x='Contract:N',
    y='Count:Q',
    color='Churn:N',
    tooltip=['Contract','Churn','Count']
)
st.altair_chart(chart5, use_container_width=True)

# 6. Internet Service vs Churn
st.write("### Internet Service vs Churn")
internet_counts = filtered_df.groupby(['InternetService','Churn']).size().reset_index(name='Count')
chart6 = alt.Chart(internet_counts).mark_bar().encode(
    x='InternetService:N',
    y='Count:Q',
    color='Churn:N',
    tooltip=['InternetService','Churn','Count']
)
st.altair_chart(chart6, use_container_width=True)

# 7. Correlation heatmap
st.write("### Correlation Heatmap")
numeric_cols_corr = filtered_df.select_dtypes(include=['float64','int64']).columns
corr = filtered_df[numeric_cols_corr].corr()
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
plt.tight_layout()
st.pyplot(fig, clear_figure=True)

# 8. Optional scatter plot
st.write("### Scatter Plot of Two Numeric Variables")
numeric_options = list(filtered_df.select_dtypes(include=['float64','int64']).columns)
x_col = st.selectbox("X-axis", numeric_options, index=numeric_options.index('tenure') if 'tenure' in numeric_options else 0)
y_col = st.selectbox("Y-axis", numeric_options, index=numeric_options.index('MonthlyCharges') if 'MonthlyCharges' in numeric_options else 0)
scatter_chart = alt.Chart(filtered_df).mark_circle(size=60, opacity=0.6).encode(
    x=f'{x_col}:Q',
    y=f'{y_col}:Q',
    color='Churn:N',
    tooltip=[x_col, y_col, 'Churn']
)
st.altair_chart(scatter_chart, use_container_width=True)

# ----------------------
# Download Filtered Dataset
# ----------------------
st.subheader("Download Filtered Dataset")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(label="Download CSV", data=csv, file_name='filtered_telco_churn.csv', mime='text/csv')
