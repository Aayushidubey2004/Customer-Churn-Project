import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Intelligence",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Customer Churn Intelligence System")

st.markdown(
    "Analyze customer churn patterns and estimate the financial "
    "impact of targeted retention offers."
)

# Load customer data (prefer local file, fallback to IBM GitHub repository)
csv_path = "Telco-Customer-Churn.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)

# Convert TotalCharges into numeric values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
).fillna(0)

# Dashboard metrics
total_customers = len(df)
churned_customers = int((df["Churn"] == "Yes").sum())
retained_customers = total_customers - churned_customers
churn_rate = churned_customers / total_customers * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Retained Customers", f"{retained_customers:,}")
col4.metric("Churn Rate", f"{churn_rate:.2f}%")

st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Customer Churn Distribution")
    churn_counts = df["Churn"].value_counts()
    st.bar_chart(churn_counts)

with col_right:
    st.subheader("Churn by Contract Type")
    contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    st.bar_chart(contract_churn)

st.markdown("---")

# ROI Simulator
st.subheader("🎯 Retention Offer ROI Simulator")
st.write("Simulate proactive retention campaigns targeting high-risk churn customers.")

slider_col, result_col = st.columns([1, 1])

with slider_col:
    discount = st.slider(
        "Select Offer Discount Percentage (%)",
        min_value=0,
        max_value=50,
        value=10,
        step=1
    )
    
    assumed_retention_rate = st.slider(
        "Assumed Campaign Success / Retention Rate (%)",
        min_value=5,
        max_value=80,
        value=30,
        step=5
    ) / 100.0

    target_customers = df[df["Churn"] == "Yes"]
    number_of_customers = len(target_customers)
    average_monthly_charge = df["MonthlyCharges"].mean()
    months = 12

    customers_retained = number_of_customers * assumed_retention_rate
    expected_savings = customers_retained * average_monthly_charge * months
    discount_cost = customers_retained * average_monthly_charge * (discount / 100) * months
    net_benefit = expected_savings - discount_cost

with result_col:
    st.markdown("### Campaign Financial Impact (12 Months)")
    st.write(f"- **Target Churned Customers:** {number_of_customers:,}")
    st.write(f"- **Expected Retained Customers:** {customers_retained:,.0f}")
    st.write(f"- **Average Monthly Charge:** ₹{average_monthly_charge:.2f}")
    st.write(f"- **Gross Revenue Retained:** ₹{expected_savings:,.2f}")
    st.write(f"- **Estimated Discount Cost:** ₹{discount_cost:,.2f}")
    st.markdown(f"### **Net Financial Benefit: ₹{net_benefit:,.2f}**")

    if net_benefit > 0:
        st.success(f"✅ The retention strategy yields an estimated positive net benefit of ₹{net_benefit:,.2f}.")
    else:
        st.warning("⚠️ The estimated discount cost exceeds retained revenue. Consider lowering discount rate.")

st.markdown("---")

# Customer Data Preview
st.subheader("📋 Customer Data Preview")
st.dataframe(df.head(20))
