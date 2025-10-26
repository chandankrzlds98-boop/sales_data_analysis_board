import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
plt.rcdefaults()
# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="📊 Generic Data Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# --- Title ---
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>
        📊 Generic Data Analysis Dashboard
    </h1>
    <p style='text-align: center; color: gray; font-size:16px;'>
        Upload any dataset (CSV/XLSX) and explore insights automatically with clean visuals
    </p>
    """,
    unsafe_allow_html=True
)

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload your data file", type=["csv", "xlsx"])

if uploaded_file:
    # Detect file type
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success(f"✅ File `{uploaded_file.name}` loaded successfully!")

    # --- Data Preview ---
    st.subheader("🔎 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Separate categorical and numeric columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    st.markdown(
        f"<p style='color:#1F618D'><b>Detected Categorical Columns:</b> {', '.join(categorical_cols) if categorical_cols else 'None'}</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='color:#117A65'><b>Detected Numerical Columns:</b> {', '.join(numeric_cols) if numeric_cols else 'None'}</p>",
        unsafe_allow_html=True
    )

    # --- KPIs ---
    st.subheader("📊 Aggregate KPIs")
    kpi_cols = st.columns(3)

    with kpi_cols[0]:
        if "Revenue" in df.columns:
            total_revenue = df["Revenue"].sum()
            st.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
    with kpi_cols[1]:
        if "Profit" in df.columns:
            total_profit = df["Profit"].sum()
            st.metric("📈 Total Profit", f"${total_profit:,.2f}")
    with kpi_cols[2]:
        if "Gross margin" in df.columns:
            avg_margin = df["Gross margin"].mean() * 100
            st.metric("📉 Avg Gross Margin", f"{avg_margin:.2f}%")

    # --- Correlation ---
    st.subheader("📈 Correlation Analysis")
    corr_method = st.selectbox("Select Correlation Method", ["pearson", "spearman", "kendall"])
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr(method=corr_method)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title(f"Correlation Matrix ({corr_method.title()})", fontsize=14, color="#2E4053")
        st.pyplot(fig)

    # --- Independent T-Test ---
    st.subheader("🎯 Independent T-Test Analysis")
    if categorical_cols and numeric_cols:
        group_col = st.selectbox("Grouping Column", categorical_cols)
        numeric_col = st.selectbox("Numeric Column for T-Test", numeric_cols)

        groups = df[group_col].dropna().unique()
        if len(groups) >= 2:
            group1 = st.selectbox("Select Group 1", groups)
            group2 = st.selectbox("Select Group 2", groups)

            data1 = df[df[group_col] == group1][numeric_col].dropna()
            data2 = df[df[group_col] == group2][numeric_col].dropna()

            if len(data1) > 1 and len(data2) > 1:
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)

                result_cols = st.columns(3)
                result_cols[0].metric("📊 t-statistic", f"{t_stat:.4f}")
                result_cols[1].metric("📐 Degrees of Freedom", f"{len(data1) + len(data2) - 2}")
                result_cols[2].metric("📉 p-value", f"{p_value:.4f}")

                mean1, mean2 = data1.mean(), data2.mean()
                st.markdown("### 📝 Observation & Insights")
                if p_value < 0.05:
                    st.success(
                        f"✅ On average, <b style='color:#1A5276'>{group1}</b> "
                        f"({mean1:.2f}) vs <b style='color:#B03A2E'>{group2}</b> "
                        f"({mean2:.2f}), the difference is <b>statistically significant</b> (p < 0.05).",
                        unsafe_allow_html=True
                    )
                else:
                    st.warning(
                        f"⚠️ On average, <b style='color:#1A5276'>{group1}</b> "
                        f"({mean1:.2f}) vs <b style='color:#B03A2E'>{group2}</b> "
                        f"({mean2:.2f}), the difference is <b>not statistically significant</b> (p ≥ 0.05).",
                        unsafe_allow_html=True
                    )

                # Mean comparison chart
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.barplot(x=[group1, group2], y=[mean1, mean2], palette="Set2", ax=ax)
                ax.set_ylabel(f"Mean of {numeric_col}")
                ax.set_title("Mean Comparison", fontsize=13, color="#2C3E50")
                st.pyplot(fig)
            else:
                st.error("Not enough data points in one of the groups.")

    # --- Profit vs Quantity Plot (Generic) ---
    if "Quantity" in df.columns and "Profit" in df.columns:
        st.subheader("📊 Profit vs Quantity")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=df, x="Quantity", y="Profit", hue=None, alpha=0.6, s=60, color="#2E86C1")
        sns.regplot(data=df, x="Quantity", y="Profit", scatter=False, color="red", ax=ax)
        ax.set_title("Profit vs Quantity with Regression Line", fontsize=13, color="#1B2631")
        st.pyplot(fig)
else:
    st.info("👆 Please upload a CSV or XLSX file to begin.")
