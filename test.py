import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
    # --- Beautiful Auto KPI Section ---
    st.subheader("🌈 Beautiful Auto KPIs Dashboard")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) == 0:
        st.warning("⚠️ No numerical columns found to calculate KPIs.")
    else:
        # Select top 3 numeric columns (you can change number)
        top_numeric = numeric_cols[:3]

        # Custom CSS for beautiful KPI cards
        st.markdown("""
            <style>
            .kpi-card {
                background: linear-gradient(135deg, #3498db, #8e44ad);
                padding: 20px;
                border-radius: 18px;
                color: white;
                text-align: center;
                box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                transition: all 0.3s ease-in-out;
            }
            .kpi-card:hover {
                transform: scale(1.05);
                box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            }
            .kpi-label {
                font-size: 18px;
                font-weight: 600;
            }
            .kpi-value {
                font-size: 26px;
                font-weight: bold;
                margin-top: 5px;
            }
            .kpi-delta {
                font-size: 15px;
                color: #f5f5f5;
            }
            </style>
        """, unsafe_allow_html=True)

        # Display KPI cards
        cols = st.columns(len(top_numeric))
        bg_colors = ["linear-gradient(135deg, #1ABC9C, #16A085)",
                     "linear-gradient(135deg, #3498DB, #2E86C1)",
                     "linear-gradient(135deg, #E67E22, #D35400)"]

        for i, col_name in enumerate(top_numeric):
            col_sum = df[col_name].sum(skipna=True)
            col_mean = df[col_name].mean(skipna=True)

            with cols[i]:
                st.markdown(
                    f"""
                    <div class="kpi-card" style="background:{bg_colors[i % len(bg_colors)]}">
                        <div class="kpi-label">📊 {col_name}</div>
                        <div class="kpi-value">{col_sum:,.2f}</div>
                        <div class="kpi-delta">Avg: {col_mean:,.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # --- Summary KPIs (Dataset Info) ---
        st.divider()
        st.markdown("### 📘 Dataset Overview")

        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.markdown(
                """
                <div class="kpi-card" style="background:linear-gradient(135deg,#9B59B6,#8E44AD);">
                    <div class="kpi-label">🧾 Total Rows</div>
                    <div class="kpi-value">{:,}</div>
                </div>
                """.format(df.shape[0]),
                unsafe_allow_html=True
            )
        with summary_cols[1]:
            st.markdown(
                """
                <div class="kpi-card" style="background:linear-gradient(135deg,#E74C3C,#C0392B);">
                    <div class="kpi-label">📚 Total Columns</div>
                    <div class="kpi-value">{:,}</div>
                </div>
                """.format(df.shape[1]),
                unsafe_allow_html=True
            )

        # Show summary KPIs for dataset
        st.divider()
        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.metric("🧾 Total Rows", f"{df.shape[0]:,}")
        with summary_cols[1]:
            st.metric("📚 Total Columns", f"{df.shape[1]:,}")
        with summary_cols[2]:
            missing = df.isnull().mean().mean() * 100
            st.metric("⚠️ Missing Data (%)", f"{missing:.2f}%")

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
                    st.success("✅ Statistically Significant Difference Found (p < 0.05)")
                    st.markdown(
                        f"""
                            On average, <b style='color:#1A5276'>{group1}</b> ({mean1:.2f}) 
                            vs <b style='color:#B03A2E'>{group2}</b> ({mean2:.2f}), 
                            the difference is <b style='color:green;'>statistically significant</b>.
                            """,
                        unsafe_allow_html=True
                    )
                else:
                    # Not statistically significant
                    st.warning("⚠️ No Statistically Significant Difference (p ≥ 0.05)")
                    st.markdown(
                        f"""
                            On average, <b style='color:#1A5276'>{group1}</b> ({mean1:.2f}) 
                            vs <b style='color:#B03A2E'>{group2}</b> ({mean2:.2f}), 
                            the difference is <b style='color:red;'>not statistically significant</b>.
                            """,
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

        st.markdown(
            "<div style='background-color:#FFF3CD; padding:15px; border-radius:10px; color:#856404;'>"
            "</div>",
            unsafe_allow_html=True
        )
else:
    st.info("👆 Please upload a CSV or XLSX file to begin.")
