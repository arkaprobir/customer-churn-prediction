"""
dashboard.py
-------------
Streamlit dashboard for the Customer Segmentation + Churn Prediction project.

Run with:
    streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT_DIR / "outputs"

st.set_page_config(page_title="Customer Churn & Segmentation Dashboard", layout="wide")

st.title("📊 Customer Segmentation & Churn Prediction Dashboard")
st.caption("K-Means (from scratch) for segmentation + XGBoost for churn prediction")

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(f"{OUT_DIR}/customers_with_segments.csv")
    profile = pd.read_csv(f"{OUT_DIR}/segment_profile.csv")
    with open(f"{OUT_DIR}/metrics.json") as f:
        metrics = json.load(f)
    return df, profile, metrics

df, profile, metrics = load_data()

SEGMENT_NAMES = {
    0: "New & High-Spend (At Risk)",
    1: "Loyal & High-Value",
    2: "New & Budget (At Risk)",
    3: "Loyal & Budget",
}
df["SegmentName"] = df["Segment"].map(SEGMENT_NAMES)

# ---------------------------------------------------------------------------
# Top-level metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Overall Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
col3.metric("XGBoost AUC", f"{metrics['xgboost']['auc']:.3f}")
col4.metric("XGBoost F1 (Churn class)", f"{metrics['xgboost']['f1']:.3f}")

st.divider()

# ---------------------------------------------------------------------------
# Segmentation view
# ---------------------------------------------------------------------------
st.subheader("🧩 Customer Segments (K-Means, implemented from scratch)")

c1, c2 = st.columns([2, 1])

with c1:
    fig = px.scatter(
        df, x="tenure", y="MonthlyCharges", color="SegmentName",
        opacity=0.6, title="Segments by Tenure vs Monthly Charges",
        labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly Charges ($)"}
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.write("**Segment Profile (avg values)**")
    profile_display = profile.copy()
    profile_display["Segment"] = profile_display["Segment"].map(SEGMENT_NAMES)
    st.dataframe(profile_display.set_index("Segment"), use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Churn risk view
# ---------------------------------------------------------------------------
st.subheader("⚠️ Churn Risk by Segment")

churn_by_segment = df.groupby("SegmentName")["Churn"].mean().sort_values(ascending=False)
fig2 = px.bar(
    churn_by_segment, orientation="h",
    title="Churn Rate by Segment",
    labels={"value": "Churn Rate", "SegmentName": "Segment"}
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------------
st.subheader("🤖 Model Performance Artifacts")
c3, c4 = st.columns(2)
with c3:
    st.image(f"{OUT_DIR}/feature_importance.png", caption="Top 10 Feature Importances (XGBoost)")
with c4:
    st.image(f"{OUT_DIR}/confusion_matrix.png", caption="Confusion Matrix (XGBoost)")

st.image(f"{OUT_DIR}/elbow_plot.png", caption="Elbow Method used to select k=4 clusters")

st.divider()

# ---------------------------------------------------------------------------
# Customer lookup / explorer
# ---------------------------------------------------------------------------
st.subheader("🔍 Customer Explorer")
selected_segment = st.selectbox("Filter by segment", ["All"] + list(SEGMENT_NAMES.values()))
filtered = df if selected_segment == "All" else df[df["SegmentName"] == selected_segment]
st.dataframe(
    filtered[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn", "SegmentName"]].head(50),
    use_container_width=True
)
