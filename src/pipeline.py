"""
pipeline.py
------------
End-to-end pipeline:
  1. Load + preprocess data
  2. Customer segmentation using our from-scratch K-Means (+ elbow method to pick k)
  3. Churn prediction using Logistic Regression (baseline) vs XGBoost
  4. Evaluation with metrics suited to imbalanced classification
  5. Save artifacts (model, scaler, cluster assignments) for the dashboard

Run: python src/pipeline.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_curve, f1_score
)
from xgboost import XGBClassifier

from kmeans_from_scratch import KMeansScratch, elbow_method

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "telco_churn.csv"
OUT_DIR = ROOT_DIR / "outputs"


# ---------------------------------------------------------------------------
# 1. Load + preprocess
# ---------------------------------------------------------------------------
def load_and_preprocess(path=DATA_PATH):
    df = pd.read_csv(path)

    # Encode binary/categorical columns
    binary_map = {"Yes": 1, "No": 0}
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]:
        df[col] = df[col].map(binary_map)

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    cat_cols = ["MultipleLines", "InternetService", "OnlineSecurity",
                "TechSupport", "StreamingTV", "Contract", "PaymentMethod"]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df, df_encoded


# ---------------------------------------------------------------------------
# 2. Segmentation (Unsupervised) — using our from-scratch K-Means
# ---------------------------------------------------------------------------
def run_segmentation(df, df_encoded):
    seg_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    X_seg = df_encoded[seg_features].values
    scaler_seg = StandardScaler()
    X_seg_scaled = scaler_seg.fit_transform(X_seg)

    # Elbow method to justify choice of k
    ks, inertias = elbow_method(X_seg_scaled, k_range=range(1, 9))
    plt.figure(figsize=(6, 4))
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/elbow_plot.png", dpi=120)
    plt.close()

    # k=4 chosen from elbow plot (visible bend around 4) — interpretable segments:
    # high-value loyal / new low-spend / new high-spend at-risk / long-tenure budget
    k_final = 4
    km = KMeansScratch(n_clusters=k_final, random_state=42, n_init=10)
    km.fit(X_seg_scaled)
    df_encoded["Segment"] = km.labels_

    # Profile each segment for business interpretation
    profile = df_encoded.groupby("Segment")[seg_features + ["Churn"]].mean().round(2)
    profile["count"] = df_encoded["Segment"].value_counts().sort_index()
    profile.to_csv(f"{OUT_DIR}/segment_profile.csv")
    print("\n--- Segment Profiles ---")
    print(profile)

    joblib.dump(scaler_seg, f"{OUT_DIR}/scaler_segmentation.pkl")
    joblib.dump(km, f"{OUT_DIR}/kmeans_model.pkl")

    return df_encoded, profile


# ---------------------------------------------------------------------------
# 3. Churn prediction (Supervised)
# ---------------------------------------------------------------------------
def run_churn_prediction(df_encoded):
    feature_cols = [c for c in df_encoded.columns
                    if c not in ["customerID", "Churn", "Segment"]]
    X = df_encoded[feature_cols]
    y = df_encoded["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # --- Baseline: Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    lr_preds = lr.predict(X_test_scaled)
    results["logistic_regression"] = {
        "auc": roc_auc_score(y_test, lr_probs),
        "f1": f1_score(y_test, lr_preds),
        "report": classification_report(y_test, lr_preds, output_dict=True),
    }

    # --- XGBoost (handles class imbalance via scale_pos_weight) ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, random_state=42,
        eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)  # tree models don't need scaling
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    xgb_preds = xgb.predict(X_test)
    results["xgboost"] = {
        "auc": roc_auc_score(y_test, xgb_probs),
        "f1": f1_score(y_test, xgb_preds),
        "report": classification_report(y_test, xgb_preds, output_dict=True),
    }

    print("\n--- Model Comparison ---")
    for name, res in results.items():
        print(f"{name}: AUC={res['auc']:.3f}  F1={res['f1']:.3f}")

    # Feature importance (top 10) from XGBoost — useful for the dashboard + interviews
    importances = pd.Series(xgb.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False).head(10)
    plt.figure(figsize=(7, 5))
    importances.sort_values().plot(kind="barh")
    plt.title("Top 10 Feature Importances (XGBoost)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=120)
    plt.close()

    # Confusion matrix for the chosen model (XGBoost)
    cm = confusion_matrix(y_test, xgb_preds)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (XGBoost)")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks([0, 1], ["No Churn", "Churn"])
    plt.yticks([0, 1], ["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=120)
    plt.close()

    joblib.dump(xgb, f"{OUT_DIR}/xgb_churn_model.pkl")
    joblib.dump(scaler, f"{OUT_DIR}/scaler_churn.pkl")
    joblib.dump(feature_cols, f"{OUT_DIR}/feature_cols.pkl")

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(
            {k: {"auc": v["auc"], "f1": v["f1"]} for k, v in results.items()},
            f, indent=2
        )

    return results, X_test, y_test, xgb_preds, xgb_probs


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, df_encoded = load_and_preprocess()
    df_encoded, profile = run_segmentation(df, df_encoded)
    results, X_test, y_test, preds, probs = run_churn_prediction(df_encoded)

    # Save the segmented + scored dataset for the dashboard
    df_encoded.to_csv(OUT_DIR / "customers_with_segments.csv", index=False)
    print(f"\nAll artifacts saved to {OUT_DIR}/")
