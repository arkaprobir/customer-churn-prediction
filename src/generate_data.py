"""
generate_data.py
-----------------
Generates a synthetic but realistic telecom customer dataset, structurally
similar to the popular Kaggle "Telco Customer Churn" dataset.

Why synthetic? It lets us:
  1. Control ground-truth patterns (useful for validating clustering/model logic)
  2. Avoid external dataset licensing/download issues
  3. Explain in interviews exactly how churn-driving features were designed

Run: python src/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N = 5000  # number of customers


def generate_dataset(n=N):
    customer_id = [f"CUST{i:05d}" for i in range(n)]

    # Demographics
    gender = np.random.choice(["Male", "Female"], n)
    senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])

    # Tenure (months) - skewed towards newer + long-term customers (bimodal-ish)
    tenure = np.clip(
        np.concatenate([
            np.random.exponential(8, int(n * 0.5)),
            np.random.normal(55, 15, int(n * 0.5))
        ])[:n], 0, 72
    ).astype(int)

    # Contract type correlates with tenure
    contract = np.where(
        tenure > 40,
        np.random.choice(["Two year", "One year", "Month-to-month"], n, p=[0.6, 0.3, 0.1]),
        np.random.choice(["Two year", "One year", "Month-to-month"], n, p=[0.05, 0.25, 0.7])
    )

    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    online_security = np.where(internet_service == "No", "No internet service",
                                np.random.choice(["Yes", "No"], n, p=[0.4, 0.6]))
    tech_support = np.where(internet_service == "No", "No internet service",
                             np.random.choice(["Yes", "No"], n, p=[0.4, 0.6]))
    streaming_tv = np.where(internet_service == "No", "No internet service",
                             np.random.choice(["Yes", "No"], n, p=[0.45, 0.55]))

    phone_service = np.random.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple_lines = np.where(phone_service == "No", "No phone service",
                               np.random.choice(["Yes", "No"], n, p=[0.45, 0.55]))

    paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.6, 0.4])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.35, 0.2, 0.22, 0.23]
    )

    # Monthly charges depend on services
    base_charge = np.where(internet_service == "Fiber optic", 70,
                   np.where(internet_service == "DSL", 45, 20))
    addon_charge = (
        (online_security == "Yes").astype(int) * 5 +
        (tech_support == "Yes").astype(int) * 5 +
        (streaming_tv == "Yes").astype(int) * 8 +
        (multiple_lines == "Yes").astype(int) * 6
    )
    monthly_charges = np.round(base_charge + addon_charge + np.random.normal(0, 5, n), 2)
    monthly_charges = np.clip(monthly_charges, 18, 120)

    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 20, n), 2)
    total_charges = np.clip(total_charges, 0, None)

    # ---- Churn logic (ground truth signal, with noise) ----
    # Higher churn probability for: month-to-month, high monthly charge,
    # low tenure, fiber optic without tech support, electronic check payment
    churn_score = (
        (contract == "Month-to-month").astype(int) * 2.0
        + (tenure < 12).astype(int) * 1.5
        + (monthly_charges > 80).astype(int) * 1.0
        + (internet_service == "Fiber optic").astype(int) * 0.5
        + (tech_support == "No").astype(int) * 0.5
        + (payment_method == "Electronic check").astype(int) * 0.7
        + (senior_citizen == 1).astype(int) * 0.3
        - (partner == "Yes").astype(int) * 0.4
        - (dependents == "Yes").astype(int) * 0.3
    )
    churn_prob = 1 / (1 + np.exp(-(churn_score - 3.4)))  # sigmoid -> probability (tuned for ~27% churn rate)
    churn = np.random.binomial(1, churn_prob)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame({
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_label,
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "telco_churn.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df["Churn"].value_counts(normalize=True))
