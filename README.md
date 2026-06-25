# Customer Segmentation & Churn Prediction

ML project combining **unsupervised learning** (customer segmentation) and
**supervised learning** (churn prediction) on telecom customer data.

## What's inside
- `src/generate_data.py` — generates a realistic synthetic Telco dataset
  (swap with the real Kaggle "Telco Customer Churn" CSV with zero code changes)
- `src/kmeans_from_scratch.py` — K-Means implemented from scratch (NumPy only,
  K-Means++ init), validated against sklearn
- `src/pipeline.py` — full pipeline: preprocessing → segmentation → churn
  models (Logistic Regression baseline vs XGBoost) → saved artifacts
- `src/dashboard.py` — Streamlit dashboard visualizing segments + churn risk

## How to run
```bash

pip install -r requirements.txt 
python src/generate_data.py      # creates data/telco_churn.csv
python src/pipeline.py           # trains models, saves outputs/
streamlit run src/dashboard.py   # launches dashboard
```

## Results
- XGBoost: AUC 0.83, F1 0.65 (class-imbalance handled via scale_pos_weight)
- 4 customer segments identified via from-scratch K-Means, validated with
  elbow method
- Top churn drivers: contract type, tenure, internet service type, tech support
.
