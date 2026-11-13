# Credit Risk & Loan Portfolio Dashboard

A Streamlit dashboard for the **Credit Portfolio Segmentation and Risk Pattern Analysis** project.

## Package structure

```text
credit_risk_streamlit_dashboard/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── credit_kmeans.pkl
│   └── credit_scaler.pkl
└── data/
    └── credit_portfolio_dashboard.csv   # add your project CSV here
```

## 1. Install dependencies

Open Command Prompt in this folder:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The saved estimators were created with scikit-learn 1.7.2, so the requirements file pins that version to reduce pickle compatibility warnings.

## 2. Add your dataset

Copy your existing `credit_portfolio_dashboard.csv` into the `data` folder.

Alternatively, start the app and upload your CSV from the sidebar.

The dashboard can use existing cluster labels named `KMeans_Cluster` or `Cluster`. If those labels are absent, it attempts to predict clusters using the saved scaler and K-Means model.

## 3. Run the dashboard

```bat
streamlit run app.py
```

## Model input order

The uploaded scaler confirms the following five feature names and order:

1. `log_revenue`
2. `dti_n`
3. `loan_amnt`
4. `fico_n`
5. `emp_years`

For new predictions, these features must be prepared exactly as they were prepared during model training. The app prefers these engineered columns when they exist. Its fallback transformations are assumptions and must be checked against the original training notebook before using them for production scoring.

## Dashboard tabs

- Portfolio Overview
- Default Analysis
- Borrower Segmentation
- Business Recommendations

## Important notes

- The dashboard does not retrain the K-Means model.
- Cluster IDs (0–3) are arbitrary group identifiers, not automatic risk grades.
- Default analysis requires a recognizable `Default`/`default_flag` column or a `loan_status` field.
- The dashboard is for analysis and education, not automated lending decisions.
- Only load trusted `.pkl` files. Pickle files can execute code when loaded.
