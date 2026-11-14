# Credit Portfolio Segmentation and Risk Pattern Analysis

Unsupervised segmentation of a 1.35-million-loan Lending Club portfolio into borrower groups, followed by an analysis of how historical default rates differ across those groups.

**Domain:** Banking and Financial Services
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, UMAP, Jupyter Notebook

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Principle: No Target Leakage](#key-principle-no-target-leakage)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Key Findings](#key-findings)
- [Business Recommendations](#business-recommendations)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Limitations](#limitations)

---

## Project Overview

Lenders need to understand who their borrowers are and where credit risk concentrates. This project:

1. Cleans and explores the loan portfolio.
2. Engineers and scales borrower features for distance-based clustering.
3. Reduces dimensionality (PCA, t-SNE, UMAP) to study structure.
4. Segments borrowers with K-Means, Hierarchical Clustering and DBSCAN.
5. Compares historical default rates across segments.
6. Detects unusual borrower profiles (anomalies).
7. Translates results into business insights and recommendations.

## Key Principle: No Target Leakage

`Default` is **never** used as a clustering input. It is kept aside (with loan ID, state, issue date, original revenue and DTI) and used only **after** clustering to profile the risk of each segment.

## Dataset

| Item | Value |
|---|---|
| Source file | `loans_granting_model_dataset.csv` (Lending Club loans) |
| Records | 1,347,681 |
| Original columns | 15 |
| Duplicate rows / loan IDs | 0 / 0 |
| Overall default rate | 19.98% (269,249 defaults) |
| Total loan value | about $19.42 billion |

| Column | Description |
|---|---|
| `id` | Unique loan identifier |
| `issue_d` | Loan issue date |
| `revenue` | Borrower annual income |
| `dti_n` | Debt-to-income ratio |
| `loan_amnt` | Loan amount |
| `fico_n` | FICO credit score |
| `experience_c` | Experience flag (near-constant, excluded) |
| `emp_length` | Employment length |
| `purpose` | Loan purpose |
| `home_ownership_n` | Home ownership status |
| `addr_state`, `zip_code` | Geography |
| `title`, `desc` | Free text (mostly missing, dropped) |
| `Default` | Historical default outcome (post-clustering analysis only) |

> The raw dataset is not included in this repository because of its size. Place it in the project folder before running the notebooks.

## Workflow

| Step | Notebook | What it does |
|---|---|---|
| 1 | `Step 1 - corrected.ipynb` | Data collection, understanding, project objectives |
| 2 | `Step 2 - corrected.ipynb` | Preprocessing: missing values, duplicates, outliers, encoding, scaling |
| 3 | `Step_3_corrected.ipynb` | Exploratory data analysis: portfolio, default, geography, time, correlation, VIF |
| 4 | `Step_4_EDA_corrected.ipynb` | Feature engineering and scaling for clustering (34-feature matrix) |
| 5 | `Step_5_Dimensionality_Reduction_corrected.ipynb` | PCA, t-SNE and UMAP |
| 6-7 | *(clustering: K-Means, Hierarchical, DBSCAN)* | Add notebooks here if included in the repository |
| 8 | `Step_8_Default_pattern_analysis.ipynb` | Default rates by cluster |
| 9 | `Step_9_Anomaly_detection.ipynb` | DBSCAN noise points and IQR feature outliers |
| 10 | `Step_10_Business_Insight.ipynb` | Segment insights, visualizations, recommendations |

### Preprocessing highlights
- No records removed; `desc` (91% missing), `title` and `zip_code` dropped.
- `revenue` log-transformed (skewness 45.4 to 0.21).
- `dti_n` capped at 65 (about the 99.9th percentile); an `dti_extreme` flag (DTI of 100 or more, default rate 26.73%) is kept for profiling rather than imputing the median.
- `loan_amnt` log-transformed; numeric features standardized with `StandardScaler`.
- `purpose`, `home_ownership_n`, `emp_length` one-hot encoded.
- Clustering matrix: 1,347,681 rows x 34 features (4 scaled numeric + 30 binary).

### Dimensionality reduction
- Incremental PCA on all rows: 10 components retain **88.95%** of variance.
- t-SNE and UMAP on a 5,000-row sample, for visual exploration only.

## Key Findings

### Default rate by borrower characteristic (full portfolio)

| Factor | Lowest-risk group | Highest-risk group |
|---|---|---|
| FICO score | 760-850: 8.32% | 640-680: 25.30% |
| DTI | 10 or below: 14.84% | Above 40: 30.67% |
| Home ownership | Mortgage: 17.23% | Rent: 23.23% |
| Employment length | 10+ years: 18.80% | Not provided (NI): 26.96% |
| Loan purpose | Wedding: 12.43% | Small business: 29.86% |
| State (at least 1,000 borrowers) | DC 13.23% | MS 26.12% |

### K-Means segments (K = 4, full data, Step 10)

| Cluster | Borrowers | Default rate | Avg FICO | Avg DTI | Avg revenue | Profile |
|---|---|---|---|---|---|---|
| 0 | 167,418 | 10.45% | 742.8 | 13.3 | $58,974 | Low risk: strong credit, low DTI |
| 3 | 220,039 | 14.12% | 731.2 | 17.9 | $130,232 | Lower risk: high income, large loans |
| 1 | 515,723 | 22.61% | 681.9 | 16.6 | $88,322 | Largest segment, moderate-to-high risk |
| 2 | 444,501 | 23.42% | 683.9 | 22.1 | $45,423 | Highest risk: low income, high DTI |

Overall portfolio default rate: 19.98%.

### Anomaly detection (Step 9, 5,000-borrower sample)
- DBSCAN (eps = 1.8, min_samples = 10, on 10 PCA components) flagged **263 potential anomalies (5.26%)**.
- Anomalies had higher average FICO (729 vs 696) and revenue, smaller loans, and a **lower** default rate (17.11% vs 20.94%). Unusual does not mean risky.

## Business Recommendations

1. **Credit assessment:** weight FICO heavily, alongside income, DTI and repayment history.
2. **DTI monitoring:** add affordability checks for DTI above 30 (default rate above 29%).
3. **Verification:** verify employment and income where employment length is not provided.
4. **Purpose risk:** apply tailored policies to small-business loans.
5. **Geographic monitoring:** track state-level performance, accounting for portfolio size.

## Repository Structure

```
.
├── README.md
├── notebooks/
│   ├── Step 1 - corrected.ipynb
│   ├── Step 2 - corrected.ipynb
│   ├── Step_3_corrected.ipynb
│   ├── Step_4_EDA_corrected.ipynb
│   ├── Step_5_Dimensionality_Reduction_corrected.ipynb
│   ├── Step_8_Default_pattern_analysis.ipynb
│   ├── Step_9_Anomaly_detection.ipynb
│   └── Step_10_Business_Insight.ipynb
└── step2_outputs/            # generated by Step 2 (not committed)
    ├── cleaned_credit_portfolio.csv / .parquet
    ├── clustering_features_scaled.parquet
    ├── audit_table.parquet
    ├── standard_scaler.joblib
    └── feature_lists.json
```

## How to Run

```bash
git clone <your-repo-url>
cd <your-repo>
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib pyarrow umap-learn jupyter
```

1. Place `loans_granting_model_dataset.csv` in the project folder.
2. Update `DATA_PATH` in Step 2 (currently a local Windows path) to your file location.
3. Run the notebooks in order. Steps 3 onward read `step2_outputs/cleaned_credit_portfolio.csv`, which Step 2 creates.

## Limitations

- Hierarchical clustering, DBSCAN, t-SNE, UMAP, default-pattern and anomaly analyses use a **5,000-row random sample** because of memory limits; results on the full 1.35M rows may differ.
- Step 8 clusters (sample-based, default rates 17.7-22.1%) and Step 10 clusters (full-data MiniBatch K-Means, 10.5-23.4%) come from different runs, so cluster numbers are not interchangeable.
- Results are descriptive, historical patterns. They do not show causation or predict future defaults.
- Small categories (for example FICO below 640 with 17 borrowers, or Iowa with 14) give unreliable rates.
- Anomalies are not automatically fraudulent, erroneous or high-risk and need further review.
- A Streamlit dashboard was planned in Step 10 but is not implemented.

## Author

Vidya Priya V
