# Credit Portfolio Segmentation – Streamlit app

Interactive companion to the project notebooks (Steps 1–10).

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Data: `cleaned_credit_portfolio.pkl` (fastest), `.csv` or `.parquet`. Upload it in the sidebar, type its path (a folder also works), or set it once:
```
set CREDIT_DATA_PATH=D:\HCL_PROJECTS\...\Pickle\cleaned_credit_portfolio.pkl   (Windows)
```

### Saved models (pickle)
Put `model_results_step6_9.pkl` (saved from the Step 6-9 notebook: a dict `{"models": {...}, "tables": {...}}`) in the same folder as the data file. The app finds it automatically (or set `CREDIT_MODEL_PATH`) and the sidebar shows **Use saved models**. It then uses the saved `preprocessor`, `ipca`, `pca_scaler` and `kmeans_final_std` instead of refitting on 1.3M rows. If the bundle is missing or does not match the data, the app shows a warning and fits from the data as before.
Only load pickle files you created yourself, with the same scikit-learn and pandas versions you saved them with.
No data handy? `python make_demo_data.py` creates a **synthetic** file to try the app (numbers are random).

## Pages
| Page | Notebook step |
|---|---|
| Overview | Steps 1–2, 4: KPIs and method |
| Portfolio insights | Step 10: default rate by purpose, employment, home ownership, FICO, DTI, state (state list filters out states with few borrowers) |
| Borrower segments (K-Means) | Steps 5–7: elbow/silhouette, K-Means on standardized PCA sample, profile table, radar, 2-D PCA, composition |
| Default pattern analysis | Step 8: default rate, share of defaults, difference from overall |
| Anomaly detection (DBSCAN) | Step 9: eps/min_samples, normal vs anomaly profile, IQR outliers |
| Business recommendations | Step 10: recommendations generated from the loaded data |

## Notes
- Same 5,000-borrower sample as the notebooks (`default_rng(42)`), `Default` never used for clustering.
- By default PCA is fitted on a 150,000-row subset for speed; tick the sidebar box to fit on every row like the notebooks (slower, labels may then match the notebooks more closely). Cluster numbers can differ from the notebooks' labels, so segment names are generated from each cluster's profile.
- Results are descriptive, historical and exploratory; they are not default predictions.

Files: `app.py` (UI), `core.py` (data/model logic), `requirements.txt`, `.streamlit/config.toml`, `make_demo_data.py`.

## Troubleshooting
- Paste the path with or without quotes, or give the folder that contains `cleaned_credit_portfolio.pkl` or `.csv`. The app also looks next to `app.py` and in `./step2_outputs`.
- The first load of ~1.35M rows takes a little while; after that pages are cached.
- Needs Python 3.9+ and the packages in `requirements.txt`. Run `streamlit run app.py` from the app folder.
