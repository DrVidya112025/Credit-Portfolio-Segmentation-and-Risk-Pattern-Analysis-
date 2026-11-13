"""Data preparation and modelling logic for the Credit Portfolio Segmentation app.

The functions mirror the notebooks:
  Step 4   feature engineering / scaling (ColumnTransformer, 34 features)
  Step 5   PCA to 10 components
  Step 6-8 K-Means on the standardized PCA sample + default pattern analysis
  Step 9   DBSCAN anomaly detection + IQR outliers
  Step 10  business insight tables
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["revenue_log", "loan_amnt_log", "fico_n", "dti_capped"]
CATEGORICAL = ["emp_length", "purpose", "home_ownership_n"]
REQUIRED = ["revenue", "dti_n", "loan_amnt", "fico_n", "emp_length", "purpose",
            "home_ownership_n", "addr_state", "Default"]
DTI_CAP = 65
SAMPLE_SIZE = 5000
SEED = 42


# ----------------------------------------------------------------------------- data
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns and add the engineered columns used in the notebooks."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    if "revenue_log" not in df.columns:
        df["revenue_log"] = np.log1p(df["revenue"])
    if "dti_capped" not in df.columns:
        df["dti_capped"] = df["dti_n"].clip(upper=DTI_CAP)
    df["loan_amnt_log"] = np.log1p(df["loan_amnt"])
    for c in CATEGORICAL + ["addr_state"]:
        df[c] = df[c].astype(str).str.strip()
    return df


def sample_indices(n_rows: int, size: int = SAMPLE_SIZE, seed: int = SEED) -> np.ndarray:
    """Same reproducible sample as the notebooks (default_rng(42).choice)."""
    size = min(size, n_rows)
    return np.random.default_rng(seed).choice(n_rows, size=size, replace=False)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


# ------------------------------------------------------------- saved (pickled) models
def load_bundle(path: str) -> dict:
    """Load model_results_step6_9.pkl ({"models": {...}, "tables": {...}})."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)  # only load pickle files you created yourself
    if not isinstance(bundle, dict) or "models" not in bundle:
        raise ValueError("Not a model bundle (expected a dict with a 'models' key).")
    return bundle


def check_bundle(df: pd.DataFrame, bundle: dict) -> None:
    """Raise if the saved preprocessor/PCA cannot be applied to this dataframe."""
    m = bundle["models"]
    for need in ("preprocessor", "ipca"):
        if need not in m:
            raise ValueError(f"'{need}' not found in the model bundle")
    cols = list(getattr(m["preprocessor"], "feature_names_in_", NUMERIC + CATEGORICAL))
    absent = [c for c in cols if c not in df.columns]
    if absent:
        raise ValueError(f"Saved preprocessor needs columns missing from the data: {absent}")


def pca_sample_saved(df: pd.DataFrame, bundle: dict):
    """Same output as pca_sample(), but with the saved ColumnTransformer + IncrementalPCA.
    Only the 5,000 sample rows are transformed, so no fitting on 1.3M rows is needed."""
    m = bundle["models"]
    pre, ipca = m["preprocessor"], m["ipca"]
    cols = list(getattr(pre, "feature_names_in_", NUMERIC + CATEGORICAL))
    idx = sample_indices(len(df))
    X = pre.transform(df.iloc[idx][cols])
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return idx, ipca.transform(X), float(ipca.explained_variance_ratio_.sum())


def standardize_saved(X_pca: np.ndarray, bundle: dict) -> np.ndarray:
    """Use the saved pca_scaler when it fits; otherwise standardize the sample."""
    sc = bundle["models"].get("pca_scaler")
    if sc is not None and getattr(sc, "n_features_in_", None) == X_pca.shape[1]:
        return sc.transform(X_pca)
    return standardize(X_pca)


def predict_saved_kmeans(X_std: np.ndarray, k: int, bundle: dict):
    """Labels from the saved K-Means (kmeans_final_std) if it has k clusters, else None."""
    km = bundle["models"].get("kmeans_final_std")
    if km is not None and km.n_clusters == k and km.cluster_centers_.shape[1] == X_std.shape[1]:
        return km.predict(X_std)
    return None


# ------------------------------------------------------------------------ clustering
def pca_sample(df: pd.DataFrame, full_fit: bool = False, n_components: int = 10,
               fit_rows: int = 150_000):
    """Standardize/encode, fit PCA, and return the PCA coordinates of the 5,000 sample.

    full_fit=True fits IncrementalPCA on every row (matches the notebooks, slower).
    full_fit=False fits PCA on a random subset of `fit_rows` rows (much faster,
    nearly identical components).
    """
    X = build_preprocessor().fit_transform(df[NUMERIC + CATEGORICAL])
    idx = sample_indices(len(df))
    n = X.shape[0]

    if full_fit or n <= fit_rows:
        batch = 50_000
        ipca = IncrementalPCA(n_components=n_components, batch_size=batch)
        for s in range(0, n, batch):
            ipca.partial_fit(X[s:s + batch].toarray())
        model = ipca
    else:
        rows = np.random.default_rng(SEED).choice(n, size=fit_rows, replace=False)
        model = PCA(n_components=n_components, random_state=SEED).fit(X[rows].toarray())

    X_pca = model.transform(X[idx].toarray())
    return idx, X_pca, float(model.explained_variance_ratio_.sum())


def standardize(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def k_selection(X_std: np.ndarray, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X_std)
        rows.append({
            "K": k,
            "Inertia": km.inertia_,
            "Silhouette": silhouette_score(X_std, km.labels_, sample_size=2000, random_state=SEED),
        })
    return pd.DataFrame(rows)


def fit_kmeans(X_std: np.ndarray, k: int = 4) -> np.ndarray:
    return KMeans(n_clusters=k, init="k-means++", n_init=50, random_state=SEED,
                  max_iter=500).fit_predict(X_std)


def default_pattern(labels: np.ndarray, default: np.ndarray) -> pd.DataFrame:
    """Step 8: cluster-wise default pattern."""
    d = pd.DataFrame({"Cluster": labels, "Default": default})
    out = d.groupby("Cluster").agg(Total_Borrowers=("Default", "count"),
                                   Default_Count=("Default", "sum"))
    out["Non_Default_Count"] = out["Total_Borrowers"] - out["Default_Count"]
    out["Default_Rate"] = out["Default_Count"] / out["Total_Borrowers"] * 100
    out["Share_of_Borrowers"] = out["Total_Borrowers"] / out["Total_Borrowers"].sum() * 100
    out["Share_of_Total_Defaults"] = out["Default_Count"] / out["Default_Count"].sum() * 100
    overall = d["Default"].mean() * 100
    out["Default_Rate_Difference"] = out["Default_Rate"] - overall
    return out


def cluster_profile(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Original-unit profile per cluster (sample_df has Cluster + original columns)."""
    prof = sample_df.groupby("Cluster").agg(
        Borrowers=("Default", "count"),
        Default_Rate=("Default", "mean"),
        Avg_Revenue=("revenue", "mean"),
        Avg_Loan_Amount=("loan_amnt", "mean"),
        Avg_FICO=("fico_n", "mean"),
        Avg_DTI=("dti_capped", "mean"),
    )
    prof["Default_Rate"] *= 100
    for col, name in [("home_ownership_n", "Top_Home_Ownership"), ("purpose", "Top_Purpose")]:
        top = sample_df.groupby("Cluster")[col].agg(lambda s: s.value_counts().idxmax())
        share = sample_df.groupby("Cluster")[col].agg(lambda s: s.value_counts(normalize=True).max() * 100)
        prof[name] = top.astype(str) + " (" + share.round(1).astype(str) + "%)"
    return prof


def radar_table(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Median z-score of each numeric feature per cluster + default-rate z-score."""
    z = pd.DataFrame(StandardScaler().fit_transform(sample_df[NUMERIC]), columns=NUMERIC,
                     index=sample_df.index)
    d = sample_df["Default"].astype(float)
    z["Default"] = (d - d.mean()) / d.std()
    t = z[NUMERIC].groupby(sample_df["Cluster"]).median()
    t["Default"] = z["Default"].groupby(sample_df["Cluster"]).mean()
    return t


def segment_names(profile_df: pd.DataFrame, overall_rate: float) -> pd.Series:
    names = {}
    for c, row in profile_df.iterrows():
        home = row["Top_Home_Ownership"].split(" (")[0]
        purpose = row["Top_Purpose"].split(" (")[0]
        if row["Default_Rate"] >= overall_rate + 1:
            risk = "Higher-risk"
        elif row["Default_Rate"] <= overall_rate - 1:
            risk = "Lower-risk"
        else:
            risk = "Average-risk"
        names[c] = f"{risk}: {home} / {purpose}"
    return pd.Series(names, name="Segment")


# ------------------------------------------------------------------------ anomalies
def dbscan_anomalies(df: pd.DataFrame, eps: float = 1.8, min_samples: int = 10,
                     bundle: dict | None = None):
    """Step 9: PCA fitted on the 5,000 sample, standardized, then DBSCAN.
    With a saved bundle only the sample is transformed by the saved preprocessor."""
    idx = sample_indices(len(df))
    if bundle is not None:
        pre = bundle["models"]["preprocessor"]
        cols = list(getattr(pre, "feature_names_in_", NUMERIC + CATEGORICAL))
        Xs = pre.transform(df.iloc[idx][cols])
        Xs = Xs.toarray() if hasattr(Xs, "toarray") else np.asarray(Xs)
    else:
        X = build_preprocessor().fit_transform(df[NUMERIC + CATEGORICAL])
        Xs = X[idx].toarray()
    pca = PCA(n_components=10, random_state=SEED)
    Xp = pca.fit_transform(Xs)
    Xstd = StandardScaler().fit_transform(Xp)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(Xstd)
    return idx, Xp, Xstd, labels, float(pca.explained_variance_ratio_.sum())


def dbscan_eps_table(Xstd: np.ndarray, eps_values, min_samples: int = 10) -> pd.DataFrame:
    rows = []
    for e in eps_values:
        lab = DBSCAN(eps=e, min_samples=min_samples).fit_predict(Xstd)
        noise = int((lab == -1).sum())
        rows.append({"Epsilon": e, "Clusters": len(set(lab)) - (1 if -1 in lab else 0),
                     "Noise_Count": noise, "Noise_%": round(noise / len(lab) * 100, 2)})
    return pd.DataFrame(rows)


def iqr_outliers(frame: pd.DataFrame, cols) -> pd.DataFrame:
    rows, flags = [], pd.DataFrame(index=frame.index)
    for c in cols:
        q1, q3 = frame[c].quantile(.25), frame[c].quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flag = (frame[c] < lo) | (frame[c] > hi)
        flags[c + "_IQR_Outlier"] = flag
        rows.append({"Feature": c, "Q1": q1, "Q3": q3, "Lower_Bound": lo, "Upper_Bound": hi,
                     "Outlier_Count": int(flag.sum()),
                     "Outlier_%": round(flag.mean() * 100, 2)})
    return pd.DataFrame(rows), flags


# ---------------------------------------------------------------- business insights
def rate_table(df: pd.DataFrame, by, observed: bool = True) -> pd.DataFrame:
    t = df.groupby(by, observed=observed).agg(
        Total_Borrowers=("Default", "count"),
        Default_Count=("Default", "sum"),
        Avg_Loan_Amount=("loan_amnt", "mean"),
    )
    t["Default_Rate"] = t["Default_Count"] / t["Total_Borrowers"] * 100
    return t


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    """FICO and DTI groups (same bins as Step 10, with the corrected labels/edges)."""
    df = df.copy()
    df["FICO_Group"] = pd.cut(df["fico_n"], bins=[0, 640, 680, 720, 760, 851],
                              labels=["Below 640", "640–680", "680–720", "720–760", "760–850"],
                              right=False)
    df["DTI_Group"] = pd.cut(df["dti_capped"], bins=[-np.inf, 10, 20, 30, 40, np.inf],
                             labels=["≤10", "10–20", "20–30", "30–40", ">40"])
    return df
