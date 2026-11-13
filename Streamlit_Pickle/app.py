"""Credit Portfolio Segmentation and Risk Pattern Analysis - Streamlit app.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

import core

st.set_page_config(page_title="Credit Portfolio Segmentation", page_icon="📊", layout="wide")

DEFAULT_PATH = os.environ.get(
    "CREDIT_DATA_PATH",
    r"D:\HCL_PROJECTS\Credit Portfolio Segmentation and Risk Pattern Analysis Project-3"
    r"\Pickle\cleaned_credit_portfolio.pkl",
)
MODEL_FILE = "model_results_step6_9.pkl"
PALETTE = px.colors.qualitative.Safe

# Works on old and new Streamlit versions
_v = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
STRETCH = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}
USECOLS = ["id", "issue_d", "revenue", "dti_n", "loan_amnt", "fico_n", "emp_length", "purpose",
           "home_ownership_n", "addr_state", "Default", "revenue_log", "dti_capped"]
FILE_NAMES = ["cleaned_credit_portfolio.pkl", "cleaned_credit_portfolio.csv",
              "cleaned_credit_portfolio.parquet"]


# ------------------------------------------------------------------------- loading
def resolve_path(text: str) -> Path | None:
    """Accept quoted paths, folders, and fall back to files next to the app."""
    text = os.path.expandvars((text or "").strip().strip('"').strip("'"))
    here = Path(__file__).resolve().parent
    candidates = []
    if text:
        p = Path(text).expanduser()
        if p.is_dir():
            candidates += [p / n for n in FILE_NAMES] + [p / "step2_outputs" / n for n in FILE_NAMES]
        else:
            candidates.append(p)
    for base in (Path.cwd(), here, here / "step2_outputs", Path.cwd() / "step2_outputs"):
        candidates += [base / n for n in FILE_NAMES]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _read(source, name: str) -> pd.DataFrame:
    keep = lambda c: c in USECOLS  # noqa: E731  (skip unused columns to save memory)
    if name.lower().endswith((".pkl", ".pickle")):
        raw = pd.read_pickle(source)  # your own pickle file only
        raw = raw[[c for c in raw.columns if keep(c)]]
    elif name.lower().endswith(".parquet"):
        raw = pd.read_parquet(source)
        raw = raw[[c for c in raw.columns if keep(c)]]
    else:
        raw = pd.read_csv(source, usecols=keep, low_memory=False)
    return core.add_groups(core.prepare(raw))


# cache_resource keeps ONE shared copy (no pickling/copying of 1.3M rows on every rerun)
@st.cache_resource(show_spinner="Loading data (about 1.35M rows, please wait)...")
def load_path(path: str, mtime: float) -> pd.DataFrame:
    return _read(path, path)


@st.cache_resource(show_spinner="Reading uploaded file...")
def load_upload(_data: bytes, name: str, size: int) -> pd.DataFrame:
    return _read(io.BytesIO(_data), name)


@st.cache_resource(show_spinner="Loading saved models...")
def load_models(path: str, mtime: float) -> dict:
    return core.load_bundle(path)


def find_model_file(data_file: Path | None) -> Path | None:
    env = os.environ.get("CREDIT_MODEL_PATH", "").strip().strip('"')
    here = Path(__file__).resolve().parent
    bases = [data_file.parent] if data_file else []
    bases += [Path(DEFAULT_PATH.strip('"')).parent, Path.cwd(), here, Path.cwd() / "Pickle", here / "Pickle"]
    cands = ([Path(env)] if env else []) + [b / MODEL_FILE for b in bases]
    return next((c for c in cands if c.is_file()), None)


@st.cache_data(show_spinner="Running PCA (this can take a while on the full data)...")
def cached_pca(_df: pd.DataFrame, key: str, full_fit: bool, _bundle=None, bkey: str = ""):
    if _bundle is not None:
        return core.pca_sample_saved(_df, _bundle)
    return core.pca_sample(_df, full_fit=full_fit)


@st.cache_data(show_spinner="Evaluating K...")
def cached_k_selection(X_std):
    return core.k_selection(X_std)


@st.cache_data(show_spinner="Fitting K-Means...")
def cached_kmeans(X_std, k, _bundle=None, bkey: str = ""):
    if _bundle is not None:
        try:
            saved = core.predict_saved_kmeans(X_std, k, _bundle)
            if saved is not None:
                return saved
        except Exception:  # noqa: BLE001  (fall back to fitting)
            pass
    return core.fit_kmeans(X_std, k)


@st.cache_data(show_spinner="Running DBSCAN...")
def cached_dbscan(_df: pd.DataFrame, key: str, eps: float, min_samples: int,
                  _bundle=None, bkey: str = ""):
    return core.dbscan_anomalies(_df, eps, min_samples, bundle=_bundle)


@st.cache_data(show_spinner="Testing epsilon values...")
def cached_eps_table(Xstd, eps_values, min_samples):
    return core.dbscan_eps_table(Xstd, eps_values, min_samples)


# ---------------------------------------------------------------------------- helpers
def fmt_pct(x):
    return f"{x:.2f}%"


def bar_with_baseline(table, x, y, title, baseline, horizontal=False, xlabel=None):
    t = table.reset_index()
    t[x] = t[x].astype(str)
    if horizontal:
        fig = px.bar(t.sort_values(y), x=y, y=x, orientation="h", title=title)
        fig.add_vline(x=baseline, line_dash="dash", annotation_text=f"Overall {baseline:.2f}%")
    else:
        fig = px.bar(t, x=x, y=y, title=title)
        fig.add_hline(y=baseline, line_dash="dash", annotation_text=f"Overall {baseline:.2f}%")
    fig.update_layout(xaxis_title=xlabel, yaxis_title=None if horizontal else "Default rate (%)",
                      margin=dict(t=50, b=10))
    return fig


def radar_figure(table: pd.DataFrame, names: pd.Series | None = None, clusters=None):
    axes = ["Revenue", "Loan amount", "FICO", "DTI", "Default rate"]
    fig = go.Figure()
    for c in (clusters if clusters is not None else table.index):
        vals = table.loc[c].to_numpy(dtype=float)
        label = f"Cluster {c + 1}" + (f" - {names[c]}" if names is not None else "")
        fig.add_trace(go.Scatterpolar(r=np.append(vals, vals[0]), theta=axes + [axes[0]],
                                      name=label, line_color=PALETTE[c % len(PALETTE)]))
    fig.update_layout(polar=dict(radialaxis=dict(range=[-1.5, 2.5])), height=520,
                      legend=dict(orientation="h", y=-0.15))
    return fig


# ------------------------------------------------------------------------- sidebar
st.sidebar.title("📊 Credit Portfolio")
st.sidebar.caption("Segmentation and Risk Pattern Analysis")

page = st.sidebar.radio("Go to", [
    "Overview",
    "Portfolio insights",
    "Borrower segments (K-Means)",
    "Default pattern analysis",
    "Anomaly detection (DBSCAN)",
    "Business recommendations",
])

st.sidebar.divider()
st.sidebar.subheader("Data")
upload = st.sidebar.file_uploader("Upload cleaned_credit_portfolio (.pkl / .csv / .parquet)",
                                  type=["pkl", "pickle", "csv", "parquet"])
path = st.sidebar.text_input("...or file path", DEFAULT_PATH)

df = None
found = None
DATA_KEY = "none"
try:
    if upload is not None:
        df = load_upload(upload.getvalue(), upload.name, upload.size)
        DATA_KEY = f"upload:{upload.name}:{upload.size}"
    else:
        found = resolve_path(path)
        if found is not None:
            df = load_path(str(found), found.stat().st_mtime)
            DATA_KEY = f"{found}:{found.stat().st_mtime}"
            st.sidebar.success(f"Loaded: {found.name}")
        elif path.strip():
            st.sidebar.error("File not found. Check the path (a folder that contains "
                             "cleaned_credit_portfolio.csv also works).")
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"Could not load data: {type(e).__name__}: {e}")
    st.sidebar.caption("Required columns: " + ", ".join(core.REQUIRED))

if df is None:
    st.title("Credit Portfolio Segmentation and Risk Pattern Analysis")
    st.info("Load the cleaned dataset (output of Step 2, `cleaned_credit_portfolio.csv` or `.parquet`) "
            "by uploading it or entering its path in the sidebar. You can also set the "
            "`CREDIT_DATA_PATH` environment variable, or put the file next to `app.py`.")
    st.stop()

overall = df["Default"].mean() * 100

# Optional: saved models from Steps 6-9 (model_results_step6_9.pkl)
BUNDLE, BKEY = None, ""
model_file = find_model_file(found)
if model_file is not None:
    use_saved = st.sidebar.checkbox(
        "Use saved models (pickle)", value=True,
        help=f"Loads {model_file.name}: skips refitting the preprocessor/PCA/K-Means on 1.3M rows.")
    if use_saved:
        try:
            b = load_models(str(model_file), model_file.stat().st_mtime)
            core.check_bundle(df, b)
            BUNDLE, BKEY = b, f"{model_file}:{model_file.stat().st_mtime}"
            st.sidebar.success(f"Saved models: {model_file.name}")
        except Exception as e:  # noqa: BLE001
            st.sidebar.warning(f"Saved models not used ({type(e).__name__}: {e}). Refitting instead.")
else:
    st.sidebar.caption(f"No {MODEL_FILE} found - models are fitted from the data.")

full_fit = st.sidebar.checkbox(
    "Fit PCA on all rows (matches notebooks, slower)", value=False,
    help="Off: PCA is fitted on a 150,000-row random subset - much faster, nearly identical result.")
st.sidebar.caption(f"{len(df):,} borrowers loaded")


def clustering_inputs():
    idx, X_pca, evr = cached_pca(df, DATA_KEY, full_fit, BUNDLE, BKEY)
    X_std = core.standardize_saved(X_pca, BUNDLE) if BUNDLE is not None else core.standardize(X_pca)
    return idx, X_pca, evr, X_std


# ---------------------------------------------------------------------------- pages
if page == "Overview":
    st.title("Credit Portfolio Segmentation and Risk Pattern Analysis")
    st.caption("Interactive companion to the project notebooks (Steps 1-10).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Borrowers", f"{len(df):,}")
    c2.metric("Defaults", f"{int(df['Default'].sum()):,}")
    c3.metric("Default rate", fmt_pct(overall))
    c4.metric("States", df["addr_state"].nunique())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average loan", f"${df['loan_amnt'].mean():,.0f}")
    c2.metric("Average revenue", f"${df['revenue'].mean():,.0f}")
    c3.metric("Average FICO", f"{df['fico_n'].mean():.1f}")
    c4.metric("Average DTI", f"{df['dti_capped'].mean():.2f}")

    st.subheader("Method in brief")
    st.markdown("""
1. **Preprocessing:** no rows removed; revenue log-transformed; DTI capped at 65; loan amount log-transformed.
2. **Features:** revenue (log), loan amount (log), FICO, capped DTI (standardized) plus employment length, purpose and home ownership (one-hot) -> 34 features. `Default` is **never** a clustering input.
3. **PCA:** 10 components (about 89% of the variance), then a reproducible 5,000-borrower sample (seed 42).
4. **K-Means (K=4)** on the standardized PCA sample; Hierarchical clustering and DBSCAN were also compared in the notebooks.
5. **Default pattern:** after clustering, historical default rates are compared across segments.
6. **Anomaly detection:** DBSCAN noise points plus IQR feature-level outliers.
""")
    st.subheader("Data preview")
    st.dataframe(df.head(100), **STRETCH)

elif page == "Portfolio insights":
    st.title("Portfolio insights")
    st.caption(f"Historical default rate by borrower characteristic (overall: {fmt_pct(overall)}).")

    tabs = st.tabs(["Loan purpose", "Employment length", "Home ownership", "FICO", "DTI", "State"])

    def simple_tab(tab, col, title, horizontal=False):
        with tab:
            t = core.rate_table(df, col).round(2)
            st.plotly_chart(bar_with_baseline(t, col, "Default_Rate", title, overall, horizontal),
                            **STRETCH)
            st.dataframe(t.sort_values("Default_Rate", ascending=False), **STRETCH)

    simple_tab(tabs[0], "purpose", "Default rate by loan purpose", True)
    simple_tab(tabs[1], "emp_length", "Default rate by employment length", True)
    simple_tab(tabs[2], "home_ownership_n", "Default rate by home ownership")
    simple_tab(tabs[3], "FICO_Group", "Default rate by FICO group")
    simple_tab(tabs[4], "DTI_Group", "Default rate by DTI group")
    with tabs[3]:
        st.caption("Very small groups can show extreme rates; check the borrower counts.")
    with tabs[4]:
        st.caption("Groups are right-closed: <=10, (10,20], (20,30], (30,40], >40.")

    with tabs[5]:
        min_n = st.slider("Minimum borrowers per state", 0, 20000, 1000, 500,
                          help="States with few borrowers have unreliable default rates.")
        st_t = core.rate_table(df, "addr_state").round(2)
        shown = st_t[st_t["Total_Borrowers"] >= min_n]
        top_n = st.slider("States to show", 5, 30, 15)
        top = shown.nlargest(top_n, "Default_Rate")
        st.plotly_chart(bar_with_baseline(top, "addr_state", "Default_Rate",
                                          f"Top {len(top)} states by default rate", overall, True),
                        **STRETCH)
        excluded = len(st_t) - len(shown)
        if excluded:
            st.caption(f"{excluded} state(s) with fewer than {min_n:,} borrowers are excluded.")
        st.dataframe(shown.sort_values("Default_Rate", ascending=False), **STRETCH)

elif page == "Borrower segments (K-Means)":
    st.title("Borrower segments (K-Means)")
    idx, X_pca, evr, X_std = clustering_inputs()
    st.caption(f"5,000-borrower sample · 10 PCA components retain {evr * 100:.2f}% of the variance "
               "· components standardized before clustering.")

    with st.expander("Choose K (elbow and silhouette)"):
        ks = cached_k_selection(X_std)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.line(ks, x="K", y="Inertia", markers=True, title="Elbow"), **STRETCH)
        c2.plotly_chart(px.line(ks, x="K", y="Silhouette", markers=True, title="Silhouette"),
                        **STRETCH)
        st.dataframe(ks.round(4), **STRETCH)
        st.caption("Silhouette scores are low and close together: the segments overlap, so treat "
                   "them as exploratory.")

    k = st.slider("Number of clusters (K)", 2, 8, 4)
    labels = cached_kmeans(X_std, k, BUNDLE, BKEY)

    sample_df = df.iloc[idx].reset_index(drop=True).copy()
    sample_df["Cluster"] = labels
    st.session_state["segments"] = (idx, labels, k)

    prof = core.cluster_profile(sample_df)
    names = core.segment_names(prof, sample_df["Default"].mean() * 100)

    tab1, tab2, tab3, tab4 = st.tabs(["Profile table", "Radar profile", "2-D view", "Composition"])
    with tab1:
        show = prof.copy()
        show.insert(0, "Segment", names)
        show["Share_%"] = (show["Borrowers"] / show["Borrowers"].sum() * 100).round(2)
        show.index = [f"Cluster {i + 1}" for i in show.index]
        st.dataframe(show.round(2), **STRETCH)
    with tab2:
        rt = core.radar_table(sample_df)
        rt_plot = rt.rename(columns=dict(zip(core.NUMERIC, ["Revenue", "Loan amount", "FICO", "DTI"])))
        which = st.multiselect("Clusters", [int(c) for c in rt.index],
                               default=[int(c) for c in rt.index], format_func=lambda c: f"Cluster {c + 1}")
        st.plotly_chart(radar_figure(rt_plot, names, which), **STRETCH)
        st.caption("Median standardized value of each feature per cluster (default axis: cluster "
                   "default-rate z-score). Default is not used to form clusters.")
        st.dataframe(rt.round(3), **STRETCH)
    with tab3:
        pcs = PCA(n_components=2, random_state=42).fit_transform(X_std)
        plot_df = pd.DataFrame({"PC 1": pcs[:, 0], "PC 2": pcs[:, 1],
                                "Cluster": [f"Cluster {c + 1}" for c in labels]})
        st.plotly_chart(px.scatter(plot_df, x="PC 1", y="PC 2", color="Cluster", opacity=.6,
                                   title="Clusters in 2-D PCA space"), **STRETCH)
    with tab4:
        feat = st.selectbox("Characteristic", ["home_ownership_n", "purpose", "emp_length"])
        comp = pd.crosstab(sample_df["Cluster"] + 1, sample_df[feat], normalize="index") * 100
        comp.index = [f"Cluster {i}" for i in comp.index]
        st.plotly_chart(px.bar(comp, barmode="stack", title=f"{feat} distribution by cluster (%)"),
                        **STRETCH)
        st.caption("Clusters are driven largely by categorical characteristics (home ownership, "
                   "loan purpose), so they are borrower/loan-type segments as well as risk levels.")

elif page == "Default pattern analysis":
    st.title("Default pattern analysis")
    st.caption("Default is used only after clustering, to compare historical default rates.")
    idx, X_pca, evr, X_std = clustering_inputs()
    k = st.slider("Number of clusters (K)", 2, 8, 4)
    labels = cached_kmeans(X_std, k, BUNDLE, BKEY)

    default = df.iloc[idx]["Default"].to_numpy()
    dp = core.default_pattern(labels, default)
    sample_overall = default.mean() * 100
    dp_show = dp.copy()
    dp_show.index = [f"Cluster {i + 1}" for i in dp_show.index]

    c1, c2, c3 = st.columns(3)
    c1.metric("Sample default rate", fmt_pct(sample_overall))
    hi = dp["Default_Rate"].idxmax()
    lo = dp["Default_Rate"].idxmin()
    c2.metric(f"Highest: Cluster {hi + 1}", fmt_pct(dp.loc[hi, "Default_Rate"]),
              f"{dp.loc[hi, 'Default_Rate_Difference']:+.2f} pp")
    c3.metric(f"Lowest: Cluster {lo + 1}", fmt_pct(dp.loc[lo, "Default_Rate"]),
              f"{dp.loc[lo, 'Default_Rate_Difference']:+.2f} pp")

    plot = dp_show.reset_index().rename(columns={"index": "Cluster"})
    fig = px.bar(plot, x="Cluster", y="Default_Rate", text=plot["Default_Rate"].round(2),
                 title="Historical default rate by cluster")
    fig.add_hline(y=sample_overall, line_dash="dash", annotation_text=f"Overall {sample_overall:.2f}%")
    fig.update_layout(yaxis_title="Default rate (%)")
    st.plotly_chart(fig, **STRETCH)

    c1, c2 = st.columns(2)
    c1.plotly_chart(px.pie(plot, names="Cluster", values="Default_Count", title="Share of total defaults"),
                    **STRETCH)
    c2.plotly_chart(px.pie(plot, names="Cluster", values="Total_Borrowers", title="Share of borrowers"),
                    **STRETCH)
    st.dataframe(dp_show.round(2), **STRETCH)
    st.caption("Descriptive, historical results from a 5,000-borrower sample. They do not establish "
               "causation or predict future defaults.")

elif page == "Anomaly detection (DBSCAN)":
    st.title("Anomaly detection (DBSCAN)")
    st.caption("PCA (10 components) fitted on the 5,000-borrower sample, standardized, then DBSCAN. "
               "Noise points (label -1) are potential anomalies, not confirmed errors or fraud.")
    c1, c2 = st.columns(2)
    eps = c1.slider("eps", 1.0, 3.0, 1.8, 0.1)
    min_samples = c2.slider("min_samples", 5, 30, 10)

    idx, Xp, Xstd, labels, evr = cached_dbscan(df, DATA_KEY, eps, min_samples, BUNDLE, BKEY)
    sdf = df.iloc[idx].reset_index(drop=True).copy()
    sdf["Anomaly_Status"] = np.where(labels == -1, "Potential Anomaly", "Normal")

    c1, c2, c3, c4 = st.columns(4)
    n_anom = int((labels == -1).sum())
    c1.metric("Sample size", f"{len(sdf):,}")
    c2.metric("Potential anomalies", f"{n_anom:,}")
    c3.metric("Anomaly share", fmt_pct(n_anom / len(sdf) * 100))
    c4.metric("DBSCAN clusters", len(set(labels)) - (1 if -1 in labels else 0))
    st.caption(f"PCA explained variance: {evr * 100:.2f}%")

    t1, t2, t3, t4 = st.tabs(["eps evaluation", "Normal vs anomaly", "Feature-level (IQR)", "2-D view"])
    with t1:
        et = cached_eps_table(Xstd, (1.5, 1.8, 2.1, 2.4, 2.6), min_samples)
        st.dataframe(et, **STRETCH)
        st.caption("Larger eps flags fewer borrowers. Noise of about 5% (eps=1.8) is the notebook choice.")
    with t2:
        prof = sdf.groupby("Anomaly_Status").agg(
            Borrowers=("Default", "count"),
            Avg_Revenue=("revenue", "mean"), Median_Revenue=("revenue", "median"),
            Avg_Loan_Amount=("loan_amnt", "mean"), Avg_FICO=("fico_n", "mean"),
            Avg_DTI=("dti_capped", "mean"), Default_Count=("Default", "sum"),
            Default_Rate=("Default", "mean"))
        prof["Default_Rate"] *= 100      # convert first, round afterwards
        st.dataframe(prof.round(2), **STRETCH)
        st.caption("Average revenue is pulled up by a few very high incomes; compare with the median.")
    with t3:
        feats = ["revenue", "loan_amnt", "fico_n", "dti_capped"]
        iqr_df, flags = core.iqr_outliers(sdf, feats)
        st.dataframe(iqr_df.round(2), **STRETCH)
        by = flags.groupby(sdf["Anomaly_Status"]).mean() * 100
        by.columns = feats
        st.plotly_chart(px.bar(by.T, barmode="group",
                               title="Share of borrowers with IQR outliers, by DBSCAN status (%)"),
                        **STRETCH)
        st.caption("Feature-specific counts overlap; do not add them together.")
    with t4:
        pcs2 = PCA(n_components=2, random_state=42).fit_transform(Xstd)
        pl = pd.DataFrame({"PC 1": pcs2[:, 0], "PC 2": pcs2[:, 1], "Status": sdf["Anomaly_Status"]})
        st.plotly_chart(px.scatter(pl, x="PC 1", y="PC 2", color="Status", opacity=.7,
                                   color_discrete_map={"Normal": "#9aa5b1", "Potential Anomaly": "#d62728"}),
                        **STRETCH)
        with st.expander("Potential anomaly records"):
            st.dataframe(sdf[sdf["Anomaly_Status"] == "Potential Anomaly"][
                ["id", "revenue", "loan_amnt", "fico_n", "dti_n", "purpose", "home_ownership_n", "Default"]
                if "id" in sdf.columns else ["revenue", "loan_amnt", "fico_n", "dti_n", "Default"]],
                **STRETCH)

else:  # Business recommendations
    st.title("Business recommendations")
    st.caption("Generated from the loaded data (full portfolio unless stated otherwise).")

    def top_bottom(col, min_n=1000):
        t = core.rate_table(df, col)
        t = t[t["Total_Borrowers"] >= min_n]
        return t["Default_Rate"].idxmax(), t["Default_Rate"].max(), t["Default_Rate"].idxmin(), t["Default_Rate"].min()

    fico = core.rate_table(df, "FICO_Group")
    dti = core.rate_table(df, "DTI_Group")
    p_hi, p_hi_r, _, _ = top_bottom("purpose")
    e_hi, e_hi_r, _, _ = top_bottom("emp_length")
    s_hi, s_hi_r, _, _ = top_bottom("addr_state")
    purpose_t = core.rate_table(df, "purpose")
    big_purpose = purpose_t["Total_Borrowers"].idxmax()
    big_state = core.rate_table(df, "addr_state")["Total_Borrowers"].idxmax()

    st.markdown(f"""
### 1. Improve credit-risk assessment
- Default rate falls steadily with FICO: **{fico['Default_Rate'].iloc[1]:.2f}%** for the 640–680 group against **{fico['Default_Rate'].iloc[-1]:.2f}%** for 760–850.
- Use FICO alongside income, DTI and repayment history rather than on its own.
- The "Below 640" group has only {int(fico['Total_Borrowers'].iloc[0]):,} borrowers, so its rate is not reliable.

### 2. Introduce DTI-based monitoring
- Default rates by DTI group: {", ".join(f"{i} {r:.2f}%" for i, r in dti['Default_Rate'].items())}.
- Apply additional affordability checks to the higher DTI groups (above 30).

### 3. Strengthen employment and income verification
- The highest observed default rate by employment length is **{e_hi}** at {e_hi_r:.2f}% (groups with at least 1,000 borrowers).
- Treat employment stability as one factor in assessment, not an independent rejection criterion.

### 4. Monitor loan-purpose risk
- Highest observed purpose-based default rate: **{p_hi}** at {p_hi_r:.2f}%.
- Largest segment by borrowers: **{big_purpose}** ({int(purpose_t.loc[big_purpose, 'Total_Borrowers']):,} borrowers, {purpose_t.loc[big_purpose, 'Default_Rate']:.2f}% default rate).

### 5. Monitor geographic credit risk
- Highest state default rate (states with at least 1,000 borrowers): **{s_hi}** at {s_hi_r:.2f}%.
- Largest borrower population: **{big_state}**.
- Always read state rates together with borrower counts.
""")

    st.markdown("### 6. Cluster-based lending strategies")
    if "segments" in st.session_state:
        idx, labels, k = st.session_state["segments"]
        sdf = df.iloc[idx].reset_index(drop=True).copy()
        sdf["Cluster"] = labels
        prof = core.cluster_profile(sdf)
        names = core.segment_names(prof, sdf["Default"].mean() * 100)
        strategy = {"Higher-risk": "Strengthen affordability and credit checks",
                    "Lower-risk": "Standard lending policy; candidates for retention offers",
                    "Average-risk": "Monitor; apply standard policy with periodic review"}
        out = pd.DataFrame({
            "Cluster": [f"Cluster {i + 1}" for i in prof.index],
            "Segment": names.values,
            "Default rate %": prof["Default_Rate"].round(2).values,
            "Suggested strategy": [strategy[n.split(":")[0]] for n in names.values],
        })
        st.dataframe(out, **STRETCH, hide_index=True)
    else:
        st.info("Open **Borrower segments (K-Means)** first; the cluster table appears here.")

    st.warning("These are descriptive observations on historical data, not causal conclusions or "
               "default predictions. Segments overlap (low silhouette scores), and anomalies flagged by "
               "DBSCAN are not automatically fraudulent or high-risk.")
