import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------- Page setup ------------------------------------
st.set_page_config(
    page_title="SmartCart | Customer Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A little CSS for a clean, premium dashboard feel.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: #0b1020; color: #edf2ff; }
      [data-testid="stHeader"] { background: rgba(11,16,32,0.85); }
      [data-testid="stSidebar"] { background: #10182b; border-right: 1px solid #25314c; }
      .hero {
        padding: 1.55rem 1.8rem; border-radius: 24px;
        background: radial-gradient(circle at 90% 10%, rgba(83,105,255,.30), transparent 35%),
                    linear-gradient(120deg, #17233d 0%, #111a30 58%, #142a36 100%);
        border: 1px solid #2a3a59; margin-bottom: 1.25rem;
      }
      .eyebrow { color: #83e5c2; font-size: .76rem; letter-spacing: .16em; font-weight: 800; text-transform: uppercase; }
      .hero h1 { font-family: 'Manrope', sans-serif; font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.08; margin: .45rem 0 .65rem; color: #f4f7ff; }
      .hero p { color: #b8c6df; max-width: 760px; font-size: 1.03rem; margin-bottom: .25rem; }
      .section-title { font-family: 'Manrope', sans-serif; font-size: 1.25rem; font-weight: 800; color: #f1f5ff; margin: .4rem 0 .2rem; }
      .muted { color: #9eacc8; font-size: .9rem; }
      div[data-testid="stMetric"] { background: #131e34; border: 1px solid #273652; padding: 1rem 1.1rem; border-radius: 16px; }
      div[data-testid="stMetricLabel"] { color: #a9b8d3; }
      div[data-testid="stMetricValue"] { color: #f6f8ff; }
      .pill { display:inline-block; padding:.32rem .7rem; border-radius:999px; background:#183b3b; color:#9df0d3; border:1px solid #27615a; font-size:.78rem; font-weight:700; margin-right:.35rem; }
      .insight { background:#121d32; border:1px solid #293957; border-radius:16px; padding:1rem 1.1rem; height:100%; }
      .insight h4 { margin:0 0 .35rem; color:#eef3ff; font-size:1rem; }
      .insight p { color:#aebbd4; margin:0; font-size:.91rem; }
      .stTabs [data-baseweb="tab-list"] { gap: .45rem; }
      .stTabs [data-baseweb="tab"] { background:#131e34; border-radius:10px 10px 0 0; padding:.65rem 1rem; }
      .stTabs [aria-selected="true"] { background:#263b60 !important; }
      .stButton button, .stDownloadButton button { border-radius:11px; font-weight:700; }
      hr { border-color:#263550; }
      footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

REQUIRED_COLUMNS = [
    "Year_Birth", "Education", "Marital_Status", "Income", "Kidhome", "Teenhome",
    "Dt_Customer", "Recency", "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds", "NumDealsPurchases",
    "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
    "NumWebVisitsMonth", "Complain", "Response",
]
SPENDING_COLUMNS = [
    "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
    "MntSweetProducts", "MntGoldProds",
]
EDUCATION_MAP = {
    "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
    "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate",
}
MARITAL_MAP = {
    "Married": "Partner", "Together": "Partner",
    "Single": "Alone", "Divorced": "Alone", "Widow": "Alone",
    "Absurd": "Alone", "YOLO": "Alone", "Alone": "Alone",
}

def money(value):
    return f"${value:,.0f}"

def prepare_features(raw: pd.DataFrame):
    """Apply the feature engineering and feature selection used in the notebook."""
    df = raw.copy()
    # Make column names resilient to accidental leading/trailing spaces.
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    numeric_cols = [
        "Income", "Year_Birth", "Kidhome", "Teenhome", "Recency",
        *SPENDING_COLUMNS, "NumDealsPurchases", "NumWebPurchases",
        "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth",
        "Complain", "Response",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df["Age"] = 2026 - df["Year_Birth"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    reference_date = df["Dt_Customer"].max()
    if pd.isna(reference_date):
        df["customer_tenure_days"] = 0
    else:
        df["customer_tenure_days"] = (reference_date - df["Dt_Customer"]).dt.days
        df["customer_tenure_days"] = df["customer_tenure_days"].fillna(
            df["customer_tenure_days"].median()
        )
    df["Total_Spending"] = df[SPENDING_COLUMNS].fillna(0).sum(axis=1)
    df["Total_Children"] = df["Kidhome"].fillna(0) + df["Teenhome"].fillna(0)
    df["Education"] = df["Education"].replace(EDUCATION_MAP).fillna("Unknown")
    df["Living_with"] = df["Marital_Status"].replace(MARITAL_MAP).fillna("Alone")

    # Match the notebook's outlier filtering.
    df = df[(df["Age"] < 90) & (df["Age"] > 0) & (df["Income"] < 600_000)].copy()
    if len(df) < 5:
        raise ValueError("Not enough valid rows remain after the notebook's outlier filters.")

    drop_cols = [
        "ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer",
        *SPENDING_COLUMNS,
    ]
    features = df.drop(columns=[c for c in drop_cols if c in df.columns])
    # Stable one-hot encoding compatible across scikit-learn versions.
    cat_cols = ["Education", "Living_with"]
    encoded = pd.get_dummies(features, columns=cat_cols, dtype=float)
    encoded = encoded.replace([np.inf, -np.inf], np.nan)
    for col in encoded.columns:
        if encoded[col].isna().any():
            encoded[col] = encoded[col].fillna(encoded[col].median())
    return df.reset_index(drop=True), encoded.reset_index(drop=True)

def fit_segmentation(encoded: pd.DataFrame, n_clusters: int, algorithm: str):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(encoded)
    n_components = min(3, scaled.shape[1], scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(scaled)
    if algorithm == "Agglomerative":
        labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(reduced)
    else:
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(reduced)
    sil = silhouette_score(reduced, labels) if len(np.unique(labels)) > 1 and len(labels) > n_clusters else np.nan
    coords = pd.DataFrame(reduced, columns=[f"PC{i+1}" for i in range(n_components)])
    return labels, coords, pca.explained_variance_ratio_, sil

def cluster_descriptions(summary):
    spend_col = "Total_Spending"
    income_col = "Income"
    overall_spend = summary[spend_col].mean() if spend_col in summary else 0
    overall_income = summary[income_col].mean() if income_col in summary else 0
    descriptions = {}
    for cluster, row in summary.iterrows():
        spend = row.get(spend_col, overall_spend)
        income = row.get(income_col, overall_income)
        if spend >= overall_spend * 1.15:
            label = "High-value shoppers"
        elif spend <= overall_spend * .75:
            label = "Low-spend shoppers"
        elif income >= overall_income * 1.1:
            label = "Affluent shoppers"
        else:
            label = "Steady shoppers"
        descriptions[cluster] = label
    return descriptions

# ------------------------------- Sidebar -------------------------------------
with st.sidebar:
    st.markdown("## 🛒 SmartCart")
    st.caption("CUSTOMER INTELLIGENCE STUDIO")
    st.markdown("---")
    st.markdown("### 1 · Add your data")
    uploaded = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
        help="Use the customer dataset used in your ML notebook (smartcart_customers.csv).",
    )
    st.markdown("### 2 · Configure model")
    algorithm = st.selectbox("Clustering algorithm", ["Agglomerative", "K-Means"], index=0)
    n_clusters = st.slider("Number of customer segments", min_value=2, max_value=8, value=4)
    st.markdown("---")
    st.markdown("**Model workflow**")
    st.caption("Feature engineering → one-hot encoding → scaling → PCA → clustering")
    st.caption("The default mirrors your notebook: Agglomerative clustering with 4 segments.")

# --------------------------------- Header ------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">SMART RETAIL · MACHINE LEARNING PROJECT</div>
      <h1>Understand your customers.<br>Make every cart smarter.</h1>
      <p>Explore customer segments, spending behaviour, channel preferences and campaign response in one interactive workspace.</p>
      <div style="margin-top:1rem">
        <span class="pill">Customer segmentation</span>
        <span class="pill">PCA visualization</span>
        <span class="pill">Behaviour analytics</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if uploaded is None:
    st.markdown("### Welcome to your SmartCart dashboard")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown(
            """
            <div class="insight">
              <h4>Start with your customer dataset</h4>
              <p>Upload <b>smartcart_customers.csv</b> from the sidebar. SmartCart will run the same core workflow as your notebook and build an interactive segmentation dashboard.</p>
              <br><p><b>Expected data:</b> customer demographics, income, purchase amounts, channel activity and campaign response.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.info("No CSV is bundled with this app. Upload your dataset to calculate real metrics; demo values are intentionally not fabricated.")
    with right:
        st.markdown("#### What you can explore")
        st.markdown("- **Segments:** compare size, income and spending")
        st.markdown("- **Customer map:** inspect the 3D PCA projection")
        st.markdown("- **Purchase channels:** web, store and catalog activity")
        st.markdown("- **Campaigns:** response rates and customer filters")
        st.markdown("- **Export:** download the customer-level segment assignments")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded)
    customer_df, feature_df = prepare_features(raw_df)
except Exception as exc:
    st.error(f"Could not prepare this dataset. {exc}")
    st.stop()

if n_clusters >= len(customer_df):
    st.error("Number of clusters must be smaller than the number of valid customers.")
    st.stop()

try:
    labels, pca_coords, explained, silhouette = fit_segmentation(feature_df, n_clusters, algorithm)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

customer_df["Cluster"] = labels.astype(int)
customer_df = customer_df.join(pca_coords)
cluster_sizes = customer_df["Cluster"].value_counts().sort_index()
summary_cols = [
    c for c in [
        "Income", "Recency", "NumDealsPurchases", "NumWebPurchases",
        "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth",
        "Complain", "Response", "Age", "customer_tenure_days",
        "Total_Spending", "Total_Children",
    ] if c in customer_df.columns
]
summary = customer_df.groupby("Cluster")[summary_cols].mean()
descriptions = cluster_descriptions(summary)

# ------------------------------- KPI row -------------------------------------
total_customers = len(customer_df)
avg_spend = customer_df["Total_Spending"].mean()
response_rate = customer_df["Response"].mean() * 100 if "Response" in customer_df else 0
top_cluster = int(cluster_sizes.idxmax())
m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers analysed", f"{total_customers:,}")
m2.metric("Average spend", money(avg_spend))
m3.metric("Campaign response", f"{response_rate:.1f}%")
m4.metric("Largest segment", f"Cluster {top_cluster}", f"{cluster_sizes[top_cluster]:,} customers")

tab_overview, tab_segments, tab_behaviour, tab_explorer, tab_method = st.tabs(
    ["✨ Overview", "🧩 Segments", "🛍️ Behaviour", "🔎 Customer explorer", "⚙️ Methodology"]
)

with tab_overview:
    st.markdown('<div class="section-title">Your customer landscape</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">A live summary calculated from the uploaded data and the selected clustering model.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.05, 1], gap="large")
    with c1:
        size_df = cluster_sizes.rename_axis("Cluster").reset_index(name="Customers")
        size_df["Segment"] = size_df["Cluster"].map(lambda x: f"Cluster {x}")
        fig = px.pie(size_df, values="Customers", names="Segment", hole=.68,
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_traces(textposition="outside", textinfo="percent+label", marker=dict(line=dict(color="#0b1020", width=3)))
        fig.update_layout(title="Segment distribution", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#eaf0ff", legend_title_text="", margin=dict(t=55,b=20,l=10,r=10), height=370)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(
            customer_df, x="Income", y="Total_Spending", color=customer_df["Cluster"].astype(str),
            size="NumWebPurchases" if "NumWebPurchases" in customer_df else None,
            hover_data=[c for c in ["Age", "Recency", "NumWebPurchases", "NumStorePurchases", "Response"] if c in customer_df],
            labels={"color":"Cluster", "Income":"Annual income", "Total_Spending":"Total product spend"},
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig.update_layout(title="Income vs. spending", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#101a2d",
                          font_color="#eaf0ff", legend_title_text="Cluster", margin=dict(t=55,b=25,l=10,r=10), height=370)
        fig.update_xaxes(gridcolor="#263550", zerolinecolor="#263550")
        fig.update_yaxes(gridcolor="#263550", zerolinecolor="#263550")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Quick insights")
    ins = st.columns(3)
    spend_cluster = int(summary["Total_Spending"].idxmax()) if "Total_Spending" in summary else top_cluster
    income_cluster = int(summary["Income"].idxmax()) if "Income" in summary else top_cluster
    response_cluster = int(summary["Response"].idxmax()) if "Response" in summary else top_cluster
    with ins[0]:
        st.markdown(f'<div class="insight"><h4>💎 Highest spend</h4><p>Cluster {spend_cluster} averages <b>{money(summary.loc[spend_cluster, "Total_Spending"])}</b> in product spend per customer.</p></div>', unsafe_allow_html=True)
    with ins[1]:
        st.markdown(f'<div class="insight"><h4>💼 Highest income</h4><p>Cluster {income_cluster} averages <b>{money(summary.loc[income_cluster, "Income"])}</b> annual income.</p></div>', unsafe_allow_html=True)
    with ins[2]:
        st.markdown(f'<div class="insight"><h4>📣 Best response</h4><p>Cluster {response_cluster} has a campaign response rate of <b>{summary.loc[response_cluster, "Response"]*100:.1f}%</b>.</p></div>', unsafe_allow_html=True)

with tab_segments:
    st.markdown('<div class="section-title">Segment profiles</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Compare the average profile of each cluster. Segment labels are descriptive heuristics, not trained labels.</div>', unsafe_allow_html=True)
    profile = summary.copy()
    profile.insert(0, "Customers", cluster_sizes.reindex(profile.index).astype(int))
    profile.insert(1, "Share", (cluster_sizes.reindex(profile.index) / total_customers * 100).round(1).astype(str) + "%")
    profile.insert(2, "Profile", [descriptions[c] for c in profile.index])
    st.dataframe(profile.style.format({
        "Income": "${:,.0f}", "Total_Spending": "${:,.0f}", "customer_tenure_days": "{:,.0f}",
        "Recency": "{:.0f}", "Age": "{:.1f}", "Response": "{:.1%}",
    }), use_container_width=True)
    chart_metric = st.selectbox("Compare segments by", ["Total_Spending", "Income", "NumWebPurchases", "NumStorePurchases", "NumCatalogPurchases", "Response", "Age"])
    if chart_metric in summary.columns:
        metric_data = summary[chart_metric].rename("Average").reset_index()
        metric_data["Cluster"] = metric_data["Cluster"].astype(str)
        fig = px.bar(metric_data, x="Cluster", y="Average", color="Cluster",
                     color_discrete_sequence=px.colors.qualitative.Vivid, text_auto=".2s",
                     labels={"Cluster":"Cluster", "Average":chart_metric.replace("_"," ")},
                     title=f"Average {chart_metric.replace('_',' ')} by segment")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#101a2d", font_color="#eaf0ff",
                          showlegend=False, margin=dict(t=55,b=20,l=10,r=10), height=360)
        fig.update_xaxes(gridcolor="#263550")
        fig.update_yaxes(gridcolor="#263550")
        st.plotly_chart(fig, use_container_width=True)
    csv_summary = profile.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download segment summary (CSV)", csv_summary, "smartcart_segment_summary.csv", "text/csv")

with tab_behaviour:
    st.markdown('<div class="section-title">Purchase behaviour</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Understand product mix and how customers interact with sales channels.</div>', unsafe_allow_html=True)
    spend_map = {
        "Wine": "MntWines", "Meat": "MntMeatProducts", "Fish": "MntFishProducts",
        "Fruit": "MntFruits", "Sweets": "MntSweetProducts", "Gold products": "MntGoldProds",
    }
    spend_cols_available = {k:v for k,v in spend_map.items() if v in customer_df.columns}
    b1, b2 = st.columns(2, gap="large")
    with b1:
        product_totals = pd.DataFrame({"Product category": list(spend_cols_available.keys()),
                                       "Total spend": [customer_df[v].fillna(0).sum() for v in spend_cols_available.values()]})
        fig = px.bar(product_totals.sort_values("Total spend", ascending=True), x="Total spend", y="Product category",
                     orientation="h", color="Total spend", color_continuous_scale="Tealgrn",
                     title="Spending by product category")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#101a2d", font_color="#eaf0ff",
                          coloraxis_showscale=False, margin=dict(t=55,b=20,l=10,r=10), height=380)
        fig.update_xaxes(gridcolor="#263550")
        st.plotly_chart(fig, use_container_width=True)
    with b2:
        channel_cols = [c for c in ["NumWebPurchases","NumCatalogPurchases","NumStorePurchases"] if c in customer_df.columns]
        if channel_cols:
            channel_labels = {"NumWebPurchases":"Web", "NumCatalogPurchases":"Catalog", "NumStorePurchases":"Store"}
            channel_df = customer_df.groupby("Cluster")[channel_cols].mean().reset_index().melt(
                id_vars="Cluster", var_name="Channel", value_name="Average purchases")
            channel_df["Channel"] = channel_df["Channel"].map(channel_labels)
            fig = px.bar(channel_df, x="Cluster", y="Average purchases", color="Channel", barmode="group",
                         title="Average purchases by channel and segment",
                         color_discrete_sequence=["#66e3c4", "#8f9cff", "#ffbe76"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#101a2d", font_color="#eaf0ff",
                              margin=dict(t=55,b=20,l=10,r=10), height=380)
            fig.update_xaxes(gridcolor="#263550")
            fig.update_yaxes(gridcolor="#263550")
            st.plotly_chart(fig, use_container_width=True)
    if "Response" in customer_df:
        resp = customer_df.groupby("Cluster")["Response"].mean().mul(100).rename("Response rate").reset_index()
        resp["Cluster"] = resp["Cluster"].astype(str)
        fig = px.bar(resp, x="Cluster", y="Response rate", color="Cluster", text_auto=".1f",
                     title="Campaign response rate by segment",
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#101a2d", font_color="#eaf0ff",
                          showlegend=False, margin=dict(t=55,b=20,l=10,r=10), height=300)
        fig.update_xaxes(gridcolor="#263550")
        fig.update_yaxes(gridcolor="#263550", ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)

with tab_explorer:
    st.markdown('<div class="section-title">Explore individual customers</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Filter the customer table and download segment assignments for follow-up analysis.</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([1,1,1])
    cluster_options = ["All clusters"] + [f"Cluster {c}" for c in sorted(customer_df["Cluster"].unique())]
    chosen_cluster = f1.selectbox("Segment", cluster_options)
    income_min, income_max = float(customer_df["Income"].min()), float(customer_df["Income"].max())
    income_range = f2.slider("Income range", min_value=int(max(0, income_min)), max_value=int(income_max),
                             value=(int(max(0, income_min)), int(income_max)))
    response_filter = f3.selectbox("Campaign response", ["All", "Responded", "Did not respond"])
    filtered = customer_df.copy()
    if chosen_cluster != "All clusters":
        filtered = filtered[filtered["Cluster"] == int(chosen_cluster.split()[-1])]
    filtered = filtered[filtered["Income"].between(income_range[0], income_range[1])]
    if response_filter == "Responded":
        filtered = filtered[filtered["Response"] == 1]
    elif response_filter == "Did not respond":
        filtered = filtered[filtered["Response"] == 0]
    st.caption(f"Showing {len(filtered):,} of {total_customers:,} customers")
    display_cols = [c for c in [
        "ID", "Cluster", "Age", "Education", "Living_with", "Income", "Total_Spending",
        "Recency", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "Response",
    ] if c in filtered.columns]
    st.dataframe(filtered[display_cols].sort_values("Total_Spending", ascending=False), use_container_width=True, height=420)
    export_cols = [c for c in customer_df.columns if not c.startswith("PC")]
    export = customer_df[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download all customer segment assignments", export, "smartcart_customer_segments.csv", "text/csv")

with tab_method:
    st.markdown('<div class="section-title">Model methodology</div>', unsafe_allow_html=True)
    st.markdown("This app follows the core workflow in your uploaded SmartCart notebook: feature engineering, category encoding, standardization, PCA, and clustering.")
    method_cols = st.columns(4)
    method_cols[0].metric("Algorithm", algorithm)
    method_cols[1].metric("Clusters", n_clusters)
    method_cols[2].metric("PCA components", pca_coords.shape[1])
    method_cols[3].metric("Silhouette score", f"{silhouette:.3f}" if np.isfinite(silhouette) else "N/A")
    st.markdown("#### PCA customer map")
    if pca_coords.shape[1] >= 3:
        fig = px.scatter_3d(customer_df, x="PC1", y="PC2", z="PC3", color=customer_df["Cluster"].astype(str),
                            hover_data=[c for c in ["Income", "Total_Spending", "Age", "Response"] if c in customer_df],
                            color_discrete_sequence=px.colors.qualitative.Vivid,
                            labels={"color":"Cluster"}, title="3D PCA projection of customer segments")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#eaf0ff",
                          scene=dict(bgcolor="#101a2d", xaxis=dict(gridcolor="#263550"), yaxis=dict(gridcolor="#263550"), zaxis=dict(gridcolor="#263550")),
                          margin=dict(t=55,b=0,l=0,r=0), height=550)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("A 3D PCA map requires at least three components; this dataset produced fewer.")
    st.markdown("#### Explained variance")
    variance_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(explained))],
        "Explained variance": explained,
    })
    fig = px.bar(variance_df, x="Component", y="Explained variance", text_auto=".1%",
                 title="Variance captured by each PCA component")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#101a2d", font_color="#eaf0ff",
                      margin=dict(t=55,b=20,l=10,r=10), height=300)
    fig.update_yaxes(tickformat=".0%", gridcolor="#263550")
    st.plotly_chart(fig, use_container_width=True)
    st.warning("Interpretation note: clustering labels are arbitrary IDs. The descriptive segment names are generated from relative averages and should be validated before business decisions.")

st.markdown("---")
st.markdown('<div class="muted" style="text-align:center">SMARTCART · Customer intelligence from your ML workflow</div>', unsafe_allow_html=True)
