from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from metrics import (
    build_item_fact,
    build_order_fact,
    cohort_ltv,
    cohort_retention_curve,
    customer_lifecycle,
    default_olist_paths,
    ltv_by_segment,
    next_best_actions,
    retention_summary,
    score_churn_risk,
)

st.set_page_config(page_title="ValueLoop", page_icon="🔁", layout="wide")

st.title("ValueLoop – Retention + LTV Intelligence")
st.caption(
    "Kaggle (Olist) dataset demo: cohort retention, customer LTV, and next-best actions. "
    "No Shopify integration yet."
)

with st.sidebar:
    st.header("Data source")
    source = st.radio(
        "Choose dataset",
        options=["Use included Kaggle dataset (recommended)", "Upload a prepared order_fact CSV"],
        index=0,
    )

    max_days = st.slider("Cohort retention horizon (days)", min_value=60, max_value=365, value=180, step=30)
    churn_days = st.slider("Churn threshold (days since last order)", min_value=30, max_value=180, value=90, step=15)
    st.divider()
    st.header("Segmentation")
    segment = st.selectbox("LTV breakdown", options=["primary_category", "customer_state"], index=0)


@st.cache_data(show_spinner=False)
def _load_order_fact_from_olist(workspace_root: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = default_olist_paths(workspace_root)
    item_fact = build_item_fact(paths)
    order_fact = build_order_fact(item_fact)
    lifecycle = customer_lifecycle(order_fact)
    return item_fact, order_fact, lifecycle


@st.cache_data(show_spinner=False)
def _load_order_fact_from_csv(upload_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(pd.io.common.BytesIO(upload_bytes))
    df["order_purchase_ts"] = pd.to_datetime(df["order_purchase_ts"], errors="coerce")
    df = df.dropna(subset=["customer_unique_id", "order_id", "order_purchase_ts"])
    if "order_revenue" not in df.columns:
        raise ValueError("Uploaded file must include 'order_revenue' column.")
    return df


if source == "Upload a prepared order_fact CSV":
    up = st.file_uploader(
        "Upload a prepared order_fact CSV",
        type=["csv"],
        help="Expected columns: customer_unique_id, order_id, order_purchase_ts, order_revenue (+ optional primary_category, customer_state).",
    )
    if not up:
        st.info("Upload a CSV to continue, or switch to the included dataset.")
        st.stop()
    order_fact = _load_order_fact_from_csv(up.getvalue())
    if "primary_category" not in order_fact.columns:
        order_fact["primary_category"] = "unknown"
    if "customer_state" not in order_fact.columns:
        order_fact["customer_state"] = "unknown"
    lifecycle = customer_lifecycle(order_fact)
else:
    item_fact, order_fact, lifecycle = _load_order_fact_from_olist(str(Path(__file__).resolve().parent))


min_date = pd.Timestamp(order_fact["order_purchase_ts"].min()).date()
max_date = pd.Timestamp(order_fact["order_purchase_ts"].max()).date()
date_start, date_end = st.date_input("Analysis window", value=(min_date, max_date), min_value=min_date, max_value=max_date)

mask = (order_fact["order_purchase_ts"].dt.date >= date_start) & (order_fact["order_purchase_ts"].dt.date <= date_end)
order_fact_f = order_fact.loc[mask].copy()

if order_fact_f.empty:
    st.warning("No orders in the selected window.")
    st.stop()

lifecycle_f = customer_lifecycle(order_fact_f)

ret = retention_summary(lifecycle_f, horizons_days=(30, 60, 90))
ltv_coh = cohort_ltv(order_fact_f)
curve = cohort_retention_curve(order_fact_f, max_days=int(max_days))
seg = ltv_by_segment(order_fact_f, segment_col=segment)
scored = next_best_actions(score_churn_risk(lifecycle_f, churn_days=int(churn_days)))


top = st.columns(5)
top[0].metric("Customers", f"{lifecycle_f.shape[0]:,}")
top[1].metric("Orders", f"{order_fact_f['order_id'].nunique():,}")
top[2].metric("Revenue", f"{order_fact_f['order_revenue'].sum():,.0f}")
top[3].metric("Repeat rate", f"{(lifecycle_f['orders'] >= 2).mean():.1%}")
top[4].metric("Avg LTV", f"{lifecycle_f['revenue'].mean():,.0f}")

ret_cols = st.columns(3)
for i, h in enumerate([30, 60, 90]):
    v = float(ret.loc[ret["horizon_days"] == h, "retention_rate"].iloc[0])
    ret_cols[i].metric(f"Retention ≤ {h}d", f"{v:.1%}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Retention cohorts", "LTV", "Next best actions", "Data preview"])

with tab1:
    left, right = st.columns([1.2, 1])
    with left:
        fig = px.line(
            curve,
            x="days_since_first",
            y="retention_rate",
            color="cohort_month",
            title="Cohort retention curves (by acquisition month)",
            labels={"cohort_month": "Cohort month"},
        )
        fig.update_layout(legend_title_text="Cohort")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Biggest retention drops")
        # Simple heuristic: compare day 0 vs day 30 retention for each cohort
        d0 = curve[curve["days_since_first"] == 0][["cohort_month", "retention_rate"]].rename(
            columns={"retention_rate": "d0"}
        )
        d30 = curve[curve["days_since_first"] == 30][["cohort_month", "retention_rate"]].rename(
            columns={"retention_rate": "d30"}
        )
        drops = d0.merge(d30, on="cohort_month", how="inner")
        drops["drop_0_to_30"] = drops["d0"] - drops["d30"]
        st.dataframe(drops.sort_values("drop_0_to_30", ascending=False), use_container_width=True)

with tab2:
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(ltv_coh, x="cohort_month", y="ltv", title="LTV by acquisition month")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(seg.head(20), x=segment, y="ltv", title=f"LTV by {segment} (top 20)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("LTV is revenue per unique customer in the selected window.")

with tab3:
    st.subheader("Who to target next")
    st.caption("Rule-based actions using churn-risk + customer value (explainable and demo-friendly).")
    show_band = st.multiselect("Risk bands", options=["high", "medium", "low"], default=["high", "medium"])
    actions = scored[scored["risk_band"].isin(show_band)].copy()
    actions = actions.sort_values(["risk_score", "revenue"], ascending=[False, False]).head(200)
    st.dataframe(
        actions[
            [
                "customer_unique_id",
                "orders",
                "revenue",
                "aov",
                "recency_days",
                "risk_score",
                "risk_band",
                "action",
                "primary_category",
                "customer_state",
            ]
        ],
        use_container_width=True,
    )

    csv = actions.to_csv(index=False).encode("utf-8")
    st.download_button("Download actions CSV", data=csv, file_name="valueloop_actions.csv", mime="text/csv")

with tab4:
    st.subheader("Orders fact preview")
    st.dataframe(order_fact_f.head(50), use_container_width=True)
    st.subheader("Customer lifecycle preview")
    st.dataframe(lifecycle_f.head(50), use_container_width=True)