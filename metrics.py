from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetPaths:
    orders: Path
    order_items: Path
    products: Path
    customers: Path
    category_translation: Optional[Path] = None


def default_olist_paths(workspace_root: str | Path) -> DatasetPaths:
    root = Path(workspace_root)
    archive = root / "archive"
    return DatasetPaths(
        orders=archive / "olist_orders_dataset.csv",
        order_items=archive / "olist_order_items_dataset.csv",
        products=archive / "olist_products_dataset.csv",
        customers=archive / "olist_customers_dataset.csv",
        category_translation=archive / "product_category_name_translation.csv",
    )


def _read_csv(path: Path, usecols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols)


def build_item_fact(paths: DatasetPaths) -> pd.DataFrame:
    """
    Build an analysis-ready item-level fact table from the Olist dataset.

    Output columns (stable contract):
    - customer_id, customer_unique_id, customer_state
    - order_id, order_status, order_purchase_ts (datetime64[ns])
    - product_id, category_raw, category
    - item_revenue (price + freight_value)
    """
    orders = _read_csv(
        paths.orders,
        usecols=[
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
        ],
    ).rename(columns={"order_purchase_timestamp": "order_purchase_ts"})

    orders["order_purchase_ts"] = pd.to_datetime(
        orders["order_purchase_ts"], errors="coerce", utc=False
    )
    orders = orders.dropna(subset=["order_purchase_ts"])

    items = _read_csv(
        paths.order_items,
        usecols=["order_id", "product_id", "price", "freight_value"],
    )
    for col in ("price", "freight_value"):
        items[col] = pd.to_numeric(items[col], errors="coerce").fillna(0.0)
    items["item_revenue"] = items["price"] + items["freight_value"]

    products = _read_csv(
        paths.products,
        usecols=["product_id", "product_category_name"],
    ).rename(columns={"product_category_name": "category_raw"})

    customers = _read_csv(
        paths.customers,
        usecols=["customer_id", "customer_unique_id", "customer_state"],
    )

    fact = (
        items.merge(orders, on="order_id", how="inner")
        .merge(products, on="product_id", how="left")
        .merge(customers, on="customer_id", how="left")
    )

    if paths.category_translation and paths.category_translation.exists():
        trans = _read_csv(
            paths.category_translation,
            usecols=["product_category_name", "product_category_name_english"],
        ).rename(
            columns={
                "product_category_name": "category_raw",
                "product_category_name_english": "category",
            }
        )
        fact = fact.merge(trans, on="category_raw", how="left")
    else:
        fact["category"] = fact["category_raw"]

    fact["category"] = fact["category"].fillna("unknown")
    fact["customer_state"] = fact["customer_state"].fillna("unknown")
    return fact[
        [
            "customer_id",
            "customer_unique_id",
            "customer_state",
            "order_id",
            "order_status",
            "order_purchase_ts",
            "product_id",
            "category_raw",
            "category",
            "item_revenue",
        ]
    ]


def build_order_fact(item_fact: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate an item fact into an order-level fact table.
    """
    order_rev = (
        item_fact.groupby(["order_id", "customer_unique_id", "order_purchase_ts"], as_index=False)[
            "item_revenue"
        ].sum()
    ).rename(columns={"item_revenue": "order_revenue"})

    # Attach a "primary category" to each order for segmentation.
    cat_rev = (
        item_fact.groupby(["order_id", "category"], as_index=False)["item_revenue"]
        .sum()
        .sort_values(["order_id", "item_revenue"], ascending=[True, False])
    )
    primary_cat = cat_rev.drop_duplicates("order_id")[["order_id", "category"]].rename(
        columns={"category": "primary_category"}
    )

    # Attach state (unique per customer_unique_id in most cases; keep mode as fallback).
    state_mode = (
        item_fact.groupby(["customer_unique_id"], as_index=False)["customer_state"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else (s.iloc[0] if len(s) else "unknown"))
        .rename(columns={"customer_state": "customer_state"})
    )

    orders = (
        order_rev.merge(primary_cat, on="order_id", how="left").merge(
            state_mode, on="customer_unique_id", how="left"
        )
    )
    orders["primary_category"] = orders["primary_category"].fillna("unknown")
    orders["customer_state"] = orders["customer_state"].fillna("unknown")
    return orders


def customer_lifecycle(order_fact: pd.DataFrame) -> pd.DataFrame:
    """
    Customer-level rollup + features used in the dashboard and churn scoring.
    """
    o = order_fact.sort_values(["customer_unique_id", "order_purchase_ts"])
    g = o.groupby("customer_unique_id", as_index=False)
    lifecycle = g.agg(
        first_order=("order_purchase_ts", "min"),
        last_order=("order_purchase_ts", "max"),
        orders=("order_id", "nunique"),
        revenue=("order_revenue", "sum"),
        primary_category=("primary_category", lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown"),
        customer_state=("customer_state", lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown"),
    )
    lifecycle["aov"] = lifecycle["revenue"] / lifecycle["orders"].clip(lower=1)

    # Days until next purchase (NaN for one-timers)
    next_order_ts = o.groupby("customer_unique_id")["order_purchase_ts"].shift(-1)
    days_to_next = (next_order_ts - o["order_purchase_ts"]).dt.days
    first_next = (
        pd.DataFrame({"customer_unique_id": o["customer_unique_id"], "days_to_next": days_to_next})
        .dropna()
        .groupby("customer_unique_id", as_index=False)["days_to_next"]
        .min()
    )
    lifecycle = lifecycle.merge(first_next, on="customer_unique_id", how="left")
    return lifecycle


def retention_summary(lifecycle: pd.DataFrame, horizons_days: Iterable[int] = (30, 60, 90)) -> pd.DataFrame:
    rows = []
    for h in horizons_days:
        rows.append(
            {
                "horizon_days": int(h),
                "retention_rate": float((lifecycle["days_to_next"] <= h).mean()),
            }
        )
    return pd.DataFrame(rows)


def cohort_retention_curve(order_fact: pd.DataFrame, max_days: int = 180) -> pd.DataFrame:
    """
    Returns per-cohort retention by day-since-first-order.
    A customer is "retained" on day d if they have any order at or after day d.
    """
    o = order_fact[["customer_unique_id", "order_purchase_ts", "order_id"]].copy()
    o["first_order"] = o.groupby("customer_unique_id")["order_purchase_ts"].transform("min")
    o["cohort_month"] = o["first_order"].dt.to_period("M").dt.to_timestamp()
    o["days_since_first"] = (o["order_purchase_ts"] - o["first_order"]).dt.days.astype(int)
    o = o[(o["days_since_first"] >= 0) & (o["days_since_first"] <= max_days)]

    # For each cohort+day, count unique active customers.
    active = (
        o.groupby(["cohort_month", "days_since_first"], as_index=False)["customer_unique_id"]
        .nunique()
        .rename(columns={"customer_unique_id": "active_customers"})
    )
    cohort_size = (
        o.groupby(["cohort_month"], as_index=False)["customer_unique_id"]
        .nunique()
        .rename(columns={"customer_unique_id": "cohort_size"})
    )
    out = active.merge(cohort_size, on="cohort_month", how="left")
    out["retention_rate"] = out["active_customers"] / out["cohort_size"].replace(0, np.nan)
    return out.sort_values(["cohort_month", "days_since_first"])


def cohort_ltv(order_fact: pd.DataFrame) -> pd.DataFrame:
    """
    Cohort-level revenue and LTV (revenue per customer) by acquisition month.
    """
    o = order_fact[["customer_unique_id", "order_purchase_ts", "order_revenue"]].copy()
    o["first_order"] = o.groupby("customer_unique_id")["order_purchase_ts"].transform("min")
    o["cohort_month"] = o["first_order"].dt.to_period("M").dt.to_timestamp()

    cohort = o.groupby("cohort_month", as_index=False).agg(
        customers=("customer_unique_id", "nunique"),
        revenue=("order_revenue", "sum"),
    )
    cohort["ltv"] = cohort["revenue"] / cohort["customers"].replace(0, np.nan)
    return cohort.sort_values("cohort_month")


def ltv_by_segment(order_fact: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    if segment_col not in order_fact.columns:
        raise KeyError(f"segment_col '{segment_col}' not in order_fact columns")
    agg = order_fact.groupby(segment_col, as_index=False).agg(
        customers=("customer_unique_id", "nunique"),
        orders=("order_id", "nunique"),
        revenue=("order_revenue", "sum"),
    )
    agg["ltv"] = agg["revenue"] / agg["customers"].replace(0, np.nan)
    agg["aov"] = agg["revenue"] / agg["orders"].replace(0, np.nan)
    return agg.sort_values("ltv", ascending=False)


def score_churn_risk(
    lifecycle: pd.DataFrame,
    as_of: Optional[pd.Timestamp] = None,
    churn_days: int = 90,
) -> pd.DataFrame:
    """
    Simple, explainable churn-risk score based on RFM-style features.
    Returns lifecycle + recency_days + risk_score (0..1) + risk_band.
    """
    df = lifecycle.copy()
    if as_of is None:
        as_of = pd.Timestamp(df["last_order"].max())
    as_of = pd.Timestamp(as_of)

    df["recency_days"] = (as_of - df["last_order"]).dt.days.clip(lower=0)
    df["tenure_days"] = (as_of - df["first_order"]).dt.days.clip(lower=0)

    # Normalize onto 0..1 with robust percentiles to reduce outlier influence.
    def _robust_norm(s: pd.Series, high_bad: bool) -> pd.Series:
        lo = float(s.quantile(0.05))
        hi = float(s.quantile(0.95))
        if hi <= lo:
            return pd.Series(0.0, index=s.index)
        x = ((s - lo) / (hi - lo)).clip(0, 1)
        return x if high_bad else (1 - x)

    recency_bad = _robust_norm(df["recency_days"], high_bad=True)
    value_good = _robust_norm(df["revenue"], high_bad=False)
    freq_good = _robust_norm(df["orders"], high_bad=False)

    # Risk increases with recency; decreases with value/frequency.
    df["risk_score"] = (0.60 * recency_bad + 0.25 * (1 - value_good) + 0.15 * (1 - freq_good)).clip(
        0, 1
    )

    df["risk_band"] = pd.cut(
        df["risk_score"],
        bins=[-0.001, 0.33, 0.66, 1.0],
        labels=["low", "medium", "high"],
    ).astype(str)

    # Helpful boolean used in UI: "already churned" by heuristic definition.
    df["churned_flag"] = df["recency_days"] >= int(churn_days)
    return df


def next_best_actions(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based actions for a resume-friendly, explainable output.
    """
    df = scored.copy()
    df["action"] = "Nurture"
    df.loc[(df["risk_band"] == "high") & (df["revenue"] >= df["revenue"].median()), "action"] = (
        "Winback: high-value"
    )
    df.loc[(df["risk_band"] == "high") & (df["revenue"] < df["revenue"].median()), "action"] = (
        "Winback: promo"
    )
    df.loc[(df["risk_band"] == "medium") & (df["orders"] <= 1), "action"] = "2nd purchase nudge"
    df.loc[(df["risk_band"] == "low") & (df["orders"] >= 3), "action"] = "VIP: loyalty / referral"
    return df

