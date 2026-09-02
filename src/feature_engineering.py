"""Phase 6 (part 1): build the customer-level modelling dataset.

A *snapshot* is built at a cut-off date and has two strictly separated parts:

    [ observation window ]  |  [ future window ]
     features come from        target comes from
     here ONLY                 here ONLY

The observation window has a FIXED length (config.OBSERVATION_DAYS) at every
cut-off. This matters: if the window grew longer at later cut-offs, volume
features such as frequency and monetary would rise for reasons that have
nothing to do with customer behaviour, and the model would not transfer
from one snapshot to the next.

Target
    1 = the customer placed NO order in the future window (became inactive)
    0 = the customer placed at least one order in the future window

No column computed after the cut-off is ever used as a feature, so the
dataset is free of look-ahead leakage by construction.
"""
import numpy as np
import pandas as pd

import config
from data_processing import load_clean, load_cancellations


def build_snapshot(sales: pd.DataFrame, cancellations: pd.DataFrame, cutoff,
                   observation_days: int = config.OBSERVATION_DAYS,
                   horizon_days: int = config.HORIZON_DAYS,
                   with_target: bool = True) -> pd.DataFrame:
    """Return one row per customer active in the observation window."""
    cutoff = pd.Timestamp(cutoff)
    obs_start = cutoff - pd.Timedelta(days=observation_days)

    obs = sales[(sales.InvoiceDate >= obs_start) & (sales.InvoiceDate < cutoff)].copy()
    if obs.empty:
        raise ValueError(f"No transactions in observation window ending {cutoff.date()}")

    # ---------------- order-level view (one row per invoice) --------------
    orders = obs.groupby(["CustomerID", "InvoiceNo"]).agg(
        order_date=("InvoiceDate", "min"),
        order_value=("Revenue", "sum"),
        order_qty=("Quantity", "sum"),
    ).reset_index().sort_values(["CustomerID", "order_date"])

    g = orders.groupby("CustomerID")
    f = pd.DataFrame({
        "frequency": g.InvoiceNo.nunique(),               # orders placed
        "monetary": g.order_value.sum().round(2),         # total spend
        "avg_order_value": g.order_value.mean().round(2),
        "max_order_value": g.order_value.max().round(2),
        "total_quantity": g.order_qty.sum(),
        "first_order": g.order_date.min(),
        "last_order": g.order_date.max(),
    })

    # Recency: days from the customer's last order to the cut-off.
    f["recency_days"] = (cutoff - f["last_order"]).dt.days
    # Tenure: how long the customer has been known within this window.
    f["tenure_days"] = (cutoff - f["first_order"]).dt.days

    # Average gap between consecutive orders. Customers with a single order
    # have no observable rhythm; they are given the full window length, which
    # correctly reads as "no repeat behaviour seen".
    gaps = orders.groupby("CustomerID").order_date.diff().dt.days
    avg_gap = gaps.groupby(orders["CustomerID"]).mean()
    f["avg_days_between_orders"] = avg_gap.reindex(f.index).fillna(observation_days).round(1)

    # Momentum: orders in the recent half of the window vs the earlier half.
    # >1 means the customer is speeding up, <1 means slowing down.
    midpoint = cutoff - pd.Timedelta(days=observation_days / 2)
    recent = orders[orders.order_date >= midpoint].groupby("CustomerID").InvoiceNo.nunique()
    early = orders[orders.order_date < midpoint].groupby("CustomerID").InvoiceNo.nunique()
    recent = recent.reindex(f.index).fillna(0)
    early = early.reindex(f.index).fillna(0)
    f["purchase_trend_ratio"] = ((recent + 1) / (early + 1)).round(3)

    # Breadth and engagement spread
    f["unique_products"] = obs.groupby("CustomerID").StockCode.nunique()
    f["active_weeks"] = (obs.assign(wk=obs.InvoiceDate.dt.to_period("W"))
                         .groupby("CustomerID").wk.nunique())
    f["avg_basket_size"] = (f["total_quantity"] / f["frequency"]).round(1)

    # ---------------- cancellation behaviour (observation window only) ----
    canc = cancellations[(cancellations.InvoiceDate >= obs_start)
                         & (cancellations.InvoiceDate < cutoff)]
    cnt = canc.groupby("CustomerID").InvoiceNo.nunique()
    f["cancel_orders"] = cnt.reindex(f.index).fillna(0).astype(int)
    f["cancel_rate"] = (f["cancel_orders"] / (f["frequency"] + f["cancel_orders"])).round(3)

    # ---------------- geography -------------------------------------------
    country = obs.groupby("CustomerID").Country.agg(lambda s: s.mode().iat[0])
    f["country"] = country
    f["is_uk"] = (country == "United Kingdom").astype(int)

    f = f.reset_index()
    f["snapshot_cutoff"] = cutoff.date().isoformat()
    f["observation_start"] = obs_start.date().isoformat()

    # ---------------- target ---------------------------------------------
    if with_target:
        fut_end = cutoff + pd.Timedelta(days=horizon_days)
        fut = sales[(sales.InvoiceDate >= cutoff) & (sales.InvoiceDate < fut_end)]
        returned = set(fut.CustomerID.unique())
        f["is_inactive"] = (~f.CustomerID.isin(returned)).astype(int)
        # Future spend is kept for business reporting ONLY - never a feature.
        fut_spend = fut.groupby("CustomerID").Revenue.sum()
        f["future_revenue_actual"] = f.CustomerID.map(fut_spend).fillna(0).round(2)

    return f.drop(columns=["first_order", "last_order"])


def build_all_snapshots(verbose: bool = True) -> dict:
    """Build the train / test / score snapshots defined in config.SNAPSHOTS."""
    sales = load_clean()
    cancellations = load_cancellations()
    snapshots = {}

    for name, cutoff in config.SNAPSHOTS.items():
        # The score snapshot sits at the end of the data: its future window
        # has not happened yet, so it carries no target.
        with_target = name != "score"
        snap = build_snapshot(sales, cancellations, cutoff, with_target=with_target)
        snapshots[name] = snap
        snap.to_csv(config.PROCESSED_DIR / f"features_{name}.csv", index=False)

        if verbose:
            line = (f"  {name:<6} cutoff={cutoff}  obs from {snap.observation_start.iat[0]}"
                    f"  customers={len(snap):>5,}")
            if with_target:
                line += (f"  inactive={snap.is_inactive.mean()*100:.1f}%")
            print(line)

    return snapshots


if __name__ == "__main__":
    snaps = build_all_snapshots()
    print("\nFeature columns:", len(config.FEATURE_COLUMNS))
    print(snaps["train"][config.FEATURE_COLUMNS].describe().T.round(2).to_string())
    print("\nMissing values per snapshot:")
    for k, v in snaps.items():
        print(f"  {k}: {int(v[config.FEATURE_COLUMNS].isna().sum().sum())}")
