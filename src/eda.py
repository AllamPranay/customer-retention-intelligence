"""Phase 3: exploratory data analysis.

Every block here answers a business question that feeds the retention
problem. Figures are written to reports/figures/ and the numbers behind
them to reports/eda_summary.json.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from data_processing import load_clean, load_cancellations

plt.rcParams.update({
    "figure.dpi": 120, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9,
})
BLUE, ORANGE, GREY = "#2E5E8C", "#D97A28", "#8C8C8C"


def _save(fig, name):
    fig.savefig(config.FIGURES_DIR / f"{name}.png")
    plt.close(fig)


def run(verbose: bool = True) -> dict:
    df = load_clean()
    canc = load_cancellations()
    out = {}

    # ---------------- Q: how big is the customer base and how loyal is it?
    orders = df.groupby("CustomerID")["InvoiceNo"].nunique()
    out["total_customers"] = int(df.CustomerID.nunique())
    out["total_orders"] = int(df.InvoiceNo.nunique())
    out["total_revenue"] = round(float(df.Revenue.sum()), 2)
    out["repeat_customers"] = int((orders >= 2).sum())
    out["repeat_rate_pct"] = round(float((orders >= 2).mean() * 100), 1)
    out["one_time_customers"] = int((orders == 1).sum())
    out["avg_orders_per_customer"] = round(float(orders.mean()), 2)
    out["median_orders_per_customer"] = float(orders.median())

    # Revenue concentration: how much of the business sits with repeat buyers?
    rev_by_cust = df.groupby("CustomerID").Revenue.sum()
    repeat_ids = orders[orders >= 2].index
    out["revenue_from_repeat_pct"] = round(
        float(rev_by_cust.loc[repeat_ids].sum() / rev_by_cust.sum() * 100), 1)
    top20 = rev_by_cust.sort_values(ascending=False).head(int(len(rev_by_cust) * 0.20))
    out["revenue_from_top20pct_customers"] = round(float(top20.sum() / rev_by_cust.sum() * 100), 1)

    # ---------------- Q: what does revenue look like over time?
    monthly = df.set_index("InvoiceDate").resample("MS").agg(
        revenue=("Revenue", "sum"),
        orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique"),
    )
    out["avg_order_value"] = round(float(df.groupby("InvoiceNo").Revenue.sum().mean()), 2)
    out["date_min"], out["date_max"] = str(df.InvoiceDate.min()), str(df.InvoiceDate.max())

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(monthly.index, monthly.revenue / 1000, marker="o", color=BLUE)
    ax[0].set_ylabel("Revenue (GBP thousands)")
    ax[0].set_title("Monthly revenue and active customers")
    ax[1].bar(monthly.index, monthly.customers, width=20, color=GREY)
    ax[1].set_ylabel("Active customers")
    _save(fig, "01_revenue_and_customers_trend")

    # ---------------- Q: how are order values distributed?
    order_val = df.groupby("InvoiceNo").Revenue.sum()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(order_val[order_val < order_val.quantile(0.99)], bins=60, color=BLUE)
    ax.axvline(order_val.median(), color=ORANGE, ls="--",
               label=f"median GBP {order_val.median():,.0f}")
    ax.set_xlabel("Order value (GBP)")
    ax.set_ylabel("Orders")
    ax.legend()
    ax.set_title("Order value distribution (below 99th percentile)")
    _save(fig, "02_order_value_distribution")

    # ---------------- Q: how many orders does a typical customer place?
    fig, ax = plt.subplots(figsize=(7, 3.5))
    counts = orders.clip(upper=15).value_counts().sort_index()
    ax.bar(counts.index, counts.values,
           color=[ORANGE if i == 1 else BLUE for i in counts.index])
    ax.set_xlabel("Orders per customer (15+ grouped)")
    ax.set_ylabel("Customers")
    ax.set_title(f"{out['repeat_rate_pct']}% of customers order more than once")
    _save(fig, "03_orders_per_customer")

    # ---------------- Q: which products drive the business?
    top_products = (df.groupby("Description")
                    .agg(revenue=("Revenue", "sum"), quantity=("Quantity", "sum"),
                         customers=("CustomerID", "nunique"))
                    .sort_values("revenue", ascending=False).head(10))
    out["top_products"] = top_products.revenue.round(0).to_dict()
    out["unique_products"] = int(df.StockCode.nunique())

    fig, ax = plt.subplots(figsize=(7, 4))
    tp = top_products.sort_values("revenue")
    ax.barh([d[:38].title() for d in tp.index], tp.revenue / 1000, color=BLUE)
    ax.set_xlabel("Revenue (GBP thousands)")
    ax.set_title("Top 10 products by revenue")
    _save(fig, "04_top_products")

    # ---------------- Q: where are the customers?
    geo = (df.groupby("Country")
           .agg(revenue=("Revenue", "sum"), customers=("CustomerID", "nunique"))
           .sort_values("revenue", ascending=False))
    out["uk_revenue_pct"] = round(
        float(geo.loc["United Kingdom", "revenue"] / geo.revenue.sum() * 100), 1)
    out["uk_customer_pct"] = round(
        float(geo.loc["United Kingdom", "customers"] / out["total_customers"] * 100), 1)
    out["n_countries"] = int(len(geo))
    out["top_countries"] = geo.head(8).revenue.round(0).to_dict()

    fig, ax = plt.subplots(figsize=(7, 4))
    g = geo.head(10).sort_values("revenue")
    ax.barh(g.index, g.revenue / 1000,
            color=[ORANGE if c == "United Kingdom" else BLUE for c in g.index])
    ax.set_xscale("log")
    ax.set_xlabel("Revenue (GBP thousands, log scale)")
    ax.set_title(f"Revenue by country - UK is {out['uk_revenue_pct']}% of the business")
    _save(fig, "05_revenue_by_country")

    # ---------------- Q: how long do customers go between orders?
    first_per_order = (df.groupby(["CustomerID", "InvoiceNo"]).InvoiceDate.min()
                       .reset_index().sort_values(["CustomerID", "InvoiceDate"]))
    gaps = first_per_order.groupby("CustomerID").InvoiceDate.diff().dt.days.dropna()
    out["median_days_between_orders"] = float(gaps.median())
    out["p75_days_between_orders"] = float(gaps.quantile(0.75))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(gaps[gaps <= 200], bins=50, color=BLUE)
    ax.axvline(gaps.median(), color=ORANGE, ls="--", label=f"median {gaps.median():.0f} days")
    ax.set_xlabel("Days between consecutive orders")
    ax.set_ylabel("Order gaps")
    ax.legend()
    ax.set_title("Repeat purchase rhythm sets the inactivity horizon")
    _save(fig, "06_days_between_orders")

    # ---------------- Q: how many customers are already dormant?
    asof = df.InvoiceDate.max()
    recency = (asof - df.groupby("CustomerID").InvoiceDate.max()).dt.days
    out["customers_inactive_90d"] = int((recency > 90).sum())
    out["customers_inactive_90d_pct"] = round(float((recency > 90).mean() * 100), 1)
    out["revenue_at_stake_inactive_90d"] = round(
        float(rev_by_cust[recency[recency > 90].index].sum()), 2)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(recency, bins=50, color=BLUE)
    ax.axvline(90, color=ORANGE, ls="--", label="90 days")
    ax.set_xlabel(f"Days since last purchase (as of {asof.date()})")
    ax.set_ylabel("Customers")
    ax.legend()
    ax.set_title(f"{out['customers_inactive_90d_pct']}% of customers have not bought in 90+ days")
    _save(fig, "07_recency_distribution")

    # ---------------- Q: do cancellations matter?
    out["customers_with_cancellations"] = int(canc.CustomerID.nunique())
    out["cancellation_line_rate_pct"] = round(float(len(canc) / (len(df) + len(canc)) * 100), 2)

    # ---------------- Monthly table for the dashboard
    monthly.reset_index().rename(columns={"InvoiceDate": "month"}).to_csv(
        config.PROCESSED_DIR / "monthly_summary.csv", index=False)

    with open(config.REPORTS_DIR / "eda_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    if verbose:
        for k, v in out.items():
            if not isinstance(v, dict):
                print(f"  {k:<36} {v}")
    return out


if __name__ == "__main__":
    run()
