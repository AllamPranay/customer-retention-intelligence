"""Phase 5: RFM customer segmentation.

Recency  - days since the customer's last purchase (lower is better)
Frequency- number of distinct orders placed
Monetary - total spend

Each dimension is scored 1-5 using quintiles of the actual data, so the
thresholds are set by this business's own customer distribution rather than
by arbitrary numbers. Recency is reversed (a recent buyer scores 5).

Segments are assigned from the Recency x Frequency grid, which is the pair
that describes *engagement*. Monetary is deliberately not used here: it
drives the separate Customer Value Score in Phase 7, which keeps
"how engaged is this customer" and "how much is this customer worth"
as two independent axes.
"""
import pandas as pd

import config
from data_processing import load_clean

# Ordered rules - the first matching rule wins. Between them they cover
# every cell of the 5x5 Recency x Frequency grid.
SEGMENT_RULES = [
    ("Champions",          lambda r, f: (r >= 4) & (f >= 4)),
    ("Loyal Customers",    lambda r, f: (r >= 3) & (f >= 3)),
    ("Potential Loyalist", lambda r, f: (r >= 4) & (f == 2)),
    ("New Customers",      lambda r, f: (r >= 4) & (f == 1)),
    ("At Risk",            lambda r, f: (r <= 2) & (f >= 3)),
    ("Lost Customers",     lambda r, f: (r == 1) & (f <= 2)),
    ("Hibernating",        lambda r, f: (r <= 3) & (f <= 2)),
]

SEGMENT_ORDER = ["Champions", "Loyal Customers", "Potential Loyalist", "New Customers",
                 "At Risk", "Hibernating", "Lost Customers"]


def _quintile_score(series: pd.Series, reverse: bool = False) -> pd.Series:
    """Score a series 1-5 by quintile.

    Ranking before cutting breaks the heavy ties (for example the many
    customers with exactly one order) that would otherwise make the
    quintile edges collapse onto each other.
    """
    ranked = series.rank(method="first")
    labels = [5, 4, 3, 2, 1] if reverse else [1, 2, 3, 4, 5]
    return pd.qcut(ranked, 5, labels=labels).astype(int)


def build_rfm(df: pd.DataFrame = None, asof=None) -> pd.DataFrame:
    """Build the full-history RFM table as of a reference date."""
    if df is None:
        df = load_clean()
    asof = pd.Timestamp(asof) if asof is not None else df.InvoiceDate.max()

    d = df[df.InvoiceDate <= asof]
    rfm = d.groupby("CustomerID").agg(
        recency_days=("InvoiceDate", lambda s: (asof - s.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary=("Revenue", "sum"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        country=("Country", lambda s: s.mode().iat[0]),
    ).reset_index()

    rfm["monetary"] = rfm["monetary"].round(2)
    rfm["avg_order_value"] = (rfm["monetary"] / rfm["frequency"]).round(2)
    rfm["tenure_days"] = (asof - rfm["first_purchase"]).dt.days

    rfm["R"] = _quintile_score(rfm["recency_days"], reverse=True)
    rfm["F"] = _quintile_score(rfm["frequency"])
    rfm["M"] = _quintile_score(rfm["monetary"])
    rfm["rfm_score"] = rfm["R"] + rfm["F"] + rfm["M"]
    rfm["rfm_cell"] = (rfm.R.astype(str) + rfm.F.astype(str) + rfm.M.astype(str))

    rfm["segment"] = assign_segments(rfm["R"], rfm["F"])
    rfm["asof_date"] = asof.date().isoformat()
    return rfm


def assign_segments(r: pd.Series, f: pd.Series) -> pd.Series:
    """Apply the ordered rule list to R and F scores."""
    seg = pd.Series("Unclassified", index=r.index, dtype=object)
    for name, rule in SEGMENT_RULES:
        mask = rule(r, f) & (seg == "Unclassified")
        seg[mask] = name
    return seg


def segment_profile(rfm: pd.DataFrame) -> pd.DataFrame:
    """Business summary of each segment - the table that goes on the dashboard."""
    prof = rfm.groupby("segment").agg(
        customers=("CustomerID", "count"),
        avg_recency_days=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        total_revenue=("monetary", "sum"),
    ).round(1)
    prof["pct_customers"] = (prof.customers / prof.customers.sum() * 100).round(1)
    prof["pct_revenue"] = (prof.total_revenue / prof.total_revenue.sum() * 100).round(1)
    prof = prof.reindex([s for s in SEGMENT_ORDER if s in prof.index])
    return prof.reset_index()


def run(verbose: bool = True):
    rfm = build_rfm()
    prof = segment_profile(rfm)
    rfm.to_csv(config.PROCESSED_DIR / "rfm_segments.csv", index=False)
    prof.to_csv(config.REPORTS_DIR / "segment_profile.csv", index=False)

    if verbose:
        print(f"RFM built for {len(rfm):,} customers as of {rfm.asof_date.iat[0]}")
        print("\nQuintile cut points (from the data):")
        for col, label in [("recency_days", "R"), ("frequency", "F"), ("monetary", "M")]:
            edges = rfm.groupby(label)[col].agg(["min", "max"]).round(0)
            print(f"  {label} ({col}):")
            for score, row in edges.iterrows():
                print(f"     score {score}: {row['min']:>10,.0f} - {row['max']:>10,.0f}")
        print("\nSegment profile:")
        print(prof.to_string(index=False))
    return rfm, prof


if __name__ == "__main__":
    run()
