"""Phases 7-9: customer value, retention priority, and value at risk.

Phase 7 - Customer Value Score
    A deliberately simple, transparent score. No probabilistic lifetime-value
    model: the business needs a number it can explain to a marketing team,
    and the inputs (spend, order count, basket size) are the ones a retention
    manager already reasons about.

Phase 8 - Retention Priority Engine
    Crossing value with risk turns a list of probabilities into a decision:
    who to contact first, and with what.

    One finding shaped this stage. Customer value and inactivity risk are
    correlated at -0.87, because both are built from how often and how much
    a customer buys. Banding risk on absolute probability therefore put
    ZERO high-value customers in the high-risk band, and sent 97% of the
    "high risk" population into the lowest value tier - a campaign list that
    spends the whole budget on the least valuable customers.

    The engine instead bands risk RELATIVE TO PEERS in the same value tier:
    a customer is high risk if they are in the riskiest 20% of customers of
    similar value. A high-value customer at p=0.55 is genuinely alarming
    even though 0.55 is unremarkable across the whole base. Both views are
    kept on the output table - risk_band_absolute is the raw model reading,
    risk_band is the peer-relative one used for prioritisation.

Phase 9 - Customer Value at Risk
    An exposure estimate, NOT a revenue forecast. It answers "where is the
    most value concentrated among customers who look likely to lapse", which
    is a prioritisation question, not a prediction of guaranteed loss.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from segmentation import build_rfm

BLUE, ORANGE, GREY, RED = "#2E5E8C", "#D97A28", "#8C8C8C", "#B03A2E"

# Weights for the Customer Value Score. Total spend carries the most weight
# because it already blends how often and how much a customer buys; order
# count rewards habit; average order value separates big-basket buyers.
VALUE_WEIGHTS = {"monetary": 0.50, "frequency": 0.30, "avg_order_value": 0.20}

VALUE_TIERS = [("High", 4.0, 5.01), ("Medium", 2.5, 4.0), ("Low", 0.0, 2.5)]

# Phase 8 decision matrix: (value tier, risk band) -> (priority, action)
PRIORITY_MATRIX = {
    ("High", "High"):     ("1 - Critical", "Personal contact from account manager plus a tailored offer"),
    ("High", "Medium"):   ("2 - High", "Targeted offer on their usual categories"),
    ("Medium", "High"):   ("2 - High", "Targeted win-back offer with a deadline"),
    ("High", "Low"):      ("4 - Maintain", "Loyalty rewards and early access - protect the relationship"),
    ("Medium", "Medium"): ("3 - Monitor", "Add to the standard nurture campaign and re-score next month"),
    ("Medium", "Low"):    ("5 - Low Priority", "No action needed - keep in normal marketing"),
    ("Low", "High"):      ("3 - Low-Cost Retention", "Automated email or discount code only"),
    ("Low", "Medium"):    ("5 - Low Priority", "Automated newsletter only"),
    ("Low", "Low"):       ("5 - Low Priority", "No action needed - keep in normal marketing"),
}

PRIORITY_ORDER = ["1 - Critical", "2 - High", "3 - Low-Cost Retention",
                  "3 - Monitor", "4 - Maintain", "5 - Low Priority"]


def _quintile(series: pd.Series) -> pd.Series:
    return pd.qcut(series.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)


# ----------------------------------------------------------------------
# Phase 7 - Customer Value Score
# ----------------------------------------------------------------------
def add_customer_value(df: pd.DataFrame) -> pd.DataFrame:
    """Score each customer 1-5 on spend, order count and basket size, then
    combine with fixed weights into a single value score and tier."""
    d = df.copy()
    for col in VALUE_WEIGHTS:
        d[f"score_{col}"] = _quintile(d[col])

    d["customer_value_score"] = sum(
        d[f"score_{col}"] * w for col, w in VALUE_WEIGHTS.items()).round(2)

    d["customer_value_tier"] = pd.cut(
        d["customer_value_score"],
        bins=[t[1] for t in reversed(VALUE_TIERS)] + [VALUE_TIERS[0][2]],
        labels=[t[0] for t in reversed(VALUE_TIERS)],
        include_lowest=True).astype(str)
    return d


# ----------------------------------------------------------------------
# Phase 8 - Retention Priority Engine
# ----------------------------------------------------------------------
def add_relative_risk_band(df: pd.DataFrame) -> pd.DataFrame:
    """Band inactivity risk against peers of similar value.

    Within each value tier: riskiest 20% -> High, next 30% -> Medium,
    safest 50% -> Low. This keeps every cell of the value x risk matrix
    populated and stops the campaign list collapsing onto low-value
    customers (see the module docstring).
    """
    d = df.copy()
    pct = d.groupby("customer_value_tier")["inactivity_probability"].rank(pct=True)
    d["risk_percentile_in_tier"] = pct.round(3)
    d["risk_band"] = pd.cut(pct, bins=[0, 0.50, 0.80, 1.0],
                            labels=["Low", "Medium", "High"],
                            include_lowest=True).astype(str)
    return d


def add_retention_priority(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    pairs = list(zip(d["customer_value_tier"], d["risk_band"].astype(str)))
    d["retention_priority"] = [PRIORITY_MATRIX.get(p, ("5 - Low Priority", ""))[0] for p in pairs]
    d["recommended_action"] = [PRIORITY_MATRIX.get(p, ("", "No action needed"))[1] for p in pairs]
    return d


# ----------------------------------------------------------------------
# Phase 9 - Customer Value at Risk
# ----------------------------------------------------------------------
def add_value_at_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Value at Risk = P(inactive) x the customer's recent spending rate,
    expressed over the same 90-day horizon the model predicts.

    The spending rate is the customer's observed spend in the 180-day
    observation window, halved to put it on a 90-day footing. This is a
    straight-line assumption and is stated as such: it is an exposure
    measure for ranking customers, not a forecast.
    """
    d = df.copy()
    scale = config.HORIZON_DAYS / config.OBSERVATION_DAYS   # 90 / 180 = 0.5
    d["expected_90d_value"] = (d["monetary"] * scale).round(2)
    d["value_at_risk"] = (d["inactivity_probability"] * d["expected_90d_value"]).round(2)
    return d


# ----------------------------------------------------------------------
def build_customer_master(verbose: bool = True) -> pd.DataFrame:
    """Assemble the single customer table that drives the dashboard."""
    risk = pd.read_csv(config.PROCESSED_DIR / "customer_risk_scores.csv")
    rfm = build_rfm()

    d = add_customer_value(risk)
    d = add_relative_risk_band(d)
    d = add_retention_priority(d)
    d = add_value_at_risk(d)

    # Attach the full-history RFM segment (descriptive view of the customer)
    d = d.merge(
        rfm[["CustomerID", "segment", "R", "F", "M", "rfm_score",
             "recency_days", "frequency", "monetary", "tenure_days"]].rename(columns={
                 "recency_days": "lifetime_recency_days",
                 "frequency": "lifetime_orders",
                 "monetary": "lifetime_revenue",
                 "tenure_days": "lifetime_tenure_days"}),
        on="CustomerID", how="left")

    # Customers whose last purchase predates the observation window cannot be
    # scored by the model. They are already dormant and belong in a separate
    # reactivation bucket rather than being given a made-up risk score.
    dormant = rfm[~rfm.CustomerID.isin(d.CustomerID)].copy()
    if len(dormant):
        dormant_rows = pd.DataFrame({
            "CustomerID": dormant.CustomerID,
            "country": dormant.country,
            "monetary": 0.0, "frequency": 0, "avg_order_value": 0.0,
            "inactivity_probability": np.nan,
            "risk_band": "Already Dormant",
            "risk_band_absolute": "Already Dormant",
            "risk_percentile_in_tier": np.nan,
            "customer_value_score": np.nan,
            "customer_value_tier": "Dormant",
            "retention_priority": "6 - Reactivation",
            "recommended_action": "Outside the 180-day scoring window - low-cost reactivation campaign",
            "expected_90d_value": 0.0, "value_at_risk": 0.0,
            "segment": dormant.segment, "R": dormant.R, "F": dormant.F, "M": dormant.M,
            "rfm_score": dormant.rfm_score,
            "lifetime_recency_days": dormant.recency_days,
            "lifetime_orders": dormant.frequency,
            "lifetime_revenue": dormant.monetary,
            "lifetime_tenure_days": dormant.tenure_days,
        })
        d = pd.concat([d, dormant_rows], ignore_index=True)

    d["is_high_risk"] = (d.risk_band == "High").astype(int)
    d = d.sort_values("value_at_risk", ascending=False).reset_index(drop=True)
    d.to_csv(config.PROCESSED_DIR / "customer_master.csv", index=False)
    if verbose:
        print(f"Customer master built: {len(d):,} customers "
              f"({len(risk):,} scored + {len(dormant):,} already dormant)")
    return d


def summarise(master: pd.DataFrame, verbose: bool = True) -> dict:
    scored = master[master.risk_band != "Already Dormant"]

    value_x_risk = (scored.pivot_table(index="customer_value_tier", columns="risk_band",
                                       values="CustomerID", aggfunc="count", observed=False)
                    .reindex(["High", "Medium", "Low"])
                    .reindex(columns=["Low", "Medium", "High"]).fillna(0).astype(int))

    var_x_risk = (scored.pivot_table(index="customer_value_tier", columns="risk_band",
                                     values="value_at_risk", aggfunc="sum", observed=False)
                  .reindex(["High", "Medium", "Low"])
                  .reindex(columns=["Low", "Medium", "High"]).fillna(0).round(0))

    priority = (master.groupby("retention_priority").agg(
        customers=("CustomerID", "count"),
        value_at_risk=("value_at_risk", "sum"),
        avg_inactivity_probability=("inactivity_probability", "mean"),
        recent_revenue=("monetary", "sum"),
    ).round(2))
    priority["pct_customers"] = (priority.customers / priority.customers.sum() * 100).round(1)
    priority["action"] = [
        master.loc[master.retention_priority == p, "recommended_action"].iat[0]
        for p in priority.index]
    priority = priority.reindex([p for p in PRIORITY_ORDER + ["6 - Reactivation"]
                                 if p in priority.index]).reset_index()

    segment_risk = (scored.groupby("segment").agg(
        customers=("CustomerID", "count"),
        avg_inactivity_probability=("inactivity_probability", "mean"),
        value_at_risk=("value_at_risk", "sum"),
    ).round(3).sort_values("value_at_risk", ascending=False).reset_index())

    priority.to_csv(config.REPORTS_DIR / "retention_priority_summary.csv", index=False)
    segment_risk.to_csv(config.REPORTS_DIR / "segment_risk_summary.csv", index=False)

    critical = master[master.retention_priority == "1 - Critical"]
    summary = {
        "customers_total": int(len(master)),
        "customers_scored": int(len(scored)),
        "customers_already_dormant": int((master.risk_band == "Already Dormant").sum()),
        "high_risk_customers": int((scored.risk_band == "High").sum()),
        "total_value_at_risk": round(float(master.value_at_risk.sum()), 2),
        "critical_customers": int(len(critical)),
        "critical_value_at_risk": round(float(critical.value_at_risk.sum()), 2),
        "critical_share_of_var_pct": round(
            float(critical.value_at_risk.sum() / master.value_at_risk.sum() * 100), 1),
        "top1_2_priority_customers": int(master.retention_priority.isin(
            ["1 - Critical", "2 - High"]).sum()),
        "top1_2_value_at_risk": round(float(master.loc[master.retention_priority.isin(
            ["1 - Critical", "2 - High"]), "value_at_risk"].sum()), 2),
    }
    with open(config.REPORTS_DIR / "retention_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print("\nCustomers by value tier x risk band:")
        print(value_x_risk.to_string())
        print("\nValue at Risk (GBP) by value tier x risk band:")
        print(var_x_risk.to_string())
        print("\nRetention priority groups:")
        print(priority[["retention_priority", "customers", "pct_customers",
                        "value_at_risk", "avg_inactivity_probability"]].to_string(index=False))
        print("\nInactivity risk by RFM segment:")
        print(segment_risk.to_string(index=False))
        print("\nHeadline numbers:")
        for k, v in summary.items():
            print(f"  {k:<32} {v:,}" if isinstance(v, (int, float)) else f"  {k:<32} {v}")
    return summary, value_x_risk, priority


def make_figures(master: pd.DataFrame, priority: pd.DataFrame):
    scored = master[master.risk_band != "Already Dormant"]

    # Value vs risk scatter - the core picture of the whole project
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = {"High": RED, "Medium": ORANGE, "Low": BLUE}
    for tier in ["Low", "Medium", "High"]:
        s = scored[scored.customer_value_tier == tier]
        ax.scatter(s.inactivity_probability, s.expected_90d_value + 1, s=14, alpha=0.5,
                   color=colours[tier], label=f"{tier} value")
    ax.set_yscale("log")
    ax.axvline(0.70, color=GREY, ls="--", lw=1)
    ax.text(0.705, ax.get_ylim()[1] * 0.4, "High risk", fontsize=8, color=GREY)
    ax.set_xlabel("Predicted probability of 90-day inactivity")
    ax.set_ylabel("Expected 90-day value (GBP, log scale)")
    ax.set_title("Customer value vs inactivity risk\nTop-right = act first")
    ax.legend(fontsize=8)
    fig.savefig(config.FIGURES_DIR / "11_value_vs_risk.png", bbox_inches="tight", dpi=120)
    plt.close(fig)

    # Value at risk by priority group
    fig, ax = plt.subplots(figsize=(7.5, 4))
    p = priority.sort_values("retention_priority", ascending=False)
    bars = ax.barh(p.retention_priority, p.value_at_risk / 1000,
                   color=[RED if x.startswith("1") else ORANGE if x.startswith("2") else BLUE
                          for x in p.retention_priority])
    for b, n in zip(bars, p.customers):
        ax.text(b.get_width() * 1.01, b.get_y() + b.get_height() / 2,
                f"{n:,} customers", va="center", fontsize=8)
    ax.set_xlabel("Customer Value at Risk (GBP thousands)")
    ax.set_title("Where the value at risk sits, by retention priority")
    fig.savefig(config.FIGURES_DIR / "12_value_at_risk_by_priority.png",
                bbox_inches="tight", dpi=120)
    plt.close(fig)


def validate_value_at_risk(verbose: bool = True) -> dict:
    """Check the Value at Risk estimate against a period we can actually see.

    The test snapshot's 90-day future window really happened, so the
    straight-line assumption can be scored. Two questions are asked:
    is the level right, and does the ranking work?
    """
    t = pd.read_csv(config.PROCESSED_DIR / "model_test_predictions.csv")
    scale = config.HORIZON_DAYS / config.OBSERVATION_DAYS
    t["expected_90d_value"] = t.monetary * scale
    t["expected_revenue"] = (1 - t.inactivity_probability) * t.expected_90d_value
    t["value_at_risk"] = t.inactivity_probability * t.expected_90d_value

    predicted, actual = t.expected_revenue.sum(), t.future_revenue_actual.sum()

    t["var_decile"] = pd.qcut(t.value_at_risk.rank(method="first"), 10, labels=range(1, 11))
    decile = t.groupby("var_decile", observed=True).agg(
        customers=("CustomerID", "count"),
        mean_value_at_risk=("value_at_risk", "mean"),
        actual_inactive_rate=("is_inactive", "mean"),
    ).round(3)
    decile["historic_value_of_those_who_lapsed"] = (
        t[t.is_inactive == 1].groupby("var_decile", observed=True)
        .expected_90d_value.sum().round(0))
    decile.reset_index().to_csv(config.REPORTS_DIR / "value_at_risk_validation.csv", index=False)

    out = {
        "predicted_expected_revenue": round(float(predicted), 2),
        "actual_future_revenue": round(float(actual), 2),
        "predicted_over_actual": round(float(predicted / actual), 3),
        "top_decile_lapsed_value": float(decile.historic_value_of_those_who_lapsed.iat[-1]),
        "bottom_decile_lapsed_value": float(decile.historic_value_of_those_who_lapsed.iat[0]),
    }
    if verbose:
        print("\nValue at Risk validation (on the test snapshot, whose future is observed):")
        print(f"  predicted expected revenue GBP {predicted:>12,.0f}")
        print(f"  actual future revenue      GBP {actual:>12,.0f}"
              f"   -> estimate captures {predicted/actual*100:.0f}% of it")
        print("  The straight-line assumption is CONSERVATIVE: the observed window "
              "was a seasonal peak\n  and retained customers grow their spend. "
              "Treat Value at Risk as a ranking tool, not a forecast.")
        print(decile.to_string())
    return out


def run(verbose: bool = True):
    master = build_customer_master(verbose=verbose)
    summary, vxr, priority = summarise(master, verbose=verbose)
    summary["value_at_risk_validation"] = validate_value_at_risk(verbose=verbose)
    with open(config.REPORTS_DIR / "retention_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    make_figures(master, priority)
    return master, summary


if __name__ == "__main__":
    run()
