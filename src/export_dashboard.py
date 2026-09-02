"""Build the flat tables that the Power BI report loads.

Power BI works best with a few wide, already-aggregated tables rather than
raw transactions, so each file here maps to one page of the report:

    customer_master.csv     one row per customer - the fact table
    kpi_summary.csv         single-row KPIs for the executive page
    monthly_trend.csv       revenue and customer trend
    segment_summary.csv     RFM segment page
    priority_summary.csv    retention strategy page
    model_performance.csv   model metrics for the inactivity page
    feature_drivers.csv     what drives the prediction
    value_risk_matrix.csv   the value x risk grid
"""
import json

import pandas as pd

import config


def run(verbose: bool = True):
    out = config.DASHBOARD_DIR
    master = pd.read_csv(config.PROCESSED_DIR / "customer_master.csv")
    eda = json.loads((config.REPORTS_DIR / "eda_summary.json").read_text())
    metrics = json.loads((config.REPORTS_DIR / "model_metrics.json").read_text())
    retention = json.loads((config.REPORTS_DIR / "retention_summary.json").read_text())

    # ---------------- 1. customer master (the fact table) ----------------
    cols = ["CustomerID", "country", "segment", "customer_value_tier",
            "customer_value_score", "risk_band", "risk_band_absolute",
            "inactivity_probability", "risk_percentile_in_tier",
            "retention_priority", "recommended_action", "expected_90d_value",
            "value_at_risk", "lifetime_revenue", "lifetime_orders",
            "lifetime_recency_days", "lifetime_tenure_days", "R", "F", "M",
            "rfm_score", "frequency", "monetary", "avg_order_value",
            "recency_days", "unique_products", "cancel_orders"]
    cm = master[[c for c in cols if c in master.columns]].copy()
    cm = cm.rename(columns={"CustomerID": "customer_id"})
    cm.to_csv(out / "customer_master.csv", index=False)

    # ---------------- 2. executive KPIs ----------------------------------
    scored = master[master.risk_band != "Already Dormant"]
    kpis = pd.DataFrame([{
        "total_customers": eda["total_customers"],
        "total_revenue": eda["total_revenue"],
        "total_orders": eda["total_orders"],
        "avg_order_value": eda["avg_order_value"],
        "repeat_customers": eda["repeat_customers"],
        "repeat_purchase_rate_pct": eda["repeat_rate_pct"],
        "revenue_from_repeat_pct": eda["revenue_from_repeat_pct"],
        "customers_scored": retention["customers_scored"],
        "high_risk_customers": retention["high_risk_customers"],
        "critical_customers": retention["critical_customers"],
        "customer_value_at_risk": retention["total_value_at_risk"],
        "critical_value_at_risk": retention["critical_value_at_risk"],
        "already_dormant_customers": retention["customers_already_dormant"],
        "model_roc_auc": metrics["results"][metrics["selected_model"]]["test_out_of_time"]["roc_auc"],
        "model_name": metrics["selected_model"],
        "data_start": eda["date_min"][:10],
        "data_end": eda["date_max"][:10],
    }])
    kpis.to_csv(out / "kpi_summary.csv", index=False)

    # ---------------- 3. monthly trend -----------------------------------
    monthly = pd.read_csv(config.PROCESSED_DIR / "monthly_summary.csv")
    monthly["avg_order_value"] = (monthly.revenue / monthly.orders).round(2)
    monthly["revenue_mom_pct"] = (monthly.revenue.pct_change() * 100).round(1)
    monthly.to_csv(out / "monthly_trend.csv", index=False)

    # ---------------- 4. segment summary ---------------------------------
    seg = pd.read_csv(config.REPORTS_DIR / "segment_profile.csv")
    seg_risk = pd.read_csv(config.REPORTS_DIR / "segment_risk_summary.csv")
    seg = seg.merge(seg_risk[["segment", "avg_inactivity_probability", "value_at_risk"]],
                    on="segment", how="left")
    seg.to_csv(out / "segment_summary.csv", index=False)

    # ---------------- 5. retention priority ------------------------------
    pd.read_csv(config.REPORTS_DIR / "retention_priority_summary.csv").to_csv(
        out / "priority_summary.csv", index=False)

    # ---------------- 6. model performance -------------------------------
    rows = []
    for name, res in metrics["results"].items():
        r = res["test_out_of_time"]
        rows.append({
            "model": name, "selected": name == metrics["selected_model"],
            "roc_auc_test": r["roc_auc"], "roc_auc_train": res["train_roc_auc"],
            "roc_auc_train_cv": res["train_cv_roc_auc"],
            "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
            "accuracy": r["accuracy"], "threshold": r["threshold"],
            "true_positives": r["true_positives"], "false_positives": r["false_positives"],
            "true_negatives": r["true_negatives"], "false_negatives": r["false_negatives"],
        })
    pd.DataFrame(rows).to_csv(out / "model_performance.csv", index=False)
    pd.DataFrame(metrics["campaign_lift"]).to_csv(out / "campaign_lift.csv", index=False)

    # ---------------- 7. feature drivers ---------------------------------
    pd.read_csv(config.REPORTS_DIR / "feature_importance.csv").head(12).to_csv(
        out / "feature_drivers.csv", index=False)

    # ---------------- 8. value x risk matrix -----------------------------
    matrix = (scored.groupby(["customer_value_tier", "risk_band"])
              .agg(customers=("CustomerID", "count"),
                   value_at_risk=("value_at_risk", "sum"),
                   avg_inactivity_probability=("inactivity_probability", "mean"))
              .round(3).reset_index())
    matrix["priority"] = [
        scored.loc[(scored.customer_value_tier == v) & (scored.risk_band == r),
                   "retention_priority"].iat[0]
        for v, r in zip(matrix.customer_value_tier, matrix.risk_band)]
    matrix.to_csv(out / "value_risk_matrix.csv", index=False)

    files = sorted(p.name for p in out.glob("*.csv"))
    if verbose:
        print(f"Dashboard tables written to {out.name}/:")
        for f in files:
            n = len(pd.read_csv(out / f))
            print(f"  {f:<28} {n:>6,} rows")
    return files


if __name__ == "__main__":
    run()
