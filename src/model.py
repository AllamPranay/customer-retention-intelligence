"""Phase 6 (part 2): predict customer inactivity.

Two models are compared on a strictly out-of-time test set:

    Logistic Regression - transparent, coefficients explain direction
    Random Forest       - captures non-linear interactions

Validation design
-----------------
The models are trained on the snapshot cut at 2011-06-09 and evaluated on
the snapshot cut at 2011-09-09. The test snapshot's target period
(Sep-Dec 2011) lies entirely after the training snapshot's target period,
so this is genuine out-of-time validation, not a random split. A random
split would let the model learn from the same weeks it is scored on and
would flatter the results.

Threshold and targeting
-----------------------
Classification metrics are reported at the neutral 0.50 cut-off. Tuning a
single threshold for recall was tried and rejected: because the training
snapshot is almost exactly balanced, recall-weighted scores such as F2 are
maximised by flagging nearly every customer, which is not a campaign a
retention team can run. The precision/recall trade-off is instead reported
as a table (reports/threshold_tradeoff.csv), and the real targeting
decision is made in Phase 8 by crossing risk with customer value.

Because a retention team can only contact a limited number of people, the
practically useful measure is lift at capacity: if we contact the top N
riskiest customers, what share of the customers who actually leave do we
catch, and how much better is that than contacting N people at random?
That analysis is in reports/campaign_lift.csv.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import config

BLUE, ORANGE, GREY = "#2E5E8C", "#D97A28", "#8C8C8C"

# Neutral operating point for reported classification metrics. Actual campaign
# targeting is done by the Phase 8 value x risk matrix, not by this cut-off.
THRESHOLD = 0.50

# Heavy right-tailed money/volume features - log-compressed before scaling so
# that a handful of wholesale-sized customers do not dominate the linear model.
SKEWED = ["monetary", "avg_order_value", "max_order_value", "total_quantity",
          "unique_products", "avg_basket_size", "frequency", "cancel_orders"]


def load_snapshot(name: str) -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED_DIR / f"features_{name}.csv")


def build_models() -> dict:
    """The two candidate models, each as a self-contained pipeline."""
    log_cols = [c for c in SKEWED if c in config.FEATURE_COLUMNS]
    passthrough = [c for c in config.FEATURE_COLUMNS if c not in log_cols]

    pre = ColumnTransformer([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), log_cols),
        ("raw", "passthrough", passthrough),
    ])

    logreg = Pipeline([
        ("pre", pre),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=config.RANDOM_STATE)),
    ])

    rf = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1,
            random_state=config.RANDOM_STATE)),
    ])
    return {"Logistic Regression": logreg, "Random Forest": rf}


def evaluate(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def threshold_tradeoff(y_true, proba) -> pd.DataFrame:
    """Precision / recall at a range of cut-offs, so the business can see
    what a stricter or looser campaign list would cost."""
    rows = []
    for t in np.arange(0.30, 0.81, 0.05):
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "customers_flagged": int(pred.sum()),
            "pct_of_base_flagged": round(float(pred.mean()) * 100, 1),
            "precision": round(precision_score(y_true, pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, pred, zero_division=0), 3),
        })
    return pd.DataFrame(rows)


def campaign_lift(y_true, proba, capacities=(250, 500, 750, 1000, 1500)) -> pd.DataFrame:
    """If the team contacts the top-N riskiest customers, how many of the
    customers who actually went inactive are captured, and how much better
    is that than contacting N customers at random?"""
    order = np.argsort(-proba)
    y = np.asarray(y_true)[order]
    base_rate = float(np.mean(y_true))
    total_inactive = int(np.sum(y_true))
    rows = []
    for n in capacities:
        n = min(n, len(y))
        caught = int(y[:n].sum())
        rows.append({
            "customers_contacted": n,
            "pct_of_base_contacted": round(n / len(y) * 100, 1),
            "inactive_customers_caught": caught,
            "recall": round(caught / total_inactive, 3),
            "precision": round(caught / n, 3),
            "lift_vs_random": round((caught / n) / base_rate, 2),
        })
    return pd.DataFrame(rows)


def run(verbose: bool = True) -> dict:
    train = load_snapshot("train")
    test = load_snapshot("test")
    score = load_snapshot("score")

    X_tr, y_tr = train[config.FEATURE_COLUMNS], train["is_inactive"]
    X_te, y_te = test[config.FEATURE_COLUMNS], test["is_inactive"]

    if verbose:
        print(f"Train: {len(X_tr):,} customers, {y_tr.mean()*100:.1f}% inactive "
              f"(cut-off {train.snapshot_cutoff.iat[0]})")
        print(f"Test : {len(X_te):,} customers, {y_te.mean()*100:.1f}% inactive "
              f"(cut-off {test.snapshot_cutoff.iat[0]})  <- out of time\n")

    # Cross-validated score on the training snapshot, to show how much of the
    # gap between train and test is overfitting rather than period change.
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)

    results, fitted = {}, {}
    for name, model in build_models().items():
        model.fit(X_tr, y_tr)
        fitted[name] = model
        proba_te = model.predict_proba(X_te)[:, 1]
        cv_proba = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        results[name] = {
            "test_out_of_time": evaluate(y_te, proba_te, THRESHOLD),
            "train_roc_auc": round(roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1]), 4),
            "train_cv_roc_auc": round(roc_auc_score(y_tr, cv_proba), 4),
        }
        if verbose:
            d = results[name]["test_out_of_time"]
            print(f"{name}")
            print(f"  ROC-AUC  train {results[name]['train_roc_auc']:.4f} | "
                  f"train 5-fold CV {results[name]['train_cv_roc_auc']:.4f} | "
                  f"TEST out-of-time {d['roc_auc']:.4f}")
            print(f"  @{THRESHOLD:.2f}  precision {d['precision']:.3f}  recall {d['recall']:.3f}  "
                  f"F1 {d['f1']:.3f}  accuracy {d['accuracy']:.3f}\n")

    # ---------------- choose the production model ------------------------
    best_name = max(results, key=lambda n: results[n]["test_out_of_time"]["roc_auc"])
    best_model = fitted[best_name]
    best_thr = THRESHOLD
    if verbose:
        print(f"Selected model: {best_name} (highest out-of-time ROC-AUC)")

    # ---------------- ROC curves -----------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for (name, model), colour in zip(fitted.items(), [BLUE, ORANGE]):
        p = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, color=colour, label=f"{name} (AUC {roc_auc_score(y_te, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-time ROC - predicting 90-day inactivity")
    ax.legend(loc="lower right")
    fig.savefig(config.FIGURES_DIR / "08_roc_curve.png", bbox_inches="tight", dpi=120)
    plt.close(fig)

    # ---------------- confusion matrix of the chosen model ---------------
    proba_best = best_model.predict_proba(X_te)[:, 1]
    cm = confusion_matrix(y_te, (proba_best >= best_thr).astype(int))
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted\nactive", "Predicted\ninactive"])
    ax.set_yticks([0, 1], ["Actually\nactive", "Actually\ninactive"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.grid(False)
    ax.set_title(f"{best_name} @ threshold {best_thr:.2f}")
    fig.savefig(config.FIGURES_DIR / "09_confusion_matrix.png", bbox_inches="tight", dpi=120)
    plt.close(fig)

    # ---------------- trade-off and campaign capacity --------------------
    tradeoff = threshold_tradeoff(y_te, proba_best)
    tradeoff.to_csv(config.REPORTS_DIR / "threshold_tradeoff.csv", index=False)
    lift = campaign_lift(y_te, proba_best)
    lift.to_csv(config.REPORTS_DIR / "campaign_lift.csv", index=False)
    if verbose:
        print("Precision / recall trade-off (out-of-time test set):")
        print(tradeoff.to_string(index=False))
        print("\nCampaign capacity - contacting the top-N riskiest customers:")
        print(lift.to_string(index=False))

    # ---------------- what drives the prediction -------------------------
    perm = permutation_importance(best_model, X_te, y_te, n_repeats=10,
                                  random_state=config.RANDOM_STATE,
                                  scoring="roc_auc", n_jobs=-1)
    imp = (pd.DataFrame({"feature": config.FEATURE_COLUMNS,
                         "importance": perm.importances_mean.round(5)})
           .sort_values("importance", ascending=False).reset_index(drop=True))

    # Direction of effect, read off the logistic regression coefficients
    lr = fitted["Logistic Regression"]
    coefs = pd.DataFrame({
        "feature": lr.named_steps["pre"].get_feature_names_out(),
        "coefficient": lr.named_steps["clf"].coef_[0].round(4),
    })
    coefs["feature"] = coefs.feature.str.replace(r"^(log|raw)__", "", regex=True)
    coefs["pushes_towards"] = np.where(coefs.coefficient > 0, "inactivity", "staying active")
    imp = imp.merge(coefs, on="feature", how="left")
    imp.to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    top = imp.head(10).sort_values("importance")
    ax.barh(top.feature, top.importance,
            color=[ORANGE if c > 0 else BLUE for c in top.coefficient.fillna(0)])
    ax.set_xlabel("Permutation importance (drop in ROC-AUC)")
    ax.set_title("What drives inactivity risk\n(orange = raises risk, blue = lowers risk)")
    fig.savefig(config.FIGURES_DIR / "10_feature_importance.png", bbox_inches="tight", dpi=120)
    plt.close(fig)

    if verbose:
        print("\nTop drivers of inactivity risk:")
        print(imp.head(8).to_string(index=False))

    # ---------------- score the live snapshot ----------------------------
    score = score.copy()
    score["inactivity_probability"] = best_model.predict_proba(
        score[config.FEATURE_COLUMNS])[:, 1].round(4)
    score["predicted_inactive"] = (score.inactivity_probability >= best_thr).astype(int)
    score["risk_band_absolute"] = pd.cut(
        score.inactivity_probability,
        bins=[b[1] for b in config.RISK_BANDS] + [config.RISK_BANDS[-1][2]],
        labels=[b[0] for b in config.RISK_BANDS], include_lowest=True)
    score.to_csv(config.PROCESSED_DIR / "customer_risk_scores.csv", index=False)

    # Test-set predictions kept so the dashboard can show real model performance
    test_out = test.copy()
    test_out["inactivity_probability"] = proba_best.round(4)
    test_out["predicted_inactive"] = (proba_best >= best_thr).astype(int)
    test_out.to_csv(config.PROCESSED_DIR / "model_test_predictions.csv", index=False)

    metrics = {
        "selected_model": best_name,
        "operating_threshold": best_thr,
        "validation": "out-of-time (train cut-off 2011-06-09, test cut-off 2011-09-09)",
        "observation_days": config.OBSERVATION_DAYS,
        "horizon_days": config.HORIZON_DAYS,
        "train_customers": int(len(X_tr)), "test_customers": int(len(X_te)),
        "train_inactive_rate": round(float(y_tr.mean()), 4),
        "test_inactive_rate": round(float(y_te.mean()), 4),
        "results": results,
        "top_features": imp.head(10).to_dict("records"),
        "threshold_tradeoff": tradeoff.to_dict("records"),
        "campaign_lift": lift.to_dict("records"),
        "scored_customers": int(len(score)),
        "scored_high_risk_absolute": int((score.risk_band_absolute == "High").sum()),
    }
    with open(config.REPORTS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump({"model": best_model, "threshold": best_thr,
                 "features": config.FEATURE_COLUMNS},
                config.PROCESSED_DIR / "inactivity_model.joblib")

    if verbose:
        print(f"\nScored {len(score):,} live customers | "
              f"absolute risk bands: {score.risk_band_absolute.value_counts().to_dict()}")
    return metrics


if __name__ == "__main__":
    run()
