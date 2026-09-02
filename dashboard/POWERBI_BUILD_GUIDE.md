# Power BI Dashboard — Build Guide

The Python pipeline exports every table this report needs as CSV into this
folder. Building the `.pbix` is a manual step in Power BI Desktop; this guide
gives the exact load order, relationships, measures and page layouts.

Re-run `python src/run_pipeline.py` to refresh the CSVs, then hit **Refresh** in
Power BI.

---

## 1. Load the data

**Home → Get Data → Text/CSV**, and load all nine files from `dashboard/`:

| File | Rows | Role |
|---|---|---|
| `customer_master.csv` | 4,334 | Fact table — one row per customer |
| `kpi_summary.csv` | 1 | Executive KPI card values |
| `monthly_trend.csv` | 13 | Revenue and customer trend |
| `segment_summary.csv` | 7 | RFM segment page |
| `priority_summary.csv` | 7 | Retention strategy page |
| `value_risk_matrix.csv` | 9 | Value x risk grid |
| `model_performance.csv` | 2 | Model comparison |
| `campaign_lift.csv` | 5 | Campaign capacity analysis |
| `feature_drivers.csv` | 12 | Model driver chart |

Check the column types on import: `customer_id` should be **Text** (it is an
identifier, not a number to sum), and `month` in `monthly_trend` should be
**Date**.

### Relationships

The summary tables are pre-aggregated, so most pages need no joins. Create one
relationship only:

```
segment_summary[segment]  1 ──── *  customer_master[segment]
```

Leave the others unrelated and drive each visual from its own table. This keeps
the model simple and avoids ambiguous filter paths.

---

## 2. Measures (DAX)

Create these in `customer_master` unless noted.

```dax
Total Customers      = DISTINCTCOUNT(customer_master[customer_id])

Total Revenue        = SUM(customer_master[lifetime_revenue])

Customer Value at Risk = SUM(customer_master[value_at_risk])

Scored Customers     =
    CALCULATE([Total Customers], customer_master[risk_band] <> "Already Dormant")

High Risk Customers  =
    CALCULATE([Total Customers], customer_master[risk_band] = "High")

Critical Customers   =
    CALCULATE([Total Customers], customer_master[retention_priority] = "1 - Critical")

Critical Value at Risk =
    CALCULATE([Customer Value at Risk],
              customer_master[retention_priority] = "1 - Critical")

Avg Inactivity Risk  = AVERAGE(customer_master[inactivity_probability])

Repeat Customers     =
    CALCULATE([Total Customers], customer_master[lifetime_orders] >= 2)

Repeat Purchase Rate = DIVIDE([Repeat Customers], [Total Customers])

Top 2 Priority Value at Risk =
    CALCULATE([Customer Value at Risk],
              customer_master[retention_priority] IN {"1 - Critical", "2 - High"})

Pct of Value at Risk =
    DIVIDE([Customer Value at Risk],
           CALCULATE([Customer Value at Risk], ALL(customer_master)))
```

In `monthly_trend`:

```dax
Monthly Revenue = SUM(monthly_trend[revenue])
Active Customers = SUM(monthly_trend[customers])
```

### Formatting

* `Customer Value at Risk`, `Total Revenue` → Currency, GBP, 0 decimals
* `Repeat Purchase Rate`, `Avg Inactivity Risk`, `Pct of Value at Risk` → Percentage, 1 decimal

### Conditional colours

Use one consistent palette across all four pages:

| Meaning | Hex |
|---|---|
| Primary / low risk | `#2E5E8C` |
| Warning / medium | `#D97A28` |
| Critical / high risk | `#B03A2E` |
| Neutral | `#8C8C8C` |

---

## 3. Page 1 — Executive Overview

> **What is happening overall?**

**KPI cards** (top row, six across) — from `kpi_summary`:

| Card | Value |
|---|---|
| Total Customers | 4,334 |
| Total Revenue | GBP 8.83m |
| Repeat Customers | 2,838 |
| Repeat Purchase Rate | 65.5% |
| High-Risk Customers | 695 |
| Customer Value at Risk | GBP 481.9k |

**Charts:**

1. **Line chart** — Revenue trend. Axis `monthly_trend[month]`, values
   `[Monthly Revenue]`.
   *Add a text note: December 2011 is partial — data ends 9 December.*
2. **Column chart** — Active customers per month. Axis `month`, values
   `[Active Customers]`.
3. **Donut** — Repeat vs one-time customers, from `customer_master`, legend on a
   calculated column:
   ```dax
   Customer Type = IF(customer_master[lifetime_orders] >= 2, "Repeat", "One-time")
   ```
4. **Stacked bar** — Customer risk distribution. Axis `customer_master[risk_band]`,
   values `[Total Customers]`.

**Slicers:** `country`, `segment`.

---

## 4. Page 2 — Customer Segmentation

> **Who are our customers?**

1. **Bar chart** — Customers by segment. From `segment_summary`: axis `segment`,
   values `customers`. Sort descending.
2. **Bar chart** — Revenue by segment. Values `total_revenue`.
3. **Scatter** — `pct_customers` (X) vs `pct_revenue` (Y), one point per segment,
   labelled. Points above the diagonal punch above their weight.
   *Only Champions do.*
4. **Matrix** — Rows `R`, columns `F` from `customer_master`, values
   `[Total Customers]`, with a background colour scale. This is the RFM grid.
5. **Bar chart** — Customer value distribution: axis `customer_value_tier`,
   values `[Total Customers]`.
6. **Map or bar** — Revenue by country. A bar chart on a log scale reads better
   than a map here, because the UK dwarfs everything else (82% of revenue).

**Key takeaway to caption:** Champions are 25.8% of customers but 66% of revenue.

---

## 5. Page 3 — Inactivity Intelligence

> **Who is becoming inactive, and why?**

1. **Histogram** — Distribution of `inactivity_probability`. Bin into 20 buckets.
2. **Card row** — from `model_performance` (filter `selected = TRUE`):
   ROC-AUC 0.730 · Precision 0.538 · Recall 0.744 · F1 0.625.
3. **Clustered bar** — Model comparison: axis `model`, values `roc_auc_train`,
   `roc_auc_train_cv`, `roc_auc_test`.
   *This visual tells the overfitting story — Random Forest scores 0.824 on
   training data but only 0.727 out of time.*
4. **Bar chart** — Average inactivity risk by RFM segment, from
   `segment_summary[avg_inactivity_probability]`.
5. **Bar chart** — Model drivers, from `feature_drivers`: axis `feature`, values
   `importance`. Colour by `pushes_towards` (orange = raises risk, blue = lowers).
6. **Table** — `campaign_lift`, showing that contacting the top 500 riskiest
   customers is 1.62x better than random.

**Caption the honest limit:** an AUC of 0.73 ranks customers well but is not a
crystal ball.

---

## 6. Page 4 — Retention Strategy

> **Where should the company spend its retention effort?**

1. **Scatter** — the core visual. X = `inactivity_probability`,
   Y = `expected_90d_value` (log scale), legend = `customer_value_tier`,
   size = `value_at_risk`, details = `customer_id`.
   Top-right of this chart is where to act first.
2. **Matrix** — Value x risk grid, from `value_risk_matrix`: rows
   `customer_value_tier`, columns `risk_band`, values `customers`, then a second
   copy with `value_at_risk`.
3. **Bar chart** — Value at risk by `retention_priority`, from
   `priority_summary`, coloured red → blue by priority.
4. **Table — the call list.** From `customer_master`, filtered to
   `retention_priority = "1 - Critical"`, sorted by `value_at_risk` descending.
   Columns: `customer_id`, `country`, `segment`, `inactivity_probability`,
   `value_at_risk`, `lifetime_revenue`, `lifetime_recency_days`,
   `recommended_action`.
5. **Table** — `priority_summary` showing the recommended action per group.

**Headline to caption:** 650 customers (15% of the base) carry GBP 189k of the
total GBP 482k exposure.

**Add a text box, verbatim:**

> Customer Value at Risk is an analytical prioritisation estimate, not a
> forecast of guaranteed revenue loss. It combines predicted inactivity
> probability with recent spending rate to rank where value is most exposed.

---

## 7. Design notes

* Four pages maximum — resist adding more.
* Every page answers one question; put that question in the page title.
* Keep slicers in the same position on every page.
* Use the same colour for the same meaning throughout.
* Label currency as **GBP** everywhere — this is UK data.

---

## 8. Saving

Save the report as `dashboard/customer_retention_intelligence.pbix` in this
folder. The CSVs use relative paths, so if you move the project you will need to
update the data source paths under **Transform data → Data source settings**.
