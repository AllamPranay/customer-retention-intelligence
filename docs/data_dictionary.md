# Data Dictionary

---

## 1. Raw dataset — `data/raw/Online Retail.xlsx`

UCI Online Retail. 541,909 rows x 8 columns, one row per invoice line.
Transactions from 2010-12-01 to 2011-12-09 for a UK-based online gift retailer.

| Column | Type | Notes |
|---|---|---|
| `InvoiceNo` | text | Order number. A leading `C` marks a cancellation. |
| `StockCode` | text | Product code. Some are administrative (see below). |
| `Description` | text | Product name. 1,454 missing. |
| `Quantity` | integer | Negative on cancellations and stock adjustments. |
| `InvoiceDate` | datetime | Timestamp of the order. |
| `UnitPrice` | float | Price per unit, GBP. |
| `CustomerID` | float | **135,080 rows (24.9%) are missing this.** |
| `Country` | text | 38 values; overwhelmingly United Kingdom. |

### Administrative stock codes removed during cleaning

`M` (manual), `D` (discount), `S` (samples), `B` (bad debt adjustment),
`PADS`, `BANK CHARGES`, `AMAZONFEE`, `CRUK` (charity donation).

Postage codes `POST`, `DOT` and `C2` are **kept** — they are genuine charges on
real customer orders.

---

## 2. Cleaned transactions — `data/processed/transactions_clean.csv`

392,398 rows, 4,334 customers, GBP 8,833,624 revenue (72.4% of raw rows retained).

| Column | Type | Description |
|---|---|---|
| `InvoiceNo` | text | Order number |
| `StockCode` | text | Product code, upper-cased and trimmed |
| `Description` | text | Product name, upper-cased |
| `Quantity` | integer | Always > 0 |
| `InvoiceDate` | datetime | Order timestamp |
| `UnitPrice` | float | Always > 0 |
| `CustomerID` | integer | Never missing |
| `Country` | text | Customer country |
| `Revenue` | float | `Quantity x UnitPrice` |
| `InvoiceDay` | date | Date component, for daily grouping |

Cancellations are written separately to `cancellations_clean.csv` with the same
layout — they are excluded from sales totals but used to build cancellation
features.

The full record-loss audit is in `reports/cleaning_log.csv`.

---

## 3. Model features — `data/processed/features_{train,test,score}.csv`

One row per customer per snapshot. All 15 features are computed **strictly
inside the observation window**.

| Feature | Description |
|---|---|
| `recency_days` | Days from the customer's last order to the cut-off |
| `frequency` | Number of distinct orders in the window |
| `monetary` | Total spend in the window |
| `avg_order_value` | Mean order value |
| `max_order_value` | Largest single order |
| `total_quantity` | Total units bought |
| `unique_products` | Distinct stock codes bought |
| `tenure_days` | Days from first order in the window to the cut-off |
| `avg_days_between_orders` | Mean gap between consecutive orders; set to 180 (the window length) for single-order customers, which reads as "no repeat rhythm observed" |
| `purchase_trend_ratio` | `(orders in recent 90d + 1) / (orders in earlier 90d + 1)`. Above 1 = speeding up |
| `active_weeks` | Distinct calendar weeks with a purchase |
| `cancel_orders` | Distinct cancelled invoices in the window |
| `cancel_rate` | `cancel_orders / (frequency + cancel_orders)` |
| `avg_basket_size` | `total_quantity / frequency` |
| `is_uk` | 1 if the customer's modal country is the United Kingdom |

Supporting columns (not features): `CustomerID`, `country`, `snapshot_cutoff`,
`observation_start`.

**Target columns** (train and test snapshots only):

| Column | Description |
|---|---|
| `is_inactive` | 1 = no order in the 90-day future window |
| `future_revenue_actual` | Revenue in the future window. **Reporting only — never used as a feature.** |

---

## 4. Customer master — `data/processed/customer_master.csv` and `dashboard/customer_master.csv`

One row per customer (4,334 rows). This is the table Power BI loads.

| Column | Description |
|---|---|
| `customer_id` | Customer identifier |
| `country` | Country |
| `segment` | RFM segment (full purchase history) |
| `R`, `F`, `M` | RFM quintile scores, 1–5 |
| `rfm_score` | `R + F + M`, range 3–15 |
| `inactivity_probability` | Model output: P(no purchase in the next 90 days). Blank for dormant customers |
| `risk_band_absolute` | Fixed bands on the probability: Low / Medium / High |
| `risk_percentile_in_tier` | Risk percentile within the customer's value tier |
| `risk_band` | Peer-relative band — **used for prioritisation** |
| `customer_value_score` | Weighted value score, 1.0–5.0 |
| `customer_value_tier` | High / Medium / Low |
| `retention_priority` | Priority group 1–6 |
| `recommended_action` | Suggested retention action |
| `expected_90d_value` | Recent 180-day spend / 2 |
| `value_at_risk` | `inactivity_probability x expected_90d_value` |
| `lifetime_revenue` | Total spend across all history |
| `lifetime_orders` | Total orders across all history |
| `lifetime_recency_days` | Days since last purchase, as of 2011-12-09 |
| `lifetime_tenure_days` | Days since first purchase |
| `frequency`, `monetary`, `avg_order_value`, `recency_days`, `unique_products`, `cancel_orders` | Observation-window values from the score snapshot |

> **Note on the two revenue columns:** `lifetime_revenue` covers all 13 months;
> `monetary` covers only the most recent 180 days. They will not match, and the
> dashboard labels which is which.

---

## 5. MySQL tables

**`transactions`** — the cleaned line-level sales, loaded from
`transactions_clean.csv`. Indexed on `customer_id`, `invoice_no`,
`invoice_date`, `stock_code`.

**`customer_scores`** — the model output that SQL cannot produce, loaded from
`customer_master.csv`. Primary key `customer_id`.

RFM, segments, cohorts and repeat rates are **derived in SQL** rather than
loaded, so the queries in `sql/02_customer_analysis.sql` do the analytical work.

---

## 6. Reports — `reports/`

| File | Contents |
|---|---|
| `cleaning_log.csv` | Rows removed at each cleaning step |
| `eda_summary.json` | Every headline EDA number |
| `segment_profile.csv` | Customers and revenue per RFM segment |
| `model_metrics.json` | Full model comparison and metrics |
| `feature_importance.csv` | Permutation importance plus logistic coefficients |
| `threshold_tradeoff.csv` | Precision/recall at cut-offs from 0.30 to 0.80 |
| `campaign_lift.csv` | Recall and lift when contacting the top N riskiest |
| `retention_priority_summary.csv` | Customers and value at risk per priority group |
| `segment_risk_summary.csv` | Average risk and value at risk per segment |
| `value_at_risk_validation.csv` | Value-at-Risk calibration check by decile |
| `retention_summary.json` | Headline retention numbers |
| `figures/` | 12 PNG charts |
