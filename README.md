# Customer Retention Intelligence

### Customer Segmentation, Inactivity Prediction & Retention Strategy

An end-to-end analytics project on real transaction data from a UK-based online
retailer. It segments customers by behaviour, predicts who is about to stop
buying, weighs that against how much each customer is worth, and produces a
ranked list of who the business should contact first.

**Stack:** Python · pandas · NumPy · scikit-learn · SQL / MySQL · Matplotlib · Power BI · Jupyter

---

## 1. Business Problem

An online retailer has thousands of customers but limited marketing budget.
Management wants to know:

1. Who are our most valuable customers?
2. How can we group customers by behaviour?
3. Which customers are going quiet?
4. Which customers are unlikely to buy again?
5. Who should we prioritise for retention?
6. What should we actually do for each group?
7. Where is the most customer value exposed?

All of which reduce to one question:

> **Which customers should the company prioritise for retention, and why?**

The project answers it as a chain:

**Understand → Segment → Predict inactivity → Measure value → Prioritise → Recommend action**

### Why retention, and not acquisition

The first analysis run settles this: **93% of revenue comes from customers who
order more than once**, and the top 20% of customers generate **74.6%** of
revenue. This business runs on people coming back, and losing one customer from
the top of that curve costs far more than losing one from the bottom.

---

## 2. Dataset

**UCI Online Retail** — <https://archive.ics.uci.edu/dataset/352/online+retail>

Real transactions from a **UK-based online gift retailer**, 2010-12-01 to
2011-12-09. 541,909 rows, one per invoice line.

> **Stated plainly: this is UK data, not Indian company data, and not synthetic.**
> Revenue is in GBP. The methodology — RFM segmentation, time-based inactivity
> modelling, value-weighted prioritisation — transfers to any e-commerce
> business, but every number in this repository describes this UK retailer.

### Was it suitable? A validation ran before any code was written

| Requirement | Finding | Verdict |
|---|---|---|
| Identifiable customers | 4,338 after basic filtering | Pass |
| Repeat purchasing to model | **65.6%** place 2+ orders | Pass |
| Enough time span | 373 days | Pass |
| A learnable, balanced target | ~43% do not return within 90 days | Pass |
| Missing customer IDs | 24.9% of rows (GBP 1.73m) | Documented, excluded |
| Cancellations | 1.7% of rows, 1,589 customers | Kept as a *feature* |

The dataset supports the objective, so **no substitute was needed.**

**Download:** the raw file is not committed. Fetch it into `data/raw/`:

```bash
curl -L -o data/raw/online_retail.zip \
  https://archive.ics.uci.edu/static/public/352/online+retail.zip
unzip data/raw/online_retail.zip -d data/raw/
```

---

## 3. Project Objective

Produce a **ranked, actionable retention list** — not just a model score — by
combining predicted inactivity risk with customer value, and to be honest about
what the numbers can and cannot support.

---

## 4. Data Cleaning

Every rule is applied as a logged step; the audit is in `reports/cleaning_log.csv`.

| Step | Rows removed | Reason |
|---|---|---|
| Exact duplicates | 5,268 | Same product, invoice, time and price repeated |
| Cancellations separated | 9,251 | Not sales — but kept, see below |
| Missing `CustomerID` | 134,658 | Cannot attribute to a customer |
| Quantity <= 0 | 0 | All were already removed above |
| Unit price <= 0 | 40 | Free items and entry errors |
| Administrative stock codes | 294 | Bank charges, fees, discounts, manual adjustments |

**Result: 392,398 rows (72.4% retained), 4,334 customers, GBP 8,833,624 revenue.**

### Decisions worth defending

* **Cancellations were kept, not deleted.** How often a customer cancels is
  behaviour, and it became a model feature. (The EDA then showed cancelling
  correlates with *more* orders and *higher* spend — it marks an active
  wholesale relationship, not a disgruntled customer. The opposite of the
  obvious guess.)
* **Missing customer IDs carry GBP 1.73m of real revenue.** That revenue exists;
  it simply cannot be assigned to anyone. It is excluded from customer-level
  analysis and reported rather than quietly dropped.
* **Postage codes (`POST`, `DOT`, `C2`) were kept** — they are genuine charges on
  real orders. Only true administrative codes were removed.
* **Negative-quantity rows that were not cancellations** (1,336) all had no
  `CustomerID`, and their descriptions (`damages`, `check`, `thrown away`)
  confirmed they were warehouse adjustments. They dropped out naturally.

---

## 5. Exploratory Analysis

Every chart answers a question that feeds the retention problem.
Figures in [`reports/figures/`](reports/figures/), numbers in `reports/eda_summary.json`.

| Finding | Number | Why it matters |
|---|---|---|
| Revenue from repeat customers | **93.0%** | Retention is the main revenue lever |
| Repeat purchase rate | 65.5% | There is real loyalty to protect |
| Revenue from top 20% of customers | 74.6% | Effort must be value-weighted |
| Median gap between orders | **21 days** | Sets the inactivity horizon |
| Customers already 90+ days silent | 33.3% (GBP 1.02m held) | A third of the base is at risk today |
| UK share of revenue | 82.2% | This is a UK business |
| Average order value | GBP 478.61 | Wholesale-sized baskets |

**The 21-day median gap is the key result.** It is the evidence behind the
90-day inactivity definition: a customer silent for 90 days has gone more than
four times their normal rhythm without buying. The threshold comes from the
data, not from convention.

---

## 6. SQL Analysis

`sql/01_schema.sql` creates a deliberately small MySQL schema — two tables:

* **`transactions`** — cleaned line-level sales
* **`customer_scores`** — the model output SQL cannot produce

RFM, segments, cohorts and repeat rates are **derived in SQL**, not loaded, so
the queries do the analytical work.

`sql/02_customer_analysis.sql` holds 14 business queries using `GROUP BY`,
`HAVING`, `CASE`, CTEs and window functions:

| Query | Question | Technique |
|---|---|---|
| Q1 | Size and shape of the data | Aggregates |
| Q2 | Month-on-month revenue growth | `LAG()` window |
| Q3 | Top products by revenue | `GROUP BY`, `LIMIT` |
| Q4 | Revenue share by country | `SUM() OVER ()` |
| Q5 | Repeat vs one-time buyers | CTE + `CASE` |
| Q6 | Revenue concentration (Pareto) | `RANK()`, running total |
| Q7 | RFM base table | `DATEDIFF`, aggregates |
| Q8 | **Full RFM scoring and segmentation** | `NTILE(5)` + `CASE` |
| Q9 | Segment summary | Nested CTEs + window |
| Q10 | Average gap between orders | `LAG()` per customer |
| Q11 | Acquisition cohorts and repeat rate | `HAVING` |
| Q12 | Who has gone quiet | `CASE` banding |
| Q13 | **Value at risk by priority** | Join to model output |
| Q14 | **The call list** | Filter + sort |

### These queries are tested

`src/test_sql.py` loads the same data into SQLite, applies a small dialect shim
for the two MySQL-only date functions, executes **all 14 queries**, and
cross-checks the results against pandas:

```
All 14 queries executed and all cross-checks passed.
  OK    total revenue        sql=8,833,623.96  pandas=8,833,623.96
  OK    customer count       sql=4,334         pandas=4,334
  OK    order count          sql=18,457        pandas=18,457
  OK    repeat customers     sql=2,838         pandas=2,838
  OK    total value at risk  sql=481,892       pandas=481,892
```

*(The SQL segment counts differ from the Python ones by 1–2 customers, because
`NTILE` and pandas `qcut` break ties on frequency differently. Immaterial, but
worth knowing rather than glossing over.)*

---

## 7. Customer Segmentation (RFM)

Recency, Frequency and Monetary, each scored 1–5 by quintile of the actual
customer distribution — so thresholds come from this business's data, not
arbitrary numbers. Recency is reversed: a recent buyer scores 5.

Segments are assigned from the **Recency × Frequency** grid. Monetary is held
back for the Customer Value Score (Phase 7), keeping *engagement* and *worth* as
two independent axes — which is what makes the priority matrix informative.

| Segment | Customers | % of base | Revenue | % of revenue | Avg recency |
|---|---|---|---|---|---|
| **Champions** | 1,119 | 25.8% | GBP 5.83m | **66.0%** | 12 days |
| Loyal Customers | 822 | 19.0% | GBP 1.36m | 15.4% | 36 days |
| **At Risk** | 659 | 15.2% | GBP 0.82m | 9.3% | **150 days** |
| Hibernating | 755 | 17.4% | GBP 0.37m | 4.1% | 86 days |
| Lost Customers | 668 | 15.4% | GBP 0.32m | 3.6% | 277 days |
| Potential Loyalist | 174 | 4.0% | GBP 0.09m | 1.0% | 17 days |
| New Customers | 137 | 3.2% | GBP 0.05m | 0.6% | 17 days |

**Champions are 26% of customers and 66% of revenue.** The 659 **At Risk**
customers used to buy often and have now been quiet for ~150 days on average,
holding GBP 817k of historic revenue — the clearest target for retention spend.

Full rules in [`docs/business_definitions.md`](docs/business_definitions.md).

### The limit of segmentation

RFM describes what has **already happened**. By the time recency flags someone,
they may already be gone — and it cannot tell a customer who is 60 days quiet
because that is their rhythm from one who is 60 days quiet because they left.
That gap is what the model fills.

---

## 8. Inactivity Prediction

### Defining the target without leaking the future

This is **not subscription data** — nobody cancels a contract, so no customer
"churns". Inactivity is defined using time instead:

```
[ 180-day observation window ]  |  [ 90-day future window ]
   features from here ONLY         target from here ONLY
                            cut-off
```

* `is_inactive = 1` — no order in the 90-day future window
* `is_inactive = 0` — the customer came back

### Three snapshots, and why the window length is fixed

| Snapshot | Observation | Future | Customers | Inactive |
|---|---|---|---|---|
| **train** | 2010-12-11 → 2011-06-08 | 2011-06-09 → 09-06 | 2,625 | 50.1% |
| **test** | 2011-03-13 → 2011-09-08 | 2011-09-09 → 12-07 | 2,775 | 38.5% |
| **score** | 2011-06-13 → 2011-12-09 | *not yet observed* | 3,471 | predicted |

The observation window is held at **exactly 180 days** at every cut-off. If it
instead ran from the start of the data, later snapshots would cover more time
and volume features (frequency, spend) would rise purely because more months
were included — and the model would not transfer between periods.

The test snapshot's future window falls **entirely after** the training
snapshot's, making this genuine **out-of-time validation**, not a random split.
A random split would let the model learn from the same weeks it is scored on.

Classes are near-balanced (50.1%), so no resampling was needed.

### 15 features, all from the observation window

RFM core · order values and basket size · product breadth · tenure and active
weeks · personal reorder rhythm · purchase momentum (recent vs earlier half) ·
cancellation behaviour · UK flag. Full definitions in the
[data dictionary](docs/data_dictionary.md).

### Results — Logistic Regression vs Random Forest

| Model | Train AUC | Train CV AUC | **Test AUC (out-of-time)** | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **Logistic Regression** *(selected)* | 0.768 | 0.762 | **0.730** | 0.538 | 0.744 | 0.625 |
| Random Forest | 0.824 | 0.756 | 0.727 | 0.538 | 0.755 | 0.628 |

**Logistic Regression was chosen**, and the reason is the most useful thing in
this table. The Random Forest looks clearly stronger on training data (0.824),
but its cross-validated score collapses to 0.756 and its out-of-time score to
0.727 — it was memorising, not learning. Selecting on training accuracy would
have picked the worse model. The simpler model also has readable coefficients
the business can interrogate.

**An AUC of 0.730 is a real but imperfect signal.** It ranks customers
substantially better than chance, which is enough to prioritise a campaign. It
is not a crystal ball and is not presented as one.

### Why accuracy is the wrong headline

At the 0.50 cut-off the model catches **74% of customers who genuinely go
inactive** (recall) at 54% precision — accuracy is only 65.6%. But missing a
valuable customer about to leave costs their future revenue, while wrongly
flagging someone costs an email. Confusion matrix (out-of-time, 2,775 customers):

|  | Predicted active | Predicted inactive |
|---|---|---|
| **Actually active** | 1,027 | 681 |
| **Actually inactive** | 273 (missed) | **794 (caught)** |

**Recall-weighted threshold tuning was tried and rejected.** Because the training
classes are balanced, F2 is maximised by flagging almost everybody — not a
campaign anyone can run. The full precision/recall trade-off is published
instead (`reports/threshold_tradeoff.csv`).

### The measure a marketing team actually needs

A retention team has limited capacity, so the real question is *if we contact N
customers, how many of the leavers do we catch?*

| Customers contacted | Leavers caught | Recall | Precision | **Lift vs random** |
|---|---|---|---|---|
| 250 | 161 | 15.1% | 64.4% | **1.67x** |
| 500 | 311 | 29.1% | 62.2% | **1.62x** |
| 1,000 | 581 | 54.5% | 58.1% | 1.51x |

Contacting the 500 riskiest customers finds 311 genuine leavers — **1.62x better
than contacting 500 at random.** That is the concrete value of the model.

### What drives inactivity

| Feature | Importance | Direction |
|---|---|---|
| `active_weeks` | 0.139 | Lowers risk |
| `avg_order_value` | 0.059 | Lowers risk |
| `max_order_value` | 0.034 | **Raises risk** |
| `avg_days_between_orders` | 0.019 | Raises risk |
| `monetary` | 0.008 | Lowers risk |

**`active_weeks` dominates: habit matters more than size.** Customers whose
buying is spread across many distinct weeks stay; customers who bought
everything in one burst leave.

The `avg_order_value` / `max_order_value` split looks contradictory but is not:
holding average order value fixed, a much larger *maximum* order marks a one-off
bulk purchase — someone who appeared, bought heavily once, and never returned.

---

## 9. Customer Value

Deliberately simple and transparent — no probabilistic lifetime-value model,
because a retention manager has to be able to explain it:

```
value score = 0.50 x quintile(total spend)
            + 0.30 x quintile(order count)
            + 0.20 x quintile(average order value)
```

| Tier | Score | Customers |
|---|---|---|
| High | >= 4.0 | 756 |
| Medium | 2.5 – 4.0 | 1,359 |
| Low | < 2.5 | 1,356 |

---

## 10. Retention Priority Strategy

### A problem worth explaining

Customer value and inactivity risk turn out to be correlated at **−0.87** —
unsurprising, since both are built from how often and how much someone buys.
Banding risk on the raw probability produced:

* **zero** high-value customers in the "high risk" band, and
* 564 of 582 high-risk customers in the **lowest** value tier.

The "Critical" group came out empty, and the campaign list would have spent the
entire budget on the least valuable customers.

**The fix:** band risk **relative to peers of similar value** — a customer is
high risk if they sit in the riskiest 20% *of their own value tier*. A
high-value customer at p = 0.55 is genuinely alarming even though 0.55 is
unremarkable across the whole base. Both readings are kept:
`risk_band_absolute` (raw model output) and `risk_band` (peer-relative, used for
prioritisation).

### The priority matrix

| Value \ Risk | Low | Medium | High |
|---|---|---|---|
| **High** | 4 - Maintain (378) | 2 - High (226) | **1 - Critical (152)** |
| **Medium** | 5 - Low Priority (679) | 3 - Monitor (408) | 2 - High (272) |
| **Low** | 5 - Low Priority (678) | 5 - Low Priority (407) | 3 - Low-Cost Retention (271) |

### The answer

| Priority | Customers | Value at Risk | Recommended action |
|---|---|---|---|
| **1 - Critical** | 152 | GBP 65,573 | Personal contact from account manager plus a tailored offer |
| **2 - High** | 498 | GBP 123,730 | Targeted offer / win-back with a deadline |
| 3 - Monitor | 408 | GBP 82,634 | Standard nurture campaign, re-score next month |
| 3 - Low-Cost Retention | 271 | GBP 12,719 | Automated email or discount code only |
| 4 - Maintain | 378 | GBP 30,242 | Loyalty rewards and early access |
| 5 - Low Priority | 1,764 | GBP 166,995 | Normal marketing |
| 6 - Reactivation | 863 | — | Already dormant beyond the scoring window |

> **650 customers — 15% of the base — carry GBP 189,303 of the total GBP 481,892
> exposure.** Start there, and handle the 152 Critical customers personally.

**Priority 6** covers 863 customers whose last purchase predates the 180-day
observation window. They have no recent behaviour to score, so inventing a
probability for them would be dishonest — they go to a separate reactivation
bucket.

---

## 11. Customer Value at Risk

```
Value at Risk = P(inactive) x (spend in the 180-day window / 2)
```

The division by two puts the customer's spending rate on the same 90-day footing
as the prediction horizon.

> **This is an analytical prioritisation estimate, not a forecast of guaranteed
> revenue loss.** A flagged customer may well return, with or without
> intervention.

### It was checked, not just asserted

The test snapshot's future actually happened, so the assumption can be scored:

| Check | Result |
|---|---|
| Predicted expected revenue | GBP 1,415,515 |
| Actual revenue in that window | GBP 2,515,102 |
| Ratio | **0.56** |

Two honest findings:

1. The straight-line estimate captures only **56%** of what actually
   materialised — it is **conservative**, because the observed window was the
   Christmas peak and retained customers grow their spend.
2. The top Value-at-Risk decile has a *lower* inactivity rate (0.309) than the
   bottom (0.385), because the measure is dominated by value rather than
   probability. It still ranks correctly on what matters: top-decile customers
   who lapsed carried **GBP 110,031** of historic value versus GBP 5,110 in the
   bottom decile.

**A sound ranking tool, a poor forecasting tool — and presented as such.**

---

## 12. Power BI Dashboard

Four pages, each answering one question. The pipeline exports nine ready-to-load
CSVs into `dashboard/`; the full build specification — load order, DAX measures,
visual-by-visual layout and colour palette — is in
[`dashboard/POWERBI_BUILD_GUIDE.md`](dashboard/POWERBI_BUILD_GUIDE.md).

| Page | Question | Contents |
|---|---|---|
| **1 · Executive Overview** | What is happening overall? | KPIs, revenue and customer trend, repeat behaviour, risk distribution |
| **2 · Customer Segmentation** | Who are our customers? | Segment sizes and revenue, RFM grid, value tiers, geography |
| **3 · Inactivity Intelligence** | Who is going quiet, and why? | Risk distribution, model comparison, drivers, campaign lift |
| **4 · Retention Strategy** | Where should effort go? | Value vs risk scatter, priority matrix, the call list, value at risk |

---

## 13. Key Insights

1. **93% of revenue comes from repeat customers** — retention, not acquisition,
   is this business's main lever.
2. **Champions are 26% of customers but 66% of revenue.** Revenue is
   concentrated; retention spending must be value-weighted.
3. **The median customer reorders every 21 days**, which is what makes a 90-day
   silence meaningful — over four times the normal rhythm.
4. **Habit beats size.** The strongest predictor of staying is `active_weeks` —
   how many distinct weeks a customer bought in — not how much they spent.
   Customers who buy in one big burst leave.
5. **Cancelling is not a warning sign.** Customers who cancel more also order
   more and spend more; it marks an active wholesale relationship.
6. **Random Forest overfitted and Logistic Regression won.** Train AUC 0.824 vs
   out-of-time 0.727 — choosing on training accuracy would have picked the worse
   model.
7. **Value and risk are −0.87 correlated**, so naive banding sends the whole
   budget to the least valuable customers. Peer-relative banding fixes it.
8. **650 customers (15%) hold 39% of the value at risk** — a genuinely
   actionable shortlist.
9. **A third of the customer base is already 90+ days silent**, holding GBP 1.02m
   of historic revenue.

---

## 14. Limitations

Stated plainly, because they affect how the results should be read.

1. **13 months of data.** No year-over-year comparison is possible, and seasonal
   effects cannot be separated from trend.
2. **The evaluation window is the Christmas peak.** The inactive rate drops from
   50.1% (training) to 38.5% (test) largely for this reason. Absolute
   probabilities are therefore calibrated to a more pessimistic period than the
   one they are scored on; **ranking metrics are more trustworthy here than
   probability levels.**
3. **24.9% of rows have no customer ID** — GBP 1.73m of real revenue that cannot
   be attributed to anyone and is excluded from all customer-level analysis.
4. **ROC-AUC 0.730 is a useful signal, not a precise prediction.** Roughly 26% of
   flagged customers at the 0.50 threshold would have returned anyway.
5. **Value at Risk is an exposure estimate, not a forecast** — and a
   conservative one, capturing 56% of the revenue that actually materialised.
6. **No marketing, web, demographic or cost data.** The model sees only
   transactions, so it cannot know whether a customer left because of price,
   service, or a competitor — and the recommended actions are not costed.
7. **No causal claim.** The model identifies who is *likely* to lapse, not
   whether contacting them will change anything. Only an A/B test can establish
   that.
8. **UK gift retail, 2010–2011.** The methodology generalises; these specific
   thresholds and segment sizes do not.
9. **863 customers cannot be scored** at all because their last purchase
   predates the observation window.

---

## 15. Future Improvements

1. **A/B test the recommended actions** — measure whether contacting the Critical
   group actually changes behaviour. This is the single most valuable next step.
2. **Roll the snapshots forward monthly** to build a larger training set and
   test stability across more periods.
3. **Add gradient boosting** as a third candidate, with proper hyperparameter
   search.
4. **Calibrate the probabilities** (Platt scaling or isotonic) so they can be
   read as genuine likelihoods rather than only as rankings.
5. **Replace the straight-line value estimate** with a BG/NBD + Gamma-Gamma
   model once more history is available.
6. **Bring in cost data** — the priority matrix currently ranks by value at risk,
   not by expected return on retention spend.
7. **Automate the refresh** so the score snapshot regenerates on a schedule.

---

## 16. Project Structure

```
customer-retention-intelligence/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                     # UCI workbook (download separately)
│   └── processed/               # cleaned data, features, scores
│
├── notebooks/
│   ├── 01_data_understanding.ipynb      # Phase 1 - is the data suitable?
│   ├── 02_data_cleaning_eda.ipynb       # Phases 2-3
│   ├── 03_customer_segmentation.ipynb   # Phase 5 - RFM
│   └── 04_inactivity_prediction.ipynb   # Phases 6-9
│
├── sql/
│   ├── 01_schema.sql                    # MySQL schema
│   └── 02_customer_analysis.sql         # 14 business queries
│
├── src/
│   ├── config.py                # paths, time design, constants
│   ├── data_processing.py       # Phases 1-2  load and clean
│   ├── eda.py                   # Phase 3     exploratory analysis
│   ├── feature_engineering.py   # Phase 6a    snapshot builder
│   ├── segmentation.py          # Phase 5     RFM
│   ├── model.py                 # Phase 6b    train, evaluate, score
│   ├── retention_strategy.py    # Phases 7-9  value, priority, value at risk
│   ├── export_dashboard.py      # Power BI tables
│   ├── load_to_mysql.py         # load into MySQL
│   ├── test_sql.py              # verifies all 14 SQL queries
│   └── run_pipeline.py          # runs everything
│
├── dashboard/                   # 9 CSVs + POWERBI_BUILD_GUIDE.md
├── reports/                     # metrics, summaries, 12 figures
└── docs/
    ├── business_definitions.md  # every term defined
    └── data_dictionary.md       # every column defined
```

---

## 17. How to Run

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download the dataset into data/raw/ (see section 2)

# 3. Run the whole pipeline (~35 seconds)
python src/run_pipeline.py

# 4. Verify the SQL queries (no MySQL server needed)
python src/test_sql.py

# 5. Optional - load into MySQL, then run the analysis queries
python src/load_to_mysql.py
mysql -u root -p retail_analytics < sql/02_customer_analysis.sql

# 6. Explore the notebooks
jupyter notebook notebooks/
```

MySQL connection settings live in `src/config.py` and can be overridden with
`MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`.

---

## 18. Summary

> Using 13 months of real UK online retail transactions, I cleaned 541,909
> records down to a reliable 392,398, segmented 4,334 customers with RFM
> analysis, and built a time-based model to predict which customers would stop
> purchasing in the following 90 days — validated out-of-time to avoid leakage.
> I combined predicted inactivity risk with a transparent customer value score
> to produce a retention priority framework, finding that 15% of customers carry
> 39% of the value at risk, and that contacting the 500 riskiest customers is
> 1.6x more effective than contacting 500 at random.
