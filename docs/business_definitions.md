# Business Definitions

Single source of truth for every term used in this project. Written so that a
non-technical stakeholder can check what a number means.

---

## Customer Inactivity (the target)

This dataset is **not subscription data**. Customers do not sign up or cancel,
so no customer ever "churns" in the contractual sense. Claiming otherwise would
be wrong.

Instead the project uses a time-based definition:

> A customer is **inactive** if they place no order at all during the 90 days
> following a chosen cut-off date.

| Label | Meaning |
|---|---|
| `is_inactive = 1` | No purchase in the 90-day future window |
| `is_inactive = 0` | At least one purchase in the 90-day future window |

### Why 90 days

The median gap between consecutive orders is **21 days**, and roughly 90% of
repeat orders arrive within about 100 days. A customer silent for 90 days has
gone more than four times their normal rhythm without buying. The threshold
comes from the data, not from convention.

---

## Observation and future windows

```
        <-- 180 days -->  |  <-- 90 days -->
        observation window |  future window
        features built     |  target measured
        from here ONLY     |  from here ONLY
                      cut-off
```

* **Observation window (180 days)** - all model features are computed here.
* **Future window (90 days)** - only the target is measured here.

Nothing measured after the cut-off is ever used as an input. This is what makes
the dataset free of look-ahead leakage.

### Why the observation window is a fixed length

Each snapshot uses exactly 180 days of history. If the window instead ran from
the start of the data to the cut-off, later snapshots would cover more time, and
volume features (frequency, monetary, unique products) would rise purely because
more months were included. The model would then fail to transfer between
periods. A fixed window keeps features comparable across snapshots.

---

## The three snapshots

| Snapshot | Observation window | Future window | Customers | Inactive rate |
|---|---|---|---|---|
| **train** | 2010-12-11 → 2011-06-08 | 2011-06-09 → 2011-09-06 | 2,625 | 50.1% |
| **test** | 2011-03-13 → 2011-09-08 | 2011-09-09 → 2011-12-07 | 2,775 | 38.5% |
| **score** | 2011-06-13 → 2011-12-09 | *not yet observed* | 3,471 | predicted |

The test snapshot's future window falls entirely **after** the training
snapshot's future window, which makes the evaluation genuinely out-of-time
rather than a random split.

The inactive rate falls from 50.1% to 38.5% between the two snapshots. This is a
real seasonal effect: the test window covers the Christmas peak, when more
customers return. It means the model's absolute probabilities are calibrated to
a more pessimistic period than the one it is scored on, and ranking metrics
(ROC-AUC, lift) are more trustworthy here than raw probability levels.

---

## RFM

| Term | Definition |
|---|---|
| **Recency** | Days between the customer's last purchase and the reference date. Lower is better. |
| **Frequency** | Count of distinct invoices (orders), not line items. |
| **Monetary** | Sum of `quantity x unit_price` across all their lines. |

Each is scored 1-5 by quintile of the actual customer distribution. Recency is
reversed so a recent buyer scores 5. Ranking breaks ties before cutting,
otherwise the many single-order customers collapse the quintile edges.

### Segment rules

Applied in order; first match wins. Based on the Recency x Frequency grid only.

| # | Segment | Rule |
|---|---|---|
| 1 | Champions | R >= 4 and F >= 4 |
| 2 | Loyal Customers | R >= 3 and F >= 3 |
| 3 | Potential Loyalist | R >= 4 and F = 2 |
| 4 | New Customers | R >= 4 and F = 1 |
| 5 | At Risk | R <= 2 and F >= 3 |
| 6 | Lost Customers | R = 1 and F <= 2 |
| 7 | Hibernating | everything remaining (R <= 3 and F <= 2) |

**Monetary is deliberately excluded from segmentation.** It drives the separate
Customer Value Score below, so that "how engaged is this customer" and "how much
is this customer worth" remain two independent axes. That independence is what
makes the priority matrix informative.

---

## Customer Value Score (Phase 7)

A transparent weighted score, not a probabilistic lifetime-value model. The
business needs something it can explain to a marketing team.

```
value score = 0.50 x quintile(total spend)
            + 0.30 x quintile(order count)
            + 0.20 x quintile(average order value)
```

Each quintile is scored 1-5, so the value score ranges from 1.0 to 5.0.

| Tier | Score |
|---|---|
| High | >= 4.0 |
| Medium | 2.5 - 4.0 |
| Low | < 2.5 |

Spend carries the most weight because it already blends how often and how much
someone buys; order count rewards habit; average order value separates
big-basket buyers.

---

## Risk bands

Two are stored, and they mean different things.

**`risk_band_absolute`** - fixed cuts on the predicted probability:
Low < 0.40, Medium 0.40-0.70, High >= 0.70. This is the raw model reading.

**`risk_band`** - the customer's risk *percentile within their own value tier*:
riskiest 20% = High, next 30% = Medium, safest 50% = Low. **This is the one used
for prioritisation.**

### Why a peer-relative band is necessary

Customer value and inactivity risk correlate at **-0.87**, because both are built
from how often and how much a customer buys. Using the absolute band in the
priority matrix produced:

* **zero** high-value customers in the high-risk band, and
* 564 of 582 high-risk customers in the *lowest* value tier.

The resulting campaign list would have spent the entire retention budget on the
least valuable customers, and the "Critical" group would have been empty. Banding
against peers of similar value fixes this: a high-value customer at p = 0.55 is
genuinely alarming even though 0.55 is unremarkable across the whole base.

---

## Retention Priority (Phase 8)

| Customer Value | Risk band | Priority | Recommended action |
|---|---|---|---|
| High | High | **1 - Critical** | Personal contact from account manager plus a tailored offer |
| High | Medium | 2 - High | Targeted offer on their usual categories |
| Medium | High | 2 - High | Targeted win-back offer with a deadline |
| Medium | Medium | 3 - Monitor | Standard nurture campaign, re-score next month |
| Low | High | 3 - Low-Cost Retention | Automated email or discount code only |
| High | Low | 4 - Maintain | Loyalty rewards and early access |
| Medium | Low | 5 - Low Priority | Normal marketing |
| Low | Medium / Low | 5 - Low Priority | Normal marketing / newsletter |
| *(not scoreable)* | Already Dormant | 6 - Reactivation | Low-cost reactivation campaign |

**Priority 6 - Reactivation** covers 863 customers whose last purchase predates
the 180-day observation window. They cannot be given a model score, because there
is no recent behaviour to score. Inventing a probability for them would be
dishonest, so they are handled as a separate reactivation bucket.

---

## Customer Value at Risk (Phase 9)

```
Value at Risk = P(inactive) x (spend in the 180-day observation window / 2)
```

The division by two converts the customer's 180-day spending rate into a 90-day
rate, matching the prediction horizon.

### What this number is NOT

It is **not** a forecast of revenue the business will lose. It is an **exposure
estimate used for ranking**. A flagged customer may return anyway, with or
without intervention, and the model is right about 73% of the time by ROC-AUC,
not always.

### How it was checked

The test snapshot's future window actually happened, so the assumption can be
scored rather than asserted:

| Check | Result |
|---|---|
| Predicted expected revenue | GBP 1,415,515 |
| Actual revenue in that window | GBP 2,515,102 |
| Ratio | **0.56** |

The straight-line assumption is **conservative** - it captures 56% of what
actually materialised, because the observed window was a seasonal peak and
retained customers grow their spend.

A second check: the top Value-at-Risk decile has a *lower* inactivity rate
(0.309) than the bottom decile (0.385), because the measure is dominated by
value rather than probability. It still ranks correctly on what matters -
customers in the top decile who lapsed carried **GBP 110,031** of historic value,
versus GBP 5,110 in the bottom decile.

**Conclusion: a sound ranking tool, a poor forecasting tool, and presented as
such.**
