-- ---------------------------------------------------------------------
-- Customer Retention Intelligence - business analysis in SQL (MySQL 8)
--
-- Each query answers one question a manager actually asked. Run against
-- the retail_analytics database created by 01_schema.sql.
--
-- Reference date: the dataset ends 2011-12-09, so "today" is that date.
-- ---------------------------------------------------------------------
USE retail_analytics;


-- =====================================================================
-- Q1. What are we looking at? Size and shape of the cleaned data.
-- =====================================================================
SELECT
    COUNT(*)                          AS line_items,
    COUNT(DISTINCT invoice_no)        AS orders,
    COUNT(DISTINCT customer_id)       AS customers,
    COUNT(DISTINCT stock_code)        AS products,
    COUNT(DISTINCT country)           AS countries,
    MIN(invoice_date)                 AS first_transaction,
    MAX(invoice_date)                 AS last_transaction,
    ROUND(SUM(revenue), 2)            AS total_revenue
FROM transactions;


-- =====================================================================
-- Q2. How is revenue trending month to month?
--     Window function LAG gives the month-on-month growth rate.
-- =====================================================================
WITH monthly AS (
    SELECT
        DATE_FORMAT(invoice_date, '%Y-%m')  AS month,
        SUM(revenue)                        AS revenue,
        COUNT(DISTINCT invoice_no)          AS orders,
        COUNT(DISTINCT customer_id)         AS active_customers
    FROM transactions
    GROUP BY DATE_FORMAT(invoice_date, '%Y-%m')
)
SELECT
    month,
    ROUND(revenue, 2)                                    AS revenue,
    orders,
    active_customers,
    ROUND(revenue / orders, 2)                           AS avg_order_value,
    ROUND(LAG(revenue) OVER (ORDER BY month), 2)         AS prev_month_revenue,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
          / LAG(revenue) OVER (ORDER BY month), 1)       AS mom_growth_pct
FROM monthly
ORDER BY month;


-- =====================================================================
-- Q3. Which products earn the most, and how many customers buy them?
-- =====================================================================
SELECT
    stock_code,
    MAX(description)                AS description,
    ROUND(SUM(revenue), 2)          AS revenue,
    SUM(quantity)                   AS units_sold,
    COUNT(DISTINCT customer_id)     AS buying_customers
FROM transactions
GROUP BY stock_code
ORDER BY revenue DESC
LIMIT 10;


-- =====================================================================
-- Q4. Where does the money come from geographically?
--     SUM() OVER () gives each country's share of the total.
-- =====================================================================
SELECT
    country,
    COUNT(DISTINCT customer_id)                    AS customers,
    COUNT(DISTINCT invoice_no)                     AS orders,
    ROUND(SUM(revenue), 2)                         AS revenue,
    ROUND(100.0 * SUM(revenue) / SUM(SUM(revenue)) OVER (), 2) AS pct_of_total_revenue
FROM transactions
GROUP BY country
ORDER BY revenue DESC
LIMIT 15;


-- =====================================================================
-- Q5. How much of the business depends on customers coming back?
--     This is the question that justifies a retention project.
-- =====================================================================
WITH customer_orders AS (
    SELECT
        customer_id,
        COUNT(DISTINCT invoice_no) AS orders,
        SUM(revenue)               AS revenue
    FROM transactions
    GROUP BY customer_id
)
SELECT
    CASE WHEN orders = 1 THEN 'One-time buyer'
         ELSE 'Repeat buyer' END                       AS customer_type,
    COUNT(*)                                           AS customers,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_customers,
    ROUND(SUM(revenue), 2)                             AS revenue,
    ROUND(100.0 * SUM(revenue) / SUM(SUM(revenue)) OVER (), 1) AS pct_of_revenue,
    ROUND(AVG(revenue), 2)                             AS avg_revenue_per_customer
FROM customer_orders
GROUP BY customer_type;


-- =====================================================================
-- Q6. Which customers are the most valuable, and how concentrated is
--     revenue at the top? Running total shows the Pareto effect.
-- =====================================================================
WITH customer_revenue AS (
    SELECT customer_id,
           SUM(revenue)               AS revenue,
           COUNT(DISTINCT invoice_no) AS orders
    FROM transactions
    GROUP BY customer_id
)
SELECT
    customer_id,
    orders,
    ROUND(revenue, 2) AS revenue,
    RANK() OVER (ORDER BY revenue DESC) AS revenue_rank,
    ROUND(100.0 * SUM(revenue) OVER (ORDER BY revenue DESC
                                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
          / SUM(revenue) OVER (), 2) AS cumulative_pct_of_revenue
FROM customer_revenue
ORDER BY revenue DESC
LIMIT 20;


-- =====================================================================
-- Q7. RFM base table: recency, frequency and monetary per customer,
--     measured against the last date in the dataset.
-- =====================================================================
WITH reference AS (
    SELECT MAX(invoice_date) AS as_of FROM transactions
)
SELECT
    t.customer_id,
    MAX(t.country)                                   AS country,
    DATEDIFF((SELECT as_of FROM reference), MAX(t.invoice_date)) AS recency_days,
    COUNT(DISTINCT t.invoice_no)                     AS frequency,
    ROUND(SUM(t.revenue), 2)                         AS monetary,
    ROUND(SUM(t.revenue) / COUNT(DISTINCT t.invoice_no), 2) AS avg_order_value
FROM transactions t
GROUP BY t.customer_id
ORDER BY monetary DESC
LIMIT 20;


-- =====================================================================
-- Q8. Full RFM scoring and segmentation, entirely in SQL.
--     NTILE(5) reproduces the quintile scoring used in the Python code:
--     recency is reversed so that a recent buyer scores 5.
-- =====================================================================
WITH reference AS (
    SELECT MAX(invoice_date) AS as_of FROM transactions
),
rfm_base AS (
    SELECT
        customer_id,
        DATEDIFF((SELECT as_of FROM reference), MAX(invoice_date)) AS recency_days,
        COUNT(DISTINCT invoice_no) AS frequency,
        SUM(revenue)               AS monetary
    FROM transactions
    GROUP BY customer_id
),
rfm_scored AS (
    SELECT
        customer_id, recency_days, frequency, monetary,
        NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,  -- reversed
        NTILE(5) OVER (ORDER BY frequency    ASC)  AS f_score,
        NTILE(5) OVER (ORDER BY monetary     ASC)  AS m_score
    FROM rfm_base
)
SELECT
    customer_id, recency_days, frequency, ROUND(monetary, 2) AS monetary,
    r_score, f_score, m_score,
    r_score + f_score + m_score AS rfm_score,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 THEN 'Champions'
        WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal Customers'
        WHEN r_score >= 4 AND f_score  = 2 THEN 'Potential Loyalist'
        WHEN r_score >= 4 AND f_score  = 1 THEN 'New Customers'
        WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
        WHEN r_score  = 1 AND f_score <= 2 THEN 'Lost Customers'
        ELSE 'Hibernating'
    END AS segment
FROM rfm_scored
ORDER BY rfm_score DESC, monetary DESC
LIMIT 25;


-- =====================================================================
-- Q9. Segment summary - the table behind the segmentation dashboard page.
-- =====================================================================
WITH reference AS (
    SELECT MAX(invoice_date) AS as_of FROM transactions
),
rfm_base AS (
    SELECT customer_id,
           DATEDIFF((SELECT as_of FROM reference), MAX(invoice_date)) AS recency_days,
           COUNT(DISTINCT invoice_no) AS frequency,
           SUM(revenue)               AS monetary
    FROM transactions
    GROUP BY customer_id
),
rfm_scored AS (
    SELECT customer_id, recency_days, frequency, monetary,
           NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,
           NTILE(5) OVER (ORDER BY frequency    ASC)  AS f_score
    FROM rfm_base
),
segmented AS (
    SELECT *,
        CASE
            WHEN r_score >= 4 AND f_score >= 4 THEN 'Champions'
            WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal Customers'
            WHEN r_score >= 4 AND f_score  = 2 THEN 'Potential Loyalist'
            WHEN r_score >= 4 AND f_score  = 1 THEN 'New Customers'
            WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
            WHEN r_score  = 1 AND f_score <= 2 THEN 'Lost Customers'
            ELSE 'Hibernating'
        END AS segment
    FROM rfm_scored
)
SELECT
    segment,
    COUNT(*)                                                   AS customers,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)         AS pct_customers,
    ROUND(AVG(recency_days), 0)                                AS avg_recency_days,
    ROUND(AVG(frequency), 1)                                   AS avg_orders,
    ROUND(AVG(monetary), 2)                                    AS avg_revenue,
    ROUND(SUM(monetary), 2)                                    AS total_revenue,
    ROUND(100.0 * SUM(monetary) / SUM(SUM(monetary)) OVER (), 1) AS pct_revenue
FROM segmented
GROUP BY segment
ORDER BY total_revenue DESC;


-- =====================================================================
-- Q10. How long do customers wait between orders?
--      LAG over each customer's order history gives the gap; the average
--      gap is what makes a 90-day silence meaningful.
-- =====================================================================
WITH orders AS (
    SELECT DISTINCT customer_id, invoice_no, DATE(invoice_date) AS order_date
    FROM transactions
),
gaps AS (
    SELECT
        customer_id,
        DATEDIFF(order_date,
                 LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date)) AS days_since_prev
    FROM orders
)
SELECT
    COUNT(*)                        AS order_gaps_observed,
    ROUND(AVG(days_since_prev), 1)  AS avg_days_between_orders,
    MIN(days_since_prev)            AS min_gap,
    MAX(days_since_prev)            AS max_gap
FROM gaps
WHERE days_since_prev IS NOT NULL;


-- =====================================================================
-- Q11. Acquisition cohorts: of the customers first seen in each month,
--      how many ever came back? HAVING filters to meaningful cohorts.
-- =====================================================================
WITH first_purchase AS (
    SELECT customer_id,
           DATE_FORMAT(MIN(invoice_date), '%Y-%m') AS cohort_month,
           COUNT(DISTINCT invoice_no)              AS lifetime_orders,
           SUM(revenue)                            AS lifetime_revenue
    FROM transactions
    GROUP BY customer_id
)
SELECT
    cohort_month,
    COUNT(*)                                                   AS customers_acquired,
    SUM(CASE WHEN lifetime_orders >= 2 THEN 1 ELSE 0 END)      AS became_repeat_buyers,
    ROUND(100.0 * SUM(CASE WHEN lifetime_orders >= 2 THEN 1 ELSE 0 END)
          / COUNT(*), 1)                                       AS repeat_rate_pct,
    ROUND(AVG(lifetime_revenue), 2)                            AS avg_lifetime_revenue
FROM first_purchase
GROUP BY cohort_month
HAVING COUNT(*) >= 20
ORDER BY cohort_month;


-- =====================================================================
-- Q12. Who has gone quiet, and how much historic revenue do they hold?
--      This is the descriptive version of the question the model answers.
-- =====================================================================
WITH reference AS (
    SELECT MAX(invoice_date) AS as_of FROM transactions
),
customer_recency AS (
    SELECT customer_id,
           DATEDIFF((SELECT as_of FROM reference), MAX(invoice_date)) AS recency_days,
           SUM(revenue) AS lifetime_revenue
    FROM transactions
    GROUP BY customer_id
)
SELECT
    CASE
        WHEN recency_days <=  30 THEN 'Active (0-30 days)'
        WHEN recency_days <=  90 THEN 'Cooling (31-90 days)'
        WHEN recency_days <= 180 THEN 'Dormant (91-180 days)'
        ELSE 'Lapsed (180+ days)'
    END                                     AS activity_status,
    COUNT(*)                                AS customers,
    ROUND(SUM(lifetime_revenue), 2)         AS historic_revenue_held,
    ROUND(100.0 * SUM(lifetime_revenue) / SUM(SUM(lifetime_revenue)) OVER (), 1)
                                            AS pct_of_revenue
FROM customer_recency
GROUP BY activity_status
ORDER BY MIN(recency_days);


-- =====================================================================
-- Q13. THE BUSINESS ANSWER: joining transaction history to the model
--      output to show where retention effort should go.
-- =====================================================================
SELECT
    s.retention_priority,
    COUNT(*)                                   AS customers,
    ROUND(AVG(s.inactivity_probability), 3)    AS avg_inactivity_risk,
    ROUND(SUM(s.value_at_risk), 2)             AS total_value_at_risk,
    ROUND(SUM(s.lifetime_revenue), 2)          AS lifetime_revenue_held,
    ROUND(100.0 * SUM(s.value_at_risk) / SUM(SUM(s.value_at_risk)) OVER (), 1)
                                               AS pct_of_value_at_risk,
    MAX(s.recommended_action)                  AS recommended_action
FROM customer_scores s
GROUP BY s.retention_priority
ORDER BY s.retention_priority;


-- =====================================================================
-- Q14. The actual call list: the highest-value customers most at risk,
--      with the behaviour that triggered the flag.
-- =====================================================================
SELECT
    s.customer_id,
    s.country,
    s.segment,
    s.customer_value_tier,
    s.inactivity_probability,
    ROUND(s.value_at_risk, 2)      AS value_at_risk,
    s.lifetime_orders,
    ROUND(s.lifetime_revenue, 2)   AS lifetime_revenue,
    s.lifetime_recency_days        AS days_since_last_order,
    s.recommended_action
FROM customer_scores s
WHERE s.retention_priority = '1 - Critical'
ORDER BY s.value_at_risk DESC
LIMIT 25;
