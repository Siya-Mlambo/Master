-- ============================================================
-- PROJECT 4: Customer Analytics SQL Pipeline
-- Author   : Siyabonga Mlambo
-- Goal     : RFM segmentation + CLV modelling in SQL
-- Database : PostgreSQL (adapts to MySQL/SQLite with minor changes)
-- ============================================================

-- ============================================================
-- SECTION 1 — DATABASE SETUP
-- ============================================================

CREATE TABLE IF NOT EXISTS customers (
    customer_id     VARCHAR(20) PRIMARY KEY,
    name            VARCHAR(100),
    email           VARCHAR(100),
    signup_date     DATE,
    region          VARCHAR(50),
    age_group       VARCHAR(20),
    acquisition_channel VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id  VARCHAR(30) PRIMARY KEY,
    customer_id     VARCHAR(20) REFERENCES customers(customer_id),
    transaction_date DATE,
    amount          DECIMAL(12, 2),
    product_category VARCHAR(50),
    payment_method  VARCHAR(30),
    store_region    VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS products (
    product_id      VARCHAR(20) PRIMARY KEY,
    product_name    VARCHAR(100),
    category        VARCHAR(50),
    unit_price      DECIMAL(10, 2)
);

-- ============================================================
-- SECTION 2 — RFM ANALYSIS
-- (Recency, Frequency, Monetary)
-- ============================================================

-- 2a. Base RFM metrics per customer
WITH rfm_base AS (
    SELECT
        customer_id,
        MAX(transaction_date)                              AS last_purchase_date,
        COUNT(DISTINCT transaction_id)                     AS frequency,
        SUM(amount)                                        AS monetary,
        CURRENT_DATE - MAX(transaction_date)               AS recency_days
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '2 years'
    GROUP BY customer_id
),

-- 2b. RFM scores (1-5 scale using quintiles)
rfm_scored AS (
    SELECT
        customer_id,
        last_purchase_date,
        recency_days,
        frequency,
        ROUND(monetary, 2)                                 AS monetary,

        -- Recency: lower days = better score
        NTILE(5) OVER (ORDER BY recency_days DESC)         AS r_score,

        -- Frequency: higher = better
        NTILE(5) OVER (ORDER BY frequency ASC)            AS f_score,

        -- Monetary: higher = better
        NTILE(5) OVER (ORDER BY monetary ASC)             AS m_score
    FROM rfm_base
),

-- 2c. Combined RFM score and segment labels
rfm_segments AS (
    SELECT
        customer_id,
        recency_days,
        frequency,
        monetary,
        r_score,
        f_score,
        m_score,
        (r_score + f_score + m_score)                      AS rfm_total,
        CONCAT(r_score::TEXT, f_score::TEXT, m_score::TEXT) AS rfm_cell,

        CASE
            WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4
                THEN 'Champions'
            WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3
                THEN 'Loyal Customers'
            WHEN r_score >= 4 AND f_score <= 2
                THEN 'Recent Customers'
            WHEN r_score >= 3 AND f_score >= 2 AND m_score >= 3
                THEN 'Potential Loyalists'
            WHEN r_score = 5 AND f_score = 1
                THEN 'New Customers'
            WHEN r_score <= 2 AND f_score >= 3 AND m_score >= 3
                THEN 'At Risk'
            WHEN r_score <= 2 AND f_score >= 4 AND m_score >= 4
                THEN 'Cannot Lose Them'
            WHEN r_score <= 2 AND f_score <= 2 AND m_score <= 2
                THEN 'Lost'
            ELSE 'Need Attention'
        END                                                AS segment
    FROM rfm_scored
)

SELECT * FROM rfm_segments
ORDER BY rfm_total DESC;

-- ============================================================
-- SECTION 3 — SEGMENT SUMMARY DASHBOARD
-- ============================================================

WITH rfm_segments AS (
    -- (same CTE as above — reuse in production)
    SELECT 'Champions'        AS segment, 450  AS customers, 4850  AS avg_monetary, 8    AS avg_frequency, 12   AS avg_recency UNION ALL
    SELECT 'Loyal Customers',              890,               2100,                  5,                    28  UNION ALL
    SELECT 'At Risk',                      320,               1800,                  4,                    95  UNION ALL
    SELECT 'Potential Loyalists',          560,               950,                   2,                    21  UNION ALL
    SELECT 'Need Attention',               280,               600,                   2,                    65  UNION ALL
    SELECT 'New Customers',                190,               400,                   1,                    8   UNION ALL
    SELECT 'Lost',                         220,               250,                   1,                    280
)
SELECT
    segment,
    customers,
    avg_monetary,
    avg_frequency,
    avg_recency                                                AS avg_recency_days,
    customers * avg_monetary                                   AS segment_revenue,
    ROUND(100.0 * customers / SUM(customers) OVER(), 1)       AS pct_of_base,
    ROUND(100.0 * (customers * avg_monetary)
          / SUM(customers * avg_monetary) OVER(), 1)           AS pct_of_revenue
FROM rfm_segments
ORDER BY segment_revenue DESC;

-- ============================================================
-- SECTION 4 — COHORT RETENTION ANALYSIS
-- ============================================================

WITH first_purchase AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', MIN(transaction_date))  AS cohort_month
    FROM transactions
    GROUP BY customer_id
),

monthly_activity AS (
    SELECT
        t.customer_id,
        DATE_TRUNC('month', t.transaction_date)     AS activity_month
    FROM transactions t
    GROUP BY t.customer_id, DATE_TRUNC('month', t.transaction_date)
),

cohort_data AS (
    SELECT
        fp.cohort_month,
        ma.activity_month,
        COUNT(DISTINCT fp.customer_id)              AS active_customers,
        EXTRACT(EPOCH FROM (ma.activity_month - fp.cohort_month))
            / (30.44 * 24 * 3600)                  AS month_number
    FROM first_purchase fp
    JOIN monthly_activity ma USING (customer_id)
    GROUP BY fp.cohort_month, ma.activity_month
),

cohort_size AS (
    SELECT cohort_month, COUNT(*) AS cohort_customers
    FROM first_purchase
    GROUP BY cohort_month
)

SELECT
    cd.cohort_month,
    cs.cohort_customers,
    cd.month_number::INT                            AS months_since_join,
    cd.active_customers,
    ROUND(100.0 * cd.active_customers
          / cs.cohort_customers, 1)                 AS retention_rate_pct
FROM cohort_data cd
JOIN cohort_size cs USING (cohort_month)
WHERE cd.month_number >= 0
ORDER BY cd.cohort_month, cd.month_number;

-- ============================================================
-- SECTION 5 — CUSTOMER LIFETIME VALUE (CLV)
-- ============================================================

WITH customer_metrics AS (
    SELECT
        customer_id,
        COUNT(DISTINCT transaction_id)                          AS total_orders,
        SUM(amount)                                             AS total_revenue,
        AVG(amount)                                             AS avg_order_value,
        MIN(transaction_date)                                   AS first_purchase,
        MAX(transaction_date)                                   AS last_purchase,
        EXTRACT(DAY FROM MAX(transaction_date) - MIN(transaction_date)) AS customer_age_days,
        COUNT(DISTINCT DATE_TRUNC('month', transaction_date))   AS active_months
    FROM transactions
    GROUP BY customer_id
),

clv_calc AS (
    SELECT
        customer_id,
        total_orders,
        total_revenue,
        avg_order_value,
        first_purchase,
        last_purchase,
        customer_age_days,
        active_months,

        -- Purchase frequency per month
        CASE WHEN active_months > 0
             THEN ROUND(total_orders::NUMERIC / active_months, 2)
             ELSE 0 END                                         AS purchase_freq_monthly,

        -- Historical CLV
        total_revenue                                           AS historical_clv,

        -- Predicted 12-month CLV (simplified BG/NBD approximation)
        ROUND(
            avg_order_value
            * (CASE WHEN active_months > 0
               THEN total_orders::NUMERIC / active_months
               ELSE 0 END)
            * 12
            * 0.6  -- assumed gross margin
        , 2)                                                    AS predicted_clv_12m,

        -- CLV segment
        CASE
            WHEN total_revenue > 10000  THEN 'High Value'
            WHEN total_revenue > 3000   THEN 'Medium Value'
            WHEN total_revenue > 1000   THEN 'Low Value'
            ELSE 'Very Low Value'
        END                                                     AS clv_segment
    FROM customer_metrics
)

SELECT
    clv_segment,
    COUNT(*)                                                     AS num_customers,
    ROUND(AVG(historical_clv), 2)                               AS avg_historical_clv,
    ROUND(AVG(predicted_clv_12m), 2)                            AS avg_predicted_clv_12m,
    ROUND(SUM(historical_clv), 2)                               AS total_segment_revenue,
    ROUND(AVG(purchase_freq_monthly), 2)                        AS avg_monthly_frequency,
    ROUND(AVG(avg_order_value), 2)                              AS avg_order_value
FROM clv_calc
GROUP BY clv_segment
ORDER BY avg_historical_clv DESC;

-- ============================================================
-- SECTION 6 — PRODUCT AFFINITY / CROSS-SELL ANALYSIS
-- ============================================================

WITH category_pairs AS (
    SELECT
        t1.customer_id,
        t1.product_category  AS category_a,
        t2.product_category  AS category_b,
        COUNT(*)             AS co_purchase_count
    FROM transactions t1
    JOIN transactions t2
        ON t1.customer_id = t2.customer_id
        AND t1.product_category < t2.product_category
        AND ABS(t1.transaction_date - t2.transaction_date) <= 30
    GROUP BY t1.customer_id, t1.product_category, t2.product_category
)
SELECT
    category_a,
    category_b,
    COUNT(DISTINCT customer_id)  AS customers_buying_both,
    SUM(co_purchase_count)       AS total_co_purchases
FROM category_pairs
GROUP BY category_a, category_b
ORDER BY total_co_purchases DESC
LIMIT 20;

-- ============================================================
-- SECTION 7 — MONTHLY REVENUE TREND + YOY GROWTH
-- ============================================================

WITH monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', transaction_date)    AS month,
        EXTRACT(YEAR FROM transaction_date)      AS yr,
        EXTRACT(MONTH FROM transaction_date)     AS mo,
        COUNT(DISTINCT customer_id)              AS unique_customers,
        COUNT(DISTINCT transaction_id)           AS total_orders,
        ROUND(SUM(amount), 2)                    AS revenue,
        ROUND(AVG(amount), 2)                    AS avg_order_value
    FROM transactions
    GROUP BY 1, 2, 3
),
with_yoy AS (
    SELECT
        *,
        LAG(revenue, 12) OVER (ORDER BY month)  AS revenue_prior_year,
        LAG(revenue, 1)  OVER (ORDER BY month)  AS revenue_prior_month
    FROM monthly_revenue
)
SELECT
    month,
    unique_customers,
    total_orders,
    revenue,
    avg_order_value,
    revenue_prior_year,
    CASE WHEN revenue_prior_year > 0
         THEN ROUND(100.0 * (revenue - revenue_prior_year) / revenue_prior_year, 1)
         ELSE NULL END                           AS yoy_growth_pct,
    CASE WHEN revenue_prior_month > 0
         THEN ROUND(100.0 * (revenue - revenue_prior_month) / revenue_prior_month, 1)
         ELSE NULL END                           AS mom_growth_pct
FROM with_yoy
ORDER BY month DESC;

-- ============================================================
-- END OF CUSTOMER ANALYTICS SQL PIPELINE
-- Author: Siyabonga Mlambo | Sol Plaatje University
-- ============================================================
