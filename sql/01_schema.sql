-- ---------------------------------------------------------------------
-- Customer Retention Intelligence - MySQL schema
--
-- Deliberately small. Two tables are loaded:
--
--   transactions       cleaned line-level sales (the analytical base)
--   customer_scores    the Python model output that SQL cannot produce
--                      (inactivity probability, value tier, priority)
--
-- Everything else - RFM, segments, repeat rates, cohorts - is DERIVED in
-- SQL in 02_customer_analysis.sql rather than loaded, so the queries do
-- the analytical work.
-- ---------------------------------------------------------------------

CREATE DATABASE IF NOT EXISTS retail_analytics
    DEFAULT CHARACTER SET utf8mb4;
USE retail_analytics;

DROP TABLE IF EXISTS customer_scores;
DROP TABLE IF EXISTS transactions;

CREATE TABLE transactions (
    line_id       INT UNSIGNED NOT NULL AUTO_INCREMENT,
    invoice_no    VARCHAR(20)    NOT NULL,
    stock_code    VARCHAR(30)    NOT NULL,
    description   VARCHAR(255),
    quantity      INT            NOT NULL,
    invoice_date  DATETIME       NOT NULL,
    unit_price    DECIMAL(10, 2) NOT NULL,
    customer_id   INT            NOT NULL,
    country       VARCHAR(60)    NOT NULL,
    revenue       DECIMAL(12, 2) NOT NULL,
    PRIMARY KEY (line_id),
    INDEX idx_customer (customer_id),
    INDEX idx_invoice  (invoice_no),
    INDEX idx_date     (invoice_date),
    INDEX idx_stock    (stock_code)
) ENGINE = InnoDB;

CREATE TABLE customer_scores (
    customer_id             INT            NOT NULL,
    country                 VARCHAR(60),
    segment                 VARCHAR(30),
    inactivity_probability  DECIMAL(6, 4),
    risk_band               VARCHAR(20),
    risk_band_absolute      VARCHAR(20),
    customer_value_score    DECIMAL(5, 2),
    customer_value_tier     VARCHAR(15),
    retention_priority      VARCHAR(30),
    recommended_action      VARCHAR(255),
    expected_90d_value      DECIMAL(12, 2),
    value_at_risk           DECIMAL(12, 2),
    lifetime_revenue        DECIMAL(12, 2),
    lifetime_orders         INT,
    lifetime_recency_days   INT,
    PRIMARY KEY (customer_id),
    INDEX idx_priority (retention_priority),
    INDEX idx_segment  (segment),
    INDEX idx_risk     (risk_band)
) ENGINE = InnoDB;
