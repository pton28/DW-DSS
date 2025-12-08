-- DuckDB schema for Financial Data Warehouse (Star Schema)

PRAGMA threads=2;

------------------------------------------------------------
-- 1) CREATE SEQUENCES FOR AUTO-INCREMENT KEYS
------------------------------------------------------------

CREATE SEQUENCE IF NOT EXISTS seq_fact_finance START 1;
CREATE SEQUENCE IF NOT EXISTS seq_fact_stock_prices START 1;


------------------------------------------------------------
-- 2) DIM_DATE
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_date (
    date_key INTEGER PRIMARY KEY,
    date DATE,
    day INTEGER,
    month INTEGER,
    month_name VARCHAR,
    quarter VARCHAR,
    year INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN
);

------------------------------------------------------------
-- 3) DIM_COMPANY
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_company (
    company_key INTEGER PRIMARY KEY,
    ticker VARCHAR,
    company_name VARCHAR,
    industry VARCHAR,
    exchange VARCHAR
);

------------------------------------------------------------
-- 4) DIM_FIN_STATEMENT_TYPE
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_fin_statement_type (
    fin_type_key INTEGER PRIMARY KEY,
    statement_type VARCHAR,
    description VARCHAR
);

------------------------------------------------------------
-- 5) DIM_FIN_METRIC
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_fin_metric (
    metric_key INTEGER PRIMARY KEY,
    metric_name VARCHAR,
    metric_group VARCHAR,
    unit VARCHAR
);

------------------------------------------------------------
-- 6) FACT_FINANCE  (USE SEQUENCE FOR AUTO-INCREMENT)
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_finance (
    fact_finance_id INTEGER PRIMARY KEY DEFAULT nextval('seq_fact_finance'),
    date_key INTEGER,
    company_key INTEGER,
    fin_type_key INTEGER,
    metric_key INTEGER,
    value DOUBLE,
    FOREIGN KEY(date_key) REFERENCES dim_date(date_key),
    FOREIGN KEY(company_key) REFERENCES dim_company(company_key),
    FOREIGN KEY(fin_type_key) REFERENCES dim_fin_statement_type(fin_type_key),
    FOREIGN KEY(metric_key) REFERENCES dim_fin_metric(metric_key)
);

------------------------------------------------------------
-- 7) DIM_STOCK_METRIC
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_stock_metric (
    stock_metric_key INTEGER PRIMARY KEY,
    metric_name VARCHAR
);

------------------------------------------------------------
-- 8) FACT_STOCK_PRICES (USE SEQUENCE FOR AUTO-INCREMENT)
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_stock_prices (
    fact_price_id INTEGER PRIMARY KEY DEFAULT nextval('seq_fact_stock_prices'),
    date_key INTEGER,
    company_key INTEGER,
    stock_metric_key INTEGER,
    value DOUBLE,
    FOREIGN KEY(date_key) REFERENCES dim_date(date_key),
    FOREIGN KEY(company_key) REFERENCES dim_company(company_key),
    FOREIGN KEY(stock_metric_key) REFERENCES dim_stock_metric(stock_metric_key)
);

------------------------------------------------------------
-- 9) Indexes for performance
------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_fact_finance_date ON fact_finance(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_price_date ON fact_stock_prices(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_finance_company ON fact_finance(company_key);
CREATE INDEX IF NOT EXISTS idx_fact_price_company ON fact_stock_prices(company_key);

------------------------------------------------------------
-- 10) Seed dim_fin_statement_type defaults
------------------------------------------------------------
INSERT INTO dim_fin_statement_type (fin_type_key, statement_type, description)
SELECT 1, 'balance_sheet', 'Balance Sheet'
WHERE NOT EXISTS (SELECT 1 FROM dim_fin_statement_type WHERE fin_type_key=1);

INSERT INTO dim_fin_statement_type (fin_type_key, statement_type, description)
SELECT 2, 'income_statement', 'Income Statement'
WHERE NOT EXISTS (SELECT 1 FROM dim_fin_statement_type WHERE fin_type_key=2);

INSERT INTO dim_fin_statement_type (fin_type_key, statement_type, description)
SELECT 3, 'cash_flow', 'Cash Flow Statement'
WHERE NOT EXISTS (SELECT 1 FROM dim_fin_statement_type WHERE fin_type_key=3);

------------------------------------------------------------
-- 11) Seed dim_stock_metric defaults
------------------------------------------------------------
INSERT INTO dim_stock_metric (stock_metric_key, metric_name)
SELECT 1, 'open' WHERE NOT EXISTS (SELECT 1 FROM dim_stock_metric WHERE stock_metric_key=1);

INSERT INTO dim_stock_metric (stock_metric_key, metric_name)
SELECT 2, 'high' WHERE NOT EXISTS (SELECT 1 FROM dim_stock_metric WHERE stock_metric_key=2);

INSERT INTO dim_stock_metric (stock_metric_key, metric_name)
SELECT 3, 'low' WHERE NOT EXISTS (SELECT 1 FROM dim_stock_metric WHERE stock_metric_key=3);

INSERT INTO dim_stock_metric (stock_metric_key, metric_name)
SELECT 4, 'close' WHERE NOT EXISTS (SELECT 1 FROM dim_stock_metric WHERE stock_metric_key=4);

INSERT INTO dim_stock_metric (stock_metric_key, metric_name)
SELECT 5, 'volume' WHERE NOT EXISTS (SELECT 1 FROM dim_stock_metric WHERE stock_metric_key=5);

INSERT INTO dim_stock_metric (stock_metric_key, metric_name)
SELECT 6, 'pct_change' WHERE NOT EXISTS (SELECT 1 FROM dim_stock_metric WHERE stock_metric_key=6);

INSERT INTO dim_stock_metric (stock_metric_key, metric_name)
SELECT 7, 'daily_return' WHERE NOT EXISTS (SELECT 1 FROM dim_stock_metric WHERE stock_metric_key=7);

INSERT INTO dim_stock_metric (stock_metric_key, metric_name)
SELECT 8, 'volatility_5d' WHERE NOT EXISTS (SELECT 1 FROM dim_stock_metric WHERE stock_metric_key=8);

INSERT INTO dim_stock_metric (stock_metric_key, metric_name)
SELECT 9, 'volatility_20d' WHERE NOT EXISTS (SELECT 1 FROM dim_stock_metric WHERE stock_metric_key=9);
