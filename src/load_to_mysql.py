"""Phase 4: load the cleaned data and model output into MySQL.

Usage
-----
    python src/load_to_mysql.py

Connection settings come from config.MYSQL and can be overridden with the
environment variables MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD,
MYSQL_DATABASE.

The schema in sql/01_schema.sql is executed first, then the two tables are
populated from data/processed/.
"""
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text

import config

SCHEMA_FILE = config.ROOT / "sql" / "01_schema.sql"

SCORE_COLUMNS = [
    "CustomerID", "country", "segment", "inactivity_probability", "risk_band",
    "risk_band_absolute", "customer_value_score", "customer_value_tier",
    "retention_priority", "recommended_action", "expected_90d_value",
    "value_at_risk", "lifetime_revenue", "lifetime_orders", "lifetime_recency_days",
]


def settings() -> dict:
    s = dict(config.MYSQL)
    for key in s:
        env = os.environ.get(f"MYSQL_{key.upper()}")
        if env:
            s[key] = int(env) if key == "port" else env
    return s


def make_engine(with_database: bool = True):
    s = settings()
    # PyMySQL is the pure-Python driver; mysqlclient is used if available.
    for driver in ("pymysql", "mysqldb", "mysqlconnector"):
        try:
            url = (f"mysql+{driver}://{s['user']}:{s['password']}"
                   f"@{s['host']}:{s['port']}/" + (s["database"] if with_database else ""))
            engine = create_engine(url, pool_pre_ping=True)
            with engine.connect():
                pass
            print(f"Connected to MySQL at {s['host']}:{s['port']} using {driver}")
            return engine
        except Exception as exc:                      # try the next driver
            last = exc
    raise SystemExit(
        f"Could not connect to MySQL ({last}).\n"
        "Check that the MySQL server is running and that a driver is installed:\n"
        "    pip install pymysql")


def run_schema():
    """Execute sql/01_schema.sql (creates the database and both tables)."""
    engine = make_engine(with_database=False)
    statements = [s.strip() for s in SCHEMA_FILE.read_text().split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    print(f"Schema applied from {SCHEMA_FILE.name} ({len(statements)} statements)")


def load_tables(chunksize: int = 20_000):
    engine = make_engine()

    tx = pd.read_csv(config.CLEAN_TRANSACTIONS, parse_dates=["InvoiceDate"])
    tx = tx.rename(columns={
        "InvoiceNo": "invoice_no", "StockCode": "stock_code",
        "Description": "description", "Quantity": "quantity",
        "InvoiceDate": "invoice_date", "UnitPrice": "unit_price",
        "CustomerID": "customer_id", "Country": "country", "Revenue": "revenue",
    })[["invoice_no", "stock_code", "description", "quantity", "invoice_date",
        "unit_price", "customer_id", "country", "revenue"]]
    tx.to_sql("transactions", engine, if_exists="append", index=False,
              chunksize=chunksize, method="multi")
    print(f"Loaded transactions:    {len(tx):,} rows")

    scores = pd.read_csv(config.PROCESSED_DIR / "customer_master.csv")
    scores = scores[[c for c in SCORE_COLUMNS if c in scores.columns]]
    scores = scores.rename(columns={"CustomerID": "customer_id"})
    scores.to_sql("customer_scores", engine, if_exists="append", index=False,
                  chunksize=chunksize, method="multi")
    print(f"Loaded customer_scores: {len(scores):,} rows")

    with engine.connect() as conn:
        for table in ("transactions", "customer_scores"):
            n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"  verified {table}: {n:,} rows in MySQL")


if __name__ == "__main__":
    run_schema()
    load_tables()
    print("\nMySQL is ready. Run the analysis with:")
    print("    mysql -u root -p retail_analytics < sql/02_customer_analysis.sql")
