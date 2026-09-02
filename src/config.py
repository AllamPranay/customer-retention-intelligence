"""Central configuration for the Customer Retention Intelligence project.

Every path and modelling constant lives here so the pipeline stages stay
consistent with each other and with the notebooks.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DASHBOARD_DIR = ROOT / "dashboard"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_EXCEL = RAW_DIR / "Online Retail.xlsx"
RAW_CACHE = RAW_DIR / "online_retail_raw.pkl"
CLEAN_TRANSACTIONS = PROCESSED_DIR / "transactions_clean.csv"

for _d in (PROCESSED_DIR, DASHBOARD_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Time design  (see docs/business_definitions.md)
#
# A snapshot = a fixed-length observation window used to build features,
# followed by a fixed-length future window used to define the target.
# The observation window is held at a CONSTANT length across snapshots so
# that volume-based features (frequency, spend) are comparable.
# ----------------------------------------------------------------------
OBSERVATION_DAYS = 180   # history used to build features
HORIZON_DAYS = 90        # future window used to define inactivity

SNAPSHOTS = {
    # name    : cut-off date (first day of the future window)
    "train": "2011-06-09",   # target measured 2011-06-09 -> 2011-09-06
    "test":  "2011-09-09",   # target measured 2011-09-09 -> 2011-12-07
    "score": "2011-12-10",   # end of data: no target, this is the live scoring run
}

# ----------------------------------------------------------------------
# Data cleaning rules
# ----------------------------------------------------------------------
# Administrative / fee stock codes that are not customer product purchases.
# Postage codes (POST, DOT, C2) are deliberately KEPT: they are genuine
# charges the customer paid on a real order.
NON_PRODUCT_STOCK_CODES = {
    "M",             # manual adjustment
    "D",             # discount
    "S",             # samples
    "B",             # adjustment for bad debt
    "PADS",          # pads to match all cushions (internal)
    "BANK CHARGES",  # bank charges
    "AMAZONFEE",     # Amazon marketplace fee
    "CRUK",          # charity donation
}

# ----------------------------------------------------------------------
# Modelling
# ----------------------------------------------------------------------
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "max_order_value",
    "unique_products",
    "total_quantity",
    "tenure_days",
    "avg_days_between_orders",
    "purchase_trend_ratio",
    "active_weeks",
    "cancel_orders",
    "cancel_rate",
    "avg_basket_size",
    "is_uk",
]

# Risk bands applied to the predicted probability of inactivity
RISK_BANDS = [("Low", 0.0, 0.40), ("Medium", 0.40, 0.70), ("High", 0.70, 1.01)]

# MySQL connection (override with environment variables)
MYSQL = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "retail_analytics",
}
