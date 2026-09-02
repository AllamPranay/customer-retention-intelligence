"""Phase 1-2: data understanding and cleaning.

Turns the raw UCI Online Retail workbook into a clean, analysis-ready
transaction table. Every cleaning rule is applied as an explicit, logged
step so the record loss can be audited (see reports/cleaning_log.csv).
"""
import pandas as pd

import config


# ----------------------------------------------------------------------
# Phase 1 - loading and understanding
# ----------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Read the raw workbook, caching it as a pickle (the .xlsx is slow)."""
    if config.RAW_CACHE.exists():
        return pd.read_pickle(config.RAW_CACHE)
    df = pd.read_excel(config.RAW_EXCEL)
    df.to_pickle(config.RAW_CACHE)
    return df


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level profile used for the data understanding summary."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(),
    })


# ----------------------------------------------------------------------
# Phase 2 - cleaning
# ----------------------------------------------------------------------
def clean(df: pd.DataFrame, verbose: bool = True):
    """Apply the cleaning rules and return (clean_df, cleaning_log).

    Design decisions
    ----------------
    * Cancellations are removed from the sales table but retained
      separately: a customer's cancellation history is legitimate
      observation-period behaviour and becomes a model feature.
    * Rows without a CustomerID cannot be attributed to a customer, so they
      are dropped for customer-level work. Their revenue is reported in the
      log rather than silently discarded.
    * Administrative stock codes (bank charges, fees, discounts, manual
      adjustments) are not product purchases and are removed. Postage codes
      are kept - they are real charges on real orders.
    """
    log = []
    n0 = len(df)

    def step(name, frame, note=""):
        log.append({
            "step": name,
            "rows_after": len(frame),
            "rows_removed": (log[-1]["rows_after"] if log else n0) - len(frame),
            "note": note,
        })
        if verbose:
            print(f"  {name:<34} rows={len(frame):>7,}  removed={log[-1]['rows_removed']:>6,}  {note}")

    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    if verbose:
        print(f"Cleaning {n0:,} raw rows")

    # 1. Drop exact duplicate rows (same invoice, product, qty, time, price)
    d = d.drop_duplicates()
    step("drop exact duplicates", d)

    # 2. Flag cancellations, then split them out of the sales table
    d["InvoiceNo"] = d["InvoiceNo"].astype(str).str.strip()
    d["is_cancellation"] = d["InvoiceNo"].str.upper().str.startswith("C")
    cancellations = d[d["is_cancellation"]].copy()
    d = d[~d["is_cancellation"]].copy()
    step("separate cancellations", d, f"{len(cancellations):,} cancelled lines kept for features")

    # 3. Drop rows with no CustomerID (cannot be attributed to a customer)
    lost_revenue = (d.loc[d.CustomerID.isna(), "Quantity"]
                    * d.loc[d.CustomerID.isna(), "UnitPrice"]).sum()
    d = d[d["CustomerID"].notna()].copy()
    step("drop missing CustomerID", d, f"unattributed revenue GBP {lost_revenue:,.0f}")

    # 4. Remove non-positive quantities (stock adjustments, write-offs)
    d = d[d["Quantity"] > 0]
    step("drop quantity <= 0", d)

    # 5. Remove non-positive prices (free gifts / data entry errors)
    d = d[d["UnitPrice"] > 0]
    step("drop unit price <= 0", d)

    # 6. Remove administrative (non-product) stock codes
    d["StockCode"] = d["StockCode"].astype(str).str.strip().str.upper()
    d = d[~d["StockCode"].isin(config.NON_PRODUCT_STOCK_CODES)]
    step("drop admin stock codes", d, "bank charges, fees, discounts, manual adj.")

    # 7. Tidy types and derive line revenue
    d["CustomerID"] = d["CustomerID"].astype(int)
    d["Description"] = d["Description"].astype(str).str.strip().str.upper()
    d["Country"] = d["Country"].str.strip()
    d["InvoiceDate"] = pd.to_datetime(d["InvoiceDate"])
    d["Revenue"] = (d["Quantity"] * d["UnitPrice"]).round(2)
    d["InvoiceDay"] = d["InvoiceDate"].dt.normalize()
    d = d.drop(columns=["is_cancellation"])

    # Clean the cancellation table the same way so it can be joined by customer
    cancellations = cancellations[cancellations["CustomerID"].notna()].copy()
    cancellations["CustomerID"] = cancellations["CustomerID"].astype(int)
    cancellations["InvoiceDate"] = pd.to_datetime(cancellations["InvoiceDate"])
    cancellations["Revenue"] = (cancellations["Quantity"] * cancellations["UnitPrice"]).round(2)

    log_df = pd.DataFrame(log)
    if verbose:
        print(f"  {'FINAL':<34} rows={len(d):>7,}  "
              f"retained={len(d)/n0*100:.1f}% of raw  customers={d.CustomerID.nunique():,}")
    return d, cancellations, log_df


def build_clean_dataset(verbose: bool = True):
    """Full Phase 1-2 run: load, clean, persist."""
    raw = load_raw()
    clean_df, cancellations, log_df = clean(raw, verbose=verbose)
    clean_df.to_csv(config.CLEAN_TRANSACTIONS, index=False)
    cancellations.to_csv(config.PROCESSED_DIR / "cancellations_clean.csv", index=False)
    log_df.to_csv(config.REPORTS_DIR / "cleaning_log.csv", index=False)
    return clean_df, cancellations


def load_clean() -> pd.DataFrame:
    """Read the cleaned transaction table back with correct dtypes."""
    df = pd.read_csv(config.CLEAN_TRANSACTIONS, parse_dates=["InvoiceDate", "InvoiceDay"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    return df


def load_cancellations() -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED_DIR / "cancellations_clean.csv",
                       parse_dates=["InvoiceDate"])


if __name__ == "__main__":
    build_clean_dataset()
