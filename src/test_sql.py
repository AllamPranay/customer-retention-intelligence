"""Verify every query in sql/02_customer_analysis.sql actually runs and
returns the right numbers.

MySQL is not required to check query logic. This harness loads the same
CSVs into an in-memory SQLite database, applies a small dialect shim for
the handful of MySQL-specific date functions, executes each statement, and
cross-checks the headline results against the values pandas computes from
the same data.

    python src/test_sql.py

The queries themselves are written for MySQL 8; the shim only exists so
they can be tested without a running server.
"""
import re
import sqlite3
import sys

import pandas as pd

import config

SQL_FILE = config.ROOT / "sql" / "02_customer_analysis.sql"


def _split_args(inner: str):
    """Split a function's argument list on top-level commas only."""
    args, depth, current = [], 0, []
    for ch in inner:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(current))
            current = []
        else:
            current.append(ch)
    args.append("".join(current))
    return [a.strip() for a in args]


def _rewrite_calls(sql: str, func: str, builder):
    """Find every call to `func` and rewrite it, matching parentheses properly.

    A regex cannot do this: the arguments themselves contain nested calls
    such as MAX(t.invoice_date).
    """
    pattern = re.compile(rf"\b{func}\s*\(", re.I)
    while True:
        m = pattern.search(sql)
        if not m:
            return sql
        start, depth, i = m.end(), 1, m.end()
        while i < len(sql) and depth:
            if sql[i] == "(":
                depth += 1
            elif sql[i] == ")":
                depth -= 1
            i += 1
        sql = sql[:m.start()] + builder(_split_args(sql[start:i - 1])) + sql[i:]


def to_sqlite_dialect(sql: str) -> str:
    """Translate the MySQL-only constructs used in the analysis file."""
    # DATE_FORMAT(col, '%Y-%m')  ->  strftime('%Y-%m', col)
    sql = _rewrite_calls(sql, "DATE_FORMAT", lambda a: f"strftime({a[1]}, {a[0]})")
    # DATEDIFF(a, b)  ->  CAST(julianday(a) - julianday(b) AS INTEGER)
    sql = _rewrite_calls(
        sql, "DATEDIFF",
        lambda a: f"CAST(julianday({a[0]}) - julianday({a[1]}) AS INTEGER)")
    sql = re.sub(r"^\s*USE\s+\w+\s*;?\s*$", "", sql, flags=re.I | re.M)
    return sql


def build_database() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")

    tx = pd.read_csv(config.CLEAN_TRANSACTIONS, parse_dates=["InvoiceDate"])
    tx = tx.rename(columns={
        "InvoiceNo": "invoice_no", "StockCode": "stock_code",
        "Description": "description", "Quantity": "quantity",
        "InvoiceDate": "invoice_date", "UnitPrice": "unit_price",
        "CustomerID": "customer_id", "Country": "country", "Revenue": "revenue",
    })[["invoice_no", "stock_code", "description", "quantity", "invoice_date",
        "unit_price", "customer_id", "country", "revenue"]]
    tx["invoice_date"] = tx["invoice_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    tx.to_sql("transactions", conn, index=False)

    scores = pd.read_csv(config.PROCESSED_DIR / "customer_master.csv")
    keep = ["CustomerID", "country", "segment", "inactivity_probability", "risk_band",
            "risk_band_absolute", "customer_value_score", "customer_value_tier",
            "retention_priority", "recommended_action", "expected_90d_value",
            "value_at_risk", "lifetime_revenue", "lifetime_orders", "lifetime_recency_days"]
    scores = scores[[c for c in keep if c in scores.columns]].rename(
        columns={"CustomerID": "customer_id"})
    scores.to_sql("customer_scores", conn, index=False)

    for stmt in ("CREATE INDEX ix_tx_cust ON transactions(customer_id)",
                 "CREATE INDEX ix_tx_date ON transactions(invoice_date)"):
        conn.execute(stmt)
    return conn


def split_statements(sql: str):
    """Split the file into (title, body) pairs.

    Comments are stripped BEFORE splitting on semicolons - a comment may
    itself contain a semicolon, which would otherwise cut a query in half.
    """
    out, title, buffer = [], None, []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            match = re.match(r"--\s*(Q\d+\..*)", stripped)
            if match and not "".join(buffer).strip():
                title = match.group(1).strip()
            continue
        buffer.append(line)
        while ";" in "".join(buffer):
            joined = "\n".join(buffer)
            head, _, tail = joined.partition(";")
            if head.strip():
                out.append((title or "(untitled)", head.strip()))
                title = None
            buffer = [tail] if tail.strip() else []
            if not tail.strip():
                break
    if "\n".join(buffer).strip():
        out.append((title or "(untitled)", "\n".join(buffer).strip()))
    return out


def main() -> int:
    conn = build_database()
    statements = split_statements(to_sqlite_dialect(SQL_FILE.read_text()))
    print(f"Executing {len(statements)} queries from {SQL_FILE.name}\n")

    failures = []
    results = {}
    for title, body in statements:
        try:
            df = pd.read_sql_query(body, conn)
            results[title] = df
            print(f"  PASS  {title[:72]:<72} {len(df):>5} rows")
        except Exception as exc:
            failures.append((title, exc))
            print(f"  FAIL  {title[:72]:<72} {exc}")

    # ---------- cross-check the SQL answers against pandas ----------------
    print("\nCross-checking SQL results against pandas:")
    tx = pd.read_csv(config.CLEAN_TRANSACTIONS)
    checks = []

    q1 = next((v for k, v in results.items() if k.startswith("Q1.")), None)
    if q1 is not None:
        checks.append(("total revenue", float(q1.total_revenue.iat[0]),
                       round(float(tx.Revenue.sum()), 2)))
        checks.append(("customer count", int(q1.customers.iat[0]),
                       int(tx.CustomerID.nunique())))
        checks.append(("order count", int(q1.orders.iat[0]),
                       int(tx.InvoiceNo.nunique())))

    q5 = next((v for k, v in results.items() if k.startswith("Q5.")), None)
    if q5 is not None:
        orders = tx.groupby("CustomerID").InvoiceNo.nunique()
        expected = int((orders >= 2).sum())
        got = int(q5.loc[q5.customer_type == "Repeat buyer", "customers"].iat[0])
        checks.append(("repeat customers", got, expected))

    q13 = next((v for k, v in results.items() if k.startswith("Q13.")), None)
    if q13 is not None:
        master = pd.read_csv(config.PROCESSED_DIR / "customer_master.csv")
        checks.append(("total value at risk", round(float(q13.total_value_at_risk.sum()), 0),
                       round(float(master.value_at_risk.sum()), 0)))

    for name, got, expected in checks:
        ok = abs(got - expected) < max(1.0, abs(expected) * 0.001)
        print(f"  {'OK  ' if ok else 'MISMATCH'}  {name:<24} sql={got:>15,}  pandas={expected:>15,}")
        if not ok:
            failures.append((name, f"sql={got} pandas={expected}"))

    # ---------- show two results in full, as evidence ---------------------
    for key in ("Q9.", "Q13."):
        r = next((v for k, v in results.items() if k.startswith(key)), None)
        if r is not None:
            print(f"\n{key} result:")
            print(r.to_string(index=False))

    if failures:
        print(f"\n{len(failures)} problem(s) found.")
        return 1
    print(f"\nAll {len(statements)} queries executed and all cross-checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
