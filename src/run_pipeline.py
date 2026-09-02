"""Run the whole project end to end.

    python src/run_pipeline.py

Stages run in order and each writes its outputs to data/processed/ and
reports/. Loading into MySQL is a separate, optional step
(python src/load_to_mysql.py) because it needs a running server.
"""
import time

import data_processing
import eda
import export_dashboard
import feature_engineering
import model
import retention_strategy
import segmentation

STAGES = [
    ("Phase 1-2  Data cleaning",          lambda: data_processing.build_clean_dataset()),
    ("Phase 3    Exploratory analysis",   lambda: eda.run()),
    ("Phase 6a   Feature engineering",    lambda: feature_engineering.build_all_snapshots()),
    ("Phase 5    RFM segmentation",       lambda: segmentation.run()),
    ("Phase 6b   Inactivity model",       lambda: model.run()),
    ("Phase 7-9  Retention strategy",     lambda: retention_strategy.run()),
    ("Export     Power BI tables",        lambda: export_dashboard.run()),
]


def main():
    started = time.time()
    for name, stage in STAGES:
        print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
        stage()
    print(f"\n{'=' * 72}")
    print(f"Pipeline finished in {time.time() - started:.1f}s")
    print("Outputs: data/processed/, reports/, dashboard/")
    print("Optional: python src/load_to_mysql.py   (needs a running MySQL server)")


if __name__ == "__main__":
    main()
