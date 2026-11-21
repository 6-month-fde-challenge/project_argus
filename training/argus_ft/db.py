"""Joined views over the Supabase table snapshots (data/db/*.parquet)."""
from __future__ import annotations

import pandas as pd

from .supabase_data import fetch_tables

AGE_BANDS = [(0, 25, "<25"), (25, 35, "25-34"), (35, 50, "35-49"), (50, 200, "50+")]


def age_band(age) -> str:
    if pd.isna(age):
        return "unknown"
    return next(label for lo, hi, label in AGE_BANDS if lo <= age < hi)


def transaction_facts(db_dir: str) -> pd.DataFrame:
    """One row per transaction: label, split, amount, time, customer age band and region."""
    p = fetch_tables(db_dir)
    tx = pd.read_parquet(p["transactions"]).merge(pd.read_parquet(p["labels"]), on="transaction_id")
    tx = tx.merge(pd.read_parquet(p["customers"]), on="customer_id", how="left")
    tx["age_band"] = tx.customer_age.map(age_band)
    tx["region"] = tx.region.fillna("unknown")
    return tx.set_index("transaction_id")
