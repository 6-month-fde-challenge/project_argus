"""Sample scoring requests for the playground: the CRD 10.2 worked example plus real validation transactions."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

CRD_EXAMPLE = {
    "transaction_id": "TX98231", "customer_id": "C_7f31a9", "transaction_amount": 50000.00,
    "merchant_category": "electronics", "transaction_time": "2025-11-14T03:12:44+05:30", "device_id": "D_44be20",
    "customer_age": 27, "customer_location": {"lat": 18.52, "lon": 73.85, "city": "Pune"},
    "merchant_location": {"lat": 28.61, "lon": 77.20, "city": "Delhi"}, "account_age": 6,
}


def _request(row) -> dict:
    loc = lambda s: json.loads(s) if isinstance(s, str) else None
    return {
        "transaction_id": row.transaction_id, "customer_id": row.customer_id,
        "transaction_amount": round(float(row.transaction_amount), 2), "merchant_category": row.merchant_category,
        "transaction_time": row.transaction_time.isoformat(),
        "device_id": row.device_id if isinstance(row.device_id, str) else None, "customer_age": int(row.customer_age),
        "customer_location": loc(row.customer_location), "merchant_location": loc(row.merchant_location),
        "account_age": int(row.account_age),
    }


@lru_cache(maxsize=1)
def _from_dataset(data_dir: str) -> list[dict]:
    import pandas as pd
    files = sorted(Path(data_dir, "raw", "parquet", "transactions", "validation").glob("*.parquet"))
    if not files:
        return []
    df = pd.concat(pd.read_parquet(f) for f in files)
    picks = [("Known fraud (validation)", df[df.fraud_label == 1].head(3)),
             ("Genuine payment (validation)", df[df.fraud_label == 0].head(3)),
             ("Night-time gaming", df[(df.merchant_category == "gaming") & (df.transaction_time.dt.hour < 2)].head(2))]
    out = []
    for label, rows in picks:
        for i, (_, row) in enumerate(rows.iterrows(), 1):
            out.append({"label": f"{label} #{i}", "request": _request(row),
                        "note": f"fraud_label={int(row.fraud_label)} (ground truth, not sent to the API)"})
    return out


def sample_requests(data_dir: Path) -> list[dict]:
    crd = {"label": "CRD 10.2 worked example (TX98231)", "request": CRD_EXAMPLE,
           "note": "not in the dataset: scored from request fields only (FEATURES_DEGRADED)"}
    invalid = {"label": "Invalid: negative amount", "request": {**CRD_EXAMPLE, "transaction_id": "TX_BAD", "transaction_amount": -5},
               "note": "expect HTTP 400 INVALID_REQUEST (FR-API-02)"}
    return [crd, *_from_dataset(str(data_dir)), invalid]
