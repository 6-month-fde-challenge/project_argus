"""Feature store stand-in (LLD 3.1 Redis) and the request-only fallback (LLD 6.1 FEATURES_DEGRADED).

Known transaction ids are served from the offline snapshot (Supabase `features` table + the serialised prompts the
training pipeline used): that is "a feature store hit". Any other transaction is scored from request fields only,
with history features `unk`, exactly the degraded path the LLD prescribes for a Redis timeout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from argus_ft import data as D
from argus_ft.stages.xgboost_baseline import FEATURES

from .schemas import TransactionIn

IST = timezone(timedelta(hours=5, minutes=30))
HIGH_RISK_MCC = {"electronics", "gaming", "gift_cards"}  # FE-08 / R08
DIST_BUCKETS = [(5, "<5"), (50, "5-50"), (200, "50-200"), (500, "200-500"), (1000, "500-1000"), (math.inf, "1000+")]


@dataclass
class FeatureVector:
    values: dict              # FE-01..FE-10 (+ data-quality flags), NaN where unknown
    prompt: str               # fmt-v1 serialisation (CRD 8.2)
    source: str               # "SNAPSHOT" (feature store hit) or "FEATURES_DEGRADED" (request fields only)
    flags: dict[str, bool] = field(default_factory=dict)  # reason code -> supporting feature flag (LLD 2.3)


def reason_flags(v: dict, mcc: str) -> dict[str, bool]:
    """The rule behind each reason code (derived from the dataset: exact on all 110k transactions)."""
    def yes(key):
        x = v.get(key)
        return x is not None and not (isinstance(x, float) and math.isnan(x)) and bool(x)

    def num(key):
        x = v.get(key)
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)

    dist, ratio, velocity = num("customer_merchant_distance_km"), num("amount_to_avg_ratio"), num("txn_count_last_1h")
    return {
        "R01_NEW_DEVICE": yes("is_new_device"),
        "R02_GEO_MISMATCH": (dist is not None and dist > 500) or yes("is_new_city"),
        "R03_ODD_HOUR": yes("is_night"),
        "R04_FAILED_ATTEMPTS": yes("has_recent_failures"),
        "R05_AMOUNT_SPIKE": ratio is not None and ratio > 5,
        "R06_NEW_ACCOUNT": yes("is_new_account"),
        "R07_VELOCITY": velocity is not None and velocity >= 4,
        "R08_HIGH_RISK_MCC": mcc in HIGH_RISK_MCC,
    }


def haversine_km(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def _two_sig(x: float) -> str:
    return f"{float(f'{x:.2g}'):.1f}"


class FeatureStore:
    def __init__(self, data_dir: str, db_dir: str):
        feats = pd.read_parquet(Path(db_dir) / "features.parquet")
        self.features = feats.set_index("transaction_id").to_dict("index")
        self.prompts: dict[str, tuple[str, str]] = {}  # transaction_id -> (fmt-v1 prompt, mcc)
        for split in ("train", "validation", "holdout"):
            for r in D.load_sft(data_dir, split):
                self.prompts[r["transaction_id"]] = (r["prompt"], r["prompt"].split(" mcc=")[1].split()[0])

    def __len__(self) -> int:
        return len(self.prompts)

    def lookup(self, txn: TransactionIn) -> FeatureVector:
        if txn.transaction_id in self.prompts and txn.transaction_id in self.features:
            prompt, mcc = self.prompts[txn.transaction_id]
            row = self.features[txn.transaction_id]
            values = {k: (np.nan if row.get(k) is None else row.get(k)) for k in FEATURES}
            return FeatureVector(values, prompt, "SNAPSHOT", reason_flags(row, mcc))
        return self.request_only(txn)

    @staticmethod
    def request_only(txn: TransactionIn) -> FeatureVector:
        """Degraded features: everything that needs customer history is unknown (`unk` / NaN)."""
        t = txn.transaction_time.astimezone(IST)
        dist = haversine_km(txn.customer_location, txn.merchant_location) if txn.customer_location else None
        values = {k: np.nan for k in FEATURES}
        values.update({
            "hour_of_day": t.hour, "is_night": t.hour < 5,
            "customer_merchant_distance_km": dist if dist is not None else np.nan,
            "is_new_account": txn.account_age < 30,
            "missing_customer_location": txn.customer_location is None, "missing_device": txn.device_id is None,
        })
        bucket = "unk" if dist is None else next(label for limit, label in DIST_BUCKETS if dist < limit)
        city = txn.customer_location.city if txn.customer_location else "unk"
        prompt = "\n".join([
            f"[TXN] id={txn.transaction_id} amt={_two_sig(txn.transaction_amount)} mcc={txn.merchant_category} "
            f"channel=unk time={t.hour:02d}h {t.strftime('%a')}",
            f"[CUST] age={txn.customer_age} acct_age_days={min(txn.account_age, 999)} avg_90d=unk txns_30d=unk fails_24h=unk",
            f"[CTX] new_device=unk city={city} merchant_city={txn.merchant_location.city} new_city=unk dist_km={bucket} "
            f"amt_ratio=unk txns_1h=unk amount_24h=unk device_users_7d=unk",
            f"[HIST] {D.NO_HISTORY}",
            "[TASK] Assess fraud risk using only the observed context.",
        ])
        return FeatureVector(values, prompt, "FEATURES_DEGRADED", reason_flags(values, txn.merchant_category))
