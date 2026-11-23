"""Transaction sources for the producer. Every event carries a CRD 10.2 request plus ground truth, so the stream can
measure live recall / precision.

    SyntheticSource  customers, merchants and devices across Indian cities, with fraud rings that run account takeover,
                     card testing, mule accounts and impossible travel. New transaction ids, so the Fraud Model API
                     scores them from request fields only (FEATURES_DEGRADED, LLD 6.1).
    HoldoutSource    replays the labelled holdout split (parquet/transactions/holdout, from a folder or s3://...).
                     Known transaction ids, so the API serves them from its feature snapshot (a feature-store hit).
    MixSource        both, interleaved.

Event: {event_id, ts, source, fraud_label, fraud_type, ring_id, request: {CRD 10.2 fields}}.
transaction_time (IST): TRANSACTION_TIME=pattern (default) keeps today's date but moves synthetic events to their
pattern's hour (night-time takeovers, daytime shopping), which the local_demo profile accepts. Set
TRANSACTION_TIME=now when the API enforces the +-15 min window (LLD 4.1; prod_a100): every event is stamped now.
Holdout replays are always stamped now.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("argus.stream.generator")

IST = timezone(timedelta(hours=5, minutes=30))
CITIES = {  # city -> (lat, lon)
    "Mumbai": (19.08, 72.88), "Delhi": (28.61, 77.21), "Bengaluru": (12.97, 77.59), "Pune": (18.52, 73.86),
    "Hyderabad": (17.39, 78.49), "Chennai": (13.08, 80.27), "Kolkata": (22.57, 88.36), "Ahmedabad": (23.02, 72.57),
    "Jaipur": (26.91, 75.79), "Lucknow": (26.85, 80.95), "Patna": (25.59, 85.14), "Kochi": (9.93, 76.27),
}
AMOUNTS = {"grocery": (150, 2500), "quick_commerce": (100, 1500), "food": (120, 1800), "electronics": (2000, 60000),
           "gaming": (100, 5000), "gift_cards": (500, 10000), "healthcare": (200, 8000), "travel": (1500, 40000),
           "education": (1000, 30000), "other": (100, 5000)}
GENUINE_MIX = {"grocery": 26, "quick_commerce": 16, "food": 18, "electronics": 5, "gaming": 4, "gift_cards": 2,
               "healthcare": 7, "travel": 5, "education": 4, "other": 13}
FRAUD_TYPES = ("account_takeover", "card_testing", "mule_account", "impossible_travel")
HOURS = [1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 9, 9, 10, 9, 8, 8, 8, 9, 10, 10, 9, 7, 4, 2]  # genuine traffic by IST hour


def _location(city: str, rnd: random.Random) -> dict:
    lat, lon = CITIES[city]
    return {"lat": round(lat + rnd.uniform(-0.05, 0.05), 4), "lon": round(lon + rnd.uniform(-0.05, 0.05), 4), "city": city}


def _time_at_hour(hour: int | None) -> str:
    """Now in IST; with `hour` (and TRANSACTION_TIME=pattern), the same minute moved to that hour today."""
    now = datetime.now(IST)
    if hour is not None and os.getenv("TRANSACTION_TIME", "pattern") == "pattern":
        now = now.replace(hour=hour)
    return now.isoformat(timespec="seconds")


class SyntheticSource:
    def __init__(self, fraud_rate: float = 0.04, seed: int | None = None, customers: int = 2000, merchants: int = 300,
                 rings: int = 8):
        self.rnd = r = random.Random(seed)
        self.fraud_rate = fraud_rate
        cities = list(CITIES)
        self.customers = [{"customer_id": f"C_SYN{i:05d}", "age": r.randint(19, 72), "account_age": r.randint(45, 3650),
                           "city": r.choice(cities), "avg": round(math.exp(r.gauss(6.8, 0.7)), 2),
                           "device_id": f"D_SYN{i:05d}"} for i in range(customers)]
        self.merchants = []
        for i in range(merchants):
            mcc = r.choices(list(GENUINE_MIX), list(GENUINE_MIX.values()))[0]
            self.merchants.append({"merchant_id": f"M_SYN{i:04d}", "mcc": mcc, "city": r.choice(cities)})
        self.rings = [{"ring_id": f"RING_{i:02d}", "device_id": f"D_RING{i:02d}", "city": r.choice(cities),
                       "type": FRAUD_TYPES[i % len(FRAUD_TYPES)],
                       "mcc": r.choice(["electronics", "gift_cards", "gaming"])} for i in range(rings)]

    # -------------------------------------------------------------- events
    def _event(self, c: dict, mcc: str, merchant_city: str, amount: float, label: int, kind: str, *,
               hour: int | None = None, device_id: str | None = None, city: str | None = None,
               account_age: int | None = None, ring_id: str | None = None) -> dict:
        r = self.rnd
        request = {
            "transaction_id": f"TX_SYN_{uuid.uuid4().hex[:12].upper()}", "customer_id": c["customer_id"],
            "transaction_amount": round(min(max(amount, 1.0), 9_999_999.0), 2), "merchant_category": mcc,
            "transaction_time": _time_at_hour(hour), "device_id": device_id or c["device_id"],
            "customer_age": c["age"], "customer_location": _location(city or c["city"], r),
            "merchant_location": _location(merchant_city, r),
            "account_age": c["account_age"] if account_age is None else account_age,
        }
        return {"event_id": uuid.uuid4().hex, "ts": time.time(), "source": "synthetic", "fraud_label": label,
                "fraud_type": kind, "ring_id": ring_id, "request": request}

    def genuine(self) -> dict:
        r = self.rnd
        c = r.choice(self.customers)
        mcc = r.choices(list(GENUINE_MIX), list(GENUINE_MIX.values()))[0]
        local = [m for m in self.merchants if m["mcc"] == mcc and m["city"] == c["city"]]
        m = r.choice(local or [m for m in self.merchants if m["mcc"] == mcc] or self.merchants)
        lo, hi = AMOUNTS[mcc]
        amount = min(hi, max(lo, r.lognormvariate(math.log(max(lo, c["avg"])), 0.6)))
        travel = r.random() < 0.04  # benign anomalies keep the false-positive rate honest
        return self._event(c, mcc, r.choice(list(CITIES)) if travel else m["city"], amount, 0, "genuine",
                           hour=r.choices(range(24), HOURS)[0])

    def fraud(self, kind: str | None = None, ring: dict | None = None) -> dict:
        r = self.rnd
        ring = ring or r.choice(self.rings)
        kind = kind or ring["type"]
        c = r.choice(self.customers)
        common = {"device_id": ring["device_id"], "ring_id": ring["ring_id"]}
        far = r.choice([x for x in CITIES if x != c["city"]])
        if kind == "account_takeover":   # new device, far from home, at night, big electronics basket
            return self._event(c, "electronics", far, c["avg"] * r.uniform(8, 45), 1, kind,
                               hour=r.choice([0, 1, 2, 3, 4]), city=ring["city"], **common)
        if kind == "card_testing":       # many small payments at a risky merchant
            return self._event(c, ring["mcc"], ring["city"], r.uniform(1, 250), 1, kind,
                               hour=r.choice([1, 2, 3, 23]), **common)
        if kind == "mule_account":       # brand-new account moving large sums
            return self._event(c, "gift_cards", far, r.uniform(20000, 95000), 1, kind, account_age=r.randint(0, 20),
                               **common)
        return self._event(c, "travel", far, c["avg"] * r.uniform(3, 15), 1, "impossible_travel",  # far, minutes later
                           city=r.choice([x for x in CITIES if x not in (c["city"], far)]), **common)

    def next(self) -> dict:
        return self.fraud() if self.rnd.random() < self.fraud_rate else self.genuine()

    def burst(self, n: int = 20, kind: str | None = None) -> list[dict]:
        """An attack wave: one ring fires n fraud transactions."""
        ring = self.rnd.choice([g for g in self.rings if not kind or g["type"] == kind] or self.rings)
        return [self.fraud(kind or ring["type"], ring) for _ in range(n)]


# ------------------------------------------------------------------ holdout replay

def _sync_s3(uri: str, dest: Path) -> Path:
    from .bus import aws_client
    bucket, _, prefix = uri[len("s3://"):].partition("/")
    s3 = aws_client("s3")
    dest.mkdir(parents=True, exist_ok=True)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet") and not (dest / Path(obj["Key"]).name).exists():
                s3.download_file(bucket, obj["Key"], str(dest / Path(obj["Key"]).name))
    return dest


def _request(row) -> dict:
    """A holdout row as a CRD 10.2 request (history fields stay with the API's feature store)."""
    loc = lambda s: json.loads(s) if isinstance(s, str) else None  # noqa: E731
    return {"transaction_id": row["transaction_id"], "customer_id": row["customer_id"],
            "transaction_amount": round(float(row["transaction_amount"]), 2), "merchant_category": row["merchant_category"],
            "transaction_time": _time_at_hour(None), "device_id": row["device_id"] if isinstance(row["device_id"], str) else None,
            "customer_age": int(row["customer_age"]), "customer_location": loc(row["customer_location"]),
            "merchant_location": loc(row["merchant_location"]), "account_age": int(row["account_age"])}


class HoldoutSource:
    def __init__(self, location: str, seed: int | None = None):
        import pandas as pd
        path = _sync_s3(location, Path("/tmp/argus-holdout")) if location.startswith("s3://") else Path(location)
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no *.parquet under {location}")
        df = pd.concat(pd.read_parquet(f) for f in files)
        self.rows = df.to_dict("records")
        self.fraud = [r for r in self.rows if r["fraud_label"] == 1]
        self.rnd = random.Random(seed)
        log.info("holdout replay: %d transactions (%d fraud) from %s", len(self.rows), len(self.fraud), location)

    def _event(self, row: dict) -> dict:
        label = int(row["fraud_label"])
        return {"event_id": uuid.uuid4().hex, "ts": time.time(), "source": "holdout", "fraud_label": label,
                "fraud_type": "holdout_fraud" if label else "genuine", "ring_id": None, "request": _request(row)}

    def next(self) -> dict:
        return self._event(self.rnd.choice(self.rows))

    def burst(self, n: int = 20, kind: str | None = None) -> list[dict]:
        return [self._event(self.rnd.choice(self.fraud or self.rows)) for _ in range(n)]


class MixSource:
    def __init__(self, holdout: HoldoutSource, synthetic: SyntheticSource, holdout_share: float = 0.5):
        self.holdout, self.synthetic, self.share, self.rnd = holdout, synthetic, holdout_share, random.Random()

    def next(self) -> dict:
        return self.holdout.next() if self.rnd.random() < self.share else self.synthetic.next()

    def burst(self, n: int = 20, kind: str | None = None) -> list[dict]:
        return self.synthetic.burst(n, kind)


def make_source(kind: str | None = None, seed: int | None = None):
    """PRODUCER_SOURCE: synthetic | holdout | mix (holdout / mix read DATASET_URI). Falls back to synthetic when the
    holdout is not available, so traffic never stops (except for an explicit `holdout`, which fails loudly)."""
    kind = (kind or os.getenv("PRODUCER_SOURCE", "mix")).lower()
    synthetic = SyntheticSource(float(os.getenv("FRAUD_RATE", "0.04")), seed)
    uri = os.getenv("DATASET_URI", "")
    if kind in ("holdout", "mix") and uri:
        try:
            holdout = HoldoutSource(uri, seed)
            return holdout if kind == "holdout" else MixSource(holdout, synthetic, float(os.getenv("HOLDOUT_SHARE", "0.5")))
        except Exception as e:  # noqa: BLE001
            if kind == "holdout":
                raise
            log.warning("holdout %s not usable (%s): synthetic traffic only", uri, e)
    return synthetic
