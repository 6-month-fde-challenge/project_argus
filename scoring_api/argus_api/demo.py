"""Demo client: drives the scoring API through the scenarios in CRD Section 10 / LLD Section 6.

  docker compose run --rm trainer python -m argus_api.demo --run argus_demo_llm_scorer

Scenarios: the CRD 10.3 worked example (new transaction, request fields only), a known fraud and a known genuine
validation transaction (feature store hit), a blocklisted device (hard rule), stricter gaming thresholds with
two-person approval and an allowlisted customer capped at REVIEW (admin API), an invalid request (400), a missing
token (401), health, and the support lookup with the asynchronous explanation from the audit log.
Results are written to outputs/<run>/api/demo_results.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import pandas as pd
import requests

from argus_ft.config import load_config

from .auth import apply_token_env

CRD_EXAMPLE = {  # CRD 10.2 sample request (transaction not in the dataset -> FEATURES_DEGRADED path)
    "transaction_id": "TX98231", "customer_id": "C_7f31a9", "transaction_amount": 50000.00,
    "merchant_category": "electronics", "transaction_time": "2025-11-14T03:12:44+05:30", "device_id": "D_44be20",
    "customer_age": 27, "customer_location": {"lat": 18.52, "lon": 73.85, "city": "Pune"},
    "merchant_location": {"lat": 28.61, "lon": 77.20, "city": "Delhi"}, "account_age": 6,
}


def to_request(row: pd.Series) -> dict:
    """A dataset row as a CRD 10.2 request (history fields are left to the feature store)."""
    loc = lambda s: json.loads(s) if isinstance(s, str) else None
    return {
        "transaction_id": row.transaction_id, "customer_id": row.customer_id,
        "transaction_amount": round(float(row.transaction_amount), 2), "merchant_category": row.merchant_category,
        "transaction_time": row.transaction_time.isoformat(), "device_id": row.device_id if isinstance(row.device_id, str) else None,
        "customer_age": int(row.customer_age), "customer_location": loc(row.customer_location),
        "merchant_location": loc(row.merchant_location), "account_age": int(row.account_age),
    }


class Client:
    def __init__(self, url: str, token: str, admin_token: str):
        self.url, self.h, self.admin = url.rstrip("/"), {"Authorization": f"Bearer {token}"}, {"Authorization": f"Bearer {admin_token}"}

    def score(self, req: dict, headers: dict | None = None) -> tuple[int, dict]:
        r = requests.post(f"{self.url}/v1/fraud/score", json=req, headers=self.h if headers is None else headers, timeout=60)
        return r.status_code, r.json()

    def get(self, path: str, admin: bool = True):
        return requests.get(f"{self.url}{path}", headers=self.admin if admin else self.h, timeout=30).json()

    def send(self, method: str, path: str, body: dict):
        return requests.request(method, f"{self.url}{path}", json=body, headers=self.admin, timeout=30).json()


def wait_until_up(url: str, timeout_s: int = 600) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/v1/health", timeout=5).ok:
                return
        except requests.ConnectionError:
            pass
        time.sleep(5)
    raise TimeoutError(f"API at {url} not up after {timeout_s}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local_demo.yaml")
    ap.add_argument("--run", required=True, help="run whose models the API serves (for the results file)")
    ap.add_argument("--url", default="http://scoring:8080")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tokens = cfg.api
    apply_token_env(tokens)  # the same tokens the server uses
    api = Client(args.url, tokens.token, tokens.admin_token)
    wait_until_up(args.url)

    val = pd.concat(pd.read_parquet(p) for p in glob.glob(f"{cfg.data_dir}/parquet/transactions/validation/*.parquet"))
    rules = api.get("/admin/v1/lists/blocked_devices")["blocked_devices"]
    frauds = val[(val.fraud_label == 1) & val.device_id.notna() & ~val.device_id.isin(rules)].head(30)
    genuine = val[(val.fraud_label == 0) & val.customer_location.notna()].iloc[0]
    gaming = val[(val.merchant_category == "gaming") & val.device_id.notna() & ~val.device_id.isin(rules)].head(30)
    results: list[dict] = []

    def record(name: str, what: str, status: int, body):
        results.append({"scenario": name, "what_it_shows": what, "http_status": status, "response": body})
        print(f"\n### {name} - {what}\nHTTP {status}\n{json.dumps(body, indent=2)}")

    record("health", "GET /v1/health (LLD 4.2)", 200, api.get("/v1/health", admin=False))
    record("crd_example", "CRD 10.2 sample: new transaction, scored from request fields only (FEATURES_DEGRADED)",
           *api.score(CRD_EXAMPLE))
    scored = [api.score(to_request(row)) for _, row in frauds.iterrows()]  # like a reviewer: top of the queue
    status, riskiest = max(scored, key=lambda s: s[1].get("fraud_probability") or 0)
    record("known_fraud", f"riskiest of {len(scored)} validation frauds: feature store hit, REVIEW/BLOCK path", status, riskiest)
    record("known_genuine", "validation transaction labelled genuine", *api.score(to_request(genuine)))
    record("hard_rule", "blocklisted device -> BLOCK before the model (FR-RE-02, decision_source HARD_RULE)",
           *api.score({**to_request(genuine), "transaction_id": "TX_DEMO_BLOCKED", "device_id": rules[0]}))

    # FR-RE-01 / 03 / 04: stricter gaming thresholds (two-person approval), allowlisted customer capped at REVIEW
    gaming_rows = [to_request(row) for _, row in gaming.iterrows()]
    base = [(req, api.score(req)[1]) for req in gaming_rows]
    gaming_req, before = max(base, key=lambda b: b[1].get("fraud_probability") or 0)
    p = before["fraud_probability"]
    record("gaming_default", "gaming transaction under the default thresholds (0.30 / 0.60 / 0.85)", 200, before)
    strict = {"medium": round(p * 0.25, 6), "review": round(p * 0.5, 6), "block": round(p * 0.9, 6), "mcc": "gaming"}
    record("admin_thresholds_rejected", "same person cannot approve their own change (two-person rule)", 400,
           api.send("POST", "/admin/v1/thresholds", {**strict, "requested_by": "karthik", "approved_by": "karthik", "reason": "demo"}))
    record("admin_thresholds", "stricter thresholds for gaming only (FR-RE-04), logged with who/why (FR-RE-01)", 200,
           api.send("POST", "/admin/v1/thresholds", {**strict, "requested_by": "karthik", "approved_by": "ananya",
                                                     "reason": "demo: stricter gaming policy"}))
    record("gaming_strict", "same transaction under the stricter gaming thresholds -> BLOCK", *api.score(gaming_req))
    api.send("POST", "/admin/v1/lists/allowlisted_customers", {"item": gaming_req["customer_id"], "by": "karthik", "reason": "demo: trusted payer"})
    record("allowlist_cap", "same transaction, customer now allowlisted -> capped at REVIEW (FR-RE-03)", *api.score(gaming_req))
    api.send("DELETE", "/admin/v1/lists/allowlisted_customers", {"item": gaming_req["customer_id"], "by": "karthik", "reason": "demo cleanup"})
    api.send("POST", "/admin/v1/thresholds", {"medium": 0.3, "review": 0.6, "block": 0.85, "mcc": "gaming",
                                              "requested_by": "karthik", "approved_by": "ananya", "reason": "demo: restore"})

    bad = {**CRD_EXAMPLE, "transaction_id": "TX_BAD", "transaction_amount": -5}
    record("invalid_request", "negative amount -> 400 with the field (FR-API-02)", *api.score(bad))
    record("unauthorised", "missing service token -> 401", *api.score(CRD_EXAMPLE, headers={}))

    time.sleep(cfg.api.async_explain_timeout_s + 1)  # FR-API-05: async explanations reach the audit log within 5 s
    record("support_lookup", "GET /admin/v1/decisions/{id}: decision + async explanation from the audit log", 200,
           api.get(f"/admin/v1/decisions/{genuine.transaction_id}"))
    record("changelog", "every config change with who and why (FR-RE-01)", 200, api.get("/admin/v1/changelog"))

    out = Path(cfg.output_dir) / args.run / "api" / "demo_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
