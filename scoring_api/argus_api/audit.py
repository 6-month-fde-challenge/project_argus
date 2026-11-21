"""Audit log (FR-API-03, LLD 3.3): one JSON line per decision, plus a line when an async explanation lands.

Stand-in for Kinesis -> S3 Parquet. Prompts are stored only as a SHA-256 hash plus the feature vector (LLD 6.2).
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path


def _clean(v):
    return None if isinstance(v, float) and math.isnan(v) else v


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _append(self, record: dict) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"logged_at": datetime.now(timezone.utc).isoformat(), **record}, default=str) + "\n")

    def decision(self, request: dict, response: dict, *, feature_vector: dict | None, prompt: str | None,
                 prompt_format: str, fraud_logit: float | None, feature_source: str | None, versions: dict,
                 warnings: list[str]) -> None:
        self._append({
            "type": "decision", "transaction_id": response["transaction_id"], "request": request,
            "feature_vector": {k: _clean(v) for k, v in (feature_vector or {}).items()},
            "feature_source": feature_source, "prompt_format": prompt_format,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest() if prompt else None,
            "fraud_logit": fraud_logit, **response, **versions, "warnings": warnings,
        })

    def explanation(self, transaction_id: str, explanation: str, source: str, latency_ms: int) -> None:
        self._append({"type": "explanation", "transaction_id": transaction_id, "explanation": explanation,
                      "explanation_source": source, "explanation_latency_ms": latency_ms})

    def recent(self, limit: int) -> list[dict]:
        """Newest decisions first, each with its async explanation merged in (reads only the file's tail)."""
        if not self.path.exists():
            return []
        with self.path.open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4_000_000))  # the tail is enough for the latest few hundred decisions
            lines = f.read().decode("utf-8", errors="replace").splitlines()[1:]
        decisions, explanations = [], {}
        for line in lines:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "decision":
                decisions.append(r)
            elif r.get("type") == "explanation":
                explanations[r["transaction_id"]] = r
        out = []
        for d in reversed(decisions[-limit:]):
            e = explanations.get(d["transaction_id"])
            if e and d.get("explanation") is None:
                d = {**d, **{k: e[k] for k in ("explanation", "explanation_source", "explanation_latency_ms")}}
            out.append({k: d.get(k) for k in ("logged_at", "transaction_id", "fraud_probability", "decision", "risk_level",
                                              "reason_codes", "explanation", "explanation_source", "decision_source",
                                              "model_version", "latency_ms", "feature_source", "warnings")})
        return out

    def lookup(self, transaction_id: str) -> dict | None:
        """Latest decision for a transaction, with any later async explanation merged in (admin support lookup)."""
        if not self.path.exists():
            return None
        decision, explanation = None, None  # both are written after the reply, in either order
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["transaction_id"] == transaction_id:
                    if r["type"] == "decision":
                        decision = r
                    else:
                        explanation = r
        if decision and explanation and decision.get("explanation") is None:
            decision.update({k: explanation[k] for k in ("explanation", "explanation_source", "explanation_latency_ms")})
        return decision
