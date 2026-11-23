"""Audit log (FR-API-03, LLD 3.3): one JSON line per decision, plus a line when an async explanation lands.

Stand-in for Kinesis -> S3 Parquet. Prompts are stored only as a SHA-256 hash plus the feature vector (LLD 6.2).
The file rotates at `max_bytes` (audit-<timestamp>.jsonl next to it); lookups search the current file, then the
rotated ones newest first. A corrupt line is skipped, never fatal. The 5-year archive (NFR-11) is S3 via Firehose
on AWS (deploy/terraform/storage.tf), not this file. Callers run reads in a worker thread (they scan files).
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


def _records(path: Path):
    """JSON records of one file; a line that does not parse (torn write) is skipped, never fatal."""
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


EXPLANATION_KEYS = ("explanation", "explanation_source", "explanation_latency_ms")


class AuditLog:
    def __init__(self, path: Path, max_bytes: int = 256 * 2**20):
        self.path, self.max_bytes = path, max_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def files(self) -> list[Path]:
        """The current file, then the rotated ones, newest first."""
        rotated = sorted(self.path.parent.glob(f"{self.path.stem}-*.jsonl"), reverse=True)
        return [p for p in [self.path, *rotated] if p.exists()]

    def _append(self, record: dict) -> None:
        line = json.dumps({"logged_at": datetime.now(timezone.utc).isoformat(), **record}, default=str) + "\n"
        with self._lock:
            if self.path.exists() and self.path.stat().st_size >= self.max_bytes:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                self.path.rename(self.path.with_name(f"{self.path.stem}-{stamp}.jsonl"))
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)

    def decision(self, request: dict, response: dict, *, feature_vector: dict | None, prompt: str | None,
                 prompt_format: str | None, fraud_logit: float | None, feature_source: str | None, versions: dict,
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
                explanations[r.get("transaction_id")] = r
        out = []
        for d in reversed(decisions[-limit:]):
            e = explanations.get(d.get("transaction_id"))
            if e and d.get("explanation") is None:
                d = {**d, **{k: e.get(k) for k in EXPLANATION_KEYS}}
            out.append({k: d.get(k) for k in ("logged_at", "transaction_id", "fraud_probability", "decision", "risk_level",
                                              "reason_codes", "explanation", "explanation_source", "decision_source",
                                              "model_version", "latency_ms", "feature_source", "warnings")})
        return out

    def lookup(self, transaction_id: str) -> dict | None:
        """Latest decision for a transaction with any later async explanation merged in (admin support lookup).
        Files are searched newest first and the search stops at the first file that holds a decision."""
        decision, explanation = None, None  # the two are written after the reply, in either order
        for path in self.files():
            found_decision = found_explanation = None
            for r in _records(path):
                if r.get("transaction_id") != transaction_id:
                    continue
                if r.get("type") == "decision":
                    found_decision = r       # within a file the last one is the latest
                elif r.get("type") == "explanation":
                    found_explanation = r
            explanation = explanation or found_explanation  # a newer file's explanation wins
            if found_decision:
                decision = found_decision
                break
        if decision and explanation and decision.get("explanation") is None:
            decision.update({k: explanation.get(k) for k in EXPLANATION_KEYS})
        return decision

    def changelog_lines(self, path: Path) -> list[dict]:
        """Every record of a JSON-lines file (the rules changelog), skipping corrupt lines."""
        return list(_records(path)) if path.exists() else []
