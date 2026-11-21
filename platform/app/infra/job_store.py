"""Job registry (contract: README.md section 10.2): outputs/_platform/jobs.json, newest first.

Small and append-mostly, so a JSON file with an in-process lock and atomic replace is enough; insertion order is
the FIFO queue order.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..core.errors import NotFound

ACTIVE = {"queued", "running"}
TERMINAL = {"succeeded", "failed", "cancelled"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    def _read(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write(self, jobs: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic: readers never see a half-written file

    def all(self) -> list[dict]:
        with self._lock:
            return self._read()

    def get(self, job_id: str) -> dict:
        job = next((j for j in self.all() if j["id"] == job_id), None)
        if job is None:
            raise NotFound(f"job '{job_id}' not found")
        return job

    def create(self, **fields) -> dict:
        with self._lock:
            job = {"id": uuid.uuid4().hex[:10], "status": "queued", "created_at": now(), "started_at": None,
                   "finished_at": None, "container_id": None, "exit_code": None, "message": None, **fields}
            self._write([job] + self._read())
            return job

    def update(self, job_id: str, **fields) -> dict:
        with self._lock:
            jobs = self._read()
            for j in jobs:
                if j["id"] == job_id:
                    j.update(fields)
                    if fields.get("status") in TERMINAL and not j.get("finished_at"):
                        j["finished_at"] = now()
                    self._write(jobs)
                    return j
            raise NotFound(f"job '{job_id}' not found")

    def active(self) -> list[dict]:
        return [j for j in self.all() if j["status"] in ACTIVE]

    def next_queued(self) -> dict | None:
        queued = [j for j in self.all() if j["status"] == "queued"]
        return queued[-1] if queued else None  # oldest first (list is newest first)

    def latest_for_run(self, run: str) -> dict | None:
        return next((j for j in self.all() if j["run_name"] == run), None)
