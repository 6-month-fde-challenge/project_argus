"""Audit trail of every change made through the UI: settings, profiles, operations, model activation, AWS.

training/outputs/_platform/changes.jsonl, one JSON object per line, newest last. Secrets are never written: a changed
secret is recorded as "(changed)". Rules & policy keep their own log in the scoring API (FR-RE-01).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .fs import adopt, mkdirs, owner


class ChangeLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def add(self, kind: str, summary: str, details: dict | None = None, approval: dict | None = None) -> dict:
        entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind, "summary": summary,
                 "details": details or {}, **(approval or {})}
        with self._lock:
            mkdirs(self.path.parent)
            new = not self.path.exists()
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            if new:
                adopt(self.path, owner(self.path.parent))  # the VM user can read / rotate it
        return entry

    def recent(self, limit: int = 100, kind: str | None = None) -> list[dict]:
        """Newest first; a damaged line is skipped, never fatal."""
        if not self.path.exists():
            return []
        out = []
        for line in reversed(self.path.read_text(encoding="utf-8", errors="replace").splitlines()):
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if kind is None or entry.get("kind") == kind:
                out.append(entry)
                if len(out) >= limit:
                    break
        return out
