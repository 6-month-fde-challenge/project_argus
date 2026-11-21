"""Read-only access to training runs on disk: outputs/<run>/{report.json, report.md, events.jsonl, ...}.

JSON files are cached by (path, mtime), so a dashboard polling many runs re-parses a report only when the
pipeline has rewritten it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from ..core.errors import NotFound
from .events import read_events, tail_events

RESERVED = {"mlruns"}  # folders under outputs/ that are not runs, besides _service (_platform, _scoring, _logs) and .hidden ones


def is_reserved(name: str) -> bool:
    return name in RESERVED or name.startswith(("_", "."))


@lru_cache(maxsize=256)
def _load_json(path: str, mtime: float) -> dict:  # mtime is part of the cache key
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json(path: Path) -> dict | None:
    try:
        return _load_json(str(path), path.stat().st_mtime)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


class RunRepository:
    def __init__(self, outputs_dir: Path):
        self.root = outputs_dir

    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(d.name for d in self.root.iterdir()
                      if d.is_dir() and not is_reserved(d.name)
                      and ((d / "report.json").exists() or (d / "events.jsonl").exists()))

    def dir(self, run: str) -> Path:
        d = (self.root / run).resolve()
        if d.parent != self.root.resolve() or is_reserved(run) or not d.is_dir():  # no path traversal
            raise NotFound(f"run '{run}' not found")
        return d

    def report(self, run: str) -> dict:
        return load_json(self.dir(run) / "report.json") or {"stages": {}}

    def report_md(self, run: str) -> str | None:
        f = self.dir(run) / "report.md"
        return f.read_text(encoding="utf-8") if f.exists() else None

    def json_file(self, run: str, rel: str) -> dict | None:
        return load_json(self.dir(run) / rel)

    def events_path(self, run: str) -> Path:
        return self.dir(run) / "events.jsonl"

    def append_event(self, run: str, type_: str, **data) -> None:
        """Append one event (same contract as the pipeline's), creating the run folder for a brand-new run.
        `run` must already be a validated run name (the job service checks it)."""
        if is_reserved(run) or "/" in run or "\\" in run or run in ("", ".", ".."):
            raise NotFound(f"run '{run}' not found")
        d = self.root / run
        d.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with (d / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": ts, "type": type_, "stage": None, "data": data}) + "\n")

    def events(self, run: str, after: int = 0, limit: int | None = None) -> tuple[list[dict], int]:
        return read_events(self.events_path(run), after, limit)

    def tail(self, run: str, n: int) -> tuple[list[dict], int]:
        return tail_events(self.events_path(run), n)

    def modified(self, run: str) -> float:
        d = self.dir(run)
        return max((p.stat().st_mtime for p in (d / "report.json", d / "events.jsonl") if p.exists()), default=0.0)

    def artifacts(self, run: str) -> list[dict]:
        """Top-level artefact folders/files with sizes (models, adapters, predictions, reports)."""
        out = []
        for p in sorted(self.dir(run).iterdir()):
            if p.name in ("trainer",):
                continue
            files = [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else [p]
            out.append({"name": p.name, "kind": "dir" if p.is_dir() else "file", "files": len(files),
                        "size_mb": round(sum(f.stat().st_size for f in files) / 1e6, 2)})
        return out

    def file_text(self, run: str, rel: str, max_bytes: int = 2_000_000) -> str:
        f = (self.dir(run) / rel).resolve()
        if not str(f).startswith(str(self.dir(run))) or not f.is_file():
            raise NotFound(f"file '{rel}' not found in run '{run}'")
        return f.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
