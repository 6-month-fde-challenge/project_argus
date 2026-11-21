"""Live run state folded incrementally from events.jsonl.

Each run keeps (byte offset, folded state); a refresh reads only the bytes appended since the last call, so
polling a dashboard of many runs costs O(new events), not O(history). Runs started from the CLI (no platform job)
are detected the same way, from their recent event activity.
"""
from __future__ import annotations

import threading
import time
from copy import deepcopy

from ...infra.runs_repo import RunRepository

# no event for this long and no run_end -> a CLI-started run is considered interrupted. Generous because some
# phases are silent for minutes (first model download, AWQ calibration); platform jobs are tracked by container.
STALE_AFTER_S = 900


def _empty() -> dict:
    return {"status": "idle", "planned": [], "current_stage": None, "stages": {}, "run_start": None,
            "run_end": None, "last_event_ts": None, "profile": None}


def fold(state: dict, ev: dict) -> None:
    """Apply one event to the state (pure, unit-tested)."""
    t, stage, d = ev.get("type"), ev.get("stage"), ev.get("data") or {}
    state["last_event_ts"] = ev.get("ts")
    st = state["stages"].setdefault(stage, {}) if stage else None
    if t == "job_queued":  # written by the platform when a job is created, before its container starts
        if state["status"] == "idle":
            state.update(planned=d.get("stages", []), profile=d.get("profile"))
    elif t == "run_start":
        state.update(status="running", planned=d.get("stages", []), run_start=ev.get("ts"), run_end=None,
                     current_stage=None, profile=d.get("profile"))
        for s in d.get("stages", []):  # a new attempt resets only the stages it will run
            state["stages"][s] = {"status": "pending"}
    elif t == "stage_start":
        state["current_stage"] = stage
        st.update(status="running", started=ev.get("ts"), progress=None, last_step=None)
    elif t == "stage_end":
        st.update(status="ok" if d.get("status") == "ok" else "failed", ended=ev.get("ts"),
                  minutes=d.get("minutes"), peak_gpu_gb=d.get("peak_gpu_gb"), error=d.get("error"))
        if state["current_stage"] == stage:
            state["current_stage"] = None
    elif t == "stage_skipped":
        st.update(status="skipped", reason=d.get("reason"))
    elif t == "progress" and st is not None:
        st["progress"] = {"label": d.get("label"), "done": d.get("done"), "total": d.get("total"), "ts": ev.get("ts")}
    elif t == "train_step" and st is not None:
        st["last_step"] = {k: d.get(k) for k in ("step", "max_steps", "loss", "learning_rate", "epoch")}
        st["steps_logged"] = st.get("steps_logged", 0) + 1
    elif t == "metric" and st is not None:
        st.setdefault("metrics", {})[d.get("name")] = d.get("value")
    elif t == "run_end":
        state.update(status="succeeded" if d.get("status") == "ok" else "failed", run_end=ev.get("ts"),
                     current_stage=None, total_minutes=d.get("total_minutes"))


class LiveStateCache:
    def __init__(self, runs: RunRepository):
        self.runs = runs
        self._cache: dict[str, tuple[int, dict, float]] = {}  # run -> (offset, state, file mtime seen)
        self._lock = threading.Lock()

    def get(self, run: str) -> dict:
        path = self.runs.events_path(run)
        with self._lock:
            offset, state, _ = self._cache.get(run, (0, _empty(), 0.0))
            size = path.stat().st_size if path.exists() else 0
            if size < offset:  # file was replaced (run deleted and re-created): start over
                offset, state = 0, _empty()
            events, offset = self.runs.events(run, after=offset)
            for ev in events:
                fold(state, ev)
            mtime = path.stat().st_mtime if path.exists() else 0.0
            self._cache[run] = (offset, state, mtime)
            out = deepcopy(state)
        out["live"] = out["status"] == "running" and (time.time() - mtime) < STALE_AFTER_S
        if out["status"] == "running" and not out["live"]:
            out["status"] = "interrupted"  # started, never finished, silent for minutes
        return out
