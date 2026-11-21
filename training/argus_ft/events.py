"""Training events for live progress in the ARGUS platform UI.

Every run appends JSON lines to outputs/<run>/events.jsonl (contract: argus_solution/README.md section 10.1):
    {"ts": "...Z", "type": "train_step", "stage": "cpt", "data": {"step": 10, "loss": 2.31, ...}}
The pipeline (parent) writes run/stage boundaries; stage processes write steps and progress. Writing an event
can never fail a run: errors are swallowed. Progress events are throttled so long loops stay cheap.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from transformers import TrainerCallback

PROGRESS_EVERY_S = 1.0  # at most one progress event per second per loop


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


class EventLog:
    def __init__(self, run_dir: Path, stage: str | None = None):
        self.path, self.stage = Path(run_dir) / "events.jsonl", stage

    def emit(self, type_: str, stage: str | None = None, **data) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                  "type": type_, "stage": stage or self.stage, "data": {k: _clean(v) for k, v in data.items()}}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:  # one short append per event: safe across processes
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass

    def progress(self, label: str):
        """A throttled `progress(done, total)` callable for long loops (scoring, generation, DPO)."""
        last = [0.0]

        def report(done: int, total: int) -> None:
            now = time.monotonic()
            if done >= total or now - last[0] >= PROGRESS_EVERY_S:
                last[0] = now
                self.emit("progress", label=label, done=done, total=total)
        return report

    def trainer_callback(self) -> "EventCallback":
        return EventCallback(self)


class EventCallback(TrainerCallback):
    """Hugging Face Trainer -> events: loss / learning rate at every log step, a progress tick at every step."""

    def __init__(self, events: EventLog):
        self.events = events
        self.tick = events.progress("training")

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" in logs or "eval_loss" in logs:
            self.events.emit("train_step", step=state.global_step, max_steps=state.max_steps, epoch=state.epoch,
                             loss=logs.get("loss", logs.get("eval_loss")), learning_rate=logs.get("learning_rate"),
                             grad_norm=logs.get("grad_norm"), eval="eval_loss" in logs)

    def on_step_end(self, args, state, control, **kwargs):
        self.tick(state.global_step, state.max_steps)
