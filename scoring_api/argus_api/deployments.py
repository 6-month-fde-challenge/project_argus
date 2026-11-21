"""Model deployments (LLD 4.4 rollback): scorers of trained runs, loaded on demand and kept warm.

Any run whose registry decision produced a scorer (FP8 LLM student, or XGBoost on a HYBRID run) can be activated;
the last `capacity` loaded scorers stay in memory (LRU), so switching between e.g. the hybrid and the LLM-scorer
run is instant after the first load.
"""
from __future__ import annotations

import gc
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from .scorer import Scorer

RESERVED = {"mlruns", "_platform", "_scoring"}


class Deployments:
    def __init__(self, outputs_dir: Path, max_length: int, warmup: Callable[[Scorer], None], capacity: int = 2):
        self.outputs_dir, self.max_length, self.warmup, self.capacity = outputs_dir, max_length, warmup, capacity
        self._loaded: OrderedDict[str, Scorer] = OrderedDict()
        self.active: str | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _kind(run_dir: Path) -> str | None:
        if not (run_dir / "pilot" / "selected.json").exists():
            return None
        if (run_dir / "scorer_student" / "fp8").exists() and (run_dir / "calibration" / "scorer_student_fp8.json").exists():
            return "llm"
        if (run_dir / "xgboost" / "model.json").exists() and (run_dir / "calibration" / "xgboost.json").exists():
            return "xgboost"
        return None

    def available(self) -> list[dict]:
        if not self.outputs_dir.exists():
            return []
        out = []
        for d in sorted(self.outputs_dir.iterdir()):
            if d.is_dir() and d.name not in RESERVED and (kind := self._kind(d)):
                out.append({"run": d.name, "scorer": kind, "loaded": d.name in self._loaded, "active": d.name == self.active})
        return out

    def activate(self, run: str) -> Scorer:
        run_dir = self.outputs_dir / run
        if run in RESERVED or not run_dir.is_dir() or self._kind(run_dir) is None:
            raise KeyError(run)
        with self._lock:
            if run not in self._loaded:
                scorer = Scorer(run_dir, self.max_length)
                self.warmup(scorer)
                self._loaded[run] = scorer
                while len(self._loaded) > self.capacity:  # evict the least recently used scorer
                    evicted, _ = self._loaded.popitem(last=False)
                    if evicted == self.active:
                        self.active = None
                    gc.collect()
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001 - CPU-only environments
                        pass
            self._loaded.move_to_end(run)
            self.active = run
            return self._loaded[run]

    @property
    def scorer(self) -> Scorer | None:
        return self._loaded.get(self.active) if self.active else None
