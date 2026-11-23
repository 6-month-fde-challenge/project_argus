"""Model deployments (LLD 4.4 rollback): scorers of trained runs, loaded on demand and kept warm.

Any run whose registry decision produced a scorer (FP8 LLM student, or XGBoost on a HYBRID run) can be activated;
the last `capacity` loaded scorers stay in memory (LRU), so switching between e.g. the hybrid and the LLM-scorer
run is instant after the first load. A shadow run (MLOPS-06) is designated by an admin, shares that cache and never
evicts the active scorer; scoring with it never loads anything. A run that forced the week-8 branch is a demo: with
`api.allow_forced_runs: false` (production) it can be shadowed for comparison but never activated.
"""
from __future__ import annotations

import gc
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from .runs import ForcedRunRefused, forced, forced_reason, run_max_length
from .scorer import Scorer

RESERVED = {"mlruns", "_platform", "_scoring"}


class Deployments:
    def __init__(self, outputs_dir: Path, max_length: int, warmup: Callable[[Scorer], None], capacity: int = 2,
                 allow_forced: bool = True):
        self.outputs_dir, self.max_length, self.warmup, self.capacity = outputs_dir, max_length, warmup, capacity
        self.allow_forced = allow_forced  # False in production: a forced-branch (demo) run is never served live
        self._loaded: OrderedDict[str, Scorer] = OrderedDict()
        self.active: str | None = None
        self.shadow_run: str | None = None
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
                out.append({"run": d.name, "scorer": kind, "loaded": d.name in self._loaded, "active": d.name == self.active,
                            "shadow": d.name == self.shadow_run, "forced": forced(d), "forced_reason": forced_reason(d)})
        return out

    def _evict_for_one_more(self) -> None:
        """Free a slot BEFORE loading another scorer (never more than `capacity` on the GPU), evicting the least
        recently used one that is not live right now. Call with the lock held."""
        while len(self._loaded) >= self.capacity:
            victim = next((r for r in self._loaded if r != self.active), None)
            if victim is None:  # only the live scorer is loaded (capacity 1): it stays until the switch
                return
            del self._loaded[victim]
            if victim == self.shadow_run:
                self.shadow_run = None  # its shadow slot was needed: shadowing stops
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - CPU-only environments
                pass

    def _load(self, run: str) -> Scorer:
        """Load (or reuse) a run's scorer; call with the lock held. The live scorer is never evicted here, so the
        service always has a model until `activate` swaps to the new one in a single assignment."""
        run_dir = self.outputs_dir / run
        if run in RESERVED or not run_dir.is_dir() or self._kind(run_dir) is None:
            raise KeyError(run)
        if run not in self._loaded:
            self._evict_for_one_more()
            scorer = Scorer(run_dir, run_max_length(run_dir, self.max_length))
            self.warmup(scorer)
            self._loaded[run] = scorer
        self._loaded.move_to_end(run)
        return self._loaded[run]

    def activate(self, run: str) -> Scorer:
        if not self.allow_forced and (why := forced_reason(self.outputs_dir / run)):
            raise ForcedRunRefused(f"run '{run}' is not a measured run ({why}): this profile serves measured runs only")
        with self._lock:
            scorer = self._load(run)
            self.active = run  # one assignment: requests see either the old scorer or the new one, never none
            if self.shadow_run == run:
                self.shadow_run = None  # the shadow was promoted
            while len(self._loaded) > self.capacity:  # capacity 1: the previous live scorer goes only now
                del self._loaded[next(r for r in self._loaded if r != run)]
            return scorer

    def set_shadow(self, run: str | None) -> Scorer | None:
        """Admin: load a run as the shadow (MLOPS-06) - kept warm next to the active scorer, never made active and
        never evicting it; None turns shadow scoring off."""
        with self._lock:
            if run is None:
                self.shadow_run = None
                return None
            if run == self.active:
                raise ValueError("the active run cannot shadow itself")
            scorer = self._load(run)
            self.shadow_run = run
            return scorer

    def shadow_scorer(self, run: str) -> Scorer | None:
        """The designated shadow scorer if `run` is it - never loads anything (the scoring path stays lock-free)."""
        return self._loaded.get(run) if run == self.shadow_run else None

    @property
    def scorer(self) -> Scorer | None:
        return self._loaded.get(self.active) if self.active else None
