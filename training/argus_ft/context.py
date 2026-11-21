"""Shared state passed to every stage: config, output paths, cached data and saved predictions.

Every stage reads its inputs from and writes its outputs to `outputs/<run_name>/`, so any stage can be
re-run on its own as long as the stages before it have produced their artefacts.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from . import data as D
from .config import Config
from .events import EventLog
from .modeling import load_tokenizer


class Ctx:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.run_dir = Path(cfg.output_dir) / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events = EventLog(self.run_dir)  # live progress for the platform UI (stage set by the runner)
        self._sft: dict[str, list[dict]] = {}
        self._tokenizers: dict[str, object] = {}
        self._prompt_cache: dict[tuple, str] = {}  # (tokenizer, format, raw prompt) -> fitted prompt

    def path(self, *parts: str) -> Path:
        p = self.run_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------ models / tokenizers

    def _pilot_choice(self, key: str, default: str) -> str:
        selected = self.run_dir / "pilot" / "selected.json"
        return json.loads(selected.read_text())[key] if selected.exists() else default

    @property
    def teacher_model(self) -> str:
        """Base model chosen by the pilot (stage 2) if it has run, else `models.teacher` from the config."""
        return self._pilot_choice("teacher", self.cfg.models.teacher)

    @property
    def prompt_format(self) -> str:
        """Prompt format chosen by the pilot (FR-FE-07), else fmt-v1. Used for every later stage."""
        return self._pilot_choice("format", "fmt-v1")

    @property
    def teacher_base(self) -> str:
        """Weights the teacher stages start from: the CPT-merged model (stage 3) if it exists."""
        merged = self.run_dir / "cpt" / "merged"
        return str(merged) if (merged / "config.json").exists() else self.teacher_model

    @property
    def tokenizer(self):
        """Teacher tokenizer: fits prompts for the teacher scorer, explainer and DPO stages."""
        name = self.teacher_model
        if name not in self._tokenizers:
            self._tokenizers[name] = load_tokenizer(name)
        return self._tokenizers[name]

    # ------------------------------------------------------------ data

    def sft(self, split: str) -> list[dict]:
        """SFT rows sorted by transaction id (= time). Validation can be capped by eval.max_rows: a seeded random
        sample, so the subset still spans the whole validation period (the calibration stage splits it by time)."""
        if split not in self._sft:
            rows = D.load_sft(self.cfg.data_dir, split)
            n = self.cfg.eval.get("max_rows")
            if split != "train" and n and n < len(rows):
                rows = random.Random(self.cfg.seed).sample(rows, n)
            self._sft[split] = sorted(rows, key=lambda r: r["transaction_id"])
        return self._sft[split]

    def fitted(self, rows: list[dict], tokenizer=None, fmt: str | None = None) -> list[dict]:
        """Copies of `rows` with the prompt converted to `fmt` (default: the pilot's choice) and trimmed
        (oldest history first) to `max_length` tokens of `tokenizer` (default: the teacher's)."""
        tok, fmt = tokenizer or self.tokenizer, fmt or self.prompt_format
        key = lambda prompt: (tok.name_or_path, fmt, prompt)
        todo = list({r["prompt"] for r in rows if key(r["prompt"]) not in self._prompt_cache})
        if todo:  # everything may already be cached (e.g. the pilot fitted the same rows)
            converted = [D.to_format(p, fmt) for p in todo]
            for raw, fitted in zip(todo, D.fit_prompts(tok, converted, self.cfg.max_length)):
                self._prompt_cache[key(raw)] = fitted
        return [{**r, "prompt": self._prompt_cache[key(r["prompt"])]} for r in rows]

    def scorer_train_rows(self, stage: str) -> list[dict]:
        c = self.cfg[stage]
        return D.downsample_negatives(self.sft("train"), c["neg_per_pos"], self.cfg.seed + c.get("seed_offset", 0))

    # ------------------------------------------------------------ artefacts

    def save_preds(self, name: str, ids: list[str], logits: np.ndarray) -> None:
        np.savez(self.path("preds", f"{name}.npz"), ids=np.array(ids), logits=logits)

    def load_preds(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        f = self.run_dir / "preds" / f"{name}.npz"
        if not f.exists():
            return None
        z = np.load(f)
        return z["ids"], z["logits"]

    def save_json(self, name: str, obj) -> None:
        self.path(name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
