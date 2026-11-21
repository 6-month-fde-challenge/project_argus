"""Scorer + calibrator (LLD 2.2), loaded from a finished training run.

The scorer is what the run registered (stage `registry`): the FP8 LLM scorer student on an LLM-scorer run, XGBoost
on a HYBRID run. Both return a raw fraud logit; the run's stage-8 calibrator maps it to a probability (FR-ML-06).
Reason codes (LLD 2.3): a code is returned only if its supporting feature flag is true and, for the LLM, its head
score is >= 0.5; at most 3, strongest first.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from xgboost import XGBClassifier

from argus_ft import data as D
from argus_ft.modeling import load_model, load_tokenizer
from argus_ft.stages.xgboost_baseline import FEATURES

from .features import FeatureVector

# XGBoost has no reason heads: flagged codes are ordered by how strongly they mark fraud in the investigators' notes
XGB_REASON_PRIORITY = ["R04_FAILED_ATTEMPTS", "R01_NEW_DEVICE", "R05_AMOUNT_SPIKE", "R07_VELOCITY", "R03_ODD_HOUR",
                       "R06_NEW_ACCOUNT", "R02_GEO_MISMATCH", "R08_HIGH_RISK_MCC"]


class Calibrator:
    """Loads calibration/<scorer>.json written by stage 8 (Platt or isotonic)."""

    def __init__(self, path: Path):
        self.params = json.loads(path.read_text())

    def __call__(self, logit: float) -> float:
        p = self.params
        if p["method"] == "platt":
            return float(1 / (1 + np.exp(-(p["slope"] * logit + p["intercept"]))))
        return float(np.interp(logit, p["x"], p["y"]))


class Scorer:
    def __init__(self, run_dir: Path, max_length: int):
        self.run_dir, self.max_length = run_dir, max_length
        selected = json.loads((run_dir / "pilot" / "selected.json").read_text())
        self.prompt_format = selected["format"]
        llm_dir = run_dir / "scorer_student" / "fp8"
        if llm_dir.exists():
            self.kind, self.name = "llm", "scorer_student_fp8"
            self.tokenizer = load_tokenizer(str(llm_dir))
            self.model = load_model(str(llm_dir), "scorer", num_labels=D.NUM_HEADS,
                                    pad_token_id=self.tokenizer.pad_token_id).eval()
            self.version = f"argus-lm-{run_dir.name}-fp8"
        else:
            self.kind, self.name = "xgboost", "xgboost"
            self.model = XGBClassifier()
            self.model.load_model(run_dir / "xgboost" / "model.json")
            self.version = f"argus-xgb-{run_dir.name}"
        self.calibrator = Calibrator(run_dir / "calibration" / f"{self.name}.json")

    @torch.no_grad()
    def score(self, fv: FeatureVector) -> tuple[float, float, list[str]]:
        """-> (raw fraud logit, calibrated probability, reason codes)."""
        if self.kind == "llm":
            prompt = D.fit_prompts(self.tokenizer, [D.to_format(fv.prompt, self.prompt_format)], self.max_length)[0]
            enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.model.device)
            logits = self.model(**enc).logits[0].float().cpu().numpy()
            logit, heads = float(logits[0]), 1 / (1 + np.exp(-logits[1:]))
            ranked = sorted(((h, c) for h, c in zip(heads, D.REASON_CODES) if h >= 0.5 and fv.flags[c]), reverse=True)
            codes = [c for _, c in ranked]
        else:
            x = np.array([[float(fv.values[k]) if fv.values[k] is not None else np.nan for k in FEATURES]])
            p = float(np.clip(self.model.predict_proba(x)[0, 1], 1e-7, 1 - 1e-7))
            logit = float(np.log(p / (1 - p)))
            codes = [c for c in XGB_REASON_PRIORITY if fv.flags[c]]
        return logit, self.calibrator(logit), codes[:3]
