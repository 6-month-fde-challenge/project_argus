"""Week-8 gate - does the LLM earn its place? (Part 3 Figure 4: "+0.03 PR-AUC?")

Compares the pilot winner with XGBoost on the same validation rows and records the measured decision:
  LLM    (uplift >= gate.min_uplift) -> every stage runs; the LLM scores.
  HYBRID (otherwise)                 -> XGBoost scores, the LLM only explains: the LLM scorer stages are skipped.
`gate.mode` decides which branch the run takes:
  auto   - follow the measured decision (Figure 4)
  hybrid - always the HYBRID branch    } for demos that must show a given branch; the measured decision
  llm    - always the LLM-scorer branch }  is still reported next to the branch taken
"""
from __future__ import annotations

import json

import numpy as np

from ..metrics import ranking

LLM_SCORER_STAGES = ["sft_scorer", "distill_scorer"]


def run(ctx) -> dict:
    c = ctx.cfg.gate
    if c.mode not in ("auto", "hybrid", "llm"):
        raise ValueError(f"gate.mode must be auto, hybrid or llm, got {c.mode!r}")
    pilot = json.loads((ctx.run_dir / "pilot" / "selected.json").read_text())
    llm_ids, llm_logits = ctx.load_preds(pilot["preds"])
    xgb_ids, xgb_logits = ctx.load_preds("xgboost_val")

    xgb_by_id = dict(zip(xgb_ids, xgb_logits[:, 0]))
    labels_by_id = {r["transaction_id"]: r["fraud_label"] for r in ctx.sft("validation")}
    labels = np.array([labels_by_id[i] for i in llm_ids])
    llm = ranking(llm_logits[:, 0], labels)
    xgb = ranking(np.array([xgb_by_id[i] for i in llm_ids]), labels)

    uplift = round((llm["pr_auc"] or 0.0) - (xgb["pr_auc"] or 0.0), 4)
    decision = "LLM" if uplift >= c.min_uplift else "HYBRID"
    branch = decision if c.mode == "auto" else c.mode.upper()
    ctx.save_json("gate/decision.json", {"decision": decision, "mode": c.mode, "branch": branch, "uplift": uplift, **pilot})
    return {
        "same_rows": len(llm_ids), "llm": f"{pilot['teacher']} | {pilot['format']}",
        "llm_pr_auc": llm["pr_auc"], "xgboost_pr_auc": xgb["pr_auc"],
        "uplift": uplift, "required_uplift": c.min_uplift,
        "decision": decision, "mode": c.mode, "branch": branch,
        "skip_stages": LLM_SCORER_STAGES if branch == "HYBRID" else [],  # read by the pipeline runner
    }
