"""Stage 8 - Calibrate every scorer and report it the same way (FR-ML-06, FR-ML-11, FR-ML-12, TR-ML-02..05).

Validation is split by time: calibrators are fitted on the earlier half and judged on the later half, so
the reported numbers are out-of-sample. Platt (temperature + bias) and isotonic are compared and the lower
ECE is kept. The decision view uses the Part 3 policy: >= 0.60 REVIEW, >= 0.85 BLOCK.
"""
from __future__ import annotations

import numpy as np

from ..db import transaction_facts
from ..metrics import (POLICY_THRESHOLDS, Calibrator, at_review_capacity, at_threshold, ece, group_error_rates, ranking,
                       threshold_report)

# saved by stages 1 (XGBoost), 4a (teacher), 6a (scorer student, bf16) and 7 (scorer student, FP8)
SCORERS = ("xgboost_val", "teacher_val", "scorer_student_val", "scorer_student_fp8_val")


def evaluate_scorer(scores: np.ndarray, facts, ids: np.ndarray) -> tuple[dict, Calibrator | None]:
    """Time-split calibration plus the full evaluation report for one scorer."""
    order = np.argsort(ids)  # transaction ids are time-ordered
    ids, scores = ids[order], scores[order]
    rows = facts.loc[ids]
    labels, amounts = rows.fraud_label.to_numpy(), rows.transaction_amount.to_numpy(dtype=float)
    fit, test = slice(0, len(ids) // 2), slice(len(ids) // 2, None)
    if labels[fit].min() == labels[fit].max():
        return {"skipped": "no fraud in the calibration half"}, None

    fitted = {m: Calibrator(m).fit(scores[fit], labels[fit]) for m in ("platt", "isotonic")}
    eces = {m: round(ece(cal(scores[test]), labels[test]), 5) for m, cal in fitted.items()}
    best = min(eces, key=eces.get)
    probs, y, amt = fitted[best](scores[test]), labels[test], amounts[test]
    _, review, block = POLICY_THRESHOLDS
    report = {
        "test_half": ranking(scores[test], y),
        "ece": eces, "chosen_calibrator": best,
        # top 0.29% -> REVIEW (12,000 of 4.2M a day); ranked by the raw score: same order as the calibrated
        # probability, but without the ties an isotonic step function creates
        "at_review_capacity": at_review_capacity(scores[test], y, amt),
        "at_review": at_threshold(probs, y, review, amt),   # REVIEW or BLOCK
        "at_block": at_threshold(probs, y, block, amt),     # BLOCK only (OBJ-02 precision@BLOCK)
        "thresholds": threshold_report(probs, y),
        "fairness_at_review": {
            "age_band": group_error_rates(probs, y, rows.age_band.to_numpy()[test], review),
            "region": group_error_rates(probs, y, rows.region.to_numpy()[test], review),
        },
    }
    return report, fitted[best]


def run(ctx) -> dict:
    facts = transaction_facts(ctx.cfg.db_dir)
    val_ids = {r["transaction_id"] for r in ctx.sft("validation")}
    report = {}
    for name in SCORERS:
        preds = ctx.load_preds(name)
        if preds is None:  # stage not run (or skipped by the week-8 gate)
            continue
        ids, logits = preds
        keep = np.array([i in val_ids for i in ids])  # XGBoost scored every DB validation row
        scorer = name.removesuffix("_val")
        report[scorer], calibrator = evaluate_scorer(logits[keep, 0], facts, ids[keep])
        if calibrator:
            ctx.save_json(f"calibration/{scorer}.json", calibrator.to_dict())
    return report
