"""Stage 9 - Acceptance on the holdout (CRD 15 acceptance gate 1; OBJ-01..04, OBJ-08..10), before registration.

Every servable scorer of the run (XGBoost always; the FP8 LLM scorer student when the LLM branch ran) is judged once
on the holdout split (Jan-Feb 2026). No stage trains on it or selects with it, and nothing here is tuned on it:

  calibrator   the stage-8 calibrator of each scorer (fitted on the earlier half of validation), loaded as served
  thresholds   (a) policy     - the rules-engine cut-offs REVIEW 0.60 / BLOCK 0.85 (what serving applies today)
               (b) recommended - picked on the later half of validation: the highest REVIEW cut-off that still reaches
                   the recall target (OBJ-01), the lowest BLOCK cut-off with precision >= its target (OBJ-02).
                   A recommendation for Rules & policy (two-person approval), never applied automatically.
  holdout      PR-AUC, ROC-AUC, ECE, recall / precision / false-decline rate / daily volume at both threshold sets,
               recall at review capacity, PR and ROC curves
  latency      p50 / p99 of single-transaction scoring in-process: the served scorer's share of the 250 ms budget
               (OBJ-10). The network, feature lookup and explainer add to it; argus_api.loadtest measures end to end.
  verdict      GO when the served scorer meets every applicable target at the policy thresholds, else NO-GO with the
               failing objectives. OBJ-04 (uplift over XGBoost) applies only when an LLM scores.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from .. import data as D
from ..branch import check
from ..db import transaction_facts
from ..metrics import (POLICY_THRESHOLDS, apply_calibration, at_review_capacity, at_threshold, curves, ece, per_code_f1,
                       ranking, threshold_for_precision, threshold_for_recall)
from ..modeling import free_gpu, load_model, load_tokenizer, predict_logits
from .xgboost_baseline import load_feature_table, to_matrix

LLM_SCORER = "scorer_student_fp8"


def served_scorer(run_dir) -> str:
    """What the registry registers and the scoring API serves: the FP8 LLM student when it exists, else XGBoost."""
    return LLM_SCORER if (run_dir / "scorer_student" / "fp8").exists() else "xgboost"


# ---------------------------------------------------------------- scoring the holdout

def _xgboost(ctx):
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(ctx.run_dir / "xgboost" / "model.json")
    return model


def _xgb_logits(model, x: np.ndarray) -> np.ndarray:
    p = np.clip(model.predict_proba(x)[:, 1], 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))  # the log-odds scale the stage-8 calibrator was fitted on


def score_xgboost(ctx, ids: list[str]) -> np.ndarray:
    df = load_feature_table(ctx).set_index("transaction_id").loc[ids].reset_index()
    return _xgb_logits(_xgboost(ctx), to_matrix(df))


def _llm(ctx):
    path = str(ctx.run_dir / "scorer_student" / "fp8")
    tok = load_tokenizer(path)
    return load_model(path, "scorer", num_labels=D.NUM_HEADS, pad_token_id=tok.pad_token_id).eval(), tok


def score_llm(ctx, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """-> (fraud logits, all head logits) of the FP8 student on the holdout rows, in the pilot's prompt format."""
    model, tok = _llm(ctx)
    fitted = ctx.fitted(rows, tok)
    logits = predict_logits(model, tok, [r["prompt"] for r in fitted], ctx.cfg.max_length, ctx.cfg.eval.batch_size,
                            progress=ctx.events.progress("scoring holdout (FP8)"))
    del model
    free_gpu()
    return logits[:, 0], logits


# ---------------------------------------------------------------- latency (OBJ-10)

def _percentiles(ms: list[float]) -> dict:
    return {"n": len(ms), "p50_ms": round(float(np.percentile(ms, 50)), 2), "p99_ms": round(float(np.percentile(ms, 99)), 2),
            "max_ms": round(float(max(ms)), 2)}


def latency(ctx, scorer: str, rows: list[dict]) -> dict:
    """Single-transaction scoring time of the served scorer, the way the API calls it (one request, batch of 1)."""
    n, warmup = ctx.cfg.acceptance.latency_samples, 5
    rows = rows[: n + warmup]
    times: list[float] = []
    if scorer == LLM_SCORER:
        model, tok = _llm(ctx)
        cuda = torch.cuda.is_available()
        for i, r in enumerate(rows):
            start = time.perf_counter()
            prompt = D.fit_prompts(tok, [D.to_format(r["prompt"], ctx.prompt_format)], ctx.cfg.max_length)[0]
            enc = tok(prompt, truncation=True, max_length=ctx.cfg.max_length, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**enc).logits.float().cpu()
            if cuda:
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - start) * 1000)
        del model
        free_gpu()
    else:
        model = _xgboost(ctx)
        df = load_feature_table(ctx).set_index("transaction_id").loc[[r["transaction_id"] for r in rows]].reset_index()
        x = to_matrix(df)
        for i in range(len(x)):
            start = time.perf_counter()
            _xgb_logits(model, x[i : i + 1])
            if i >= warmup:
                times.append((time.perf_counter() - start) * 1000)
    return {"scorer": scorer, "where": "in-process, batch of 1, no network / feature lookup / explainer",
            **_percentiles(times)}


# ---------------------------------------------------------------- evaluation

def _later_half_validation(ctx, name: str, val_ids: set[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """Validation scores the calibrator did not see (stage 8 fits on the earlier half): where thresholds are chosen."""
    preds = ctx.load_preds(f"{name}_val")
    if preds is None:
        return None
    ids, logits = preds
    keep = np.array([i in val_ids for i in ids])
    ids, scores = ids[keep], logits[keep, 0]
    order = np.argsort(ids)
    return ids[order][len(ids) // 2:], scores[order][len(ids) // 2:]


def _at(probs, labels, amounts, threshold) -> dict | None:
    return None if threshold is None else at_threshold(probs, labels, round(threshold, 6), amounts)


def evaluate(ctx, name: str, holdout_scores: np.ndarray, labels: np.ndarray, amounts: np.ndarray,
             facts, val_ids: set[str]) -> dict | None:
    calib_path = ctx.run_dir / "calibration" / f"{name}.json"
    if not calib_path.exists():  # stage 8 skipped it (e.g. no fraud in the calibration half)
        return None
    params = json.loads(calib_path.read_text())
    targets = ctx.cfg.acceptance.targets
    _, review, block = POLICY_THRESHOLDS

    recommended = {"review": None, "block": None}
    val = _later_half_validation(ctx, name, val_ids)
    if val is not None:
        v_ids, v_scores = val
        v_probs, v_labels = apply_calibration(params, v_scores), facts.loc[v_ids].fraud_label.to_numpy()
        r = threshold_for_recall(v_probs, v_labels, targets.recall)
        b = threshold_for_precision(v_probs, v_labels, targets.precision_at_block)
        recommended = {"review": r, "block": None if b is None else max(b, r or 0.0)}  # BLOCK never below REVIEW

    probs = apply_calibration(params, holdout_scores)
    return {
        "holdout": ranking(holdout_scores, labels),
        "calibrator": params["method"],
        "ece": round(ece(probs, labels), 5),
        "policy": {"review_threshold": review, "block_threshold": block,
                   "at_review": _at(probs, labels, amounts, review), "at_block": _at(probs, labels, amounts, block)},
        "recommended": {"chosen_on": "later half of validation", "review_threshold": recommended["review"],
                        "block_threshold": recommended["block"],
                        "at_review": _at(probs, labels, amounts, recommended["review"]),
                        "at_block": _at(probs, labels, amounts, recommended["block"])},
        "at_review_capacity": at_review_capacity(holdout_scores, labels, amounts),
        "curves": curves(holdout_scores, labels),
    }


def objectives(served: dict, xgb: dict | None, served_name: str, lat: dict, targets) -> list[dict]:
    """CRD 4 success criteria that an offline holdout run can measure, at the policy thresholds."""
    at_review, at_block = served["policy"]["at_review"], served["policy"]["at_block"]
    llm_serves = served_name != "xgboost"
    uplift = (round(served["holdout"]["pr_auc"] - xgb["holdout"]["pr_auc"], 4)
              if llm_serves and xgb and served["holdout"]["pr_auc"] is not None and xgb["holdout"]["pr_auc"] is not None
              else None)

    def row(obj, name, value, target, ok, note=None):
        return {"id": obj, "name": name, "value": value, "target": target, "pass": ok, **({"note": note} if note else {})}

    pr_auc = served["holdout"]["pr_auc"]
    return [
        row("OBJ-01", "fraud recall at REVIEW", at_review["recall"], f">= {targets.recall}", at_review["recall"] >= targets.recall),
        row("OBJ-02", "precision at BLOCK", at_block["precision"], f">= {targets.precision_at_block}",
            at_block["precision"] >= targets.precision_at_block if at_block["flagged"] else False,
            None if at_block["flagged"] else "nothing reached the BLOCK threshold"),
        row("OBJ-03", "PR-AUC", pr_auc, f">= {targets.pr_auc}", None if pr_auc is None else pr_auc >= targets.pr_auc),
        row("OBJ-04", "PR-AUC uplift over XGBoost", uplift, f">= +{targets.uplift}",
            None if uplift is None else uplift >= targets.uplift,
            None if llm_serves else "HYBRID run: XGBoost scores, so there is no LLM uplift to measure"),
        row("OBJ-08", "false declines (genuine payments blocked)", at_block["false_positive_rate"],
            f"<= {targets.false_decline_rate}", at_block["false_positive_rate"] <= targets.false_decline_rate),
        row("OBJ-09", "REVIEW cases a day", at_review["flagged_per_day"], f"<= {targets.review_per_day}",
            at_review["flagged_per_day"] <= targets.review_per_day),
        row("OBJ-10", "p99 scoring latency (ms, in-process)", lat["p99_ms"], f"<= {targets.p99_latency_ms}",
            lat["p99_ms"] <= targets.p99_latency_ms, "model only; end to end: python -m argus_api.loadtest"),
    ]


def run(ctx) -> dict:
    facts = transaction_facts(ctx.cfg.db_dir)
    rows = ctx.sft("holdout")  # eval.max_rows caps it the same way as validation (seeded sample across the period)
    ids = [r["transaction_id"] for r in rows]
    labels = facts.loc[ids].fraud_label.to_numpy()
    amounts = facts.loc[ids].transaction_amount.to_numpy(dtype=float)
    val_ids = {r["transaction_id"] for r in ctx.sft("validation")}
    served_name = served_scorer(ctx.run_dir)

    holdout = {"xgboost": score_xgboost(ctx, ids)}
    reason_f1 = None
    if served_name == LLM_SCORER:
        holdout[LLM_SCORER], heads = score_llm(ctx, rows)
        reason_f1 = per_code_f1(heads, np.array([D.head_targets(r) for r in rows]))
    for name, scores in holdout.items():
        ctx.save_preds(f"{name}_holdout", ids, scores[:, None])

    scorers = {name: report for name, scores in holdout.items()
               if (report := evaluate(ctx, name, scores, labels, amounts, facts, val_ids)) is not None}
    if served_name not in scorers:
        raise RuntimeError(f"no calibrator for the served scorer '{served_name}': run the calibrate stage first")
    if reason_f1:
        scorers[LLM_SCORER]["reason_f1_per_code"] = reason_f1

    lat = latency(ctx, served_name, rows)
    checks = objectives(scorers[served_name], scorers.get("xgboost"), served_name, lat, ctx.cfg.acceptance.targets)
    failing = [c["id"] for c in checks if c["pass"] is False]
    verdict = "GO" if not failing else "NO-GO"
    # A run that is not a measured one (argus_ft.branch: forced gate.mode, a missing or forced gate decision, or a
    # served scorer that differs from the measured decision) is a demo: its numbers are real, but it is never
    # promotable - whatever the objectives say.
    branch = check(ctx.run_dir, ctx.cfg.gate.mode)
    forced = branch["forced"]
    promotable = verdict == "GO" and not forced
    ctx.save_json("acceptance/verdict.json", {"served": served_name, "verdict": verdict, "failing": failing,
                                              "gate_mode": ctx.cfg.gate.mode, "promotable": promotable,
                                              "forced_reason": branch["reason"], "objectives": checks})
    ctx.events.emit("metric", name="acceptance_verdict", value=verdict)
    return {
        "split": "holdout", "rows": len(ids), "fraud": int(labels.sum()), "served": served_name,
        "scorers": scorers, "latency": lat, "objectives": checks, "verdict": verdict, "failing": failing,
        "gate_mode": ctx.cfg.gate.mode, "forced_branch": forced, "forced_reason": branch["reason"],
        "measured_branch": branch["measured"], "promotable": promotable,
    }


def forced_branch(cfg) -> bool:
    """True when the profile overrides the week-8 gate (gate.mode hybrid / llm): demo profiles only, never production.
    The full run-level rule (decision file, served vs measured branch) is argus_ft.branch.check."""
    return (cfg.get("gate") or {}).get("mode", "auto") != "auto"
