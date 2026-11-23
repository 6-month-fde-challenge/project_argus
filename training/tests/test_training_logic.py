"""Training-package logic that needs no GPU, no model and no dataset: metrics, thresholds, calibration as served,
prompt formats, label maturity, profile resolution, the acceptance checklist and the reports."""
from __future__ import annotations

import json

import numpy as np
import pytest

from argus_ft import data as D
from argus_ft.config import load_config
from argus_ft.metrics import (Calibrator, apply_calibration, at_review_capacity, at_threshold, curves, ece,
                              per_code_f1, ranking, threshold_for_precision, threshold_for_recall)

CONFIGS = __import__("pathlib").Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def scored():
    """2,000 rows, 2% fraud, scores that separate the classes well but not perfectly."""
    rng = np.random.default_rng(0)
    labels = (rng.random(2000) < 0.02).astype(int)
    scores = rng.normal(0, 1, 2000) + 2.5 * labels
    return scores, labels


# ------------------------------------------------------------------ metrics
def test_ranking_handles_single_class_slices():
    assert ranking(np.array([0.1, 0.2]), np.array([0, 0]))["pr_auc"] is None


def test_threshold_for_recall_reaches_the_target(scored):
    scores, labels = scored
    t = threshold_for_recall(scores, labels, 0.85)
    assert at_threshold(scores, labels, t)["recall"] >= 0.85
    assert at_threshold(scores, labels, t + 1e-6)["recall"] < 0.85  # the highest cut-off that still does


def test_threshold_for_precision_reaches_the_target_or_the_best(scored):
    scores, labels = scored
    t = threshold_for_precision(scores, labels, 0.70)
    assert at_threshold(scores, labels, t)["precision"] >= 0.70
    best = threshold_for_precision(scores, labels, 1.01)  # unreachable: the best-precision cut-off instead
    assert at_threshold(scores, labels, best)["precision"] > 0
    assert threshold_for_recall(scores, np.zeros_like(labels), 0.85) is None
    assert threshold_for_precision(scores, np.zeros_like(labels), 0.7) is None


def test_curves_are_thin_and_bounded(scored):
    c = curves(*scored, points=50)
    assert len(c["pr"]["recall"]) <= 50 and len(c["roc"]["fpr"]) <= 50
    assert all(0 <= v <= 1 for v in c["pr"]["precision"] + c["roc"]["tpr"])
    assert curves(np.array([0.1, 0.2]), np.array([1, 1])) == {}


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_a_saved_calibrator_gives_what_training_measured(scored, method):
    scores, labels = scored
    cal = Calibrator(method).fit(scores, labels)
    params = json.loads(json.dumps(cal.to_dict()))  # what calibration/<scorer>.json holds
    np.testing.assert_allclose(apply_calibration(params, scores), cal(scores), atol=1e-9)
    assert ece(apply_calibration(params, scores), labels) < 0.05


def test_review_capacity_and_per_code_f1(scored):
    scores, labels = scored
    cap = at_review_capacity(scores, labels, np.full(len(labels), 100.0))
    assert cap["reviewed"] >= 1 and 0 <= cap["recall"] <= 1
    logits = np.zeros((4, D.NUM_HEADS))
    logits[:, 1] = [3, 3, -3, -3]  # R01 head
    targets = np.zeros((4, D.NUM_HEADS))
    targets[:2, 1] = 1
    f1 = per_code_f1(logits, targets)
    assert f1["R01_NEW_DEVICE"] == 1.0 and set(f1) == set(D.REASON_CODES)


# ------------------------------------------------------------------ data
PROMPT = ("[TXN] id=TX_1 amt=12000.0 mcc=gift_cards channel=UPI time=00h Wed\n[CUST] age=57\n"
          "[HIST] -3d:UPI 500.0 grocery Pune | -9d:card 900.0 food Delhi\n[TASK] Assess fraud risk.")


def test_fmt_v2_adds_a_summary_and_drops_history_cities():
    v2 = D.to_format(PROMPT, "fmt-v2")
    assert "[SUMMARY] hist_n=2 hist_cities=2" in v2 and "Pune" not in v2.split("[SUMMARY]")[1]
    assert D.to_format(PROMPT, "fmt-v1") == PROMPT
    with pytest.raises(ValueError):
        D.to_format(PROMPT, "fmt-v9")


def test_label_maturity_and_downsampling():
    rows = [{"transaction_id": f"T{i}", "fraud_label": int(i < 3), "eligible_at_next_split_start": i % 2 == 0}
            for i in range(20)] + [{"transaction_id": "TX", "fraud_label": 0}]  # no flag: counted as mature
    mature = D.mature_labels_only(rows)
    assert len(mature) == 11 and all(D.label_mature(r) for r in mature)
    sampled = D.downsample_negatives(rows, 2, seed=1)
    assert sum(r["fraud_label"] for r in sampled) == 3 and len(sampled) == 9


# ------------------------------------------------------------------ profiles
@pytest.mark.parametrize("profile", ["base", "smoke", "local_demo", "demo_hybrid", "demo_llm_scorer", "prod_a100"])
def test_every_profile_resolves_and_lists_known_stages(profile):
    from argus_ft.prefetch import profile_models
    from argus_ft.stages import LADDER
    cfg = load_config(CONFIGS / f"{profile}.yaml")
    assert set(cfg.stages) <= set(LADDER)
    assert "acceptance" in cfg.stages and cfg.stages.index("acceptance") < cfg.stages.index("registry")
    assert cfg.acceptance.targets.recall == 0.85 and cfg.label_maturity.eligible_only is False
    assert len(profile_models(cfg)) >= 3


DEMO_PROFILES = {"demo_hybrid", "demo_llm_scorer", "smoke"}  # the ONLY profiles allowed to force the week-8 branch


def test_only_demo_profiles_force_the_gate_and_production_never_serves_them():
    from argus_ft.stages.acceptance import forced_branch
    for path in sorted(CONFIGS.glob("*.yaml")):
        cfg = load_config(path)
        assert forced_branch(cfg) == (path.stem in DEMO_PROFILES), \
            f"{path.name}: gate.mode={cfg.gate.mode} - only {sorted(DEMO_PROFILES)} may force the branch"
    for profile in ("base", "prod_a100"):
        assert load_config(CONFIGS / f"{profile}.yaml").api.allow_forced_runs is False
    assert load_config(CONFIGS / "local_demo.yaml").api.allow_forced_runs is True


def test_overrides_are_typed():
    cfg = load_config(CONFIGS / "smoke.yaml", ["acceptance.latency_samples=7", "label_maturity.eligible_only=true"])
    assert cfg.acceptance.latency_samples == 7 and cfg.label_maturity.eligible_only is True


# ------------------------------------------------------------------ acceptance checklist + report
def _scorer(pr_auc, recall, precision, fpr, per_day, flagged=10):
    at = {"threshold": 0.6, "flagged": flagged, "precision": precision, "recall": recall,
          "false_positive_rate": fpr, "flagged_per_day": per_day}
    return {"holdout": {"pr_auc": pr_auc}, "policy": {"at_review": at, "at_block": at},
            "recommended": {"review_threshold": 0.3, "block_threshold": 0.7, "at_review": at, "at_block": at}}


def test_acceptance_objectives_and_verdict():
    from argus_ft.stages.acceptance import objectives
    cfg = load_config(CONFIGS / "base.yaml")
    good = objectives(_scorer(0.9, 0.9, 0.8, 0.001, 9000), _scorer(0.5, 0, 0, 0, 0), "scorer_student_fp8",
                      {"p99_ms": 40.0}, cfg.acceptance.targets)
    assert all(c["pass"] for c in good) and {c["id"] for c in good} >= {"OBJ-01", "OBJ-02", "OBJ-04", "OBJ-10"}
    hybrid = objectives(_scorer(0.3, 0.5, 0.2, 0.05, 30000, flagged=0), None, "xgboost", {"p99_ms": 900.0},
                        cfg.acceptance.targets)
    by_id = {c["id"]: c for c in hybrid}
    assert by_id["OBJ-04"]["pass"] is None and "HYBRID" in by_id["OBJ-04"]["note"]  # no LLM uplift to measure
    assert by_id["OBJ-02"]["pass"] is False and by_id["OBJ-10"]["pass"] is False


def test_report_renders_the_acceptance_section():
    from argus_ft.compare import acceptance_rows
    acc = {"verdict": "NO-GO", "served": "xgboost", "rows": 5000, "fraud": 30, "failing": ["OBJ-03"],
           "objectives": [{"id": "OBJ-03", "name": "PR-AUC", "value": 0.31, "target": ">= 0.8", "pass": False},
                          {"id": "OBJ-09", "name": "REVIEW cases a day", "value": 9000, "target": "<= 12000", "pass": True}],
           "scorers": {"xgboost": _scorer(0.31, 0.5, 0.4, 0.01, 9000)},
           "latency": {"scorer": "xgboost", "where": "in-process", "p50_ms": 1.0, "p99_ms": 2.0}}
    text = "\n".join(acceptance_rows({"acceptance": acc}))
    assert "**NO-GO**" in text and "| OBJ-03 | PR-AUC | 0.31 | >= 0.8 | **FAIL** |" in text and "9000" in text
    assert acceptance_rows({}) == []


def test_comparison_shows_every_runs_holdout_verdict(tmp_path):
    from argus_ft.compare import comparison
    run = tmp_path / "argus_x"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"stages": {
        "gate": {"decision": "HYBRID", "branch": "LLM"}, "quantize": {"scorer_fp8": {}, "explainer_awq": {}}, "acceptance": {
        "verdict": "NO-GO", "failing": ["OBJ-03"], "served": "xgboost", "rows": 10, "fraud": 1,
        "objectives": [], "scorers": {"xgboost": _scorer(0.3, 0.5, 0.4, 0.01, 9000)},
        "latency": {"scorer": "xgboost", "where": "x", "p50_ms": 1, "p99_ms": 2}}}}))
    text = comparison([run])
    assert "| Holdout acceptance |" in text and "| NO-GO (OBJ-03) |" in text and "`argus_x` - acceptance" in text
    # measured vs taken, both shown; serving an LLM the gate never measured makes it a demo run
    assert "| HYBRID | LLM (demo) | LLM (FP8 student) | LLM (AWQ student, vLLM) |" in text


# ------------------------------------------------------------------ measured vs forced runs (one rule everywhere)
def _run(root, name, mode="auto", decision=None, llm=False):
    d = root / name
    d.mkdir()
    (d / "report.json").write_text(json.dumps({"config": {"gate": {"mode": mode}}, "stages": {}}))
    if decision:
        (d / "gate").mkdir()
        (d / "gate" / "decision.json").write_text(json.dumps(decision))
    if llm:
        (d / "scorer_student" / "fp8").mkdir(parents=True)
    return d


def test_only_a_run_whose_serving_matches_its_measured_auto_gate_is_measured(tmp_path):
    from argus_ft.branch import check
    auto = {"decision": "HYBRID", "mode": "auto"}
    cases = {
        "measured_hybrid": (_run(tmp_path, "a", decision=auto), False),
        "measured_llm": (_run(tmp_path, "b", decision={"decision": "LLM", "mode": "auto"}, llm=True), False),
        "forced_profile": (_run(tmp_path, "c", mode="llm", decision=auto, llm=True), True),
        "no_decision": (_run(tmp_path, "d"), True),
        "reused_forced_decision": (_run(tmp_path, "e", decision={"decision": "HYBRID", "mode": "hybrid"}), True),
        "stale_llm_on_hybrid": (_run(tmp_path, "f", decision=auto, llm=True), True),
        "llm_measured_xgb_served": (_run(tmp_path, "g", decision={"decision": "LLM", "mode": "auto"}), True),
    }
    for name, (d, want) in cases.items():
        got = check(d)
        assert got["forced"] is want, name
        assert (got["reason"] is None) is (not want), name
    assert check(tmp_path / "missing")["forced"]


def test_a_hybrid_gate_removes_every_stale_llm_scorer_artefact(tmp_path):
    from argus_ft.branch import served_kind
    from argus_ft.runs import drop_llm_scorer
    d = tmp_path / "run"
    for rel in ("scorer_student/fp8/model.safetensors", "scorer_teacher/adapter/a.bin", "preds/teacher_val.npz",
                "preds/scorer_student_fp8_val.npz", "calibration/scorer_student_fp8.json", "calibration/teacher.json",
                "preds/xgboost_val.npz", "calibration/xgboost.json", "explainer_student/awq/x"):
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text("x")
    assert served_kind(d) == "LLM"
    removed = drop_llm_scorer(d)
    assert served_kind(d) == "HYBRID" and "scorer_student" in removed and "calibration/teacher.json" in removed
    # XGBoost and the explainer are untouched
    for rel in ("preds/xgboost_val.npz", "calibration/xgboost.json", "explainer_student/awq/x"):
        assert (d / rel).exists()


def test_registry_status_needs_a_measured_run_and_a_go(tmp_path):
    from argus_ft.stages.registry import registry_status
    d = _run(tmp_path, "r", decision={"decision": "HYBRID", "mode": "auto"})
    assert registry_status(d, "auto", {"status": "ok", "verdict": "GO"}) == ("candidate", None)
    assert registry_status(d, "auto", {"status": "ok", "verdict": "NO-GO", "failing": ["OBJ-03"]})[0] == "rejected"
    assert registry_status(d, "auto", None)[0] == "unverified"
    assert registry_status(d, "auto", {"status": "failed", "error": "boom"})[0] == "unverified"  # no KeyError
    assert registry_status(d, "auto", {"status": "ok", "verdict": "GO", "reused_from": "x"})[0] == "unverified"
    assert registry_status(d, "llm", {"status": "ok", "verdict": "GO"})[0] == "demo"


def test_threshold_for_precision_never_splits_a_tie():
    # three rows tie at 0.9 (one fraud): no cut-off flags only the first of them, so precision 1.0 is not reachable
    probs = np.array([0.9, 0.9, 0.9, 0.5, 0.1])
    labels = np.array([1, 0, 0, 1, 0])
    t = threshold_for_precision(probs, labels, 0.9)
    assert at_threshold(probs, labels, t)["precision"] == max(
        at_threshold(probs, labels, c)["precision"] for c in (0.9, 0.5, 0.1))
    assert t in (0.9, 0.5)
