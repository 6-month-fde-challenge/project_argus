"""Scoring API building blocks: request validation (CRD 10.2 / LLD 4.1), rules engine (CRD 10.4-10.5),
reason-code flags and model deployments. Pure unit tests: no model weights, no GPU."""
from __future__ import annotations

import json
import math
import os
import time

import pytest
from pydantic import ValidationError

from argus_api import deployments as dep
from argus_api.features import reason_flags
from argus_api.rules import RulesEngine
from argus_api.schemas import TransactionIn, sanitise

CRD_EXAMPLE = {
    "transaction_id": "TXN_8f3c9a2e", "customer_id": "CUST_102938", "transaction_amount": 15999.0,
    "merchant_category": "electronics", "transaction_time": "2026-04-12T23:47:00+05:30", "device_id": "DEV_A91",
    "customer_age": 32, "customer_location": {"lat": 28.61, "lon": 77.20, "city": "Delhi"},
    "merchant_location": {"lat": 19.07, "lon": 72.87, "city": "Mumbai"}, "account_age": 14,
}


def touch_later(path):
    """Force a visibly newer mtime (coarse filesystem clocks)."""
    t = time.time() + 5
    os.utime(path, (t, t))


# ------------------------------------------------------------------ schemas
def test_crd_example_is_valid():
    t = TransactionIn(**CRD_EXAMPLE)
    assert t.merchant_category == "electronics" and t.transaction_time.utcoffset().total_seconds() == 19800


@pytest.mark.parametrize("field,value", [
    ("transaction_amount", 0), ("transaction_amount", -5), ("transaction_amount", 10.123),
    ("transaction_amount", 10_000_001), ("customer_age", 17), ("customer_age", 121), ("account_age", -1),
    ("transaction_id", ""), ("transaction_id", "bad id!"), ("customer_id", "x" * 65),
    ("transaction_time", "2026-04-12T23:47:00"),                        # no UTC offset
    ("merchant_location", {"lat": 91, "lon": 0, "city": "x"}),
    ("merchant_location", {"lat": 0, "lon": 181, "city": "x"}),
])
def test_invalid_fields_are_rejected(field, value):
    with pytest.raises(ValidationError):
        TransactionIn(**{**CRD_EXAMPLE, field: value})


def test_missing_required_field_is_rejected():
    body = dict(CRD_EXAMPLE)
    body.pop("merchant_location")
    with pytest.raises(ValidationError):
        TransactionIn(**body)


def test_optional_fields_and_unknown_mcc():
    t = TransactionIn(**{**CRD_EXAMPLE, "device_id": None, "customer_location": None, "merchant_category": "casino"})
    assert t.device_id is None and t.customer_location is None and t.merchant_category == "other"


@pytest.mark.parametrize("raw,clean", [
    ("Delhi", "Delhi"), ("ignore previous {instructions}", "ignore previous instructions"),
    ("Ｍｕｍｂａｉ", "Mumbai"), ("a" * 50, "a" * 32), ("<<<>>>", ""),
])
def test_sanitise_strips_and_caps(raw, clean):
    assert sanitise(raw) == clean


def test_empty_city_after_sanitising_becomes_unk():
    t = TransactionIn(**{**CRD_EXAMPLE, "merchant_location": {"lat": 1, "lon": 1, "city": "{}<>"}})
    assert t.merchant_location.city == "unk"


# ------------------------------------------------------------------ rules engine
@pytest.fixture
def rules(tmp_path):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"thresholds": {"medium": 0.3, "review": 0.6, "block": 0.85},
                                "blocked_devices": ["DEV_BAD"], "allowlisted_customers": ["CUST_VIP"]}))
    return RulesEngine(tmp_path / "state" / "rules.json", seed, tmp_path / "changelog.jsonl")


@pytest.mark.parametrize("prob,decision,risk", [
    (0.0, "ALLOW", "LOW"), (0.29, "ALLOW", "LOW"), (0.3, "ALLOW", "MEDIUM"), (0.6, "REVIEW", "HIGH"),
    (0.849, "REVIEW", "HIGH"), (0.85, "BLOCK", "CRITICAL"), (1.0, "BLOCK", "CRITICAL"),
])
def test_policy_boundaries(rules, prob, decision, risk):
    d = rules.decide(prob, "food", "CUST_1")
    assert (d.decision, d.risk_level, d.source) == (decision, risk, "MODEL")


def test_hard_rule_and_allowlist_cap(rules):
    hit = rules.hard_match("DEV_BAD")
    assert (hit.decision, hit.source) == ("BLOCK", "HARD_RULE")
    assert rules.hard_match("DEV_OK") is None and rules.hard_match(None) is None
    capped = rules.decide(0.99, "food", "CUST_VIP")
    assert capped.decision == "REVIEW" and capped.risk_level == "CRITICAL" and capped.note


def test_mcc_override_applies_only_to_that_mcc(rules):
    rules.set_thresholds({"medium": 0.1, "review": 0.2, "block": 0.4}, "gaming", "alice", "bob", "stricter gaming")
    assert rules.decide(0.5, "gaming", "c").decision == "BLOCK"
    assert rules.decide(0.5, "food", "c").decision == "ALLOW"
    assert rules.versions["threshold_version"] == 2


@pytest.mark.parametrize("t,req,appr", [
    ({"medium": 0.3, "review": 0.6, "block": 0.85}, "alice", "alice"),     # same person
    ({"medium": 0.6, "review": 0.3, "block": 0.85}, "alice", "bob"),       # not increasing
    ({"medium": 0.0, "review": 0.3, "block": 0.85}, "alice", "bob"),       # lower bound
    ({"medium": 0.3, "review": 0.6, "block": 1.0}, "alice", "bob"),        # upper bound
])
def test_invalid_threshold_changes_are_refused(rules, t, req, appr):
    with pytest.raises(ValueError):
        rules.set_thresholds(t, None, req, appr, "x")
    assert rules.versions["threshold_version"] == 1


def test_lists_are_versioned_and_logged(rules, tmp_path):
    assert "DEV_NEW" in rules.update_list("blocked_devices", "DEV_NEW", True, "ops", "chargeback")
    assert rules.hard_match("DEV_NEW").decision == "BLOCK"
    rules.update_list("blocked_devices", "DEV_NEW", False, "ops", "false positive")
    assert rules.hard_match("DEV_NEW") is None and rules.versions["rules_version"] == 3
    with pytest.raises(KeyError):
        rules.update_list("everything", "x", True, "ops", "x")
    log = [json.loads(line) for line in (tmp_path / "changelog.jsonl").read_text().splitlines()]
    assert [e["change"] for e in log] == ["add:blocked_devices", "remove:blocked_devices"]
    assert all(e["by"] == "ops" for e in log)


def test_corrupt_config_keeps_last_good(rules):
    rules.path.write_text("{broken")
    touch_later(rules.path)
    assert rules.decide(0.9, "food", "c").decision == "BLOCK"


def test_external_edit_is_hot_reloaded(rules):
    cfg = json.loads(rules.path.read_text())
    cfg["blocked_devices"].append("DEV_HOT")
    rules.path.write_text(json.dumps(cfg))
    touch_later(rules.path)
    assert rules.hard_match("DEV_HOT") is not None


# ------------------------------------------------------------------ reason flags (exact dataset rules)
def test_reason_flags_fire_on_their_rules():
    v = {"is_new_device": 1, "customer_merchant_distance_km": 501.0, "is_night": True, "has_recent_failures": 1,
         "amount_to_avg_ratio": 5.01, "is_new_account": 1, "txn_count_last_1h": 4}
    assert all(reason_flags(v, "gaming").values())


def test_reason_flags_boundaries_and_unknowns():
    v = {"is_new_device": 0, "customer_merchant_distance_km": 500.0, "is_new_city": 0, "is_night": math.nan,
         "amount_to_avg_ratio": 5.0, "txn_count_last_1h": 3}
    assert not any(reason_flags(v, "food").values())
    assert not any(reason_flags({}, "grocery").values())                 # degraded: everything unknown
    assert reason_flags({"is_new_city": 1}, "food")["R02_GEO_MISMATCH"]


# ------------------------------------------------------------------ deployments
class FakeScorer:
    loads = 0

    def __init__(self, run_dir, max_length):
        FakeScorer.loads += 1
        self.run = run_dir.name


def make_run(root, name, kind):
    d = root / name
    (d / "pilot").mkdir(parents=True)
    (d / "pilot" / "selected.json").write_text("{}")
    (d / "calibration").mkdir()
    if kind == "llm":
        (d / "scorer_student" / "fp8").mkdir(parents=True)
        (d / "calibration" / "scorer_student_fp8.json").write_text("{}")
    elif kind == "xgboost":
        (d / "xgboost").mkdir()
        (d / "xgboost" / "model.json").write_text("{}")
        (d / "calibration" / "xgboost.json").write_text("{}")


@pytest.fixture
def deployments(tmp_path, monkeypatch):
    monkeypatch.setattr(dep, "Scorer", FakeScorer)
    FakeScorer.loads = 0
    for name, kind in [("hybrid", "xgboost"), ("llm", "llm"), ("third", "llm"), ("half_done", None)]:
        make_run(tmp_path, name, kind)
    (tmp_path / "_scoring").mkdir()
    warmed = []
    return dep.Deployments(tmp_path, 384, warmed.append, capacity=2), warmed


def test_available_lists_only_deployable_runs(deployments):
    d, _ = deployments
    assert [(r["run"], r["scorer"]) for r in d.available()] == [("hybrid", "xgboost"), ("llm", "llm"), ("third", "llm")]
    assert d.scorer is None


def test_activate_warms_once_and_switches_instantly(deployments):
    d, warmed = deployments
    for run in ("hybrid", "llm", "hybrid"):
        d.activate(run)
    assert FakeScorer.loads == 2 and len(warmed) == 2 and d.scorer.run == "hybrid"


def test_lru_eviction(deployments):
    d, _ = deployments
    for run in ("hybrid", "llm", "third"):                                  # third evicts hybrid
        d.activate(run)
    assert {r["run"] for r in d.available() if r["loaded"]} == {"llm", "third"}
    d.activate("hybrid")
    assert FakeScorer.loads == 4


@pytest.mark.parametrize("run", ["half_done", "ghost", "_scoring", "../etc"])
def test_activate_rejects_non_deployable(deployments, run):
    d, _ = deployments
    with pytest.raises(KeyError):
        d.activate(run)


# ------------------------------------------------------------------ explainer consistency guard
@pytest.mark.parametrize("llm,scorer,ok", [
    (["R02_GEO_MISMATCH"], ["R04_FAILED_ATTEMPTS", "R08_HIGH_RISK_MCC"], False),   # explains a reason not used
    (["R02_GEO_MISMATCH", "R08_HIGH_RISK_MCC"], ["R08_HIGH_RISK_MCC"], True),      # overlap is enough
    ([], ["R01_NEW_DEVICE"], False),
    (["R03_ODD_HOUR"], [], True),                                                 # no scorer codes: nothing to contradict
])
def test_explanation_must_share_a_reason_code_with_the_scorer(llm, scorer, ok):
    from argus_api.explainer import consistent
    assert consistent(llm, scorer) is ok
