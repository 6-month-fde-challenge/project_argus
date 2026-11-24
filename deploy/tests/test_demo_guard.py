"""deploy/ops.py never connects AWS to a Fraud Model API that serves a demo setup (unless told so explicitly)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ops  # noqa: E402

PROD = {"scorer": "ok", "serving_mode": "production", "active_run": "argus_prod_a100", "active_run_forced": False}


def test_production_with_a_measured_run_is_accepted(monkeypatch):
    monkeypatch.delenv("ARGUS_ALLOW_DEMO", raising=False)
    assert ops.demo_serving(PROD) is None


def test_demo_mode_or_a_demo_run_is_refused(monkeypatch):
    monkeypatch.delenv("ARGUS_ALLOW_DEMO", raising=False)
    assert "--allow-demo" in ops.demo_serving({**PROD, "serving_mode": "demo"})
    assert "--allow-demo" in ops.demo_serving({**PROD, "serving_mode": None})          # an older API: unknown = refuse
    why = ops.demo_serving({**PROD, "active_run_forced": True, "active_run_forced_reason": "gate.mode: llm"})
    assert "argus_prod_a100" in why and "gate.mode: llm" in why


def test_allow_demo_is_an_explicit_choice(monkeypatch):
    monkeypatch.setenv("ARGUS_ALLOW_DEMO", "1")
    assert ops.demo_serving({**PROD, "serving_mode": "demo"}) is None
