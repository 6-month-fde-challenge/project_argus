"""Is a run a measured one? The single rule that acceptance, the registry and the scoring API apply.

A run is promotable / servable in production only when every one of these holds; otherwise it is a demo ("forced"):
  1. its own config ran the week-8 gate in `gate.mode: auto` (hybrid / llm override the measurement: demo profiles)
  2. it has the gate's decision file (gate/decision.json), written by a gate that also ran in auto mode - a run that
     reuses another run's artefacts carries that run's decision, which may have been forced
  3. what it serves matches what the gate measured: the FP8 LLM scorer only on an LLM decision, XGBoost only on a
     HYBRID decision (a stale LLM scorer left in a re-used folder would otherwise be served on a HYBRID run)
Light on purpose (json + pathlib only): the scoring API imports it.
"""
from __future__ import annotations

import json
from pathlib import Path


def served_kind(run_dir: Path) -> str:
    """LLM when the run has the FP8 LLM scorer (what the registry registers and the API serves), else HYBRID."""
    return "LLM" if (Path(run_dir) / "scorer_student" / "fp8").exists() else "HYBRID"


def check(run_dir: Path, config_mode: str | None = None) -> dict:
    """-> {"forced": bool, "reason": str | None, "measured": decision | None, "served": "LLM" | "HYBRID"}.
    config_mode: the run's gate.mode when the caller has its config; else read from report.json. Anything that
    cannot be read counts as forced: production must be able to prove a run is a measured one."""
    run_dir = Path(run_dir)
    served = served_kind(run_dir)
    out = {"forced": True, "reason": None, "measured": None, "served": served}
    if config_mode is None:
        try:
            cfg = json.loads((run_dir / "report.json").read_text()).get("config") or {}
            config_mode = (cfg.get("gate") or {}).get("mode", "auto")
        except (OSError, ValueError):
            return {**out, "reason": "report.json is missing or unreadable"}
    if config_mode != "auto":
        return {**out, "reason": f"the run forced the branch (gate.mode: {config_mode})"}
    try:
        decision = json.loads((run_dir / "gate" / "decision.json").read_text())
    except (OSError, ValueError):
        return {**out, "reason": "no measured gate decision (gate/decision.json)"}
    out["measured"] = decision.get("decision")
    if decision.get("mode", "auto") != "auto":
        return {**out, "reason": f"its gate decision was taken with gate.mode: {decision.get('mode')} (reused)"}
    if out["measured"] != served:
        return {**out, "reason": f"it serves the {served} scorer but the gate measured {out['measured']}"}
    return {**out, "forced": False}
