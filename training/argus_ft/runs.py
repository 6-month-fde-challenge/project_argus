"""Start a run from another run's artefacts (config key `reuse_from: <run_name>`).

Used to build a variant without retraining shared stages - e.g. the LLM-scorer run reuses the pilot, CPT and
explainer branch of the HYBRID run and trains only the scorer stages. The new run gets its own copy of the
artefacts (runs never share folders), and every copied stage entry in its report is marked `reused_from`.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

SKIP = shutil.ignore_patterns("trainer", "report.json", "events.jsonl", "api")  # rebuilt / per-run state


def seed_from(source: Path, target: Path, stages_to_run: list[str]) -> dict:
    """Copy `source` into an empty `target`; return the report stages that are reused (not re-run)."""
    if not (source / "report.json").exists():
        raise SystemExit(f"reuse_from: {source} has no report.json - run it first")
    if (target / "report.json").exists():
        return json.loads((target / "report.json").read_text())["stages"]  # already seeded earlier
    shutil.copytree(source, target, ignore=SKIP, dirs_exist_ok=True)
    source_stages = json.loads((source / "report.json").read_text())["stages"]
    return {name: {**entry, "reused_from": source.name}
            for name, entry in source_stages.items() if name not in stages_to_run and entry.get("status") == "ok"}
