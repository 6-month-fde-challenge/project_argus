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


# everything the LLM-scorer branch writes (sft_scorer, distill_scorer, the scorer half of quantize and calibrate)
LLM_SCORER_OUTPUTS = ["scorer_teacher", "scorer_student", "preds/teacher_*.npz", "preds/scorer_student*.npz",
                      "calibration/teacher.json", "calibration/scorer_student*.json"]


def drop_llm_scorer(run_dir: Path) -> list[str]:
    """The gate took the HYBRID branch: remove LLM-scorer artefacts an earlier attempt or a reused run left in this
    folder, so nothing downstream (quantize, calibrate, acceptance, registry, the API) can pick them up."""
    removed = []
    for pattern in LLM_SCORER_OUTPUTS:
        for p in sorted(run_dir.glob(pattern)):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
            removed.append(p.relative_to(run_dir).as_posix())
    return removed


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
