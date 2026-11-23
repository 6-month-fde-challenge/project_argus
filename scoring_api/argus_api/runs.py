"""Facts about a trained run that serving decisions depend on. Light to import (no torch), unlike deployments."""
from __future__ import annotations

import json
from pathlib import Path

from argus_ft.branch import check


class ForcedRunRefused(Exception):
    """Activating a forced-branch (demo) run where the profile does not allow it (api.allow_forced_runs: false)."""


def forced(run_dir: Path) -> bool:
    """Not a measured run (argus_ft.branch): the gate was overridden, its decision is missing or forced, or what it
    serves differs from what the gate measured. Unreadable counts as forced."""
    return check(Path(run_dir))["forced"]


def run_max_length(run_dir: Path, default: int) -> int:
    """The prompt token cap the run was trained and calibrated with (its report.json config): a serving profile with
    another cap would trim the history differently from training and shift the probabilities."""
    try:
        value = (json.loads((Path(run_dir) / "report.json").read_text()).get("config") or {}).get("max_length")
        return int(value) if value else default
    except (OSError, ValueError, TypeError):
        return default


def forced_reason(run_dir: Path) -> str | None:
    return check(Path(run_dir))["reason"]
