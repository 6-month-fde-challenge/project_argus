"""Rules engine (CRD 10.4-10.5, LLD 2.2 / 4.4): hard rules, lists, thresholds, MCC overrides, change log.

Config lives in a JSON file seeded from the dataset's rules_config.json; it is re-read whenever the file changes
(stand-in for Aurora + 30 s hot reload). Every admin change is appended to a change log with who and why (FR-RE-01).
"""
from __future__ import annotations

import copy
import json
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .schemas import MCC_GROUPS  # CRD 10.2 merchant categories

RISK_LEVELS = [("LOW", "ALLOW"), ("MEDIUM", "ALLOW"), ("HIGH", "REVIEW"), ("CRITICAL", "BLOCK")]
LISTS = {"blocked_devices", "allowlisted_customers"}


def person(name: str) -> str:
    """One spelling per person for the two-person rule: trimmed, inner spaces collapsed, case-folded."""
    return " ".join((name or "").split()).casefold()


@dataclass
class Decision:
    decision: str
    risk_level: str
    source: str            # MODEL or HARD_RULE (FR-RE-05)
    note: str | None = None


class RulesEngine:
    def __init__(self, config_path: Path, seed_from: Path, changelog: Path):
        self.path, self.changelog = config_path, changelog
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            cfg = json.loads(seed_from.read_text())
            cfg.update({"mcc_thresholds": {}, "threshold_version": 1, "rules_version": 1})
            self.path.write_text(json.dumps(cfg, indent=2))
        self._lock, self._mtime, self.cfg = threading.Lock(), 0.0, {}
        self._reload()

    # ------------------------------------------------------------ config

    def _reload(self) -> None:
        mtime = self.path.stat().st_mtime
        if mtime != self._mtime:
            try:
                self.cfg, self._mtime = json.loads(self.path.read_text()), mtime
            except (OSError, json.JSONDecodeError):
                pass  # LLD 6.1: config reload fails -> keep the last good config

    def log_change(self, change: dict) -> None:
        with self.changelog.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), **change}) + "\n")

    def _commit(self, new_cfg: dict, change: dict) -> None:
        """Write the new config to disk first (atomic rename), then serve it: a failed write changes nothing."""
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(new_cfg, indent=2))
        shutil.move(tmp, self.path)
        self.cfg, self._mtime = new_cfg, self.path.stat().st_mtime
        self.log_change(change)

    @property
    def versions(self) -> dict:
        self._reload()
        return {"rules_version": self.cfg["rules_version"], "threshold_version": self.cfg["threshold_version"]}

    # ------------------------------------------------------------ decisions

    def hard_match(self, device_id: str | None) -> Decision | None:
        """FR-RE-02: a blocklisted device always returns BLOCK, before the model runs."""
        self._reload()
        if device_id and device_id in self.cfg["blocked_devices"]:
            return Decision("BLOCK", "CRITICAL", "HARD_RULE", "blocked_device")
        return None

    def thresholds(self, mcc: str) -> dict:
        return self.cfg["mcc_thresholds"].get(mcc, self.cfg["thresholds"])  # FR-RE-04

    def decide(self, prob: float, mcc: str, customer_id: str) -> Decision:
        self._reload()
        t = self.thresholds(mcc)
        level = sum(prob >= t[k] for k in ("medium", "review", "block"))
        risk, action = RISK_LEVELS[level]
        if action == "BLOCK" and customer_id in self.cfg["allowlisted_customers"]:
            return Decision("REVIEW", risk, "MODEL", "allowlisted_customer_capped_at_review")  # FR-RE-03
        return Decision(action, risk, "MODEL")

    # ------------------------------------------------------------ admin (LLD 4.4)

    def set_thresholds(self, thresholds: dict, mcc: str | None, requested_by: str, approved_by: str, reason: str) -> dict:
        # Two-person rule on normalised names ("Alice " and "alice" are one person). With a shared admin token this is
        # an audit control, not authentication: production puts each name behind SSO + MFA (NFR-09).
        requested, approved = person(requested_by), person(approved_by)
        if not requested or not approved:
            raise ValueError("requested_by and approved_by are required")
        if requested == approved:
            raise ValueError("two-person approval: approved_by must differ from requested_by")
        if not reason.strip():
            raise ValueError("a reason is required for every threshold change")
        if not 0 < thresholds["medium"] < thresholds["review"] < thresholds["block"] < 1:
            raise ValueError("thresholds must satisfy 0 < medium < review < block < 1")
        if mcc and mcc not in MCC_GROUPS:
            raise ValueError(f"unknown merchant category {mcc!r}: one of {sorted(MCC_GROUPS)}")
        with self._lock:
            self._reload()
            new = copy.deepcopy(self.cfg)
            if mcc:
                new["mcc_thresholds"][mcc] = thresholds
            else:
                new["thresholds"] = thresholds
            new["threshold_version"] += 1
            self._commit(new, {"change": "thresholds", "mcc": mcc, "thresholds": thresholds, "requested_by": requested,
                               "approved_by": approved, "reason": reason, "threshold_version": new["threshold_version"]})
        return self.cfg

    def update_list(self, name: str, item: str, add: bool, by: str, reason: str) -> list[str]:
        if name not in LISTS:
            raise KeyError(name)
        with self._lock:
            self._reload()
            new = copy.deepcopy(self.cfg)
            items = set(new[name])
            items.add(item) if add else items.discard(item)
            new[name] = sorted(items)
            new["rules_version"] += 1
            self._commit(new, {"change": f"{'add' if add else 'remove'}:{name}", "item": item, "by": person(by),
                               "reason": reason, "rules_version": new["rules_version"]})
        return self.cfg[name]
