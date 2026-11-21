"""Rules engine (CRD 10.4-10.5, LLD 2.2 / 4.4): hard rules, lists, thresholds, MCC overrides, change log.

Config lives in a JSON file seeded from the dataset's rules_config.json; it is re-read whenever the file changes
(stand-in for Aurora + 30 s hot reload). Every admin change is appended to a change log with who and why (FR-RE-01).
"""
from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

RISK_LEVELS = [("LOW", "ALLOW"), ("MEDIUM", "ALLOW"), ("HIGH", "REVIEW"), ("CRITICAL", "BLOCK")]
LISTS = {"blocked_devices", "allowlisted_customers"}


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

    def _save(self, change: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cfg, indent=2))
        shutil.move(tmp, self.path)
        self._mtime = self.path.stat().st_mtime
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
        if requested_by == approved_by:
            raise ValueError("two-person approval: approved_by must differ from requested_by")
        if not 0 < thresholds["medium"] < thresholds["review"] < thresholds["block"] < 1:
            raise ValueError("thresholds must satisfy 0 < medium < review < block < 1")
        with self._lock:
            self._reload()
            if mcc:
                self.cfg["mcc_thresholds"][mcc] = thresholds
            else:
                self.cfg["thresholds"] = thresholds
            self.cfg["threshold_version"] += 1
            self._save({"change": "thresholds", "mcc": mcc, "thresholds": thresholds, "requested_by": requested_by,
                        "approved_by": approved_by, "reason": reason, "threshold_version": self.cfg["threshold_version"]})
        return self.cfg

    def update_list(self, name: str, item: str, add: bool, by: str, reason: str) -> list[str]:
        if name not in LISTS:
            raise KeyError(name)
        with self._lock:
            self._reload()
            items = set(self.cfg[name])
            items.add(item) if add else items.discard(item)
            self.cfg[name] = sorted(items)
            self.cfg["rules_version"] += 1
            self._save({"change": f"{'add' if add else 'remove'}:{name}", "item": item, "by": by, "reason": reason,
                        "rules_version": self.cfg["rules_version"]})
        return self.cfg[name]
