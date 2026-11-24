"""Operations use cases: what the page can run (with sensible defaults), and queueing one action with its approval.

Validation and the exact command come from the repository's catalogue through infra.ops_queue.OpsQueue (the same
code the agent runs); this service adds the governance: approval in production mode and the change log.
"""
from __future__ import annotations

from ...core.container import Container
from ...core.governance import Approval, check_approval, serving_mode


class OpsService:
    def __init__(self, c: Container):
        self.c = c

    async def catalog(self) -> dict:
        actions = self.c.ops.actions()
        mode = await serving_mode(self.c)
        env = self.c.env.values() if self.c.env.exists() else {}
        defaults = {  # what the forms start with: the live state first, then .env
            "run": mode.get("active_run") or env.get("RUN_NAME"),
            "config": env.get("SCORING_CONFIG") or mode["configured"]["config"],
            "profile": env.get("ARGUS_PROFILE"),
        }
        listed = [{"name": name, "label": a["label"], "group": a["group"], "description": a["description"],
                   "params": a["params"], "linux": a.get("linux") is not None, "windows": a.get("windows") is not None,
                   "approval_needed": a.get("approval") == "production" and mode["production"],
                   "timeout_min": a.get("timeout_min")} for name, a in actions.items()]
        return {"actions": listed, "agent": self.c.ops.agent(), "serving": mode, "defaults": defaults,
                "runs": self.c.runs.names(), "profiles": self.c.profiles.names()}

    async def submit(self, action: str, params: dict, approval: Approval | None) -> dict:
        spec = self.c.ops.actions().get(action) or {}
        mode = await serving_mode(self.c)
        who = check_approval(approval, spec.get("approval") == "production" and mode["production"],
                             spec.get("label", action))
        record = self.c.ops.submit(action, params, {"origin": "operations", **who})
        self.c.changes.add("operation", f"queued {record['label']}", {"action": record["id"], "params": record["params"]}, who)
        return record

    def cancel(self, rid: str) -> dict:
        record = self.c.ops.cancel(rid)
        self.c.changes.add("operation", f"cancel requested: {record['label']}", {"action": rid})
        return record
