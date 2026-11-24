"""The platform's side of the operations agent (deploy/agent.py): queue requests, read their status and logs.

The platform runs in a container; the host's scripts (deploy/hyperstack/*.sh, argus.ps1) need the host. So a request
is a file in training/outputs/_platform/ops/actions/ that the agent on the host picks up. Validation and the exact
command come from the repository's own catalogue (deploy/ops_actions.json + deploy/ops_catalog.py), the same code the
agent uses, loaded from the mounted repository: the UI can never offer or describe something the agent would not run.
"""
from __future__ import annotations

import importlib.util
import json
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

from ..core.errors import BadRequest, Conflict, NotFound, UpstreamError
from .fs import mkdirs, write_atomic

ID_RE = re.compile(r"^[0-9a-f]{12,20}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


class OpsQueue:
    def __init__(self, ops_dir: Path, repo_dir: Path, stale_s: float = 15.0):
        self.ops, self.repo, self.stale_s = Path(ops_dir), Path(repo_dir), stale_s
        self.actions_dir = self.ops / "actions"
        self._catalog: tuple[float, ModuleType] | None = None

    # ------------------------------------------------------------ catalogue (the repository's own code)
    def catalog(self) -> ModuleType:
        path = self.repo / "deploy" / "ops_catalog.py"
        if not path.is_file():
            raise UpstreamError(f"the repository is not visible to the platform ({path} missing): "
                                "docker-compose.yml mounts it at /workspace/repo")
        mtime = path.stat().st_mtime
        if self._catalog is None or self._catalog[0] != mtime:  # reloaded after a `git pull`
            spec = importlib.util.spec_from_file_location("argus_ops_catalog", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            self._catalog = (mtime, module)
        return self._catalog[1]

    def actions(self) -> dict:
        try:
            return self.catalog().load(self.repo)
        except (OSError, ValueError, KeyError) as e:
            raise UpstreamError(f"deploy/ops_actions.json cannot be read: {e}") from e

    def prepare(self, action: str, params: dict | None) -> dict:
        """Validate exactly as the agent will -> {params, commands: {linux, windows}} (None where unsupported)."""
        cat, actions = self.catalog(), self.actions()
        try:
            clean = cat.validate(actions, action, params, self.repo)
        except cat.ActionError as e:
            raise BadRequest(str(e)) from e
        commands = {}
        for os_name in ("linux", "windows"):
            try:
                commands[os_name] = cat.preview(*cat.render(actions, action, clean, os_name), os_name)
            except cat.ActionError:
                commands[os_name] = None
        return {"params": clean, "commands": commands}

    # ------------------------------------------------------------ agent
    def agent(self) -> dict:
        try:
            beat = json.loads((self.ops / "agent.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"online": False, "seen": False}
        age = time.time() - float(beat.get("beat", 0))
        return {**beat, "online": age < self.stale_s, "seen": True, "age_s": round(age, 1)}

    # ------------------------------------------------------------ requests
    def submit(self, action: str, params: dict | None, origin: dict | None = None) -> dict:
        prepared = self.prepare(action, params)
        agent = self.agent()
        if agent.get("seen") and agent.get("os") and prepared["commands"].get(agent["os"]) is None:
            raise BadRequest(f"'{self.actions()[action]['label']}' does not run on this machine ({agent['os']}): "
                             "it is a Linux GPU VM action (deploy/hyperstack)")
        mkdirs(self.actions_dir)  # owned like training/outputs: the agent (the VM user) writes here too
        rid = f"{time.time_ns() // 1_000_000:x}{secrets.token_hex(2)}"
        record = {"id": rid, "action": action, "label": self.actions()[action]["label"], "params": prepared["params"],
                  "commands": prepared["commands"], "status": "queued", "created": _now(), "started": None,
                  "finished": None, "exit_code": None, "message": None, **(origin or {})}
        write_atomic(self.actions_dir / f"{rid}.json", json.dumps(record, indent=2).encode("utf-8"))
        return record

    def _path(self, rid: str) -> Path:
        if not ID_RE.match(rid):
            raise NotFound(f"action '{rid}' not found")
        p = self.actions_dir / f"{rid}.json"
        if not p.is_file():
            raise NotFound(f"action '{rid}' not found")
        return p

    def get(self, rid: str) -> dict:
        p = self._path(rid)
        try:
            record = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            raise UpstreamError(f"action {rid} is being written, try again") from e
        record["cancel_requested"] = p.with_suffix(".cancel").exists()
        return record

    def list(self, limit: int = 50) -> list[dict]:
        if not self.actions_dir.exists():
            return []
        out = []
        for p in sorted(self.actions_dir.glob("*.json"), key=lambda f: f.stem, reverse=True)[:limit]:
            try:
                out.append(self.get(p.stem))
            except (NotFound, UpstreamError):
                continue
        return out

    def active(self) -> list[dict]:
        return [r for r in self.list(200) if r["status"] in ("queued", "running")]

    def cancel(self, rid: str) -> dict:
        record = self.get(rid)
        if record["status"] not in ("queued", "running"):
            raise Conflict(f"action is already {record['status']}")
        write_atomic(self._path(rid).with_suffix(".cancel"), _now().encode("utf-8"))  # the agent stops it
        return self.get(rid)

    def log(self, rid: str, offset: int = 0, max_bytes: int = 256_000) -> dict:
        self._path(rid)
        p = self.actions_dir / f"{rid}.log"
        if not p.exists():
            return {"text": "", "offset": 0, "size": 0}
        size = p.stat().st_size
        offset = max(0, min(offset, size))
        with p.open("rb") as f:
            f.seek(offset)
            chunk = f.read(max_bytes)
        return {"text": chunk.decode("utf-8", errors="replace"), "offset": offset + len(chunk), "size": size}
