"""Settings use cases: read the repository's .env for the UI, validate and write changes, queue "apply".

The router only translates HTTP; everything here works on the Container's adapters (EnvFile, ChangeLog, OpsQueue,
ProfileRepository), so the rules are the same whoever calls them.
"""
from __future__ import annotations

import secrets as pysecrets

from ...core.container import Container
from ...core.errors import AppError, BadRequest, UpstreamError
from ...core.governance import Approval, check_approval, serving_mode
from .catalog import BY_KEY, PORT_KEYS, SETTINGS, Setting, validate

GENERATE = "@generate"                 # a token value the platform generates (never shown)
VM_PROFILES = ["prod", "demo", "smoke"]  # deploy/hyperstack/_common.sh runs_for_profile


def new_token() -> str:
    """40 URL-safe characters without '-' / '_': safe in .env, bash and every header."""
    return pysecrets.token_urlsafe(30).replace("-", "A").replace("_", "B")


class SettingsService:
    def __init__(self, c: Container):
        self.c = c

    # ------------------------------------------------------------ choices offered by the page
    def serving_configs(self) -> dict[str, str]:
        """configs/<profile>.yaml -> "production" | "demo" for every profile the API can run with."""
        out = {}
        for name in self.c.profiles.names():
            try:
                api = self.c.profiles.resolve(name).get("api")
            except AppError:
                continue  # a broken profile is simply not offered
            if api:
                out[f"configs/{name}.yaml"] = "demo" if api.get("allow_forced_runs", False) else "production"
        return out

    def vm_profiles(self) -> list[str]:
        """What ARGUS_PROFILE accepts: the scripts' three names, or any profile but base."""
        return list(dict.fromkeys(VM_PROFILES + [n for n in self.c.profiles.names() if n != "base"]))

    def require_repo(self) -> None:
        repo = self.c.settings.repo_dir
        if not repo.is_dir() or not (repo / "docker-compose.yml").is_file():
            raise UpstreamError("the repository is not visible to the platform: docker-compose.yml mounts it at "
                                "/workspace/repo (recreate the platform after updating the compose file)")

    def env(self) -> dict[str, str]:
        return self.c.env.values() if self.c.env.exists() else {}

    # ------------------------------------------------------------ read
    def _item(self, s: Setting, value: str | None, configs: dict[str, str], vm_profiles: list[str]) -> dict:
        warn = None
        if s.key == "ARGUS_BIND" and value and value != "127.0.0.1":
            warn = f"set to {value} by hand: the UI, platform and MLflow (no login) are reachable from the network"
        options = list(s.options) or (list(configs) if s.type == "serving_config"
                                      else vm_profiles if s.type == "vm_profile" else [])
        return {"key": s.key, "label": s.label, "type": s.type, "help": s.help, "note": s.note, "default": s.default,
                "value": None if s.secret else value, "is_set": bool(value), "secret": s.secret,
                "restart": list(s.restart), "approval": s.approval, "locked": s.locked, "options": options,
                "min": s.min, "max": s.max, "warn": warn,
                "option_modes": configs if s.type == "serving_config" else None}

    async def view(self) -> dict:
        self.require_repo()
        env, configs, vm_profiles = self.env(), self.serving_configs(), self.vm_profiles()
        groups: dict[str, list] = {}
        for s in SETTINGS:
            groups.setdefault(s.group, []).append(self._item(s, env.get(s.key), configs, vm_profiles))
        return {
            "env_path": str(self.c.settings.env_path), "env_exists": self.c.env.exists(),
            "groups": [{"name": g, "items": items} for g, items in groups.items()],
            "unknown_keys": sorted(k for k in env if k not in BY_KEY),  # kept as they are; listed so nothing hides
            "serving": await serving_mode(self.c), "agent": self.c.ops.agent(),
        }

    # ------------------------------------------------------------ write
    def _new_values(self, changes: dict[str, str | None]) -> dict[str, str | None]:
        configs, vm_profiles = list(self.serving_configs()), self.vm_profiles()
        out: dict[str, str | None] = {}
        for key, value in changes.items():
            s = BY_KEY.get(key)
            if s is None:
                raise BadRequest(f"{key} is not a setting the UI manages (edit .env by hand)")
            if s.locked:
                raise BadRequest(f"{key} cannot be changed here: {s.locked}")
            if value is None:
                out[key] = None
            elif value == GENERATE:
                if s.type != "token":
                    raise BadRequest(f"{key}: only tokens can be generated")
                out[key] = new_token()
            else:
                out[key] = validate(s, value, configs, vm_profiles)
        return out

    @staticmethod
    def _check_ports(env: dict[str, str], changed: dict[str, str | None]) -> None:
        ports = {k: (changed[k] if k in changed else env.get(k)) or BY_KEY[k].default for k in PORT_KEYS}
        seen: dict[str, str] = {}
        for key, port in ports.items():
            if port in seen:
                raise BadRequest(f"port {port} is used twice: {seen[port]} and {key}")
            seen[port] = key

    async def save(self, changes: dict[str, str | None], approval: Approval | None, apply: bool) -> dict:
        self.require_repo()
        if not changes:
            raise BadRequest("no changes")
        env = self.env()
        changed = {k: v for k, v in self._new_values(changes).items() if env.get(k) != v}
        if not changed:
            return {"saved": [], "restart": [], "action": None, "commands": {"linux": None, "windows": None},
                    "message": "nothing changed"}
        self._check_ports(env, changed)
        mode = await serving_mode(self.c)
        needs = [k for k in changed if BY_KEY[k].approval]
        who = check_approval(approval, bool(needs) and mode["production"], f"changing {', '.join(needs)}")

        self.c.env.update(changed)
        restart = sorted({svc for k in changed for svc in BY_KEY[k].restart})
        shown = {k: "(removed)" if v is None else "(changed)" if BY_KEY[k].secret else v for k, v in changed.items()}
        self.c.changes.add("settings", f"changed {', '.join(changed)}", {"values": shown, "restart": restart}, who)
        action = self.c.ops.submit("apply", {"services": restart}, {"origin": "settings", **who}) if apply and restart else None
        return {"saved": sorted(changed), "restart": restart, "action": action, "commands": apply_commands(restart)}

    def apply(self, services: list[str], approval: Approval | None) -> dict:
        who = check_approval(approval, False, "apply")
        record = self.c.ops.submit("apply", {"services": services}, {"origin": "settings", **who})
        self.c.changes.add("operation", f"queued {record['label']}: {', '.join(record['params']['services'])}",
                           {"action": record["id"]}, who)
        return record


def apply_commands(services: list[str]) -> dict[str, str | None]:
    """The same apply by hand (deploy/hyperstack/apply.sh on the VM, argus.ps1 apply on Windows)."""
    if not services:
        return {"linux": None, "windows": None}
    names = " ".join(services)
    return {"linux": f"bash deploy/hyperstack/apply.sh {names}", "windows": f".\\argus.ps1 apply {names}"}
