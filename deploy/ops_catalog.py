"""The operations catalogue (deploy/ops_actions.json): parameter validation and command rendering.

One implementation, used by both sides so they can never disagree:
  deploy/agent.py                the operations agent on the host: re-validates every request, then runs it
  the ARGUS platform             validates before queueing, shows the exact command (it loads this file from the
                                 repository it administers: /workspace/repo/deploy/ops_catalog.py)
Standard library only (the agent runs on the bare host). Nothing here runs a command.
"""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

RUN_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")
PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")
CONFIG_RE = re.compile(r"^configs/[A-Za-z0-9][A-Za-z0-9_-]{0,62}\.yaml$")
VM_PROFILES = ("prod", "demo", "smoke")          # deploy/hyperstack/_common.sh runs_for_profile
RESERVED_RUNS = ("mlruns",)                      # and every name starting with "_" (platform / scoring state)
PLACEHOLDER = re.compile(r"\{(\w+)(\*|\?[^}]*)?\}")
POWERSHELL = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "argus.ps1"]


class ActionError(ValueError):
    """A request the catalogue does not allow (unknown action or parameter, bad value, unsupported OS)."""


def load(repo: Path) -> dict:
    return json.loads((Path(repo) / "deploy" / "ops_actions.json").read_text(encoding="utf-8"))["actions"]


def _runs_dir(repo: Path) -> Path:
    return Path(repo) / "training" / "outputs"


def run_exists(repo: Path, run: str) -> bool:
    return (bool(RUN_RE.match(run)) and not run.startswith("_") and run not in RESERVED_RUNS
            and (_runs_dir(repo) / run).is_dir())


def profile_exists(repo: Path, name: str) -> bool:
    return bool(PROFILE_RE.match(name)) and (Path(repo) / "training" / "configs" / f"{name}.yaml").is_file()


def _check(name: str, spec: dict, value, repo: Path):
    t = spec["type"]
    if t == "bool":
        if not isinstance(value, bool):
            raise ActionError(f"{name}: true or false")
        return value
    if t == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ActionError(f"{name}: a whole number")
        if not spec.get("min", value) <= value <= spec.get("max", value):
            raise ActionError(f"{name}: between {spec.get('min')} and {spec.get('max')}")
        return value
    if t in ("services", "runs"):
        if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
            raise ActionError(f"{name}: a non-empty list")
        items = list(dict.fromkeys(value))  # de-duplicated, order kept
        if t == "services":
            bad = [v for v in items if v not in spec["options"]]
            if bad:
                raise ActionError(f"{name}: unknown {bad}; allowed {spec['options']}")
        else:
            bad = [v for v in items if not run_exists(repo, v)]
            if bad or len(items) > 8:
                raise ActionError(f"{name}: unknown runs {bad}" if bad else f"{name}: at most 8 runs")
        return items
    if not isinstance(value, str):
        raise ActionError(f"{name}: text expected")
    if t == "run":
        if not run_exists(repo, value):
            raise ActionError(f"{name}: no trained run '{value}' in training/outputs")
    elif t == "profile":
        if not profile_exists(repo, value):
            raise ActionError(f"{name}: no profile training/configs/{value}.yaml")
    elif t == "vm_profile":
        if value not in VM_PROFILES and not profile_exists(repo, value):
            raise ActionError(f"{name}: {', '.join(VM_PROFILES)} or a training/configs/<name>.yaml")
    elif t == "serving_config":
        if not CONFIG_RE.match(value) or not (Path(repo) / "training" / value).is_file():
            raise ActionError(f"{name}: configs/<profile>.yaml of an existing profile")
    else:
        raise ActionError(f"{name}: unknown parameter type {t}")
    return value


def validate(actions: dict, action: str, params: dict | None, repo: Path) -> dict:
    """-> clean parameters (only known ones, each checked; missing optional ones are None)."""
    if action not in actions:
        raise ActionError(f"unknown action '{action}'")
    spec, params = actions[action], dict(params or {})
    unknown = sorted(set(params) - set(spec["params"]))
    if unknown:
        raise ActionError(f"unknown parameters {unknown} for '{action}'")
    clean = {}
    for name, p in spec["params"].items():
        value = params.get(name, p.get("default"))
        if _empty(value):
            if p.get("required"):
                raise ActionError(f"{name} is required")
            clean[name] = False if p["type"] == "bool" else None
            continue
        clean[name] = _check(name, p, value, repo)
    return clean


def _empty(value) -> bool:
    """None, "", False or [] (but never 0: 0 == False in Python, and 0 is a real value)."""
    return value is None or value is False or (isinstance(value, (str, list)) and len(value) == 0)


def _render_token(tok: str, p: dict) -> list[str]:
    whole = PLACEHOLDER.fullmatch(tok)
    if whole:
        name, flag = whole.group(1), whole.group(2) or ""
        value = p.get(name)
        if flag == "*":
            return [str(v) for v in value or []]
        if flag.startswith("?"):
            return [flag[1:]] if value else []
        return [] if _empty(value) else [str(value)]
    missing = [m.group(1) for m in PLACEHOLDER.finditer(tok) if _empty(p.get(m.group(1)))]
    if missing:
        return []
    return [PLACEHOLDER.sub(lambda m: str(p[m.group(1)]), tok)]


def render(actions: dict, action: str, params: dict, os_name: str) -> tuple[list[str], dict[str, str]]:
    """-> (argv, extra environment) for os_name "linux" | "windows". params must come from validate()."""
    spec = actions[action]
    template = spec.get(os_name)
    if not template:
        raise ActionError(f"'{spec['label']}' runs on the Linux GPU VM only (deploy/hyperstack), not on {os_name}")
    argv = [t for tok in template for t in _render_token(tok, params)]
    env = {}
    for key, tok in (spec.get("env") or {}).items():
        out = _render_token(tok, params)
        if out and out[0] != "":
            env[key] = out[0]
    if os_name == "windows" and argv and argv[0] == "argus.ps1":
        argv = POWERSHELL + argv[1:]
    return argv, env


def preview(argv: list[str], env: dict[str, str], os_name: str) -> str:
    """The same command as a person would type it (shown in the UI next to every action)."""
    if os_name == "windows":
        if argv[: len(POWERSHELL)] == POWERSHELL:
            return ".\\argus.ps1 " + " ".join(_ps_quote(a) for a in argv[len(POWERSHELL):])
        return " ".join(_ps_quote(a) for a in argv)
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    return (prefix + " " if prefix else "") + shlex.join(argv)


def _ps_quote(arg: str) -> str:
    return arg if re.fullmatch(r"[A-Za-z0-9_./:=,-]+", arg) else "'" + arg.replace("'", "''") + "'"
