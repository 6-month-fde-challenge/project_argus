"""Training profiles (training/configs/*.yaml): raw YAML (with its comments) and the resolved config.

Resolution follows the trainer exactly: a profile may `extends:` another and overrides only what differs; nested
sections deep-merge, everything else replaces.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from ..core.errors import BadRequest, Conflict, NotFound
from .fs import write_atomic

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]{1,40}$")
# shipped with the repository: editable (changes show in git), never deleted from the UI
BUILT_IN = ("base", "local_demo", "smoke", "demo_hybrid", "demo_llm_scorer", "prod_a100")


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def set_dotted(cfg: dict, key: str, value) -> None:
    *parents, leaf = key.split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise BadRequest(f"override '{key}': '{p}' is not a section")
    node[leaf] = value


class ProfileRepository:
    def __init__(self, configs_dir: Path):
        self.root = configs_dir

    def names(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.yaml"))

    def path(self, name: str) -> Path:
        p = self.root / f"{name}.yaml"
        if name not in self.names():
            raise NotFound(f"profile '{name}' not found")
        return p

    def raw(self, name: str) -> str:
        return self.path(name).read_text(encoding="utf-8")

    def resolve(self, name: str, overrides: list[str] | None = None, _seen: tuple = ()) -> dict:
        if name in _seen:
            raise BadRequest(f"profile '{name}' extends itself")
        doc = yaml.safe_load(self.raw(name)) or {}
        parent = doc.pop("extends", None)
        cfg = deep_merge(self.resolve(Path(parent).stem, None, _seen + (name,)), doc) if parent else doc
        for item in overrides or []:
            key, sep, value = item.partition("=")
            if not sep or not key:
                raise BadRequest(f"override '{item}' must look like key=value")
            set_dotted(cfg, key.strip(), yaml.safe_load(value))
        return cfg

    def check_text(self, name: str, text: str) -> dict:
        """Validate a profile's YAML before it is written -> its resolved config. The same resolution as the trainer
        (extends + deep merge), and the keys every stage relies on."""
        if not NAME_RE.match(name):
            raise BadRequest("profile name: 2-41 characters, lowercase letters, digits and '_'")
        try:
            doc = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            raise BadRequest(f"not valid YAML: {e}") from e
        if not isinstance(doc, dict):
            raise BadRequest("a profile is a mapping of settings (key: value)")
        parent = doc.pop("extends", None)
        if parent is not None:
            parent = Path(str(parent)).stem
            if parent == name:
                raise BadRequest("a profile cannot extend itself")
            if parent not in self.names():
                raise BadRequest(f"extends: no profile '{parent}'")
            if name in self._ancestors(parent):
                raise BadRequest(f"extends: '{parent}' already extends '{name}' (a loop)")
            cfg = deep_merge(self.resolve(parent), doc)
        elif name != "base":
            raise BadRequest("a profile extends another one (extends: base.yaml): only base.yaml stands alone")
        else:
            cfg = doc
        for key in ("run_name", "stages", "gate"):
            if key not in cfg:
                raise BadRequest(f"the resolved profile has no '{key}'")
        if (cfg.get("gate") or {}).get("mode", "auto") not in ("auto", "hybrid", "llm"):
            raise BadRequest("gate.mode: auto, hybrid or llm")
        if not isinstance(cfg.get("stages"), list) or not cfg["stages"]:
            raise BadRequest("stages: a non-empty list")
        if not re.match(r"^[a-z0-9][a-z0-9_-]{1,62}$", str(cfg.get("run_name"))):
            raise BadRequest("run_name: lowercase letters, digits, '_' or '-'")
        return cfg

    def _ancestors(self, name: str) -> list[str]:
        out, seen = [], set()
        while name and name not in seen:
            seen.add(name)
            out.append(name)
            try:
                name = Path(str((yaml.safe_load(self.raw(name)) or {}).get("extends") or "")).stem
            except (NotFound, yaml.YAMLError):
                break
        return out

    def write(self, name: str, text: str) -> dict:
        """Validate, then write atomically (LF). Returns the resolved config."""
        cfg = self.check_text(name, text)
        text = text.replace("\r\n", "\n")
        if not text.endswith("\n"):
            text += "\n"
        path = self.root / f"{name}.yaml"
        old = path.read_bytes() if path.exists() else None
        write_atomic(path, text.encode("utf-8"))  # keeps the VM user as owner: editable by hand, updatable by git
        # every profile that extends this one must still resolve: otherwise put the old file back
        for child in (n for n in self.names() if n != name and name in self._ancestors(n)[1:]):
            try:
                self.check_text(child, self.raw(child))
            except BadRequest as e:
                if old is None:
                    path.unlink()
                else:
                    path.write_bytes(old)
                raise BadRequest(f"not saved: profile '{child}' extends '{name}' and would break: {e.message}") from e
        return cfg

    def create_from(self, name: str, parent: str, overrides: list[str], description: str = "") -> dict:
        """A new profile that extends `parent` and lists only `overrides` (key=value), as YAML with a comment header."""
        if not NAME_RE.match(name):
            raise BadRequest("profile name: 2-41 characters, lowercase letters, digits and '_'")
        if name in self.names():
            raise Conflict(f"profile '{name}' exists: pick another name, or edit it on the Profiles page")
        self.path(parent)  # NotFound when unknown
        doc: dict = {}
        for item in overrides:
            key, sep, value = item.partition("=")
            if not sep or not key.strip():
                raise BadRequest(f"override '{item}' must look like key=value")
            if key.strip() == "extends":
                raise BadRequest("extends is chosen separately")
            set_dotted(doc, key.strip(), yaml.safe_load(value))
        header = description.strip() or f"Saved from the Launch page: {parent} with {len(overrides)} change(s)"
        text = f"# {header}" + chr(10) + f"extends: {parent}.yaml" + chr(10)
        if doc:
            text += yaml.safe_dump(doc, sort_keys=False)
        return self.write(name, text)

    def delete(self, name: str) -> None:
        self.path(name)
        if name in BUILT_IN:
            raise Conflict(f"'{name}' ships with the repository: edit it, do not delete it")
        children = [n for n in self.names() if n != name and name in self._ancestors(n)[1:]]
        if children:
            raise Conflict(f"'{name}' is extended by {children}: change or delete those first")
        (self.root / f"{name}.yaml").unlink()

    def summary(self, name: str) -> dict:
        doc = yaml.safe_load(self.raw(name)) or {}
        first = next((line.lstrip("# ").strip() for line in self.raw(name).splitlines() if line.startswith("#")), "")
        cfg = self.resolve(name)
        api = cfg.get("api") or {}
        return {"name": name, "extends": doc.get("extends"), "description": first, "run_name": cfg.get("run_name"),
                "stages": cfg.get("stages", []), "gate_mode": (cfg.get("gate") or {}).get("mode"),
                "reuse_from": cfg.get("reuse_from"), "built_in": name in BUILT_IN,
                "serving_mode": "demo" if api.get("allow_forced_runs", False) else "production",
                "models": cfg.get("models") or {}}
