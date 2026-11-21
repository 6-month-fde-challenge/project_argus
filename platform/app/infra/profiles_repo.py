"""Training profiles (training/configs/*.yaml): raw YAML (with its comments) and the resolved config.

Resolution follows the trainer exactly: a profile may `extends:` another and overrides only what differs; nested
sections deep-merge, everything else replaces.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from ..core.errors import BadRequest, NotFound


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

    def summary(self, name: str) -> dict:
        doc = yaml.safe_load(self.raw(name)) or {}
        first = next((line.lstrip("# ").strip() for line in self.raw(name).splitlines() if line.startswith("#")), "")
        cfg = self.resolve(name)
        return {"name": name, "extends": doc.get("extends"), "description": first, "run_name": cfg.get("run_name"),
                "stages": cfg.get("stages", []), "gate_mode": (cfg.get("gate") or {}).get("mode"),
                "reuse_from": cfg.get("reuse_from")}
