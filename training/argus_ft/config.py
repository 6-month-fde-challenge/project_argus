"""Pipeline configuration: one YAML file, dotted --set overrides (e.g. sft_scorer.epochs=2)."""
from __future__ import annotations

from pathlib import Path

import yaml


class Config(dict):
    """dict with attribute access for nested sections: cfg.sft_scorer.lr"""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as e:
            raise AttributeError(f"config has no key '{name}'") from e
        return Config(value) if isinstance(value, dict) else value


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path: Path) -> dict:
    """A profile can `extends: base.yaml` and override only what differs for its environment."""
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    return _deep_merge(_read(path.parent / parent), raw) if parent else raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    raw = _read(Path(path))
    for item in overrides or []:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = raw
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return Config(raw)
