"""/api/v1/profiles - training profiles: summaries, raw YAML (with comments) and the resolved config."""
from __future__ import annotations

import yaml
from fastapi import APIRouter, Depends, Query

from ...core.container import Container, get_container
from ...core.errors import AppError

router = APIRouter(prefix="/profiles", tags=["profiles"])


@router.get("")
def list_profiles(c: Container = Depends(get_container)):
    """Every profile; a broken one (bad YAML, extends loop) is listed with its error instead of failing the list."""
    out = []
    for n in c.profiles.names():
        try:
            out.append(c.profiles.summary(n))
        except (AppError, yaml.YAMLError) as e:
            out.append({"name": n, "extends": None, "description": "", "run_name": None, "stages": [],
                        "gate_mode": None, "reuse_from": None, "error": str(e)})
    return out


@router.get("/{name}")
def get_profile(name: str, override: list[str] = Query(default=[]), c: Container = Depends(get_container)):
    return {**c.profiles.summary(name), "yaml": c.profiles.raw(name), "resolved": c.profiles.resolve(name, override)}
