"""/api/v1/profiles - training profiles (training/configs/*.yaml): read, edit, create, delete.

GET    /profiles                  every profile (summary)
GET    /profiles/{name}           raw YAML (with comments) + the resolved config (+ ?override=key=value)
POST   /profiles/{name}/validate  {yaml} -> the resolved config, or what is wrong (nothing is written)
PUT    /profiles/{name}           {yaml, reason?} -> written (new or changed), logged
POST   /profiles                  {name, extends, overrides: ["key=value"], description?} -> "save as profile"
DELETE /profiles/{name}           profiles created here only (the shipped ones are edited, never deleted)
The files are tracked by git: a change made here shows in `git status` (commit it to keep it). By hand: edit the YAML.
"""
from __future__ import annotations

import yaml
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from ...core.container import Container, get_container
from ...core.errors import AppError, Conflict

router = APIRouter(prefix="/profiles", tags=["profiles"])


class ProfileText(BaseModel):
    yaml: str = Field(max_length=200_000)
    reason: str = ""


class SaveAs(BaseModel):
    name: str
    extends: str
    overrides: list[str] = Field(default_factory=list)
    description: str = ""


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


@router.post("/{name}/validate")
def validate_profile(name: str, body: ProfileText, c: Container = Depends(get_container)):
    return {"ok": True, "resolved": c.profiles.check_text(name, body.yaml)}


@router.put("/{name}")
def save_profile(name: str, body: ProfileText, c: Container = Depends(get_container)):
    if any(j.get("profile") == name for j in c.jobs.active()):
        raise Conflict(f"a queued or running job uses '{name}': wait for it (the job reads the file when it starts)")
    existed = name in c.profiles.names()
    old = c.profiles.raw(name) if existed else ""
    cfg = c.profiles.write(name, body.yaml)
    c.changes.add("profile", f"{'changed' if existed else 'created'} profile {name}",
                  {"lines_before": len(old.splitlines()), "lines_after": len(body.yaml.splitlines()),
                   "gate_mode": (cfg.get("gate") or {}).get("mode")}, {"reason": body.reason.strip()} if body.reason else None)
    return c.profiles.summary(name)


@router.post("")
def save_as(body: SaveAs, c: Container = Depends(get_container)):
    """The Launch page's "save as profile": a new profile = extends + only the changed values."""
    c.profiles.create_from(body.name, body.extends, body.overrides, body.description)
    c.changes.add("profile", f"created profile {body.name} (extends {body.extends})", {"overrides": body.overrides})
    return c.profiles.summary(body.name)


@router.delete("/{name}")
def delete_profile(name: str, c: Container = Depends(get_container)):
    if any(j.get("profile") == name for j in c.jobs.active()):
        raise Conflict(f"a queued or running job uses '{name}'")
    c.profiles.delete(name)
    c.changes.add("profile", f"deleted profile {name}")
    return {"deleted": name}
