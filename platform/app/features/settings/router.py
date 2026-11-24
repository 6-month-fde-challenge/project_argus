"""/api/v1/settings - the repository's .env through the UI (the same file compose, the scripts and ops.py read).

GET  /settings            every known setting with its value (secrets: only whether set), the serving mode, the agent
PUT  /settings            {changes: {KEY: value | null | "@generate"}, approval?, apply?} -> validated, written, logged;
                          apply: queue "Apply settings" (the agent recreates the affected services)
POST /settings/apply      {services: [...]} -> queue "Apply settings" for these services
GET  /settings/changes    the UI change log (settings, profiles, operations, activation, AWS)
By hand it is the same: edit .env, then `bash deploy/hyperstack/apply.sh <services>` (VM) / `.\\argus.ps1 apply ...`.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from ...core.container import Container, get_container
from ...core.governance import Approval
from .service import SettingsService

router = APIRouter(prefix="/settings", tags=["settings"])


class SettingsChange(BaseModel):
    changes: dict[str, str | None] = Field(description='KEY -> new value, null to remove, "@generate" for a new token')
    approval: Approval | None = None
    apply: bool = Field(False, description="queue Apply settings for the affected services right away")


class ApplyRequest(BaseModel):
    services: list[str]
    approval: Approval | None = None


def service(c: Container = Depends(get_container)) -> SettingsService:
    return SettingsService(c)


@router.get("")
async def get_settings(svc: SettingsService = Depends(service)):
    return await svc.view()


@router.put("")
async def save_settings(req: SettingsChange, svc: SettingsService = Depends(service)):
    return await svc.save(req.changes, req.approval, req.apply)


@router.post("/apply")
def apply(req: ApplyRequest, svc: SettingsService = Depends(service)):
    return svc.apply(req.services, req.approval)


@router.get("/changes")
def changes(limit: int = Query(100, ge=1, le=1000), kind: str | None = None, c: Container = Depends(get_container)):
    return c.changes.recent(limit, kind)
