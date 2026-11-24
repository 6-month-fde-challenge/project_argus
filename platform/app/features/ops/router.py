"""/api/v1/ops - run the host's operations (deploy/hyperstack/*.sh, argus.ps1) from the UI, through the agent.

GET  /ops/catalog                  the actions (deploy/ops_actions.json) with defaults, availability and approval
GET  /ops/agent                    is the operations agent running on the host, and where (linux VM / windows PC)
POST /ops/preview                  {action, params} -> the exact command (validated; nothing is queued)
POST /ops/actions                  {action, params, approval?} -> queued (validated exactly as the agent will)
GET  /ops/actions                  history, newest first
GET  /ops/actions/{id}             one action; GET .../log?offset= its output so far; POST .../cancel stops it
Every action shows the command it runs: the same command works by hand, without the UI.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from ...core.container import Container, get_container
from ...core.governance import Approval
from .service import OpsService

router = APIRouter(prefix="/ops", tags=["operations"])


class ActionRequest(BaseModel):
    action: str
    params: dict = Field(default_factory=dict)
    approval: Approval | None = None


def service(c: Container = Depends(get_container)) -> OpsService:
    return OpsService(c)


@router.get("/agent")
def agent(c: Container = Depends(get_container)):
    return c.ops.agent()


@router.get("/catalog")
async def catalog(svc: OpsService = Depends(service)):
    return await svc.catalog()


@router.post("/preview")
def preview(req: ActionRequest, c: Container = Depends(get_container)):
    return c.ops.prepare(req.action, req.params)


@router.post("/actions")
async def submit(req: ActionRequest, svc: OpsService = Depends(service)):
    return await svc.submit(req.action, req.params, req.approval)


@router.get("/actions")
def history(limit: int = Query(50, ge=1, le=500), c: Container = Depends(get_container)):
    return c.ops.list(limit)


@router.get("/actions/{rid}")
def get_action(rid: str, c: Container = Depends(get_container)):
    return c.ops.get(rid)


@router.get("/actions/{rid}/log")
def get_log(rid: str, offset: int = Query(0, ge=0), c: Container = Depends(get_container)):
    return {**c.ops.log(rid, offset), "status": c.ops.get(rid)["status"]}


@router.post("/actions/{rid}/cancel")
def cancel(rid: str, svc: OpsService = Depends(service)):
    return svc.cancel(rid)
