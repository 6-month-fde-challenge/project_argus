"""/api/v1/stream - the real-time stream (Kinesis -> Fraud Model API -> decision -> action) for the Live stream page.

Reads are relayed as-is; traffic control adds the stream service's control key here, so the browser never holds it.
The stream service runs locally (docker compose --profile stream up -d) or on AWS (ARGUS_STREAM_URL = its URL).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...core.container import Container, get_container

router = APIRouter(prefix="/stream", tags=["stream"])


def _relay(result: tuple[int, object]) -> JSONResponse:
    status, body = result
    return JSONResponse(status_code=status, content=body)


class Traffic(BaseModel):
    action: str = Field(pattern="^(start|stop)$")
    rate: float | None = Field(None, gt=0, le=50, description="transactions per second")


class Attack(BaseModel):
    n: int = Field(20, ge=1, le=100)
    fraud_type: str | None = Field(None, description="account_takeover, card_testing, mule_account, impossible_travel")


class Shadow(BaseModel):
    run: str | None = Field(None, pattern=r"^[a-z0-9][a-z0-9_-]{1,62}$", description="trained run to shadow; null = off")


@router.post("/shadow")
async def shadow(body: Shadow, c: Container = Depends(get_container)):
    """Score every stream transaction with a second run too, for a live comparison (MLOPS-06); it never decides.
    First the Fraud Model API loads the run as its shadow (admin token, held here), then the stream starts using it."""
    status, reply = await (c.scoring.admin_call("POST", f"models/{body.run}/shadow", {"by": "ui", "reason": "shadow"})
                           if body.run else c.scoring.admin_call("DELETE", "models/shadow"))
    if status != 200:
        return JSONResponse(status_code=status, content=reply)
    return _relay(await c.stream.control_call("shadow", body.model_dump()))


@router.get("/state")
async def state(c: Container = Depends(get_container)):
    """Pipeline status: Kinesis, the Fraud Model API it scores with, the traffic producer, lag."""
    return _relay(await c.stream.get("state"))


@router.get("/summary")
async def summary(minutes: int = Query(60, ge=1, le=10080), c: Container = Depends(get_container)):
    return _relay(await c.stream.get("summary", minutes=minutes))


@router.get("/events")
async def events(limit: int = Query(50, ge=1, le=500), since: int = Query(0, ge=0), c: Container = Depends(get_container)):
    return _relay(await c.stream.get("events", limit=limit, since=since))


@router.post("/producer")
async def producer(cmd: Traffic, c: Container = Depends(get_container)):
    return _relay(await c.stream.control_call("producer", cmd.model_dump(exclude_none=True)))


@router.post("/inject")
async def inject(attack: Attack, c: Container = Depends(get_container)):
    """One attack wave from a fraud ring (shows the BLOCK path and, on AWS, the SNS alert)."""
    return _relay(await c.stream.control_call("inject", attack.model_dump(exclude_none=True)))
