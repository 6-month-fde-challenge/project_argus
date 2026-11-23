"""Stream service API (:8003). Everything lives under /live so one load-balancer rule routes it; docs at /live/docs.

    uvicorn argus_stream.api:app --host 0.0.0.0 --port 8003

  GET  /live/health                     liveness (+ stream state, stored events)            open (load balancer)
  GET  /live/api/state                  pipeline: Kinesis, Fraud Model API, producer, lag  X-Control-Key
  GET  /live/api/summary?minutes=60     analytics: decisions, actions, live accuracy, ...  X-Control-Key
  GET  /live/api/events?limit=50&since= latest scored transactions (newest first)         X-Control-Key
  GET  /live/api/producer               traffic status                                     X-Control-Key
  POST /live/api/producer               {"action": "start" | "stop", "rate": 2}         X-Control-Key
  POST /live/api/inject                 {"n": 20, "fraud_type": "account_takeover"}     X-Control-Key
  POST /live/api/shadow                 {"run": "<trained run>" | null}                 X-Control-Key (MLOPS-06)
The ARGUS platform proxies these for the UI and holds the control key server-side (ARGUS_STREAM_CONTROL_KEY). Reads
need the key too: events carry customer, device and transaction data.
Env: KINESIS_ENABLED=1 runs the processor and the producer; STORE_PATH, RETENTION_HOURS, CONTROL_API_KEY, AUTO_START.
"""
from __future__ import annotations

import hmac
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from .generator import FRAUD_TYPES
from .processor import Processor
from .producer import Traffic
from .store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("argus.stream.api")
S: dict = {}


def store() -> Store:
    if "store" not in S:
        S["store"] = Store(os.getenv("STORE_PATH", "/data/argus_stream.db"), float(os.getenv("RETENTION_HOURS", "72")))
    return S["store"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.getenv("KINESIS_ENABLED") == "1":
        S["processor"] = Processor(store())
        S["processor"].start()
        S["traffic"] = Traffic()
        if os.getenv("AUTO_START", "0") == "1":
            S["traffic"].start()
    else:
        log.warning("KINESIS_ENABLED is not 1: API only, no stream processing")
    yield
    if S.get("traffic"):
        S["traffic"].stop()
    if S.get("processor"):
        S["processor"].stop()


app = FastAPI(title="ARGUS stream", version="1.0.0", docs_url="/live/docs", openapi_url="/live/openapi.json",
              lifespan=lifespan)


def pipeline() -> dict:
    p: Processor | None = S.get("processor")
    t: Traffic | None = S.get("traffic")
    if not p:
        return {"stream": "disabled", "consumed": 0, "scoring": None, "producer": {"running": False}}
    c = p.client
    return {**p.state, "uptime_s": round(time.time() - p.state["started"]),
            "scoring": {"url": c.url, "status": c.status, "model_version": c.model_version, "active_run": c.active_run,
                        "scored": c.scored, "errors": c.errors, "breaker_open": time.monotonic() < c.open_until,
                        "shadow_run": c.shadow_run, "shadow_status": c.shadow_status},
            "producer": t.status() if t else {"running": False}}


@app.get("/live/health")
def health():
    return {"status": "ok", "stream": pipeline().get("stream"), "events": store().head()}


def _key(x_control_key: str | None = Header(None)) -> None:
    """X-Control-Key must match CONTROL_API_KEY; with no key configured nothing but /live/health answers."""
    want = os.getenv("CONTROL_API_KEY", "")
    if not want or not hmac.compare_digest((x_control_key or "").encode(), want.encode()):
        raise HTTPException(401, "missing or invalid X-Control-Key")


keyed = [Depends(_key)]


@app.get("/live/api/state", dependencies=keyed)
def state():
    return {**pipeline(), "head": store().head()}


@app.get("/live/api/summary", dependencies=keyed)
def summary(minutes: int = Query(60, ge=1, le=60 * 24 * 7)):
    return store().summary(minutes)


@app.get("/live/api/events", dependencies=keyed)
def events(limit: int = Query(50, ge=1, le=1000), since: int = Query(0, ge=0)):
    return {"head": store().head(), "events": store().recent(limit, since)}


def _control(key: str | None) -> Traffic:
    _key(key)
    if "traffic" not in S:
        raise HTTPException(503, "Kinesis streaming is not enabled (KINESIS_ENABLED=1)")
    return S["traffic"]


class ProducerCommand(BaseModel):
    action: str = Field(pattern="^(start|stop)$")
    rate: float | None = Field(None, gt=0, le=50, description="transactions per second")


class Attack(BaseModel):
    n: int = Field(20, ge=1, le=100)
    fraud_type: str | None = Field(None, description=f"one of {FRAUD_TYPES}; omitted = a random ring's pattern")


class Shadow(BaseModel):
    run: str | None = Field(None, pattern=r"^[a-z0-9][a-z0-9_-]{1,62}$", description="trained run to shadow; null = off")


@app.post("/live/api/shadow")
def set_shadow(shadow: Shadow, x_control_key: str | None = Header(None)):
    """Score every transaction with a second run as well, for comparison (MLOPS-06). It never decides."""
    _control(x_control_key)
    if "processor" not in S:
        raise HTTPException(503, "Kinesis streaming is not enabled (KINESIS_ENABLED=1)")
    client = S["processor"].client
    client.shadow_run, client.shadow_open_until = shadow.run, 0.0
    client.shadow_status = "starting" if shadow.run else "off"
    log.info("shadow run: %s", shadow.run or "off")
    return {"shadow_run": shadow.run}


@app.get("/live/api/producer", dependencies=keyed)
def producer_status():
    t: Traffic | None = S.get("traffic")
    return t.status() if t else {"running": False}


@app.post("/live/api/producer")
def producer_control(cmd: ProducerCommand, x_control_key: str | None = Header(None)):
    t = _control(x_control_key)
    return t.start(cmd.rate) if cmd.action == "start" else t.stop()


@app.post("/live/api/inject")
def inject(attack: Attack, x_control_key: str | None = Header(None)):
    t = _control(x_control_key)
    if attack.fraud_type and attack.fraud_type not in FRAUD_TYPES:
        raise HTTPException(400, f"fraud_type must be one of {FRAUD_TYPES}")
    if time.time() - S.get("last_inject", 0) < 3:
        raise HTTPException(429, "one attack wave every 3 seconds")
    S["last_inject"] = time.time()
    try:
        sent = t.burst(attack.n, attack.fraud_type)
    except Exception as e:  # noqa: BLE001 - Kinesis unreachable / misconfigured: say so instead of a bare 500
        log.exception("attack wave failed")
        raise HTTPException(503, f"could not publish to Kinesis: {type(e).__name__}: {e}") from e
    return {"sent": sent, "fraud_type": attack.fraud_type or "mixed"}
