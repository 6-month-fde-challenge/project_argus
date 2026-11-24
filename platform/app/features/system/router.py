"""/api/v1/health, /system, /services - platform health, GPU telemetry, and the serving stack."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from ...core.container import Container, get_container
from ...core.errors import BadRequest, UpstreamError
from ...core.governance import configured_mode

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok", "service": "argus-platform"}


@router.get("/system/gpu")
def gpu(c: Container = Depends(get_container)):
    return c.gpu.snapshot()


@router.get("/services")
async def services(c: Container = Depends(get_container)):
    """Health of every service the UI depends on, checked concurrently. A serving container that Docker reports
    as stopped is not probed over HTTP, so a stopped stack answers instantly instead of after a probe timeout."""
    try:
        containers = await asyncio.to_thread(c.runner.service_status)
        docker = "ok"
    except UpstreamError as e:
        containers, docker = {}, e.message

    async def probe(name: str, client, path: str) -> bool:
        if containers.get(name, "running") != "running":  # unknown (no Docker) -> probe anyway
            return False
        return await client.up(path)

    async def scoring_health() -> dict | None:  # the body too: serving mode and whether the live run is a demo
        if containers.get("scoring", "running") != "running":
            return None
        try:
            status, body = await c.scoring.request("GET", "/v1/health", timeout=2.0)
        except UpstreamError:
            return None
        return body if status < 500 and isinstance(body, dict) else None

    health, mlflow, explainer, stream = await asyncio.gather(scoring_health(), c.mlflow.up("/health", timeout=5.0),
                                                             probe("explainer", c.explainer, "/health"),
                                                             c.stream.up("/live/health"))
    scoring = health is not None
    return {
        "platform": "ok", "docker": docker, "containers": containers,
        "scoring": "ok" if scoring else "down", "explainer": "ok" if explainer else "down",
        "mlflow": "ok" if mlflow else "down", "mlflow_url": c.settings.mlflow_public_url,
        "stream": "ok" if stream else "down",
        "gpu_jobs": [j["run_name"] for j in c.jobs.active() if j["status"] == "running"],
        # production = demo runs refused; demo_model = the API serves a demo setup right now (top-bar badge)
        "serving_mode": (health or {}).get("serving_mode") or configured_mode(c)["mode"],
        "active_run": (health or {}).get("active_run"),
        "demo_model": bool(health) and ((health or {}).get("serving_mode") != "production" or bool(health.get("active_run_forced"))),
        "agent": c.ops.agent().get("online", False),
    }


@router.post("/services/{service}/{action}")
async def control_service(service: str, action: str, c: Container = Depends(get_container)):
    """Start / stop the serving containers (scoring API, vLLM) to hand the GPU to training and back."""
    if service not in c.settings.gpu_service_names or action not in ("start", "stop"):
        raise BadRequest(f"service must be one of {list(c.settings.gpu_service_names)} and action start|stop")
    if action == "start" and any(j["status"] == "running" for j in c.jobs.active()):
        raise BadRequest("a training job is using the GPU: wait for it or cancel it first")
    await asyncio.to_thread(c.runner.set_service, service, action == "start")
    return {"service": service, "action": action, "containers": await asyncio.to_thread(c.runner.service_status)}
