"""/api/v1/scoring - the Fraud Model API through the platform (tokens stay server-side).

Responses keep the scoring API's own status codes and bodies (e.g. 400 INVALID_REQUEST with the field), so the
UI shows exactly what the Payment API would see.
"""
from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse

from ...core.container import Container, get_container
from ...core.errors import BadRequest
from .samples import sample_requests

router = APIRouter(prefix="/scoring", tags=["scoring"])
LISTS = {"blocked_devices", "allowlisted_customers"}


def _relay(result: tuple[int, object]) -> JSONResponse:
    status, body = result
    return JSONResponse(status_code=status, content=body)


@router.post("/score")
async def score(body: dict = Body(...), c: Container = Depends(get_container)):
    return _relay(await c.scoring.score(body))


@router.get("/health")
async def health(c: Container = Depends(get_container)):
    return _relay(await c.scoring.health())


@router.get("/samples")
def samples(c: Container = Depends(get_container)):
    """Ready-made CRD 10.2 requests: the CRD worked example plus real validation transactions."""
    return sample_requests(c.settings.data_dir)


@router.get("/models")
async def models(c: Container = Depends(get_container)):
    return _relay(await c.scoring.admin_call("GET", "models"))


@router.post("/models/{run}/activate")
async def activate(run: str, c: Container = Depends(get_container)):
    return _relay(await c.scoring.admin_call("POST", f"models/{run}/activate"))


@router.get("/admin/thresholds")
async def get_thresholds(c: Container = Depends(get_container)):
    return _relay(await c.scoring.admin_call("GET", "thresholds"))


@router.post("/admin/thresholds")
async def set_thresholds(body: dict = Body(...), c: Container = Depends(get_container)):
    return _relay(await c.scoring.admin_call("POST", "thresholds", body))


@router.get("/admin/lists/{name}")
async def get_list(name: str, c: Container = Depends(get_container)):
    if name not in LISTS:
        raise BadRequest(f"list must be one of {sorted(LISTS)}")
    return _relay(await c.scoring.admin_call("GET", f"lists/{name}"))


@router.post("/admin/lists/{name}")
async def add_to_list(name: str, body: dict = Body(...), c: Container = Depends(get_container)):
    if name not in LISTS:
        raise BadRequest(f"list must be one of {sorted(LISTS)}")
    return _relay(await c.scoring.admin_call("POST", f"lists/{name}", body))


@router.delete("/admin/lists/{name}")
async def remove_from_list(name: str, body: dict = Body(...), c: Container = Depends(get_container)):
    if name not in LISTS:
        raise BadRequest(f"list must be one of {sorted(LISTS)}")
    return _relay(await c.scoring.admin_call("DELETE", f"lists/{name}", body))


@router.get("/admin/decisions/{transaction_id}")
async def decision(transaction_id: str, c: Container = Depends(get_container)):
    return _relay(await c.scoring.admin_call("GET", f"decisions/{transaction_id}"))


@router.get("/admin/decisions")
async def recent_decisions(c: Container = Depends(get_container)):
    return _relay(await c.scoring.admin_call("GET", "decisions"))


@router.get("/admin/changelog")
async def changelog(c: Container = Depends(get_container)):
    return _relay(await c.scoring.admin_call("GET", "changelog"))
