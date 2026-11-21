"""/api/v1/runs - training runs: list, detail (ladder + sections), report, loss series, events (JSON + SSE)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import PlainTextResponse

from ...core.container import Container, get_container
from ...core.errors import BadRequest, NotFound
from ...core.sse import sse_message, sse_response
from ...infra.events import follow_events
from .service import RunService

router = APIRouter(prefix="/runs", tags=["runs"])


def service(c: Container = Depends(get_container)) -> RunService:
    return RunService(c)


@router.get("")
def list_runs(svc: RunService = Depends(service)):
    return svc.list()


@router.get("/{run}")
def get_run(run: str, svc: RunService = Depends(service)):
    return svc.detail(run)


@router.get("/{run}/report.md", response_class=PlainTextResponse)
def report_markdown(run: str, c: Container = Depends(get_container)):
    md = c.runs.report_md(run)
    if md is None:
        raise NotFound(f"run '{run}' has no report.md yet (it is written by `argus_ft.compare`)")
    return md


@router.get("/{run}/series")
def loss_series(run: str, svc: RunService = Depends(service)):
    return svc.loss_series(run)


@router.get("/{run}/artifacts")
def artifacts(run: str, c: Container = Depends(get_container)):
    return c.runs.artifacts(run)


@router.get("/{run}/files/{path:path}", response_class=PlainTextResponse)
def run_file(run: str, path: str, c: Container = Depends(get_container)):
    """Small text artefacts (samples, calibrators, audit), path-checked to stay inside the run folder."""
    return c.runs.file_text(run, path)


@router.get("/{run}/events")
def events(run: str, after: int = Query(0, ge=0), limit: int = Query(500, ge=1, le=5000),
           tail: int | None = Query(None, ge=1, le=2000, description="last N events instead of reading from `after`"),
           c: Container = Depends(get_container)):
    evs, nxt = c.runs.tail(run, tail) if tail else c.runs.events(run, after=after, limit=limit)
    return {"events": evs, "next": nxt}


@router.get("/{run}/events/stream")
async def events_stream(run: str, after: int = Query(0, ge=0), last_event_id: str | None = Header(None),
                        svc: RunService = Depends(service), c: Container = Depends(get_container)):
    """SSE: batches of new events; the id is the byte offset, so EventSource resumes exactly after a reconnect."""
    offset = int(last_event_id) if last_event_id and last_event_id.isdigit() else after
    path = c.runs.events_path(run)

    async def stream():
        async for batch, nxt in follow_events(path, offset, lambda: svc.is_live(run)):
            yield sse_message(batch, event="events", id_=nxt)
        yield sse_message({"run": run}, event="end")

    return sse_response(stream())


compare_router = APIRouter(tags=["runs"])


@compare_router.get("/compare")
def compare(runs: str = Query(..., description="comma-separated run names"), svc: RunService = Depends(service)):
    names = [r for r in runs.split(",") if r]
    if not 1 <= len(names) <= 4:
        raise BadRequest("compare 1 to 4 runs")
    return svc.compare(names)
