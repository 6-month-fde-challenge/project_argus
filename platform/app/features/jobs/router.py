"""/api/v1/jobs - launch, list, cancel training jobs and stream their container logs."""
from __future__ import annotations

import asyncio
import re

from fastapi import APIRouter, Depends, Query

from ...core.container import Container, get_container
from ...core.sse import sse_message, sse_response
from .service import JobRequest, JobService

router = APIRouter(prefix="/jobs", tags=["jobs"])
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def service(c: Container = Depends(get_container)) -> JobService:
    return JobService(c)


@router.get("")
def list_jobs(c: Container = Depends(get_container)):
    return c.jobs.all()


@router.post("", status_code=201)
def create_job(req: JobRequest, svc: JobService = Depends(service)):
    """Queue a pipeline run; it starts as soon as no other job is running and serving does not hold the GPU."""
    return svc.create(req)


@router.post("/preview")
def preview(req: JobRequest, svc: JobService = Depends(service)):
    """Validate a request and show the exact command and resolved config, without queueing anything."""
    run_name, cfg = svc.validate(req)
    return {"run_name": run_name, "command": svc.build_command(req, run_name), "config": cfg}


@router.get("/{job_id}")
def get_job(job_id: str, c: Container = Depends(get_container)):
    return c.jobs.get(job_id)


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, svc: JobService = Depends(service)):
    return svc.cancel(job_id)


@router.get("/{job_id}/logs/stream")
async def stream_logs(job_id: str, tail: int = Query(300, ge=0, le=5000), c: Container = Depends(get_container)):
    """SSE of the job container's stdout/stderr (progress-bar carriage returns become separate lines)."""
    job = c.jobs.get(job_id)

    async def stream():
        if not job["container_id"]:
            yield sse_message({"line": f"job is {job['status']}: {job.get('message') or 'waiting to start'}"}, event="log")
            yield sse_message({}, event="end")
            return
        it = c.runner.logs(job["container_id"], follow=True, tail=tail)
        buffer = ""
        while True:
            chunk = await asyncio.to_thread(next, it, None)  # docker's log iterator blocks: keep it off the event loop
            if chunk is None:
                break
            buffer += ANSI.sub("", chunk.decode("utf-8", errors="replace"))
            *lines, buffer = re.split(r"[\r\n]", buffer)
            for line in lines:
                if line.strip():
                    yield sse_message({"line": line}, event="log")
        if buffer.strip():
            yield sse_message({"line": buffer}, event="log")
        yield sse_message({}, event="end")

    return sse_response(stream())
