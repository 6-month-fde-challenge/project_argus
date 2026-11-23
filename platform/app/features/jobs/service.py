"""Training jobs: validate a run request, queue it, and keep job state in sync with its container.

One GPU job at a time (FIFO). A queued job also waits while the serving stack (scoring API / vLLM) holds the
GPU, with the reason shown to the user, instead of starting and running out of memory on a shared 4 GB card.
"""
from __future__ import annotations

import asyncio
import logging
import re

from pydantic import BaseModel, Field

from ...core.container import Container
from ...core.errors import BadRequest, Conflict, UpstreamError
from ...infra.job_store import ACTIVE, now
from ..runs.service import DEFAULT_LADDER

log = logging.getLogger("argus.jobs")
RUN_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")


class JobRequest(BaseModel):
    profile: str = Field(description="training profile, e.g. smoke, demo_hybrid, demo_llm_scorer")
    run_name: str | None = Field(None, description="defaults to the profile's run_name")
    overrides: list[str] = Field(default_factory=list, description="dotted key=value, e.g. dpo.max_pairs=200")
    stages: list[str] | None = Field(None, description="subset of stages to (re)run; default = the profile's")
    reuse_from: str | None = Field(None, description="seed this run from another run's artefacts")
    skip_data_sync: bool = Field(True, description="use the cached Supabase data (sync downloads only changes)")
    allow_existing: bool = Field(False, description="continue in an existing run folder instead of refusing")


class JobService:
    def __init__(self, c: Container):
        self.c = c

    # ------------------------------------------------------------ create / cancel

    def build_command(self, req: JobRequest, run_name: str) -> list[str]:
        cmd = ["python", "-m", "argus_ft.pipeline", "--config", f"configs/{req.profile}.yaml", "--set", f"run_name={run_name}"]
        for o in req.overrides:
            cmd += ["--set", o]
        if req.reuse_from:
            cmd += ["--set", f"reuse_from={req.reuse_from}"]
        if req.stages:
            cmd += ["--stages", ",".join(req.stages)]
        if req.skip_data_sync:
            cmd.append("--no-sync")
        return cmd

    def validate(self, req: JobRequest) -> tuple[str, dict]:
        cfg = self.c.profiles.resolve(req.profile, req.overrides)  # unknown profile / bad override -> 4xx
        run_name = req.run_name or cfg.get("run_name")
        if not run_name or not RUN_NAME.match(run_name):
            raise BadRequest("run_name must be 2-63 chars: lowercase letters, digits, '_' or '-'")
        if req.stages:
            unknown = set(req.stages) - set(DEFAULT_LADDER)
            if unknown:
                raise BadRequest(f"unknown stages: {sorted(unknown)}; known: {DEFAULT_LADDER}")
        reuse = req.reuse_from or cfg.get("reuse_from")
        if reuse:
            if reuse == run_name:
                raise BadRequest("a run cannot reuse itself")
            if reuse not in self.c.runs.names():
                raise BadRequest(f"reuse_from: run '{reuse}' does not exist yet - train it first")
        if any(j["run_name"] == run_name for j in self.c.jobs.active()):
            raise Conflict(f"run '{run_name}' already has a queued or running job")
        if run_name in self.c.runs.names() and not req.allow_existing and not req.stages:
            raise Conflict(f"run '{run_name}' already exists: pick another name, re-run selected stages, "
                           "or set allow_existing to continue in it")
        return run_name, {**cfg, "run_name": run_name}  # what the job will really run with

    def create(self, req: JobRequest) -> dict:
        run_name, cfg = self.validate(req)
        job = self.c.jobs.create(run_name=run_name, profile=req.profile, overrides=req.overrides, stages=req.stages,
                                 reuse_from=req.reuse_from or cfg.get("reuse_from"),
                                 command=self.build_command(req, run_name))
        # the run exists from this moment (listed, its page loads with the planned ladder), not only once the
        # container has imported torch and written run_start
        self.c.runs.append_event(run_name, "job_queued", job_id=job["id"], profile=f"configs/{req.profile}.yaml",
                                 stages=req.stages or cfg.get("stages", []))
        return job

    def cancel(self, job_id: str) -> dict:
        # mark it cancelled first (only if still active), then stop whatever container it has by now: if the scheduler
        # launches it meanwhile, its own compare-and-set to "running" fails and it stops the container itself
        job = self.c.jobs.update(job_id, expect=ACTIVE, status="cancelled", message="cancelled by user")
        if job is None:
            raise Conflict(f"job is already {self.c.jobs.get(job_id)['status']}")
        if job["container_id"]:
            self.c.runner.stop(job["container_id"])
        return job

    # ------------------------------------------------------------ scheduler

    def tick(self) -> None:
        """One reconciliation pass: finish exited jobs, then start the next queued one if the GPU is free."""
        for job in self.c.jobs.active():
            if job["status"] != "running":
                continue
            st = self.c.runner.state(job["container_id"])
            if st is None:
                self.c.jobs.update(job["id"], expect={"running"}, status="failed", message="container disappeared")
            elif not st.running:
                self.c.jobs.update(job["id"], expect={"running"}, status="succeeded" if st.exit_code == 0 else "failed",
                                   exit_code=st.exit_code,
                                   message=None if st.exit_code == 0 else f"exited with code {st.exit_code}")
        if any(j["status"] == "running" for j in self.c.jobs.active()):
            return
        nxt = self.c.jobs.next_queued()
        if nxt is None:
            return
        holders = self.c.runner.gpu_holders()
        if holders:
            msg = f"waiting for the GPU: stop serving ({', '.join(holders)}) to start training"
            if nxt.get("message") != msg:
                self.c.jobs.update(nxt["id"], message=msg)
            return
        cid = self.c.runner.launch(nxt["id"], nxt["run_name"], nxt["command"])
        if self.c.jobs.update(nxt["id"], expect={"queued"}, status="running", container_id=cid, started_at=now(),
                              message=None) is None:  # cancelled while its container was starting
            self.c.runner.stop(cid)
            log.info("job %s was cancelled during launch: stopped container %s", nxt["id"], cid[:12])
            return
        log.info("started job %s (%s) in container %s", nxt["id"], nxt["run_name"], cid[:12])


async def scheduler_loop(c: Container, stop: asyncio.Event) -> None:
    svc = JobService(c)
    while not stop.is_set():
        try:
            await asyncio.to_thread(svc.tick)
        except UpstreamError as e:  # Docker not reachable: keep the API up, retry later
            log.warning("scheduler: %s", e.message)
        except Exception:  # noqa: BLE001 - the loop must survive any single failure
            log.exception("scheduler tick failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=c.settings.job_poll_seconds)
        except asyncio.TimeoutError:
            pass
