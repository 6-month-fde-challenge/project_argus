"""Profiles (extends / overrides) and the job service (validation, commands, scheduler, cancel)."""
from __future__ import annotations

import pytest

from app.core.errors import BadRequest, Conflict, NotFound
from app.features.jobs.service import JobRequest, JobService
from app.infra.job_runner import ContainerState
from app.infra.profiles_repo import ProfileRepository


def test_profile_extends_and_overrides(tree):
    repo = ProfileRepository(tree["configs"])
    cfg = repo.resolve("smoke", ["pilot.lr=0.001", "gate.mode=llm", "new.section.key=7"])
    assert cfg["run_name"] == "smoke" and cfg["pilot"]["max_steps"] == 3          # child wins
    assert cfg["pilot"]["lr"] == 0.001 and cfg["gate"]["min_uplift"] == 0.03      # deep merge + override
    assert cfg["gate"]["mode"] == "llm" and cfg["new"]["section"]["key"] == 7     # typed YAML values, new paths


@pytest.mark.parametrize("override", ["no_equals", "=5", "max_length.sub=1"])
def test_bad_overrides_are_rejected(tree, override):
    with pytest.raises(BadRequest):
        ProfileRepository(tree["configs"]).resolve("smoke", [override])


def test_unknown_and_self_extending_profiles(tree):
    repo = ProfileRepository(tree["configs"])
    with pytest.raises(NotFound):
        repo.resolve("missing")
    with pytest.raises(BadRequest):
        repo.resolve("loop")


def test_profile_summary_reads_first_comment(tree):
    s = ProfileRepository(tree["configs"]).summary("smoke")
    assert s["description"] == "Smoke profile: tiny sizes" and s["extends"] == "base.yaml" and len(s["stages"]) == 4


def test_command_contains_every_choice(container):
    req = JobRequest(profile="smoke", run_name="r1", overrides=["pilot.lr=0.001"], stages=["pilot", "gate"],
                     reuse_from="demo_run", skip_data_sync=True)
    cmd = JobService(container).build_command(req, "r1")
    assert cmd[:5] == ["python", "-m", "argus_ft.pipeline", "--config", "configs/smoke.yaml"]
    for part in ("run_name=r1", "pilot.lr=0.001", "reuse_from=demo_run", "pilot,gate", "--no-sync"):
        assert part in cmd


@pytest.mark.parametrize("req,error", [
    (dict(profile="smoke", run_name="Bad Name!"), BadRequest),
    (dict(profile="smoke", run_name="ok_run", stages=["not_a_stage"]), BadRequest),
    (dict(profile="smoke", run_name="ok_run", reuse_from="ghost"), BadRequest),
    (dict(profile="smoke", run_name="ok_run", reuse_from="ok_run"), BadRequest),
    (dict(profile="smoke", run_name="demo_run"), Conflict),            # exists, no stages, no allow_existing
    (dict(profile="nope", run_name="ok_run"), NotFound),
])
def test_validation_errors(container, req, error):
    with pytest.raises(error):
        JobService(container).validate(JobRequest(**req))


def test_existing_run_allowed_for_stage_reruns_or_explicitly(container):
    svc = JobService(container)
    assert svc.validate(JobRequest(profile="smoke", run_name="demo_run", stages=["gate"]))[0] == "demo_run"
    assert svc.validate(JobRequest(profile="smoke", run_name="demo_run", allow_existing=True))[0] == "demo_run"


def test_duplicate_active_job_is_a_conflict(container):
    svc = JobService(container)
    svc.create(JobRequest(profile="smoke", run_name="fresh"))
    with pytest.raises(Conflict):
        svc.create(JobRequest(profile="smoke", run_name="fresh"))


def test_scheduler_waits_for_gpu_then_runs_fifo(container, runner):
    svc = JobService(container)
    a = svc.create(JobRequest(profile="smoke", run_name="first"))
    b = svc.create(JobRequest(profile="smoke", run_name="second"))
    runner.holders = ["scoring"]
    svc.tick()
    assert runner.launched == [] and "stop serving" in container.jobs.get(a["id"])["message"]
    runner.holders = []
    svc.tick()
    assert [r for _, r, _ in runner.launched] == ["first"]          # oldest first, one at a time
    assert container.jobs.get(a["id"])["status"] == "running" and container.jobs.get(b["id"])["status"] == "queued"
    runner.states[f"cid-{a['id']}"] = ContainerState(running=False, exit_code=0)
    svc.tick()
    assert container.jobs.get(a["id"])["status"] == "succeeded"
    assert [r for _, r, _ in runner.launched] == ["first", "second"]


def test_scheduler_marks_failures_and_vanished_containers(container, runner):
    svc = JobService(container)
    a = svc.create(JobRequest(profile="smoke", run_name="crashy"))
    svc.tick()
    runner.states[f"cid-{a['id']}"] = ContainerState(running=False, exit_code=2)
    svc.tick()
    assert container.jobs.get(a["id"])["status"] == "failed" and container.jobs.get(a["id"])["exit_code"] == 2
    b = svc.create(JobRequest(profile="smoke", run_name="ghost_run"))
    svc.tick()
    runner.states.pop(f"cid-{b['id']}")
    svc.tick()
    assert container.jobs.get(b["id"])["message"] == "container disappeared"


def test_cancel_running_and_finished_jobs(container, runner):
    svc = JobService(container)
    a = svc.create(JobRequest(profile="smoke", run_name="stop_me"))
    svc.tick()
    assert svc.cancel(a["id"])["status"] == "cancelled" and runner.stopped == [f"cid-{a['id']}"]
    with pytest.raises(Conflict):
        svc.cancel(a["id"])
