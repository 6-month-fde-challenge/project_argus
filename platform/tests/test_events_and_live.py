"""Event tailing (byte offsets, partial / corrupt lines) and the incremental live-state fold."""
from __future__ import annotations

import json
import os
import time

import pytest

from app.features.runs.live import LiveStateCache, fold, _empty
from app.infra.events import follow_events, read_events, tail_events
from app.infra.runs_repo import RunRepository

from .conftest import ev, write_events


def test_read_events_resumes_from_offset(tmp_path):
    p = tmp_path / "events.jsonl"
    write_events(p, [ev("run_start"), ev("stage_start", "a")])
    first, off = read_events(p, 0, limit=1)
    rest, end = read_events(p, off)
    assert [e["type"] for e in first] == ["run_start"]
    assert [e["type"] for e in rest] == ["stage_start"]
    assert end == p.stat().st_size


def test_partial_last_line_is_left_for_the_next_read(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text(json.dumps(ev("run_start")) + "\n" + '{"ts": "x", "type": "stage_st')  # writer mid-line
    events, off = read_events(p, 0)
    assert len(events) == 1 and off < p.stat().st_size
    with p.open("a") as f:
        f.write('art", "stage": "a", "data": {}}\n')
    more, _ = read_events(p, off)
    assert more[0]["type"] == "stage_start"


def test_corrupt_line_is_skipped_not_fatal(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text(json.dumps(ev("run_start")) + "\n{not json}\n" + json.dumps(ev("run_end", status="ok")) + "\n")
    events, _ = read_events(p, 0)
    assert [e["type"] for e in events] == ["run_start", "run_end"]


def test_missing_file_reads_as_empty(tmp_path):
    assert read_events(tmp_path / "nope.jsonl", 7) == ([], 7)
    assert tail_events(tmp_path / "nope.jsonl", 5) == ([], 0)


def test_tail_reads_only_the_end_and_drops_a_partial_first_line(tmp_path):
    p = tmp_path / "events.jsonl"
    write_events(p, [ev("train_step", "s", step=i) for i in range(500)])
    last, end = tail_events(p, 3, window=400)  # window cuts mid-line
    assert [e["data"]["step"] for e in last] == [497, 498, 499]
    assert end == p.stat().st_size


async def test_follow_stops_at_run_end(tmp_path):
    p = tmp_path / "events.jsonl"
    write_events(p, [ev("run_start"), ev("run_end", status="ok")])
    batches = [b async for b, _ in follow_events(p, 0, is_live=lambda: True, poll_s=0.01)]
    assert batches and batches[-1][-1]["type"] == "run_end"


async def test_follow_stops_when_run_is_no_longer_live(tmp_path):
    p = tmp_path / "events.jsonl"
    write_events(p, [ev("run_start")])
    batches = [b async for b, _ in follow_events(p, 0, is_live=lambda: False, poll_s=0.01)]
    assert len(batches) == 1


def test_fold_full_lifecycle():
    s = _empty()
    for e in [ev("run_start", stages=["a", "b"], profile="p"), ev("stage_start", "a"),
              ev("train_step", "a", step=3, max_steps=10, loss=0.5), ev("progress", "a", label="scoring", done=1, total=4),
              ev("metric", "a", name="ppl", value=12.0), ev("stage_end", "a", status="ok", minutes=1.0, peak_gpu_gb=1.2),
              ev("stage_skipped", "b", reason="gate"), ev("run_end", status="ok", total_minutes=2.0)]:
        fold(s, e)
    assert s["status"] == "succeeded" and s["current_stage"] is None
    assert s["stages"]["a"]["status"] == "ok" and s["stages"]["a"]["last_step"]["loss"] == 0.5
    assert s["stages"]["a"]["metrics"] == {"ppl": 12.0}
    assert s["stages"]["b"]["status"] == "skipped"


def test_fold_failed_stage_and_rerun_resets_only_its_stages():
    s = _empty()
    for e in [ev("run_start", stages=["a", "b"]), ev("stage_start", "a"), ev("stage_end", "a", status="ok"),
              ev("stage_start", "b"), ev("stage_end", "b", status="failed", error="boom"), ev("run_end", status="failed")]:
        fold(s, e)
    assert s["status"] == "failed" and s["stages"]["b"]["error"] == "boom"
    fold(s, ev("run_start", stages=["b"]))  # re-run of stage b only
    assert s["status"] == "running" and s["stages"]["a"]["status"] == "ok" and s["stages"]["b"]["status"] == "pending"


def test_live_cache_is_incremental_and_detects_replacement(tree):
    repo = RunRepository(tree["outputs"])
    cache = LiveStateCache(repo)
    assert cache.get("demo_run")["status"] == "succeeded"
    path = repo.events_path("demo_run")
    write_events(path, [ev("run_start", stages=["x"])])  # file replaced by a shorter one: start over
    st = cache.get("demo_run")
    assert st["status"] == "running" and st["live"] is True


def test_silent_running_run_becomes_interrupted(tree):
    repo = RunRepository(tree["outputs"])
    path = repo.events_path("demo_run")
    write_events(path, [ev("run_start", stages=["x"]), ev("stage_start", "x")])
    old = time.time() - 3600
    os.utime(path, (old, old))
    st = LiveStateCache(repo).get("demo_run")
    assert st["status"] == "interrupted" and st["live"] is False


@pytest.mark.parametrize("name", ["../etc", "mlruns", "_platform", "_scoring", ".hidden", "does_not_exist"])
def test_run_dir_rejects_traversal_reserved_and_unknown(tree, name):
    from app.core.errors import NotFound
    with pytest.raises(NotFound):
        RunRepository(tree["outputs"]).dir(name)


def test_job_queued_sets_the_plan_only_before_the_run_starts():
    s = _empty()
    fold(s, ev("job_queued", stages=["a", "b"], profile="configs/smoke.yaml"))
    assert s["planned"] == ["a", "b"] and s["status"] == "idle" and s["profile"] == "configs/smoke.yaml"
    fold(s, ev("run_start", stages=["a"]))
    fold(s, ev("job_queued", stages=["z"]))                       # a later queued re-run never rewrites a live plan
    assert s["planned"] == ["a"] and s["status"] == "running"
