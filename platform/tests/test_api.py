"""HTTP contract of the platform API: every router, its happy path and its edge cases."""
from __future__ import annotations

import pytest


def test_health_and_openapi(client):
    assert client.get("/api/v1/health").json() == {"status": "ok", "service": "argus-platform"}
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/runs/{run}/events/stream" in paths and "/api/v1/jobs" in paths


def test_profiles(client):
    listed = {p["name"]: p for p in client.get("/api/v1/profiles").json()}
    assert {"base", "smoke", "loop"} <= set(listed) and "error" not in listed["smoke"]
    assert "extends itself" in listed["loop"]["error"]                    # a broken profile never breaks the list
    d = client.get("/api/v1/profiles/smoke", params={"override": "pilot.lr=0.5"}).json()
    assert d["resolved"]["pilot"]["lr"] == 0.5 and "extends: base.yaml" in d["yaml"]
    assert client.get("/api/v1/profiles/ghost").json()["error"] == "NOT_FOUND"


def test_runs_list_and_detail(client):
    runs = client.get("/api/v1/runs").json()
    assert [r["name"] for r in runs] == ["demo_run"]                      # mlruns is not a run
    r = runs[0]
    assert r["status"] == "succeeded" and r["gate"]["decision"] == "HYBRID" and r["stages_done"] == 4
    d = client.get("/api/v1/runs/demo_run").json()
    assert [row["name"] for row in d["ladder"]] == ["data_checks", "xgboost", "pilot", "gate"]
    assert d["sections"]["pilot"]["selected"]["teacher"] == "org/model-a" and d["has_report_md"] is True


def test_run_not_found_and_traversal(client):
    assert client.get("/api/v1/runs/ghost").status_code == 404
    assert client.get("/api/v1/runs/demo_run/files/../../configs/base.yaml").status_code in (404, 400)


def test_report_series_events(client):
    assert client.get("/api/v1/runs/demo_run/report.md").text.startswith("# Run")
    s = client.get("/api/v1/runs/demo_run/series").json()
    assert [p["loss"] for p in s["series"]["pilot"]] == [1.5, 0.9] and s["offset"] > 0
    tail = client.get("/api/v1/runs/demo_run/events", params={"tail": 2}).json()["events"]
    assert [e["type"] for e in tail] == ["stage_end", "run_end"]
    page = client.get("/api/v1/runs/demo_run/events", params={"after": 0, "limit": 1}).json()
    assert len(page["events"]) == 1 and page["next"] > 0


def test_event_stream_of_a_finished_run_ends(client):
    with client.stream("GET", "/api/v1/runs/demo_run/events/stream") as resp:
        body = "".join(resp.iter_text())
    assert "event: events" in body and "event: end" in body and '"run_end"' in body


def test_compare_validation(client):
    assert client.get("/api/v1/compare", params={"runs": "demo_run"}).status_code == 200
    assert client.get("/api/v1/compare", params={"runs": ","}).status_code == 400
    assert client.get("/api/v1/compare", params={"runs": "a,b,c,d,e"}).status_code == 400


def test_jobs_lifecycle_over_http(client, runner):
    bad = client.post("/api/v1/jobs", json={"profile": "smoke", "run_name": "X Y"})
    assert bad.status_code == 400 and bad.json()["error"] == "INVALID_REQUEST"
    missing = client.post("/api/v1/jobs", json={"run_name": "x"})
    assert missing.status_code == 400 and "profile" in missing.json()["message"]
    prev = client.post("/api/v1/jobs/preview", json={"profile": "smoke", "run_name": "new_run"}).json()
    assert prev["run_name"] == "new_run" and "--no-sync" in prev["command"]
    job = client.post("/api/v1/jobs", json={"profile": "smoke", "run_name": "new_run"})
    assert job.status_code == 201 and job.json()["status"] == "queued"
    assert client.post("/api/v1/jobs", json={"profile": "smoke", "run_name": "new_run"}).status_code == 409
    jid = job.json()["id"]
    assert client.get(f"/api/v1/jobs/{jid}").json()["run_name"] == "new_run"
    assert client.post(f"/api/v1/jobs/{jid}/cancel").json()["status"] == "cancelled"
    assert client.post(f"/api/v1/jobs/{jid}/cancel").status_code == 409
    assert client.get("/api/v1/jobs/nope").status_code == 404


def test_job_logs_stream_strips_ansi_and_splits_progress(client, container):
    from app.features.jobs.service import JobRequest, JobService
    svc = JobService(container)
    job = svc.create(JobRequest(profile="smoke", run_name="logged"))
    svc.tick()
    with client.stream("GET", f"/api/v1/jobs/{job['id']}/logs/stream") as resp:
        body = "".join(resp.iter_text())
    assert '"line one"' in body and '"progress 1/2"' in body and '"progress 2/2"' in body
    assert '"done"' in body and "\\u001b" not in body


def test_services_and_control(client, container, runner):
    s = client.get("/api/v1/services").json()
    assert s["platform"] == "ok" and s["scoring"] == "down" and s["containers"]["scoring"] == "exited"
    assert client.post("/api/v1/services/postgres/start").status_code == 400
    assert client.post("/api/v1/services/scoring/explode").status_code == 400
    assert client.post("/api/v1/services/scoring/start").json()["containers"]["scoring"] == "running"
    from app.features.jobs.service import JobRequest, JobService
    svc = JobService(container)
    svc.create(JobRequest(profile="smoke", run_name="busy"))
    runner.services["scoring"] = "exited"
    svc.tick()
    blocked = client.post("/api/v1/services/scoring/start")
    assert blocked.status_code == 400 and "training job" in blocked.json()["message"]


def test_upstreams_down_give_502_with_message(client):
    for path in ("/api/v1/scoring/health", "/api/v1/registry/models"):
        r = client.get(path)
        assert r.status_code == 502 and r.json()["error"] == "UPSTREAM_UNAVAILABLE"
    assert client.post("/api/v1/scoring/score", json={"a": 1}).status_code == 502


def test_scoring_list_name_is_validated_before_any_call(client):
    assert client.get("/api/v1/scoring/admin/lists/everything").status_code == 400


def test_knowledge_endpoints(client):
    data = client.get("/api/v1/data/overview").json()
    assert data["dataset_id"] == "argus-x" and data["data_card"] == "# card"
    diagrams = client.get("/api/v1/docs/diagrams").json()
    assert [d["name"] for d in diagrams] == ["fig04_model_ladder.svg"]       # SVG preferred, docx copies skipped
    svg = client.get("/api/v1/docs/diagrams/fig04_model_ladder.svg")
    assert svg.status_code == 200 and svg.headers["content-type"].startswith("image/svg+xml")
    assert client.get("/api/v1/docs/diagrams/../secret.svg").status_code == 404
    assert client.get("/api/v1/docs/documents").json() == []
    assert client.get("/api/v1/docs/documents/x.docx").status_code == 404


def test_gpu_endpoint_degrades_without_nvml(client):
    assert set(client.get("/api/v1/system/gpu").json()) == {"available", "gpus"}


def test_ladder_includes_stages_reused_from_another_run(client, tree):
    import json
    path = tree["outputs"] / "demo_run" / "report.json"
    report = json.loads(path.read_text())
    report["config"]["stages"] = ["gate"]                          # e.g. demo_llm_scorer: only its own stages
    report["stages"]["pilot"]["reused_from"] = "other_run"
    path.write_text(json.dumps(report))
    rows = {r["name"]: r["status"] for r in client.get("/api/v1/runs/demo_run").json()["ladder"]}
    assert list(rows) == ["data_checks", "xgboost", "pilot", "gate"] and rows["pilot"] == "reused"


def test_a_new_job_is_a_visible_queued_run_before_its_container_starts(client):
    job = client.post("/api/v1/jobs", json={"profile": "smoke", "run_name": "fresh_run", "stages": ["xgboost", "pilot"]})
    assert job.status_code == 201
    assert "fresh_run" in [r["name"] for r in client.get("/api/v1/runs").json()]
    d = client.get("/api/v1/runs/fresh_run").json()
    assert d["status"] == "queued" and [r["name"] for r in d["ladder"]] == ["xgboost", "pilot"]
    assert d["profile"] == "configs/smoke.yaml" and all(r["status"] == "pending" for r in d["ladder"])


def test_a_run_cannot_read_a_sibling_run_with_the_same_prefix(container):
    from app.core.errors import NotFound
    outputs = container.runs.root
    (outputs / "demo_run2").mkdir()
    (outputs / "demo_run2" / "secret.txt").write_text("other run")
    with pytest.raises(NotFound):
        container.runs.file_text("demo_run", "../demo_run2/secret.txt")


def test_samples_pick_up_a_dataset_synced_after_start(tmp_path):
    """The playground's samples must not stay empty after a dataset sync (no platform restart needed)."""
    import pandas as pd
    from app.features.scoring.samples import sample_requests
    assert [s["label"] for s in sample_requests(tmp_path)] == ["CRD 10.2 worked example (TX98231)", "Invalid: negative amount"]
    folder = tmp_path / "raw" / "parquet" / "transactions" / "validation"
    folder.mkdir(parents=True)
    rows = [{"transaction_id": f"T{i}", "customer_id": "C1", "transaction_amount": 100.0 + i, "merchant_category": "gaming",
             "transaction_time": pd.Timestamp("2025-11-01T01:00:00+05:30"), "device_id": "D1", "customer_age": 30,
             "customer_location": '{"lat": 1, "lon": 2, "city": "Pune"}', "merchant_location": '{"lat": 1, "lon": 2, "city": "Pune"}',
             "account_age": 10, "fraud_label": i % 2} for i in range(4)]
    pd.DataFrame(rows).to_parquet(folder / "part-00001.parquet")
    labels = [s["label"] for s in sample_requests(tmp_path)]
    assert len(labels) > 2 and any(lbl.startswith("Known fraud") for lbl in labels)
