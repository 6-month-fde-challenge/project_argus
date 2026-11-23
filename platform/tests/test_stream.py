"""/api/v1/stream: relays to the stream service; the control key is added server-side; inputs validated first."""
from __future__ import annotations

import httpx

from app.infra.http_clients import StreamClient


def test_stream_down_gives_502_with_message(client):
    for path in ("/api/v1/stream/state", "/api/v1/stream/summary?minutes=30", "/api/v1/stream/events"):
        r = client.get(path)
        assert r.status_code == 502 and r.json()["error"] == "UPSTREAM_UNAVAILABLE"
    assert client.post("/api/v1/stream/producer", json={"action": "start"}).status_code == 502


def test_stream_inputs_are_validated_before_any_call(client):
    assert client.post("/api/v1/stream/producer", json={"action": "explode"}).status_code == 400
    assert client.post("/api/v1/stream/producer", json={"action": "start", "rate": 500}).status_code == 400
    assert client.post("/api/v1/stream/inject", json={"n": 0}).status_code == 400
    assert client.get("/api/v1/stream/summary?minutes=0").status_code == 400
    assert client.post("/api/v1/stream/shadow", json={"run": "../../etc"}).status_code == 400
    assert client.post("/api/v1/stream/shadow", json={"run": "argus_demo_hybrid"}).status_code == 502  # valid, stream down


def test_ports_and_origins_come_from_the_environment(monkeypatch):
    """docker-compose.yml passes ARGUS_CORS_ORIGINS as JSON and ARGUS_MLFLOW_PUBLIC_URL with the host ports."""
    from app.core.config import Settings
    monkeypatch.setenv("ARGUS_CORS_ORIGINS", '["http://localhost:3001", "http://127.0.0.1:3001"]')
    monkeypatch.setenv("ARGUS_MLFLOW_PUBLIC_URL", "http://localhost:5001")
    s = Settings()
    assert s.cors_origins == ["http://localhost:3001", "http://127.0.0.1:3001"]
    assert s.mlflow_public_url == "http://localhost:5001"


def test_branch_is_what_the_run_did_not_only_what_the_gate_measured(client, tree):
    import json
    for name, (mode, stages) in {
        "forced_llm": ("llm", {"gate": {"status": "ok", "decision": "HYBRID", "branch": "HYBRID", "mode": "hybrid",  # reused
                                        "uplift": -0.05}, "sft_scorer": {"status": "ok"}}),
        "hybrid": ("auto", {"gate": {"status": "ok", "decision": "HYBRID", "branch": "HYBRID", "uplift": -0.05}}),
        "reuses_llm": ("auto", {"gate": {"status": "ok", "decision": "LLM", "uplift": 0.04},
                                "sft_scorer": {"status": "ok", "reused_from": "x"}}),
        "auto_on_forced_gate": ("auto", {"gate": {"status": "ok", "decision": "HYBRID", "mode": "llm", "uplift": -0.05},
                                         "sft_scorer": {"status": "ok"}}),
    }.items():
        (tree["outputs"] / name).mkdir()
        (tree["outputs"] / name / "report.json").write_text(json.dumps({"config": {"gate": {"mode": mode}}, "stages": stages}))
    gates = {r["name"]: r["gate"] for r in client.get("/api/v1/runs").json() if r["name"] != "demo_run"}
    assert gates["forced_llm"]["decision"] == "HYBRID" and gates["forced_llm"]["branch"] == "LLM"
    assert gates["hybrid"]["branch"] == "HYBRID" and gates["reuses_llm"]["branch"] == "LLM"
    # "forced" comes from the run's own config, not from the (reused) gate entry
    assert gates["forced_llm"]["forced"] and gates["forced_llm"]["mode"] == "llm"
    assert not gates["hybrid"]["forced"] and not gates["reuses_llm"]["forced"]
    # an auto run that reuses a forced gate decision (and so serves a branch the gate never measured) is a demo too
    assert gates["auto_on_forced_gate"]["forced"] and "reused" in gates["auto_on_forced_gate"]["forced_reason"]
    assert gates["hybrid"]["forced_reason"] is None


def test_services_report_the_stream(client):
    assert client.get("/api/v1/services").json()["stream"] == "down"


async def test_every_stream_call_carries_the_key(container):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    stream = StreamClient(container.settings)
    stream.http = httpx.AsyncClient(base_url="http://stream:8003", transport=httpx.MockTransport(handler))
    assert await stream.control_call("producer", {"action": "start"}) == (200, {"ok": True})
    await stream.get("summary", minutes=5)
    assert seen[0].url.path == "/live/api/producer" and seen[0].headers["x-control-key"] == "local-control-key"
    assert seen[1].url.params["minutes"] == "5" and seen[1].headers["x-control-key"] == "local-control-key"
