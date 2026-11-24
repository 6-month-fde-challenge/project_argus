"""Settings (.env), Operations (the agent's queue), profile editing, AWS through GitHub, approval in production."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.container import Container
from app.infra.env_file import EnvFile
from app.main import create_app

from .conftest import BASE_YAML, SMOKE_YAML, FakeRunner

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"  # the real catalogue (CI: the checkout; Docker: mounted at /deploy)
needs_catalogue = pytest.mark.skipif(not (DEPLOY / "ops_catalog.py").exists(), reason="deploy/ not available")
APPROVAL = {"requested_by": "Karthik", "approved_by": "Ananya", "reason": "rotation"}


def make_repo(tmp_path: Path, serving: str = "configs/local_demo.yaml") -> Path:
    repo = tmp_path / "repo"
    configs, outputs = repo / "training" / "configs", repo / "training" / "outputs"
    configs.mkdir(parents=True)
    (outputs / "argus_prod_a100").mkdir(parents=True)
    (repo / "docker-compose.yml").write_text("services: {}\n")
    (configs / "base.yaml").write_text(BASE_YAML + "api: {allow_forced_runs: false}\n")
    (configs / "smoke.yaml").write_text(SMOKE_YAML)
    (configs / "local_demo.yaml").write_text("# local\nextends: base.yaml\nrun_name: argus_local_demo\napi: {allow_forced_runs: true}\n")
    (configs / "prod_a100.yaml").write_text("# prod\nextends: base.yaml\nrun_name: argus_prod_a100\n")
    (repo / ".env").write_text(f"# local settings\nSUPABASE_URL=https://x.supabase.co\nARGUS_SCORING_TOKEN=old-token-123\n"
                               f"SCORING_CONFIG={serving}\nCUSTOM_KEY=kept\n")
    if DEPLOY.exists():
        (repo / "deploy").mkdir()
        for f in ("ops_catalog.py", "ops_actions.json"):
            shutil.copy(DEPLOY / f, repo / "deploy" / f)
    return repo


def client_for(repo: Path, github: httpx.MockTransport | None = None) -> tuple[TestClient, Container]:
    outputs = repo / "training" / "outputs"
    settings = Settings(outputs_dir=outputs, configs_dir=repo / "training" / "configs", data_dir=repo / "data",
                        docs_dir=repo / "docs", diagrams_dir=repo / "diagrams", repo_dir=repo,
                        scoring_url="http://127.0.0.1:9", mlflow_url="http://127.0.0.1:9", explainer_url="http://127.0.0.1:9",
                        stream_url="http://127.0.0.1:9", github_api="https://github.test")
    container = Container(settings, runner_override=FakeRunner(), github_transport=github)
    return TestClient(create_app(container, run_scheduler=False)), container


@pytest.fixture
def demo(tmp_path):
    with client_for(make_repo(tmp_path))[0] as c:
        yield c


@pytest.fixture
def prod(tmp_path):
    repo = make_repo(tmp_path, "configs/prod_a100.yaml")
    client, container = client_for(repo)
    with client as c:
        yield c, repo, container


# ------------------------------------------------------------------ .env
def test_env_file_keeps_comments_order_and_unknown_keys(tmp_path):
    env = EnvFile(tmp_path / ".env")
    (tmp_path / ".env").write_text("# head\nA=1\n\n# b\nB='two words'\nA=dup\nC=3 # note\n")
    assert env.values() == {"A": "dup", "B": "two words", "C": "3"}  # last wins, quotes / inline comment like compose
    env.update({"A": "9", "C": None, "D": "4"})
    text = (tmp_path / ".env").read_text()
    assert text.startswith("# head\nA=9\n\n# b\nB='two words'\n") and "A=dup" not in text and "C=" not in text
    assert text.rstrip().endswith("D=4") and env.values() == {"A": "9", "B": "two words", "D": "4"}


def test_env_file_survives_windows_encodings(tmp_path):
    p = tmp_path / ".env"
    p.write_bytes("\ufeffX=1\n".encode("utf-8"))
    assert EnvFile(p).values() == {"X": "1"}
    p.write_bytes("Y=2\n".encode("utf-16"))  # PowerShell 5.1 `>`
    assert EnvFile(p).values() == {"Y": "2"}


# ------------------------------------------------------------------ settings
def test_settings_show_values_but_never_secrets(demo):
    body = demo.get("/api/v1/settings").json()
    items = {i["key"]: i for g in body["groups"] for i in g["items"]}
    assert items["SCORING_CONFIG"]["value"] == "configs/local_demo.yaml"
    assert items["ARGUS_SCORING_TOKEN"]["value"] is None and items["ARGUS_SCORING_TOKEN"]["is_set"]
    assert "configs/prod_a100.yaml" in items["SCORING_CONFIG"]["options"]
    assert items["SCORING_CONFIG"]["option_modes"]["configs/prod_a100.yaml"] == "production"
    assert body["unknown_keys"] == ["CUSTOM_KEY"] and body["serving"]["mode"] == "demo"
    assert "old-token-123" not in json.dumps(body)


def test_settings_are_validated_before_anything_is_written(demo, tmp_path):
    put = lambda changes: demo.put("/api/v1/settings", json={"changes": changes})  # noqa: E731
    assert put({"NOT_A_SETTING": "1"}).status_code == 400
    assert put({"RUN_NAME": "x1"}).status_code == 400                 # locked: Serve a run sets it
    assert put({"ARGUS_BIND": "0.0.0.0"}).status_code == 400          # the UI is never opened to the network
    assert put({"ARGUS_UI_PORT": "80"}).status_code == 400
    assert put({"ARGUS_UI_PORT": "8090"}).status_code == 400          # the platform API's port already
    assert put({"SUPABASE_URL": "https://a\nB=1"}).status_code == 400  # never breaks the .env format
    assert put({"TF_VAR_allowed_cidrs": "1.2.3.4"}).status_code == 400
    assert put({"SCORING_CONFIG": "configs/nope.yaml"}).status_code == 400
    env = (tmp_path / "repo" / ".env").read_text()
    assert "ARGUS_UI_PORT" not in env and "CUSTOM_KEY=kept" in env


def test_saving_writes_env_logs_the_change_and_hides_secrets(demo, tmp_path):
    r = demo.put("/api/v1/settings", json={"changes": {"ARGUS_SCORING_TOKEN": "@generate", "ARGUS_UI_PORT": "3001",
                                                       "TF_VAR_allowed_cidrs": '["203.0.113.7"]'}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["saved"] == ["ARGUS_SCORING_TOKEN", "ARGUS_UI_PORT", "TF_VAR_allowed_cidrs"]
    assert body["restart"] == ["frontend", "mlflow", "platform", "scoring", "stream"] and body["action"] is None
    assert body["commands"]["linux"] == "bash deploy/hyperstack/apply.sh frontend mlflow platform scoring stream"
    env = EnvFile(tmp_path / "repo" / ".env").values()
    assert env["ARGUS_UI_PORT"] == "3001" and env["TF_VAR_allowed_cidrs"] == '["203.0.113.7/32"]'
    assert env["ARGUS_SCORING_TOKEN"] != "old-token-123" and len(env["ARGUS_SCORING_TOKEN"]) >= 30
    assert env["ARGUS_SCORING_TOKEN"] not in r.text
    log = demo.get("/api/v1/settings/changes").json()[0]
    assert log["kind"] == "settings" and log["details"]["values"]["ARGUS_SCORING_TOKEN"] == "(changed)"
    assert env["ARGUS_SCORING_TOKEN"] not in json.dumps(log)


def test_production_mode_needs_two_people_for_what_live_traffic_meets(prod):
    client, repo, _ = prod
    change = {"changes": {"ARGUS_SCORING_TOKEN": "@generate"}}
    r = client.put("/api/v1/settings", json=change)
    assert r.status_code == 428 and r.json()["error"] == "APPROVAL_REQUIRED"
    same = {**change, "approval": {**APPROVAL, "approved_by": " karthik "}}
    assert client.put("/api/v1/settings", json=same).status_code == 428       # one person is not two
    assert client.put("/api/v1/settings", json={**change, "approval": APPROVAL}).status_code == 200
    assert client.get("/api/v1/settings/changes").json()[0]["approved_by"] == "ananya"
    # a change that does not touch live scoring needs no approval, even in production
    assert client.put("/api/v1/settings", json={"changes": {"PRODUCER_SOURCE": "holdout"}}).status_code == 200


@needs_catalogue
def test_save_and_apply_queues_the_agent_action(demo):
    r = demo.put("/api/v1/settings", json={"changes": {"PRODUCER_SOURCE": "synthetic"}, "apply": True}).json()
    assert r["action"]["action"] == "apply" and r["action"]["params"] == {"services": ["stream"]}
    assert r["action"]["commands"]["linux"] == "bash deploy/hyperstack/apply.sh stream"
    assert r["action"]["commands"]["windows"] == ".\\argus.ps1 apply stream"


# ------------------------------------------------------------------ operations
@needs_catalogue
def test_operations_validate_like_the_agent_and_queue_files(demo, tmp_path):
    cat = demo.get("/api/v1/ops/catalog").json()
    assert {a["name"] for a in cat["actions"]} >= {"serve", "pipeline", "apply", "check"}
    assert cat["agent"]["online"] is False
    assert demo.post("/api/v1/ops/actions", json={"action": "serve", "params": {"run": "argus_missing"}}).status_code == 400
    assert demo.post("/api/v1/ops/actions", json={"action": "shell", "params": {}}).status_code == 400
    pv = demo.post("/api/v1/ops/preview", json={"action": "pipeline", "params": {"profile": "prod"}}).json()
    assert pv["commands"]["linux"].endswith("bash deploy/hyperstack/pipeline.sh") and pv["commands"]["windows"] is None
    rec = demo.post("/api/v1/ops/actions", json={"action": "serve", "params": {"run": "argus_prod_a100"}}).json()
    assert rec["status"] == "queued" and rec["commands"]["linux"] == "bash deploy/hyperstack/serve.sh argus_prod_a100"
    on_disk = tmp_path / "repo" / "training" / "outputs" / "_platform" / "ops" / "actions" / f"{rec['id']}.json"
    assert json.loads(on_disk.read_text())["action"] == "serve"
    assert demo.get("/api/v1/ops/actions").json()[0]["id"] == rec["id"]
    assert demo.post(f"/api/v1/ops/actions/{rec['id']}/cancel").json()["cancel_requested"] is True
    on_disk.with_suffix(".log").write_bytes(b"line 1\nline 2\n")  # bytes: the agent's log, LF on every OS
    log = demo.get(f"/api/v1/ops/actions/{rec['id']}/log?offset=7").json()
    assert log["text"] == "line 2\n" and log["offset"] == 14
    assert demo.get("/api/v1/ops/actions/not-an-id").status_code == 404


@needs_catalogue
def test_a_linux_only_action_is_refused_when_the_agent_runs_on_windows(demo, tmp_path):
    ops = tmp_path / "repo" / "training" / "outputs" / "_platform" / "ops"
    ops.mkdir(parents=True)
    import time
    (ops / "agent.json").write_text(json.dumps({"os": "windows", "beat": time.time(), "host": "pc"}))
    r = demo.post("/api/v1/ops/actions", json={"action": "pipeline", "params": {"profile": "prod"}})
    assert r.status_code == 400 and "Linux GPU VM" in r.json()["message"]


@needs_catalogue
def test_serving_actions_need_approval_in_production(prod):
    client, _, _ = prod
    body = {"action": "serve", "params": {"run": "argus_prod_a100"}}
    assert client.post("/api/v1/ops/actions", json=body).status_code == 428
    assert client.post("/api/v1/ops/actions", json={**body, "approval": APPROVAL}).status_code == 200
    assert client.post("/api/v1/ops/actions", json={"action": "check", "params": {}}).status_code == 200  # read-only


# ------------------------------------------------------------------ activation
def test_activating_a_model_in_production_needs_approval(prod):
    client, _, _ = prod
    assert client.post("/api/v1/scoring/models/argus_prod_a100/activate").status_code == 428
    # approved: the call goes on to the scoring API (not running in this test -> 502)
    assert client.post("/api/v1/scoring/models/argus_prod_a100/activate", json={"approval": APPROVAL}).status_code == 502


def test_services_report_the_serving_mode(prod):
    client, _, _ = prod
    body = client.get("/api/v1/services").json()
    assert body["serving_mode"] == "production" and body["demo_model"] is False and body["agent"] is False


# ------------------------------------------------------------------ profiles
def test_profiles_can_be_saved_validated_and_deleted_safely(demo, tmp_path):
    configs = tmp_path / "repo" / "training" / "configs"
    r = demo.post("/api/v1/profiles", json={"name": "prod_longer", "extends": "prod_a100",
                                            "overrides": ["pilot.max_steps=30", "lora.r=32"], "description": "longer pilot"})
    assert r.status_code == 200, r.text
    text = (configs / "prod_longer.yaml").read_text()
    assert text.startswith("# longer pilot\nextends: prod_a100.yaml\n") and "max_steps: 30" in text
    resolved = demo.get("/api/v1/profiles/prod_longer").json()["resolved"]
    assert resolved["pilot"]["max_steps"] == 30 and resolved["run_name"] == "argus_prod_a100"
    assert demo.post("/api/v1/profiles", json={"name": "prod_longer", "extends": "base"}).status_code == 409
    assert demo.put("/api/v1/profiles/bad", json={"yaml": "extends: base.yaml\n: :\n"}).status_code == 400
    assert demo.put("/api/v1/profiles/bad", json={"yaml": "run_name: x\n"}).status_code == 400   # must extend
    loop = demo.put("/api/v1/profiles/prod_a100", json={"yaml": "extends: prod_longer.yaml\n"})
    assert loop.status_code == 400 and "loop" in loop.json()["message"]
    # an edit that would break a profile extending it is rolled back
    before = (configs / "prod_a100.yaml").read_text()
    broken = demo.put("/api/v1/profiles/prod_a100", json={"yaml": "extends: base.yaml\ngate: {mode: sideways}\n"})
    assert broken.status_code == 400 and (configs / "prod_a100.yaml").read_text() == before
    assert demo.delete("/api/v1/profiles/prod_a100").status_code == 409       # shipped: edit, never delete
    assert demo.delete("/api/v1/profiles/prod_longer").status_code == 200
    assert not (configs / "prod_longer.yaml").exists()
    kinds = [c["summary"] for c in demo.get("/api/v1/settings/changes?kind=profile").json()]
    assert kinds[:2] == ["deleted profile prod_longer", "created profile prod_longer (extends prod_a100)"]


# ------------------------------------------------------------------ AWS through GitHub
def fake_github(calls: list, variables: dict | None = None):
    from nacl import public
    key = public.PrivateKey.generate().public_key
    import base64

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        path = request.url.path
        if path.endswith("/actions/variables") and request.method == "GET":
            return httpx.Response(200, json={"variables": [{"name": k, "value": v} for k, v in (variables or {}).items()]})
        if "/actions/variables/" in path and request.method == "PATCH":
            return httpx.Response(404)
        if path.endswith("/actions/secrets/public-key"):
            return httpx.Response(200, json={"key_id": "k1", "key": base64.b64encode(bytes(key)).decode()})
        if path.endswith("/runs"):
            return httpx.Response(200, json={"workflow_runs": [{"id": 1, "status": "completed", "conclusion": "success",
                                                                "html_url": "https://x", "created_at": "2026-11-24T00:00:00Z",
                                                                "event": "push", "head_branch": "main", "head_sha": "abc"}]})
        return httpx.Response(204 if request.method in ("PUT", "PATCH") or "dispatches" in path else 201)
    return httpx.MockTransport(handler)


def test_aws_page_without_a_token_explains_what_to_set(demo):
    body = demo.get("/api/v1/aws").json()
    assert body["github"]["ok"] is False and "ARGUS_GITHUB" in body["github"]["error"]


def test_aws_deploy_is_refused_for_a_demo_setup_and_needs_two_people(tmp_path):
    calls: list = []
    repo = make_repo(tmp_path)
    EnvFile(repo / ".env").update({"ARGUS_GITHUB_REPO": "acme/argus", "ARGUS_GITHUB_TOKEN": "ghp_1234567890",
                                   "TF_VAR_allowed_cidrs": '["203.0.113.7/32"]', "TF_VAR_scoring_host": "198.51.100.4"})
    client, _ = client_for(repo, fake_github(calls, {"AWS_ROLE_ARN": "arn:aws:iam::1:role/argus-github-deploy"}))
    with client as c:
        status = c.get("/api/v1/aws").json()
        assert status["github"]["ok"] and status["bootstrapped"] and status["runs"]["deploy"][0]["sha"] == "abc"
        # the API is not reachable here: its mode is unknown, so a deploy is refused unless a demo is allowed
        assert c.post("/api/v1/aws/deploy", json={"approval": APPROVAL}).status_code == 409
        assert c.post("/api/v1/aws/deploy", json={"allow_demo": True}).status_code == 428
        assert c.post("/api/v1/aws/deploy", json={"allow_demo": True, "approval": APPROVAL}).status_code == 200
        dispatch = [x for x in calls if "dispatches" in x[1]][-1]
        assert dispatch[1].endswith("/deploy.yml/dispatches") and dispatch[2] == {"ref": "main", "inputs": {"allow_demo": "true"}}
        # sync: variables + the sealed secret, never the token in clear
        assert c.post("/api/v1/aws/sync", json={}).status_code == 200
        sent = {x[2]["name"]: x[2]["value"] for x in calls if x[0] == "POST" and x[1].endswith("/actions/variables")}
        assert sent["SCORING_HOST"] == "198.51.100.4" and json.loads(sent["TF_VARS"])["scoring_host"] == "198.51.100.4"
        secret = [x for x in calls if x[1].endswith("/actions/secrets/SCORING_TOKEN")][0][2]
        assert secret["key_id"] == "k1" and "old-token-123" not in json.dumps(calls)
        # destroy: typed confirmation of the stack name
        assert c.post("/api/v1/aws/destroy", json={"approval": APPROVAL, "confirm": "argus"}).status_code == 400
        assert c.post("/api/v1/aws/destroy", json={"approval": APPROVAL, "confirm": "argus-prod"}).status_code == 200
