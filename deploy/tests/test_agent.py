"""The operations agent and its catalogue: what the UI can make the host run, and nothing else."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY))
import agent as agent_mod  # noqa: E402
import ops_catalog as cat  # noqa: E402


@pytest.fixture
def repo(tmp_path):
    """A repository layout with two runs and two profiles."""
    (tmp_path / "deploy").mkdir()
    for run in ("argus_prod_a100", "argus_demo_hybrid", "_scoring", "mlruns"):
        (tmp_path / "training" / "outputs" / run).mkdir(parents=True)
    (tmp_path / "training" / "configs").mkdir(parents=True)
    for name in ("prod_a100", "local_demo"):
        (tmp_path / "training" / "configs" / f"{name}.yaml").write_text("run_name: x\n")
    return tmp_path


def real_actions():
    return json.loads((DEPLOY / "ops_actions.json").read_text(encoding="utf-8"))["actions"]


# ------------------------------------------------------------------ catalogue
def test_every_action_of_the_catalogue_is_well_formed():
    known_types = {"run", "runs", "profile", "vm_profile", "serving_config", "int", "bool", "services"}
    for name, a in real_actions().items():
        assert a["label"] and a["group"] and a["description"], name
        assert a.get("approval") in (None, "production"), name
        assert a.get("linux") or a.get("windows"), name
        for p, spec in a["params"].items():
            assert spec["type"] in known_types, (name, p)
            if spec["type"] == "services":
                assert spec["options"], (name, p)
        for os_name in ("linux", "windows"):  # every placeholder names a parameter
            for tok in (a.get(os_name) or []) + list((a.get("env") or {}).values()):
                for m in cat.PLACEHOLDER.finditer(tok):
                    assert m.group(1) in a["params"], (name, tok)


def test_validation_rejects_anything_the_catalogue_does_not_allow(repo):
    actions = real_actions()
    with pytest.raises(cat.ActionError, match="unknown action"):
        cat.validate(actions, "rm_rf", {}, repo)
    with pytest.raises(cat.ActionError, match="unknown parameters"):
        cat.validate(actions, "serve", {"run": "argus_prod_a100", "cmd": "id"}, repo)
    for bad in ("argus_missing", "../etc", "_scoring", "mlruns", "a; reboot"):
        with pytest.raises(cat.ActionError):
            cat.validate(actions, "serve", {"run": bad}, repo)
    with pytest.raises(cat.ActionError, match="required"):
        cat.validate(actions, "serve", {}, repo)
    with pytest.raises(cat.ActionError):
        cat.validate(actions, "apply", {"services": ["scoring", "postgres"]}, repo)
    with pytest.raises(cat.ActionError):
        cat.validate(actions, "loadtest", {"run": "argus_prod_a100", "config": "configs/prod_a100.yaml", "rps": 0, "duration": 30}, repo)
    with pytest.raises(cat.ActionError):
        cat.validate(actions, "loadtest", {"run": "argus_prod_a100", "config": "../../etc/passwd", "rps": 5, "duration": 30}, repo)
    with pytest.raises(cat.ActionError):
        cat.validate(actions, "check", {"full": "yes"}, repo)  # a bool must be a bool
    with pytest.raises(cat.ActionError):
        cat.validate(actions, "pipeline", {"profile": "nope"}, repo)
    assert cat.validate(actions, "pipeline", {"profile": "prod"}, repo)["profile"] == "prod"
    assert cat.validate(actions, "pipeline", {"profile": "local_demo"}, repo)["skip_train"] is False


def test_commands_are_rendered_the_same_way_the_ui_shows_them(repo):
    actions = real_actions()
    p = cat.validate(actions, "pipeline", {"profile": "prod", "skip_train": True}, repo)
    argv, env = cat.render(actions, "pipeline", p, "linux")
    assert argv == ["bash", "deploy/hyperstack/pipeline.sh"]
    assert env == {"PROFILE": "prod", "SKIP_TRAIN": "1", "ARGUS_PIPELINE_INSIDE": "1"}  # empty SERVE_RUN not set
    assert cat.preview(argv, env, "linux") == "PROFILE=prod SKIP_TRAIN=1 ARGUS_PIPELINE_INSIDE=1 bash deploy/hyperstack/pipeline.sh"
    with pytest.raises(cat.ActionError, match="Linux GPU VM only"):
        cat.render(actions, "pipeline", p, "windows")

    p = cat.validate(actions, "apply", {"services": ["scoring", "platform", "scoring"]}, repo)
    assert p["services"] == ["scoring", "platform"]  # de-duplicated, order kept
    argv, _ = cat.render(actions, "apply", p, "windows")
    assert argv == cat.POWERSHELL + ["apply", "scoring", "platform"]
    assert cat.preview(argv, {}, "windows") == ".\\argus.ps1 apply scoring platform"

    p = cat.validate(actions, "check", {}, repo)
    assert cat.render(actions, "check", p, "linux")[0] == ["bash", "deploy/hyperstack/check.sh"]  # no --full
    p = cat.validate(actions, "prefetch", {"profile": "prod_a100"}, repo)
    assert "configs/prod_a100.yaml" in cat.render(actions, "prefetch", p, "linux")[0]


def test_a_zero_is_a_value_not_an_empty_one():
    assert cat._render_token("{n}", {"n": 0}) == ["0"]
    assert cat._render_token("x={n}", {"n": 0}) == ["x=0"]
    assert cat._render_token("{n}", {"n": None}) == [] and cat._render_token("{f?--full}", {"f": False}) == []


# ------------------------------------------------------------------ agent
def fake_catalogue(repo: Path, extra: dict | None = None) -> None:
    py = sys.executable
    actions = {
        "hello": {"label": "Hello", "group": "Test", "description": "prints", "approval": None, "timeout_min": 5,
                  "params": {"run": {"type": "run", "required": True, "label": "run"}},
                  "env": {"WHO": "{run}"},
                  "linux": [py, "-c", "import os; print('hello', os.environ['WHO'])"],
                  "windows": [py, "-c", "import os; print('hello', os.environ['WHO'])"]},
        "fail": {"label": "Fail", "group": "Test", "description": "exits 3", "approval": None, "timeout_min": 5,
                 "params": {}, "linux": [py, "-c", "import sys; sys.exit(3)"], "windows": [py, "-c", "import sys; sys.exit(3)"]},
        **(extra or {}),
    }
    (repo / "deploy" / "ops_actions.json").write_text(json.dumps({"actions": actions}))


def queue(ops: Path, rid: str, action: str, params: dict, created: str) -> Path:
    (ops / "actions").mkdir(parents=True, exist_ok=True)
    f = ops / "actions" / f"{rid}.json"
    f.write_text(json.dumps({"id": rid, "action": action, "params": params, "status": "queued", "created": created}))
    return f


def test_the_agent_runs_the_queue_in_order_and_reports_each_result(repo):
    fake_catalogue(repo)
    ops = repo / "training" / "outputs" / "_platform" / "ops"
    a = queue(ops, "000000000001aaaa", "hello", {"run": "argus_prod_a100"}, "2026-01-01T00:00:01Z")
    b = queue(ops, "000000000002bbbb", "fail", {}, "2026-01-01T00:00:02Z")
    c = queue(ops, "000000000003cccc", "hello", {"run": "../etc"}, "2026-01-01T00:00:03Z")   # refused by the agent
    d = queue(ops, "000000000004dddd", "hello", {"run": "argus_demo_hybrid"}, "2026-01-01T00:00:04Z")
    d.with_suffix(".cancel").write_text("x")                                                    # cancelled before it ran
    ag = agent_mod.Agent(ops, repo)
    while item := ag.next_queued():
        ag.execute(*item)
    ra, rb, rc, rd = (json.loads(f.read_text()) for f in (a, b, c, d))
    assert ra["status"] == "succeeded" and ra["exit_code"] == 0 and "hello argus_prod_a100" in a.with_suffix(".log").read_text()
    assert ra["command"] and ra["started"] and ra["finished"]  # the exact command is recorded for the UI
    assert rb["status"] == "failed" and rb["exit_code"] == 3
    assert rc["status"] == "failed" and "refused by the agent" in rc["message"]
    assert rd["status"] == "cancelled"
    ag.beat()
    beat = json.loads((ops / "agent.json").read_text())
    assert beat["running"] is None and beat["os"] in ("linux", "windows")


def test_an_action_cut_off_by_a_restart_is_reported_never_rerun(repo):
    fake_catalogue(repo)
    ops = repo / "training" / "outputs" / "_platform" / "ops"
    f = queue(ops, "000000000005eeee", "hello", {"run": "argus_prod_a100"}, "2026-01-01T00:00:05Z")
    f.write_text(json.dumps({**json.loads(f.read_text()), "status": "running"}))
    ag = agent_mod.Agent(ops, repo)
    ag.recover()
    assert json.loads(f.read_text())["status"] == "failed" and ag.next_queued() is None


def test_agent_status_reads_the_heartbeat(repo, capsys):
    ops = repo / "ops"
    assert agent_mod.status(ops) == 1
    ops.mkdir()
    agent_mod.write_json(ops / "agent.json", {"pid": 1, "host": "vm", "beat": 0, "running": None})
    assert agent_mod.status(ops) == 1 and "NOT running" in capsys.readouterr().out
