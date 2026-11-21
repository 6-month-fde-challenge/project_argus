"""Shared fixtures: a temporary outputs/configs/docs tree, a fake job runner, and an app wired to them."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.container import Container
from app.infra.job_runner import ContainerState
from app.main import create_app

BASE_YAML = """# Base profile: full ladder
run_name: argus
seed: 42
max_length: 384
stages: [data_checks, xgboost, pilot, gate]
gate: {mode: auto, min_uplift: 0.03}
pilot: {max_steps: 60, lr: 0.0003}
"""
SMOKE_YAML = """# Smoke profile: tiny sizes
extends: base.yaml
run_name: smoke
pilot: {max_steps: 3}
"""
LOOP_YAML = "extends: loop.yaml\nrun_name: loop\n"


class FakeRunner:
    """In-memory JobRunner: records launches, lets tests decide container state and GPU holders."""

    def __init__(self):
        self.launched: list[tuple[str, str, list[str]]] = []
        self.states: dict[str, ContainerState | None] = {}
        self.holders: list[str] = []
        self.stopped: list[str] = []
        self.services: dict[str, str] = {"scoring": "exited", "explainer": "exited"}

    def launch(self, job_id, run_name, command):
        cid = f"cid-{job_id}"
        self.launched.append((job_id, run_name, command))
        self.states[cid] = ContainerState(running=True, exit_code=None)
        return cid

    def state(self, container_id):
        return self.states.get(container_id)

    def stop(self, container_id):
        self.stopped.append(container_id)
        self.states[container_id] = ContainerState(running=False, exit_code=137)

    def logs(self, container_id, follow, tail):
        return iter([b"line one\n", b"progress 1/2\rprogress 2/2\n", b"\x1b[32mdone\x1b[0m\n"])

    def gpu_holders(self):
        return list(self.holders)

    def service_status(self):
        return dict(self.services)

    def set_service(self, service, running):
        self.services[service] = "running" if running else "exited"


def write_events(path: Path, events: list[dict]) -> None:
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def ev(type_, stage=None, ts="2026-11-08T00:00:00.000Z", **data):
    return {"ts": ts, "type": type_, "stage": stage, "data": data}


@pytest.fixture
def tree(tmp_path: Path) -> dict[str, Path]:
    outputs, configs, docs, diagrams, data = (tmp_path / n for n in ("outputs", "configs", "docs", "diagrams", "data"))
    for d in (outputs, configs, docs, diagrams, data / "raw"):
        d.mkdir(parents=True)
    (configs / "base.yaml").write_text(BASE_YAML)
    (configs / "smoke.yaml").write_text(SMOKE_YAML)
    (configs / "loop.yaml").write_text(LOOP_YAML)
    (diagrams / "fig04_model_ladder.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    (diagrams / "fig04_model_ladder.png").write_bytes(b"\x89PNG")
    (diagrams / "fig04_model_ladder.fit-docx.png").write_bytes(b"\x89PNG")
    (data / "raw" / "manifest.json").write_text(json.dumps({"dataset_id": "argus-x", "version": "1.0.0", "counts": {"gold": 8000}}))
    (data / "raw" / "quality_report.json").write_text(json.dumps({"main_fraud_rate": 0.006, "observed": {"splits": {}}}))
    (data / "raw" / "DATA_CARD.md").write_text("# card")

    run = outputs / "demo_run"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({
        "profile": "configs/smoke.yaml", "gpu": "Fake GPU", "total_minutes": 12.5, "overrides": [],
        "config": {"stages": ["data_checks", "xgboost", "pilot", "gate"]},
        "stages": {
            "data_checks": {"status": "ok", "minutes": 0.2},
            "xgboost": {"status": "ok", "minutes": 0.1, "validation": {"pr_auc": 0.27}},
            "pilot": {"status": "ok", "minutes": 4.0, "selected": {"teacher": "org/model-a", "format": "fmt-v1", "pr_auc": 0.3}},
            "gate": {"status": "ok", "minutes": 0.1, "decision": "HYBRID", "mode": "auto", "branch": "HYBRID", "uplift": 0.01,
                     "llm_pr_auc": 0.28, "xgboost_pr_auc": 0.27, "skip_stages": []},
        }}))
    (run / "report.md").write_text("# Run demo_run")
    write_events(run / "events.jsonl", [
        ev("run_start", stages=["data_checks", "xgboost", "pilot", "gate"], profile="configs/smoke.yaml"),
        ev("stage_start", "pilot", index=3, total=4),
        ev("train_step", "pilot", step=5, max_steps=10, loss=1.5, learning_rate=0.0003, epoch=0.1),
        ev("train_step", "pilot", step=10, max_steps=10, loss=0.9, learning_rate=0.0001, epoch=0.2),
        ev("stage_end", "pilot", status="ok", minutes=4.0, peak_gpu_gb=1.6),
        ev("run_end", status="ok", total_minutes=12.5),
    ])
    (outputs / "mlruns").mkdir()  # reserved: never listed as a run
    return {"outputs": outputs, "configs": configs, "docs": docs, "diagrams": diagrams, "data": data}


@pytest.fixture
def runner() -> FakeRunner:
    return FakeRunner()


@pytest.fixture
def container(tree, runner) -> Container:
    settings = Settings(outputs_dir=tree["outputs"], configs_dir=tree["configs"], data_dir=tree["data"],
                        docs_dir=tree["docs"], diagrams_dir=tree["diagrams"],
                        scoring_url="http://127.0.0.1:9", mlflow_url="http://127.0.0.1:9", explainer_url="http://127.0.0.1:9")
    return Container(settings, runner_override=runner)


@pytest.fixture
def client(container) -> TestClient:
    with TestClient(create_app(container, run_scheduler=False)) as c:
        yield c
