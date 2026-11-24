"""Registry - version the run in MLflow (FR-ML-08, TR-DEP-03).

Each run records the data snapshot, prompt format, full config, every stage's metrics and the artefacts,
and registers new versions of `argus-scorer` (FP8 scorer student, or XGBoost on the HYBRID path) and
`argus-explainer` (AWQ explainer student)
with a status (registry_status): "candidate" only for a measured run whose holdout acceptance said GO; "rejected" on
NO-GO; "unverified" when acceptance did not run or failed; "demo" when the run is not a measured one (argus_ft.branch:
forced gate, missing / forced gate decision, or served scorer != measured decision) - never promotable. Promotion
happens after shadow testing, outside this pipeline. The scorer version also carries the holdout verdict of stage 9
(tag `acceptance`: GO / NO-GO).
Local store: metadata in outputs/mlruns/mlflow.db (SQLite), artefacts under outputs/mlruns/artifacts.
Browse with `mlflow ui --backend-store-uri sqlite:///outputs/mlruns/mlflow.db`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException

from ..branch import check

ARTEFACTS = [  # paths inside outputs/<run_name>/
    "report.json", "xgboost", "pilot/selected.json", "gate", "calibration", "acceptance",
    "explainer/sft_samples.json", "explainer/dpo_samples.json",
    "explainer_student/samples.json", "explainer_student/awq_samples.json",
    "scorer_student/fp8", "explainer_student/awq",
]


def registered_models(run_dir: Path) -> dict[str, str]:
    """What production would deploy: the FP8 scorer student, or XGBoost when the run took the HYBRID path
    (no LLM scorer was trained), plus the AWQ explainer student."""
    scorer = "scorer_student/fp8" if (run_dir / "scorer_student" / "fp8").exists() else "xgboost"
    return {"argus-scorer": scorer, "argus-explainer": "explainer_student/awq"}


def _key(name: str) -> str:
    """MLflow keys allow only letters, digits, _ - . / and spaces (e.g. age band "<25" -> "_25")."""
    return re.sub(r"[^\w\-. /]", "_", name)


def _flatten(obj: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in obj.items():
        out.update(_flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def _log_artefact(run_dir: Path, rel: str) -> bool:
    path = run_dir / rel
    if path.is_dir():
        mlflow.log_artifacts(str(path), artifact_path=rel)
    elif path.exists():
        folder = str(Path(rel).parent)
        mlflow.log_artifact(str(path), artifact_path=None if folder == "." else folder)
    return path.exists()


def _register(client, name: str, source: str, run_id: str, status: str) -> str:
    try:
        client.create_registered_model(name)
    except MlflowException:
        pass  # already registered by an earlier run
    version = client.create_model_version(name, source=source, run_id=run_id)
    client.set_model_version_tag(name, version.version, "status", status)
    return version.version


def registry_status(run_dir: Path, gate_mode: str, acceptance: dict | None) -> tuple[str, str | None]:
    """-> (status, why). Only a measured run with a GO on the holdout becomes a promotion candidate."""
    branch = check(run_dir, gate_mode)
    if branch["forced"]:
        return "demo", branch["reason"]
    if not acceptance or acceptance.get("status") == "failed" or "verdict" not in acceptance:
        return "unverified", "holdout acceptance did not run or failed"
    if acceptance.get("reused_from"):
        return "unverified", f"holdout acceptance was reused from {acceptance['reused_from']}"
    if acceptance["verdict"] != "GO":
        return "rejected", "holdout acceptance: " + ", ".join(acceptance.get("failing") or ["NO-GO"])
    return "candidate", None


def _acceptance_tags(client, name: str, version: str, report: dict) -> None:
    """The holdout verdict on the scorer version, so a reviewer sees GO / NO-GO next to the model (MLOPS-06)."""
    acc = report["stages"].get("acceptance")
    if acc and acc.get("verdict") and name == "argus-scorer":
        client.set_model_version_tag(name, version, "acceptance", acc["verdict"])
        if acc.get("failing"):
            client.set_model_version_tag(name, version, "acceptance_failing", ",".join(acc["failing"]))


def run(ctx) -> dict:
    report = json.loads((ctx.run_dir / "report.json").read_text())
    manifest = json.loads((Path(ctx.cfg.data_dir) / "manifest.json").read_text())
    root = (ctx.run_dir.parent / "mlruns").resolve()
    root.mkdir(parents=True, exist_ok=True)
    store = f"sqlite:///{root / 'mlflow.db'}"
    mlflow.set_tracking_uri(store)
    mlflow.set_registry_uri(store)
    if mlflow.get_experiment_by_name("argus-ladder") is None:
        mlflow.create_experiment("argus-ladder", artifact_location=(root / "artifacts").as_uri())
    mlflow.set_experiment("argus-ladder")

    params = {_key(k): str(v)[:500] for k, v in _flatten(ctx.cfg).items()}
    metrics = {_key(k): float(v) for k, v in _flatten(report["stages"]).items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}

    with mlflow.start_run(run_name=ctx.cfg.run_name) as run:
        for i in range(0, len(params), 100):  # MLflow accepts at most 100 params per call
            mlflow.log_params(dict(list(params.items())[i : i + 100]))
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "data_snapshot": f"{manifest['dataset_id']} v{manifest['version']} (labels as of {manifest['label_snapshot']})",
            "prompt_format": ctx.prompt_format, "teacher": ctx.teacher_model,
            "scorer_student": ctx.cfg.models.student_scorer, "explainer_student": ctx.cfg.models.student_explainer,
            "gate_decision": report["stages"].get("gate", {}).get("decision", "n/a"),
            "gate_mode": ctx.cfg.gate.mode,  # auto = the measured branch; hybrid / llm = forced (demo)
            "acceptance_verdict": report["stages"].get("acceptance", {}).get("verdict", "n/a"),  # holdout GO / NO-GO
            "scorer": "LLM (FP8 scorer student)" if (ctx.run_dir / "scorer_student" / "fp8").exists() else "XGBoost",
        })
        logged = [rel for rel in ARTEFACTS if _log_artefact(ctx.run_dir, rel)]
        client = mlflow.MlflowClient()
        status, why = registry_status(ctx.run_dir, ctx.cfg.gate.mode, report["stages"].get("acceptance"))
        versions = {name: _register(client, name, f"{run.info.artifact_uri}/{rel}", run.info.run_id, status)
                    for name, rel in registered_models(ctx.run_dir).items() if rel in logged}
        for name, version in versions.items():
            _acceptance_tags(client, name, version, report)
            if why:
                client.set_model_version_tag(name, version, "status_reason", why[:500])

    return {"tracking_uri": store, "run_id": run.info.run_id, "metrics_logged": len(metrics),
            # never the key "status": the runner merges this into the stage result, whose status is ok / failed
            "artefacts": logged, "registered_versions": versions, "version_status": status, "version_status_reason": why}
