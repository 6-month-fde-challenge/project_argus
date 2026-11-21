"""Registry - version the run in MLflow (FR-ML-08, TR-DEP-03).

Each run records the data snapshot, prompt format, full config, every stage's metrics and the artefacts,
and registers new versions of `argus-scorer` (FP8 scorer student, or XGBoost on the HYBRID path) and
`argus-explainer` (AWQ explainer student)
with status "candidate" - promotion happens after shadow testing, outside this pipeline.
Local store: metadata in outputs/mlruns/mlflow.db (SQLite), artefacts under outputs/mlruns/artifacts.
Browse with `mlflow ui --backend-store-uri sqlite:///outputs/mlruns/mlflow.db`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException

ARTEFACTS = [  # paths inside outputs/<run_name>/
    "report.json", "xgboost", "pilot/selected.json", "gate", "calibration",
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


def _register(client, name: str, source: str, run_id: str) -> str:
    try:
        client.create_registered_model(name)
    except MlflowException:
        pass  # already registered by an earlier run
    version = client.create_model_version(name, source=source, run_id=run_id)
    client.set_model_version_tag(name, version.version, "status", "candidate")
    return version.version


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
            "scorer": "LLM (FP8 scorer student)" if (ctx.run_dir / "scorer_student" / "fp8").exists() else "XGBoost",
        })
        logged = [rel for rel in ARTEFACTS if _log_artefact(ctx.run_dir, rel)]
        client = mlflow.MlflowClient()
        versions = {name: _register(client, name, f"{run.info.artifact_uri}/{rel}", run.info.run_id)
                    for name, rel in registered_models(ctx.run_dir).items() if rel in logged}

    return {"tracking_uri": store, "run_id": run.info.run_id, "metrics_logged": len(metrics),
            "artefacts": logged, "registered_versions": versions}
