"""/api/v1/registry - MLflow tracking + model registry, shaped for the UI (MLflow's own UI is also embedded)."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ...core.container import Container, get_container

router = APIRouter(prefix="/registry", tags=["registry"])
EXPERIMENT = "argus-ladder"
KEY_METRICS = ("calibrate.teacher.test_half.pr_auc", "calibrate.xgboost.test_half.pr_auc",
               "calibrate.scorer_student_fp8.test_half.pr_auc", "quantize.explainer_awq.awq_size_mb",
               "quantize.scorer_fp8.fp8.size_mb", "gate.uplift")


def _tags(entity: dict) -> dict:
    return {t["key"]: t["value"] for t in entity.get("tags", []) or []}


@router.get("/info")
def info(c: Container = Depends(get_container)):
    return {"mlflow_url": c.settings.mlflow_public_url, "experiment": EXPERIMENT}


@router.get("/models")
async def models(c: Container = Depends(get_container)):
    out = []
    for m in await c.mlflow.registered_models():
        versions = await c.mlflow.model_versions(m["name"])
        out.append({"name": m["name"], "created": m.get("creation_timestamp"), "updated": m.get("last_updated_timestamp"),
                    "versions": sorted(({"version": v["version"], "run_id": v.get("run_id"), "source": v.get("source"),
                                         "status": _tags(v).get("status", v.get("status")),
                                         "created": v.get("creation_timestamp")} for v in versions),
                                       key=lambda v: -int(v["version"]))})
    return out


@router.get("/runs")
async def runs(c: Container = Depends(get_container)):
    exp = await c.mlflow.experiment(EXPERIMENT)
    if exp is None:
        return []
    out = []
    for r in await c.mlflow.runs(exp["experiment_id"]):
        info, data = r["info"], r.get("data", {})
        metrics = {m["key"]: m["value"] for m in data.get("metrics", [])}
        out.append({"run_id": info["run_id"], "run_name": info.get("run_name"), "status": info.get("status"),
                    "start_time": info.get("start_time"), "end_time": info.get("end_time"), "tags": _tags(data),
                    "key_metrics": {k: metrics[k] for k in KEY_METRICS if k in metrics}, "metric_count": len(metrics)})
    return out
