"""ARGUS Fraud Model API (CRD Section 10, LLD Sections 4-6) - local demo build.

  POST /v1/fraud/score            score one transaction (CRD 10.2 request -> 10.3 response)
  GET  /v1/health                 service, scorer, explainer and config status (LLD 4.2)
  GET/POST /admin/v1/thresholds   threshold sets, global or per MCC; two-person approval (LLD 4.4)
  GET/POST/DELETE /admin/v1/lists/{type}   blocked_devices / allowlisted_customers
  GET  /admin/v1/decisions/{id}   support lookup: decision, reasons, explanation (from the audit log)
  GET  /admin/v1/changelog        every config change with who / why (FR-RE-01)
  GET  /admin/v1/decisions        most recent decisions (audit log)
  GET  /admin/v1/models           trained runs that can serve; POST /admin/v1/models/{run}/activate (LLD 4.4 rollback)

Run: uvicorn argus_api.app:app  (env: ARGUS_CONFIG profile, ARGUS_RUN run name, VLLM_URL explainer)
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from argus_ft import data as D
from argus_ft.config import load_config

from .audit import AuditLog
from .explainer import Explainer
from .features import FeatureStore
from .rules import LISTS, RulesEngine
from .schemas import ScoreResponse, TransactionIn

state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cfg = load_config(os.environ.get("ARGUS_CONFIG", "configs/local_demo.yaml"))
    api, home = cfg.api, Path(cfg.output_dir) / "_scoring"  # service-level state, independent of any one run
    from .deployments import Deployments  # heavy imports (torch, transformers) only once the app starts
    store = FeatureStore(cfg.data_dir, cfg.db_dir)
    warm = TransactionIn(transaction_id="WARMUP", customer_id="WARMUP", transaction_amount=100.0,
                         merchant_category="other", transaction_time=datetime.now(timezone.utc), customer_age=30,
                         merchant_location={"lat": 0, "lon": 0, "city": "unk"}, account_age=100)
    models = Deployments(Path(cfg.output_dir), cfg.max_length, lambda s: s.score(store.request_only(warm)))
    state.update(
        cfg=cfg, api=api, store=store, models=models,
        rules=RulesEngine(home / "rules_config.json", Path(cfg.data_dir) / "rules_config.json", home / "changelog.jsonl"),
        explainer=Explainer(os.environ.get("VLLM_URL", api.vllm_url), cfg.data_dir,
                            cfg.sft_explainer.val_rows, cfg.sft_explainer.answer_budget),
        audit=AuditLog(home / "audit.jsonl"),
    )
    initial = os.environ.get("ARGUS_RUN")
    if initial:
        try:  # warm-up happens inside activate: CUDA kernels + weights initialised before traffic
            await run_in_threadpool(models.activate, initial)
        except KeyError:
            logging.getLogger("argus.api").warning("ARGUS_RUN=%s has no servable scorer yet; activate one later", initial)
    await state["explainer"].healthy()  # opens the circuit breaker now if the explainer is not up
    yield


app = FastAPI(title="ARGUS Fraud Model API", version="1.0.0-demo", lifespan=lifespan)


# ---------------------------------------------------------------- errors and auth (FR-API-02, LLD 4.1)

@app.exception_handler(RequestValidationError)
async def invalid_request(_request: Request, exc: RequestValidationError):
    err = exc.errors()[0]
    field = ".".join(str(p) for p in err["loc"] if p != "body") or "body"
    return JSONResponse(status_code=400, content={"error": "INVALID_REQUEST", "field": field, "message": err["msg"]})


def _unauthorised(authorization: str | None, token: str) -> JSONResponse | None:
    if authorization != f"Bearer {token}":
        return JSONResponse(status_code=401, content={"error": "UNAUTHORISED"})
    return None


def _invalid(field: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": "INVALID_REQUEST", "field": field, "message": message})


# ---------------------------------------------------------------- scoring (LLD 5.2 orchestrator)

async def _explain_async(txn_id: str, prompt: str, codes: list[str]) -> None:
    """ALLOW decisions, or REVIEW/BLOCK that missed the deadline: explanation lands in the audit log (FR-API-05)."""
    explanation, source, latency = await state["explainer"].explain(prompt, codes, timeout_s=state["api"].async_explain_timeout_s)
    if explanation is None:  # even the async budget ran out: template, so the audit always has an explanation
        explanation, source = state["explainer"].template(codes), "TEMPLATE"
    state["audit"].explanation(txn_id, explanation, source, latency)


@app.post("/v1/fraud/score", response_model=ScoreResponse)
async def score(txn: TransactionIn, background: BackgroundTasks, authorization: str | None = Header(None),
                x_request_deadline_ms: int | None = Header(None)):
    start = time.perf_counter()
    api, rules, scorer = state["api"], state["rules"], state["models"].scorer
    if denied := _unauthorised(authorization, api.token):
        return denied
    if scorer is None:  # nothing deployed yet: the caller applies fail-open (FR-API-01)
        return JSONResponse(status_code=503, content={"error": "UNAVAILABLE", "retry": False,
                                                      "message": "no model is active: POST /admin/v1/models/{run}/activate"})
    window = api.time_window_minutes
    if window and abs((datetime.now(timezone.utc) - txn.transaction_time).total_seconds()) > window * 60:
        return _invalid("transaction_time", f"must be within {window} minutes of server time")
    deadline_ms = min(x_request_deadline_ms or api.deadline_ms, api.max_deadline_ms)
    elapsed_ms = lambda: int((time.perf_counter() - start) * 1000)

    fv, logit, prob, explanation = None, None, None, None
    if hit := rules.hard_match(txn.device_id):  # FR-RE-02, before any I/O
        decision, codes = hit, []
        explanation = "Declined by a hard rule: the device is on the VKPayments blocklist."
    else:
        try:
            fv = state["store"].lookup(txn)
            logit, prob, codes = await run_in_threadpool(scorer.score, fv)
        except Exception:  # LLD 6.1: scorer error -> 503, the caller applies fail-open (FR-API-01)
            return JSONResponse(status_code=503, content={"error": "UNAVAILABLE", "retry": False})
        decision = rules.decide(prob, txn.merchant_category, txn.customer_id)
        prompt = D.to_format(fv.prompt, scorer.prompt_format)
        if decision.decision in ("REVIEW", "BLOCK"):
            budget_s = (min(api.explain_cutoff_ms, deadline_ms) - elapsed_ms()) / 1000
            if budget_s > 0:
                explanation, source, latency = await state["explainer"].explain(prompt, codes, timeout_s=budget_s)
                if explanation is None:  # LLD 6.1: explainer slow -> reply without it, finish asynchronously
                    background.add_task(_explain_async, txn.transaction_id, prompt, codes)
                else:
                    background.add_task(state["audit"].explanation, txn.transaction_id, explanation, source, latency)
            else:
                background.add_task(_explain_async, txn.transaction_id, prompt, codes)  # EXPLANATION_ASYNC
        else:
            background.add_task(_explain_async, txn.transaction_id, prompt, codes)

    response = {
        "transaction_id": txn.transaction_id,
        "fraud_probability": None if prob is None else round(prob, 4),
        "decision": decision.decision, "risk_level": decision.risk_level, "reason_codes": codes,
        "explanation": explanation, "decision_source": decision.source, "model_version": scorer.version,
        "latency_ms": elapsed_ms(),
    }
    background.add_task(  # after the reply leaves (LLD 5.1)
        state["audit"].decision, txn.model_dump(mode="json"), response,
        feature_vector=fv.values if fv else None, prompt=fv.prompt if fv else None, prompt_format=scorer.prompt_format,
        fraud_logit=logit, feature_source=fv.source if fv else None, versions=rules.versions,
        warnings=[w for w in [decision.note] if w])
    return response


@app.get("/v1/health")
async def health():
    scorer, explainer_ok = state["models"].scorer, await state["explainer"].healthy()
    return {
        "status": "ok" if scorer and explainer_ok else ("degraded" if scorer else "down"),
        "model_version": scorer.version if scorer else None, "scorer": "ok" if scorer else "down",
        "scorer_kind": scorer.kind if scorer else None, "active_run": state["models"].active,
        "explainer": "ok" if explainer_ok else "degraded",
        "feature_store": f"snapshot ({len(state['store'])} transactions)", **state["rules"].versions,
    }


# ---------------------------------------------------------------- admin API (LLD 4.4)

class ThresholdChange(BaseModel):
    medium: float
    review: float
    block: float
    mcc: str | None = None
    requested_by: str
    approved_by: str
    reason: str


class ListChange(BaseModel):
    item: str
    by: str
    reason: str


def _admin(authorization: str | None) -> JSONResponse | None:
    return _unauthorised(authorization, state["api"].admin_token)


@app.get("/admin/v1/thresholds")
async def get_thresholds(authorization: str | None = Header(None)):
    if denied := _admin(authorization):
        return denied
    cfg = state["rules"].cfg
    return {"thresholds": cfg["thresholds"], "mcc_thresholds": cfg["mcc_thresholds"], **state["rules"].versions}


@app.post("/admin/v1/thresholds")
async def set_thresholds(change: ThresholdChange, authorization: str | None = Header(None)):
    if denied := _admin(authorization):
        return denied
    try:
        cfg = state["rules"].set_thresholds({k: getattr(change, k) for k in ("medium", "review", "block")},
                                            change.mcc, change.requested_by, change.approved_by, change.reason)
    except ValueError as e:
        return _invalid("thresholds", str(e))
    return {"thresholds": cfg["thresholds"], "mcc_thresholds": cfg["mcc_thresholds"], **state["rules"].versions}


@app.get("/admin/v1/lists/{name}")
async def get_list(name: str, authorization: str | None = Header(None)):
    if denied := _admin(authorization):
        return denied
    if name not in LISTS:
        return _invalid("type", f"must be one of {sorted(LISTS)}")
    return {name: state["rules"].cfg[name]}


@app.post("/admin/v1/lists/{name}")
@app.delete("/admin/v1/lists/{name}")
async def change_list(name: str, change: ListChange, request: Request, authorization: str | None = Header(None)):
    if denied := _admin(authorization):
        return denied
    try:
        items = state["rules"].update_list(name, change.item, request.method == "POST", change.by, change.reason)
    except KeyError:
        return _invalid("type", f"must be one of {sorted(LISTS)}")
    return {name: items, **state["rules"].versions}


@app.get("/admin/v1/decisions/{transaction_id}")
async def lookup_decision(transaction_id: str, authorization: str | None = Header(None)):
    if denied := _admin(authorization):
        return denied
    record = state["audit"].lookup(transaction_id)
    return record or JSONResponse(status_code=404, content={"error": "NOT_FOUND"})


@app.get("/admin/v1/decisions")
async def recent_decisions(limit: int = 50, authorization: str | None = Header(None)):
    if denied := _admin(authorization):
        return denied
    return state["audit"].recent(max(1, min(limit, 500)))


@app.get("/admin/v1/models")
async def list_models(authorization: str | None = Header(None)):
    if denied := _admin(authorization):
        return denied
    return {"active": state["models"].active, "models": state["models"].available()}


class Activation(BaseModel):
    by: str = "admin"
    reason: str = "deploy"


@app.post("/admin/v1/models/{run}/activate")
async def activate_model(run: str, body: Activation | None = None, authorization: str | None = Header(None)):
    """LLD 4.4 rollback, generalised: make any trained run's scorer the live one (kept warm for instant switching)."""
    if denied := _admin(authorization):
        return denied
    body = body or Activation()
    try:
        scorer = await run_in_threadpool(state["models"].activate, run)
    except KeyError:
        return JSONResponse(status_code=404, content={"error": "NOT_FOUND", "message": f"run '{run}' has no servable scorer"})
    state["rules"].log_change({"change": "activate_model", "run": run, "model_version": scorer.version,
                               "by": body.by, "reason": body.reason})
    return {"active": run, "model_version": scorer.version, "scorer_kind": scorer.kind}


@app.get("/admin/v1/changelog")
async def changelog(authorization: str | None = Header(None)):
    if denied := _admin(authorization):
        return denied
    path = state["rules"].changelog
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
