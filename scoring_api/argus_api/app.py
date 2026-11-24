"""ARGUS Fraud Model API (CRD Section 10, LLD Sections 4-6) - local demo build.

  POST /v1/fraud/score            score one transaction (CRD 10.2 request -> 10.3 response)
  POST /v1/fraud/shadow/{run}     the same with the designated shadow run, for comparison only (MLOPS-06)
  POST /admin/v1/models/{run}/shadow, DELETE /admin/v1/models/shadow   designate / clear the shadow run
  GET  /v1/health                 service, scorer, explainer and config status (LLD 4.2)
  GET/POST /admin/v1/thresholds   threshold sets, global or per MCC; two-person approval (LLD 4.4)
  GET/POST/DELETE /admin/v1/lists/{type}   blocked_devices / allowlisted_customers
  GET  /admin/v1/decisions/{id}   support lookup: decision, reasons, explanation (from the audit log)
  GET  /admin/v1/changelog        every config change with who / why (FR-RE-01)
  GET  /admin/v1/decisions        most recent decisions (audit log)
  GET  /admin/v1/models           trained runs that can serve; POST /admin/v1/models/{run}/activate (LLD 4.4 rollback)

Run: uvicorn argus_api.app:app  (env: ARGUS_CONFIG profile, ARGUS_RUN run name, VLLM_URL explainer,
ARGUS_SCORING_TOKEN / ARGUS_SCORING_ADMIN_TOKEN override the profile's demo tokens: always set them on a reachable host)
"""
from __future__ import annotations

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
from .admission import Admission
from .auth import apply_token_env, authorised, bearer_ok
from .runs import ForcedRunRefused, forced_reason
from .explainer import Explainer
from .features import FeatureStore
from .rules import LISTS, RulesEngine
from .schemas import ScoreResponse, TransactionIn

state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cfg = load_config(os.environ.get("ARGUS_CONFIG", "configs/local_demo.yaml"))
    api, home = cfg.api, Path(cfg.output_dir) / "_scoring"  # service-level state, independent of any one run
    apply_token_env(api)
    from .deployments import Deployments  # heavy imports (torch, transformers) only once the app starts
    store = FeatureStore(cfg.data_dir, cfg.db_dir)
    warm = TransactionIn(transaction_id="WARMUP", customer_id="WARMUP", transaction_amount=100.0,
                         merchant_category="other", transaction_time=datetime.now(timezone.utc), customer_age=30,
                         merchant_location={"lat": 0, "lon": 0, "city": "unk"}, account_age=100)
    models = Deployments(Path(cfg.output_dir), cfg.max_length, lambda s: s.score(store.request_only(warm)),
                         allow_forced=bool(api.get("allow_forced_runs", False)))
    state.update(
        cfg=cfg, api=api, store=store, models=models, admission=Admission(api.get("max_concurrent_scoring", 4)),
        rules=RulesEngine(home / "rules_config.json", Path(cfg.data_dir) / "rules_config.json", home / "changelog.jsonl"),
        explainer=Explainer(os.environ.get("VLLM_URL", api.vllm_url), cfg.data_dir,
                            cfg.sft_explainer.val_rows, cfg.sft_explainer.answer_budget,
                            int(api.get("explainer_max_in_flight", 4))),
        audit=AuditLog(home / "audit.jsonl", int(api.get("audit_max_mb", 256)) * 2**20),
    )
    initial = os.environ.get("ARGUS_RUN")
    if initial:
        try:  # warm-up happens inside activate: CUDA kernels + weights initialised before traffic
            await run_in_threadpool(models.activate, initial)
        except KeyError:
            logging.getLogger("argus.api").warning("ARGUS_RUN=%s has no servable scorer yet; activate one later", initial)
        except ForcedRunRefused as e:
            logging.getLogger("argus.api").warning("ARGUS_RUN not activated: %s", e)
    await state["explainer"].healthy()  # opens the circuit breaker now if the explainer is not up
    yield


app = FastAPI(title="ARGUS Fraud Model API", version="1.0.0-demo", lifespan=lifespan)


# ---------------------------------------------------------------- errors and auth (FR-API-02, LLD 4.1)

@app.middleware("http")
async def authenticate(request: Request, call_next):
    """Bearer token before anything else (LLD 4.1): an unauthenticated request gets 401 and nothing about validation."""
    api = state.get("api")
    if api is not None and not authorised(request.url.path, request.headers.get("authorization"), api):
        return JSONResponse(status_code=401, content={"error": "UNAUTHORISED"})
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def invalid_request(_request: Request, exc: RequestValidationError):
    err = exc.errors()[0]
    field = ".".join(str(p) for p in err["loc"] if p != "body") or "body"
    return JSONResponse(status_code=400, content={"error": "INVALID_REQUEST", "field": field, "message": err["msg"]})


def _unauthorised(authorization: str | None, token: str) -> JSONResponse | None:
    if not bearer_ok(authorization, token):
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
    window = api.time_window_minutes
    if window and abs((datetime.now(timezone.utc) - txn.transaction_time).total_seconds()) > window * 60:
        return _invalid("transaction_time", f"must be within {window} minutes of server time")
    deadline_ms = min(x_request_deadline_ms or api.deadline_ms, api.max_deadline_ms)
    elapsed_ms = lambda: int((time.perf_counter() - start) * 1000)

    fv, logit, prob, explanation = None, None, None, None
    # FR-RE-02: hard rules first - before the model, and even when no model is active (a blocklisted device must
    # never be approved by the caller's fail-open rule during a deploy, a rollback or a scorer outage)
    if hit := rules.hard_match(txn.device_id):
        decision, codes = hit, []
        explanation = "Declined by a hard rule: the device is on the VKPayments blocklist."
    elif scorer is None:  # nothing deployed: the caller applies fail-open (FR-API-01)
        return JSONResponse(status_code=503, content={"error": "UNAVAILABLE", "retry": False,
                                                      "message": "no model is active: POST /admin/v1/models/{run}/activate"})
    else:
        admission = state["admission"]
        if not await admission.acquire((deadline_ms - elapsed_ms()) / 1000):  # LLD 6.3: no slot before the deadline
            return JSONResponse(status_code=503, content={
                "error": "UNAVAILABLE", "retry": True,
                "message": "deadline passed while waiting for a scoring slot (overloaded): apply fail-open (FR-API-01)"})
        try:
            fv = state["store"].lookup(txn)
            logit, prob, codes = await run_in_threadpool(scorer.score, fv)
        except Exception:  # LLD 6.1: scorer error -> 503, the caller applies fail-open (FR-API-01)
            return JSONResponse(status_code=503, content={"error": "UNAVAILABLE", "retry": False})
        finally:
            admission.release()
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
        "explanation": explanation, "decision_source": decision.source,
        "model_version": scorer.version if scorer else "none (hard rule)", "latency_ms": elapsed_ms(),
    }
    background.add_task(  # after the reply leaves (LLD 5.1)
        state["audit"].decision, txn.model_dump(mode="json"), response,
        feature_vector=fv.values if fv else None, prompt=fv.prompt if fv else None,
        prompt_format=scorer.prompt_format if scorer else None,
        fraud_logit=logit, feature_source=fv.source if fv else None, versions=rules.versions,
        warnings=[w for w in [decision.note] if w])
    return response


@app.post("/v1/fraud/shadow/{run}")
async def shadow_score(run: str, txn: TransactionIn, authorization: str | None = Header(None)):
    """Shadow scoring (MLOPS-06): the same transaction scored by the run an admin designated as shadow - same features,
    calibrator and rules - to compare a candidate with the live model on real traffic. It changes nothing: no
    explanation, no audit record, and it never loads a model (POST /admin/v1/models/{run}/shadow does that)."""
    start = time.perf_counter()
    if denied := _unauthorised(authorization, state["api"].token):
        return denied
    scorer = state["models"].shadow_scorer(run)
    if scorer is None:
        return JSONResponse(status_code=409, content={"error": "NOT_SHADOW", "message":
                            f"run '{run}' is not the designated shadow run: POST /admin/v1/models/{run}/shadow first"})
    rules = state["rules"]
    prob, codes = None, []
    if hit := rules.hard_match(txn.device_id):
        decision = hit
    else:
        admission = state["admission"]
        if not await admission.try_acquire():  # only a free slot: the shadow never makes live traffic wait
            return JSONResponse(status_code=503, content={"error": "UNAVAILABLE", "retry": True,
                                                          "message": "no free scoring slot for shadow scoring"})
        try:
            _, prob, codes = await run_in_threadpool(scorer.score, state["store"].lookup(txn))
        except Exception:  # noqa: BLE001 - the shadow must never take the service down
            return JSONResponse(status_code=503, content={"error": "UNAVAILABLE", "retry": False})
        finally:
            admission.release()
        decision = rules.decide(prob, txn.merchant_category, txn.customer_id)
    return {"transaction_id": txn.transaction_id, "shadow_run": run,
            "fraud_probability": None if prob is None else round(prob, 4), "decision": decision.decision,
            "risk_level": decision.risk_level, "reason_codes": codes, "decision_source": decision.source,
            "model_version": scorer.version, "latency_ms": int((time.perf_counter() - start) * 1000)}


@app.get("/v1/health")
async def health():
    models = state["models"]
    scorer, explainer_ok = models.scorer, await state["explainer"].healthy()
    # serving_mode: "production" refuses demo (forced / unmeasured) runs, "demo" serves them (api.allow_forced_runs).
    # The AWS stream alarms and ops.py refuses to connect AWS when live traffic would meet a demo model.
    demo_reason = await run_in_threadpool(forced_reason, Path(models.outputs_dir) / models.active) if models.active else None
    return {
        "status": "ok" if scorer and explainer_ok else ("degraded" if scorer else "down"),
        "model_version": scorer.version if scorer else None, "scorer": "ok" if scorer else "down",
        "scorer_kind": scorer.kind if scorer else None, "active_run": models.active,
        "serving_mode": "demo" if models.allow_forced else "production",
        "active_run_forced": demo_reason is not None, "active_run_forced_reason": demo_reason,
        "explainer": "ok" if explainer_ok else "degraded",
        "feature_store": f"snapshot ({len(state['store'])} transactions)", **state["rules"].versions,
        "admission": state["admission"].stats(), "shadow_run": models.shadow_run,
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
    record = await run_in_threadpool(state["audit"].lookup, transaction_id)  # scans files: off the event loop
    return record or JSONResponse(status_code=404, content={"error": "NOT_FOUND"})


@app.get("/admin/v1/decisions")
async def recent_decisions(limit: int = 50, authorization: str | None = Header(None)):
    if denied := _admin(authorization):
        return denied
    return await run_in_threadpool(state["audit"].recent, max(1, min(limit, 500)))


@app.get("/admin/v1/models")
async def list_models(authorization: str | None = Header(None)):
    if denied := _admin(authorization):
        return denied
    return {"active": state["models"].active, "shadow": state["models"].shadow_run, "models": state["models"].available()}


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
    except ForcedRunRefused as e:
        return JSONResponse(status_code=409, content={"error": "FORCED_RUN", "message": str(e)})
    except Exception as e:  # noqa: BLE001 - e.g. CUDA out of memory while loading: the previous state stays served
        logging.getLogger("argus.api").exception("activating %s failed", run)
        return JSONResponse(status_code=503, content={"error": "UNAVAILABLE", "message": f"loading '{run}' failed: {type(e).__name__}"})
    state["rules"].log_change({"change": "activate_model", "run": run, "model_version": scorer.version,
                               "by": body.by, "reason": body.reason})
    return {"active": run, "model_version": scorer.version, "scorer_kind": scorer.kind}


@app.post("/admin/v1/models/{run}/shadow")
async def shadow_model(run: str, body: Activation | None = None, authorization: str | None = Header(None)):
    """MLOPS-06: load a run as the shadow, scored alongside the active run on the stream; it never decides."""
    if denied := _admin(authorization):
        return denied
    body = body or Activation(reason="shadow")
    try:
        scorer = await run_in_threadpool(state["models"].set_shadow, run)
    except KeyError:
        return JSONResponse(status_code=404, content={"error": "NOT_FOUND", "message": f"run '{run}' has no servable scorer"})
    except ValueError as e:
        return _invalid("run", str(e))
    except Exception as e:  # noqa: BLE001 - e.g. CUDA out of memory: the active scorer keeps serving
        logging.getLogger("argus.api").exception("loading shadow %s failed", run)
        return JSONResponse(status_code=503, content={"error": "UNAVAILABLE", "message": f"loading '{run}' failed: {type(e).__name__}"})
    state["rules"].log_change({"change": "shadow_model", "run": run, "model_version": scorer.version,
                               "by": body.by, "reason": body.reason})
    return {"shadow": run, "model_version": scorer.version, "scorer_kind": scorer.kind}


@app.delete("/admin/v1/models/shadow")
async def stop_shadow(authorization: str | None = Header(None)):
    if denied := _admin(authorization):
        return denied
    state["models"].set_shadow(None)
    state["rules"].log_change({"change": "shadow_model", "run": None, "by": "admin", "reason": "shadow off"})
    return {"shadow": None}


@app.get("/admin/v1/changelog")
async def changelog(authorization: str | None = Header(None)):
    if denied := _admin(authorization):
        return denied
    path = state["rules"].changelog
    return await run_in_threadpool(state["audit"].changelog_lines, path)
