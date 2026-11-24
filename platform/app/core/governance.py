"""Serving mode and two-person approval: the rules every change made through the UI follows.

Serving mode (what the Fraud Model API does with demo runs):
  production  the API runs a profile with api.allow_forced_runs: false - demo (forced / unmeasured) runs are refused
  demo        the API serves demo runs too (local_demo.yaml): fine for demonstrations, never behind live traffic
It is read from both the configuration (.env SCORING_CONFIG -> that profile's api section) and the live API's
/v1/health. When either says production, the stricter rules apply.

Approval: while in production, a change that affects live scoring (serving mode, active run, serving actions, tokens,
AWS deploys) needs two different named people and a reason, like thresholds on Rules & policy. There is no login in
this UI (access is the SSH tunnel), so this is an audit control: names and reasons land in the change log.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .errors import AppError

DEFAULT_SERVING_CONFIG = "configs/local_demo.yaml"   # docker-compose.yml: ARGUS_CONFIG default


class ApprovalRequired(AppError):
    status, code = 428, "APPROVAL_REQUIRED"


class Approval(BaseModel):
    requested_by: str = ""
    approved_by: str = ""
    reason: str = ""


def person(name: str) -> str:
    """One spelling per person: trimmed, inner spaces collapsed, case-folded (as Rules & policy)."""
    return " ".join((name or "").split()).casefold()


def check_approval(approval: Approval | None, needed: bool, what: str) -> dict:
    """-> the approval fields for the change log ({} when not needed). Raises ApprovalRequired otherwise."""
    if not needed:
        if approval and (approval.requested_by or approval.reason):
            return {"requested_by": person(approval.requested_by), "approved_by": person(approval.approved_by),
                    "reason": approval.reason.strip()}
        return {}
    a = approval or Approval()
    req, appr, reason = person(a.requested_by), person(a.approved_by), a.reason.strip()
    if not req or not appr or not reason:
        raise ApprovalRequired(f"{what} affects live scoring in production mode: requested by, approved by and a "
                               "reason are required")
    if req == appr:
        raise ApprovalRequired("two-person approval: approved by must be a different person than requested by")
    return {"requested_by": req, "approved_by": appr, "reason": reason}


def configured_mode(container) -> dict:
    """From .env SCORING_CONFIG (what the API runs with after its next start)."""
    env = container.env.values() if container.env.exists() else {}
    config = env.get("SCORING_CONFIG") or DEFAULT_SERVING_CONFIG
    name = Path(config).stem
    try:
        api = container.profiles.resolve(name).get("api") or {}
        mode = "demo" if api.get("allow_forced_runs", False) else "production"
    except AppError as e:
        return {"config": config, "mode": "unknown", "error": e.message}
    return {"config": config, "mode": mode}


async def serving_mode(container) -> dict:
    """{configured, live, production, active_run, active_run_forced, ...}: production when either side says so."""
    configured = configured_mode(container)
    live = None
    try:
        status, body = await container.scoring.request("GET", "/v1/health", timeout=3.0)  # never hang a page on it
        if status == 200 and isinstance(body, dict):
            live = body
    except AppError:
        pass
    live_mode = (live or {}).get("serving_mode")
    production = configured["mode"] == "production" or live_mode == "production"
    return {
        "configured": configured, "live_mode": live_mode, "production": production,
        "mode": "production" if production else ("demo" if "demo" in (configured["mode"], live_mode) else "unknown"),
        "active_run": (live or {}).get("active_run"), "active_run_forced": bool((live or {}).get("active_run_forced")),
        "active_run_forced_reason": (live or {}).get("active_run_forced_reason"),
        "mismatch": bool(live_mode and configured["mode"] in ("production", "demo") and live_mode != configured["mode"]),
    }
