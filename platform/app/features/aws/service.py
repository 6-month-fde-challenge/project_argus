"""AWS use cases: status, settings sync, deploy, destroy - all through the repository's GitHub Actions.

No AWS key is stored or used here. The workflows authenticate to AWS through the OIDC role `python deploy/ops.py up`
created once; this service only sets the repository variables / secret they read and starts them.
The rules mirror deploy/ops.py exactly: a demo setup never goes behind live traffic unless explicitly allowed.
"""
from __future__ import annotations

import json

import httpx

from ...core.container import Container
from ...core.errors import AppError, BadRequest, Conflict
from ...core.governance import Approval, check_approval, serving_mode

WORKFLOWS = {"deploy": "deploy.yml", "destroy": "destroy.yml", "ci": "ci.yml"}
SECRET_TF_VARS = {"scoring_token"}                    # deploy/ops.py: a GitHub secret, never a variable
STACK_TF_VARS = {"region", "project", "environment"}  # deploy/ops.py: from AWS_REGION / PROJECT / ENVIRONMENT
SHOWN_VARIABLES = ("AWS_ROLE_ARN", "AWS_REGION", "PROJECT", "ENVIRONMENT", "SCORING_HOST", "ALLOWED_CIDRS",
                   "ALERT_EMAIL", "ALLOW_DEMO", "TF_VARS")


async def public_ip() -> str | None:
    """This host's public address as AWS sees it (the GPU VM's IP)."""
    try:
        async with httpx.AsyncClient(timeout=4.0) as http:
            r = await http.get("https://api.ipify.org")
            return r.text.strip() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def stack_name(variables: dict) -> str:
    return f"{variables.get('PROJECT') or 'argus'}-{variables.get('ENVIRONMENT') or 'prod'}"


def demo_problem(mode: dict) -> str | None:
    """Why live AWS traffic would meet a demo setup (the same rule as deploy/ops.py demo_serving)."""
    if mode.get("live_mode") != "production":
        return (f"the Fraud Model API runs in {mode.get('live_mode') or 'an unknown'} mode (demo runs allowed): serve "
                "with the production profile (Settings -> Serving profile), or allow a demo deploy explicitly")
    if mode.get("active_run_forced"):
        return (f"the active run {mode.get('active_run')} is a demo run ({mode.get('active_run_forced_reason')}): "
                "serve a measured run first")
    return None


class AwsService:
    def __init__(self, c: Container):
        self.c = c

    def _env(self) -> dict[str, str]:
        return self.c.env.values() if self.c.env.exists() else {}

    async def status(self) -> dict:
        env = self._env()
        url = env.get("ARGUS_STREAM_URL") or self.c.settings.stream_url
        out: dict = {"serving": await serving_mode(self.c), "github": {"ok": False}, "variables": {}, "runs": {},
                     "stream": {"url": url, "remote": not url.startswith("http://stream:"),
                                "up": await self.c.stream.up("/live/health")}}
        try:
            gh = self.c.github()
        except AppError as e:
            out["github"]["error"] = e.message
            return out
        try:
            variables = await gh.variables()
            out["github"] = {"ok": True, "repo": gh.repo}
            out["variables"] = {k: variables.get(k) for k in SHOWN_VARIABLES}
            out["bootstrapped"] = bool(variables.get("AWS_ROLE_ARN"))
            out["stack"] = stack_name(variables)
            out["runs"] = {key: await gh.runs(wf) for key, wf in WORKFLOWS.items()}
        except AppError as e:
            out["github"] = {"ok": False, "repo": gh.repo, "error": e.message}
        finally:
            await gh.close()
        return out

    async def deployment_values(self) -> tuple[dict[str, str], dict, str]:
        """(repository variables, TF_VARS, the scoring token) from this machine's .env, as `ops.py up` would set."""
        env = self._env()
        host = env.get("TF_VAR_scoring_host") or await public_ip()
        cidrs, token = env.get("TF_VAR_allowed_cidrs"), env.get("ARGUS_SCORING_TOKEN")
        missing = [k for k, v in (("TF_VAR_scoring_host (or a detectable public IP)", host),
                                  ("TF_VAR_allowed_cidrs", cidrs), ("ARGUS_SCORING_TOKEN", token)) if not v]
        if missing:
            raise BadRequest(f"set on the Settings page first: {', '.join(missing)}")
        tf_vars = {k[len("TF_VAR_"):]: v for k, v in env.items()
                   if k.startswith("TF_VAR_") and v and k[len("TF_VAR_"):] not in SECRET_TF_VARS | STACK_TF_VARS}
        tf_vars["scoring_host"] = host
        if env.get("ARGUS_SCORING_PORT"):
            tf_vars["scoring_port"] = env["ARGUS_SCORING_PORT"]
        values = {"SCORING_HOST": host, "ALLOWED_CIDRS": cidrs, "TF_VARS": json.dumps(tf_vars)}
        if env.get("TF_VAR_alert_email"):
            values["ALERT_EMAIL"] = env["TF_VAR_alert_email"]
        return values, tf_vars, token

    async def sync(self, approval: Approval | None) -> dict:
        mode = await serving_mode(self.c)
        who = check_approval(approval, mode["production"], "syncing the AWS deployment settings")
        values, tf_vars, token = await self.deployment_values()
        gh = self.c.github()
        try:
            for name, value in values.items():
                await gh.set_variable(name, value)
            await gh.set_secret("SCORING_TOKEN", token)
        finally:
            await gh.close()
        self.c.changes.add("aws", "synced the AWS deployment settings to GitHub",
                           {"variables": {**values, "TF_VARS": tf_vars}, "secrets": ["SCORING_TOKEN (changed)"]}, who)
        return {"variables": values, "secrets": ["SCORING_TOKEN"]}

    async def deploy(self, approval: Approval | None, allow_demo: bool) -> dict:
        if not allow_demo and (why := demo_problem(await serving_mode(self.c))):
            raise Conflict(why)
        who = check_approval(approval, True, "deploying to AWS")  # always two people: it changes production
        gh = self.c.github()
        try:
            if not (await gh.variables()).get("AWS_ROLE_ARN"):
                raise Conflict("AWS is not bootstrapped yet: run `python deploy/ops.py up` once from your PC (it needs "
                               "AWS credentials: state bucket + the GitHub deploy role)")
            # workflow_dispatch inputs travel as strings; GitHub converts them to the declared type (boolean)
            await gh.dispatch(WORKFLOWS["deploy"], "main", {"allow_demo": "true" if allow_demo else "false"})
        finally:
            await gh.close()
        self.c.changes.add("aws", "started an AWS deploy (deploy.yml on main)", {"allow_demo": allow_demo}, who)
        return {"started": "deploy.yml", "ref": "main", "allow_demo": allow_demo}

    async def destroy(self, approval: Approval | None, confirm: str) -> dict:
        who = check_approval(approval, True, "destroying the AWS stack")
        gh = self.c.github()
        try:
            stack = stack_name(await gh.variables())
            if confirm != stack:
                raise BadRequest(f"type the stack name '{stack}' to confirm")
            await gh.dispatch(WORKFLOWS["destroy"], "main", {"confirm": stack})
        finally:
            await gh.close()
        self.c.changes.add("aws", f"started an AWS destroy of {stack} (destroy.yml)", {}, who)
        return {"started": "destroy.yml", "stack": stack}
