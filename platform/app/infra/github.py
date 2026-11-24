"""GitHub REST API for the AWS page: the repository's deploy / destroy workflows and their settings.

AWS is never called from here and no AWS key is stored on the VM: deploys run in GitHub Actions through the OIDC role
that `python deploy/ops.py up` created once. This client only (a) sets the repository variables / the SCORING_TOKEN
secret the workflows read, (b) starts deploy.yml / destroy.yml, (c) reads their runs.
Token: a fine-grained personal access token for this repository with Actions, Variables and Secrets read/write
(ARGUS_GITHUB_TOKEN in .env, set on the Settings page, never shown again).
"""
from __future__ import annotations

import base64
import re

import httpx

from ..core.errors import BadRequest, UpstreamError

REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def repo_from_git_config(repo_dir) -> str | None:
    """owner/repo of the `origin` remote in .git/config (https or ssh URL), None when there is none."""
    try:
        text = (repo_dir / ".git" / "config").read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r'\[remote "origin"\][^\[]*?url\s*=\s*(\S+)', text, re.S)
    if not m:
        return None
    found = re.search(r"github\.com[:/]([^/\s]+/[^/\s]+?)(?:\.git)?$", m.group(1))
    return found.group(1) if found else None


class GitHub:
    def __init__(self, api: str, repo: str, token: str, transport: httpx.AsyncBaseTransport | None = None):
        if not REPO_RE.match(repo or ""):
            raise BadRequest("GitHub repository not set: ARGUS_GITHUB_REPO (owner/repo) on the Settings page")
        if not token:
            raise BadRequest("GitHub token not set: ARGUS_GITHUB_TOKEN on the Settings page (fine-grained token for "
                             f"{repo} with Actions, Variables and Secrets read/write)")
        self.repo = repo
        self.http = httpx.AsyncClient(base_url=api.rstrip("/"), timeout=20.0, transport=transport, headers={
            "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "argus-platform"})

    async def _call(self, method: str, path: str, ok=(200, 201, 204), **kw) -> httpx.Response:
        try:
            resp = await self.http.request(method, f"/repos/{self.repo}{path}", **kw)
        except httpx.HTTPError as e:
            raise UpstreamError(f"GitHub is not reachable ({type(e).__name__})") from e
        if resp.status_code not in ok:
            try:
                msg = resp.json().get("message", resp.text[:200])
            except ValueError:
                msg = resp.text[:200]
            hint = " (the token lacks a permission, or the repository name is wrong)" if resp.status_code in (403, 404) else ""
            raise UpstreamError(f"GitHub {method} {path}: HTTP {resp.status_code} {msg}{hint}")
        return resp

    async def close(self) -> None:
        await self.http.aclose()

    # ------------------------------------------------------------ variables / secrets
    async def variables(self) -> dict[str, str]:
        resp = await self._call("GET", "/actions/variables", params={"per_page": 30})
        return {v["name"]: v["value"] for v in resp.json().get("variables", [])}

    async def set_variable(self, name: str, value: str) -> None:
        resp = await self._call("PATCH", f"/actions/variables/{name}", ok=(204, 404), json={"name": name, "value": value})
        if resp.status_code == 404:  # not created yet
            await self._call("POST", "/actions/variables", json={"name": name, "value": value})

    async def set_secret(self, name: str, value: str) -> None:
        from nacl import encoding, public  # sealed box with the repository's public key (GitHub's required scheme)
        key = (await self._call("GET", "/actions/secrets/public-key")).json()
        box = public.SealedBox(public.PublicKey(key["key"].encode(), encoding.Base64Encoder()))
        sealed = base64.b64encode(box.encrypt(value.encode())).decode()
        await self._call("PUT", f"/actions/secrets/{name}", json={"encrypted_value": sealed, "key_id": key["key_id"]})

    # ------------------------------------------------------------ workflows
    async def dispatch(self, workflow: str, ref: str = "main", inputs: dict | None = None) -> None:
        await self._call("POST", f"/actions/workflows/{workflow}/dispatches", json={"ref": ref, "inputs": inputs or {}})

    async def runs(self, workflow: str, limit: int = 5) -> list[dict]:
        resp = await self._call("GET", f"/actions/workflows/{workflow}/runs", params={"per_page": limit}, ok=(200, 404))
        if resp.status_code == 404:
            return []
        return [{k: r.get(k) for k in ("id", "status", "conclusion", "html_url", "created_at", "event", "head_branch",
                                       "display_title")} | {"sha": (r.get("head_sha") or "")[:10]}
                for r in resp.json().get("workflow_runs", [])]
