"""HTTP adapters for the neighbouring services. The scoring tokens stay here: the browser never sees them."""
from __future__ import annotations

import httpx

from ..core.config import Settings
from ..core.errors import UpstreamError


class _Client:
    name = "service"

    def __init__(self, base_url: str, timeout: float = 30.0, headers: dict | None = None):
        self.http = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout, headers=headers or {})

    async def request(self, method: str, path: str, *, expect_json: bool = True, **kw):
        try:
            resp = await self.http.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise UpstreamError(f"{self.name} is not reachable ({type(e).__name__}); is it running?") from e
        if expect_json:
            try:
                return resp.status_code, resp.json()
            except ValueError:
                return resp.status_code, {"error": "UPSTREAM_BAD_RESPONSE", "message": resp.text[:300]}
        return resp.status_code, resp.text

    async def up(self, path: str, timeout: float = 2.0) -> bool:
        try:
            return (await self.http.get(path, timeout=timeout)).status_code < 500
        except httpx.HTTPError:
            return False


class ScoringClient(_Client):
    """The Fraud Model API (scoring_api): scoring with the service token, admin with the admin token."""
    name = "scoring API"

    def __init__(self, s: Settings):
        super().__init__(s.scoring_url, timeout=60.0)
        self.service = {"Authorization": f"Bearer {s.scoring_token}"}
        self.admin = {"Authorization": f"Bearer {s.scoring_admin_token}"}

    async def score(self, body: dict):
        return await self.request("POST", "/v1/fraud/score", json=body, headers=self.service)

    async def health(self):
        return await self.request("GET", "/v1/health")

    async def admin_call(self, method: str, path: str, body: dict | None = None):
        return await self.request(method, f"/admin/v1/{path}", json=body, headers=self.admin)


class MlflowClient(_Client):
    """MLflow REST API 2.0 (tracking + model registry)."""
    name = "MLflow"

    def __init__(self, s: Settings):
        super().__init__(s.mlflow_url)

    async def _json(self, method: str, path: str, **kw) -> dict:
        status, body = await self.request(method, f"/api/2.0/mlflow/{path}", **kw)
        if status >= 400:
            raise UpstreamError(f"MLflow {path}: {body.get('message', body) if isinstance(body, dict) else body}")
        return body

    async def registered_models(self) -> list[dict]:
        return (await self._json("GET", "registered-models/search", params={"max_results": 100})).get("registered_models", [])

    async def model_versions(self, name: str) -> list[dict]:
        return (await self._json("GET", "model-versions/search", params={"filter": f"name='{name}'"})).get("model_versions", [])

    async def experiment(self, name: str) -> dict | None:
        status, body = await self.request("GET", "/api/2.0/mlflow/experiments/get-by-name", params={"experiment_name": name})
        return body.get("experiment") if status == 200 else None

    async def runs(self, experiment_id: str) -> list[dict]:
        body = await self._json("POST", "runs/search", json={"experiment_ids": [experiment_id], "max_results": 200,
                                                              "order_by": ["attributes.start_time DESC"]})
        return body.get("runs", [])


class ExplainerClient(_Client):
    name = "vLLM explainer"

    def __init__(self, s: Settings):
        super().__init__(s.explainer_url, timeout=5.0)
