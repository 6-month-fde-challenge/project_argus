"""Explainer client (LLD 2.2 / 4.3): the AWQ explainer student served by vLLM, schema-constrained JSON.

Every LLM answer goes through the training pipeline's guard (schema + grounding, TR-LLM-07) and must agree with the
scorer's reason codes. If either check rejects it, vLLM is unreachable or the deadline passes, the explanation is built from reason-code templates instead
(FR-API-06, LLD 6.1). Reason codes in the response always come from the scorer; the explainer writes the sentence.
"""
from __future__ import annotations

import time

import httpx

from argus_ft import data as D
from argus_ft.grounding import ExplanationGuard

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason_codes": {"type": "array", "items": {"type": "string", "enum": D.REASON_CODES}},
        "explanation": {"type": "string", "maxLength": 400},
    },
    "required": ["reason_codes", "explanation"],
    "additionalProperties": False,
}
GUARD_STATUS = {"invalid_json": "SCHEMA_FAILED", "schema": "SCHEMA_FAILED", "ungrounded_number": "GROUNDING_FAILED"}
BREAKER_OPEN_S = 5  # LLD 6.1: after a failure, skip the dependency and try again (half-open) after 5 s


def consistent(llm_codes: list[str], scorer_codes: list[str]) -> bool:
    """The sentence must explain the scorer's decision: when the scorer reports reason codes, the explainer's codes
    must share at least one of them (otherwise an investigator reads a reason the decision was not based on)."""
    return not scorer_codes or bool(set(llm_codes) & set(scorer_codes))


class Explainer:
    def __init__(self, url: str, data_dir: str, val_rows: int, max_tokens: int):
        self.url, self.max_tokens = url.rstrip("/"), max_tokens
        gold_train, _ = D.load_gold(data_dir, val_rows)
        self.guard = ExplanationGuard(gold_train)
        self.client = httpx.AsyncClient()
        self._open_until = 0.0  # circuit breaker: while open, explanations come straight from templates

    def template(self, codes: list[str]) -> str:
        return self.guard.template(codes)["explanation"]

    def _trip(self) -> None:
        self._open_until = time.monotonic() + BREAKER_OPEN_S

    async def healthy(self) -> bool:
        try:
            ok = (await self.client.get(f"{self.url}/health", timeout=1.0)).status_code == 200
        except httpx.HTTPError:
            ok = False
        if not ok:
            self._trip()
        return ok

    async def explain(self, prompt: str, codes: list[str], timeout_s: float) -> tuple[str | None, str, int]:
        """-> (explanation, explanation_source, latency_ms). Never raises. A timeout returns (None, "EXPLANATION_ASYNC")
        so the caller can reply without it; any other failure falls back to the reason-code template."""
        start = time.perf_counter()
        if time.monotonic() < self._open_until:  # breaker open: explainer known to be down
            return self.template(codes), "TEMPLATE", 0
        try:
            resp = await self.client.post(f"{self.url}/v1/chat/completions", timeout=timeout_s, json={
                "model": "argus-explainer", "temperature": 0, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": D.SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "argus_explanation", "schema": RESPONSE_SCHEMA}},
            })
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout):  # slow: reply now, explain async
            return None, "EXPLANATION_ASYNC", int((time.perf_counter() - start) * 1000)
        except (httpx.HTTPError, KeyError, ValueError):  # down / bad reply (incl. connect timeout): template
            self._trip()
            return self.template(codes), "TEMPLATE", int((time.perf_counter() - start) * 1000)
        obj, status = self.guard.check(text, prompt)
        latency = int((time.perf_counter() - start) * 1000)
        if obj is None:
            return self.template(codes), GUARD_STATUS.get(status, "SCHEMA_FAILED"), latency
        if not consistent(obj["reason_codes"], codes):
            return self.template(codes), "INCONSISTENT", latency
        return obj["explanation"], "LLM", latency
