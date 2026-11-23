"""Load test of the Fraud Model API: end-to-end latency at a fixed arrival rate (OBJ-10, CRD 15 acceptance gate 4).

    docker compose --profile serve exec scoring python -m argus_api.loadtest --run argus_demo_llm_scorer --rps 5 --duration 60
    python -m argus_api.loadtest --url http://<gpu-vm>:8080 --rps 20 --duration 120 --fresh-time   # from anywhere

Open loop: requests start on a fixed schedule whatever the latency (like real payment traffic), so a slow server
shows up as queueing instead of a politely lower rate. Each request is a real CRD 10.2 transaction replayed from the
holdout split (feature-store hits) mixed with the CRD worked example (request-only path). --fresh-time sets
transaction_time to now, for profiles that enforce the +-15 min window (prod_a100).

Reports p50 / p95 / p99 latency (client side, including the network), status codes, decisions, explanation sources
and the achieved rate, checks p99 against the profile's deadline (250 ms in production), and writes
outputs/<run>/api/loadtest.json.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from argus_ft.config import load_config

from .auth import apply_token_env
from .demo import CRD_EXAMPLE, to_request


def load_requests(data_dir: str, n: int, seed: int) -> list[dict]:
    files = glob.glob(f"{data_dir}/parquet/transactions/holdout/*.parquet")
    rows = pd.concat(pd.read_parquet(p) for p in files).sample(frac=1.0, random_state=seed).head(n) if files else []
    out = [to_request(r) for _, r in rows.iterrows()] if len(rows) else []
    return out + [{**CRD_EXAMPLE, "transaction_id": f"TX_LOAD_{i:04d}"} for i in range(max(1, n // 20))]


def percentiles(ms: list[float]) -> dict:
    if not ms:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    return {f"p{q}_ms": round(float(np.percentile(ms, q)), 1) for q in (50, 95, 99)} | {"max_ms": round(max(ms), 1)}


async def run_load(url: str, token: str, requests: list[dict], rps: float, duration_s: float, fresh_time: bool,
                   timeout_s: float = 10.0) -> dict:
    total = int(rps * duration_s)
    lat, status, decisions, sources = [], Counter(), Counter(), Counter()
    headers = {"Authorization": f"Bearer {token}"}
    # enough connections for every request that can be in flight (rate x timeout): the client must never queue
    # requests itself, or its own waiting would be measured as API latency
    limits = httpx.Limits(max_connections=int(rps * timeout_s) + 10, max_keepalive_connections=int(rps * timeout_s) + 10)

    async with httpx.AsyncClient(base_url=url.rstrip("/"), headers=headers, timeout=timeout_s, limits=limits) as client:
        async def one(req: dict) -> None:
            if fresh_time:
                req = {**req, "transaction_time": datetime.now(timezone.utc).isoformat()}
            start = time.perf_counter()
            try:
                resp = await client.post("/v1/fraud/score", json=req)
                lat.append((time.perf_counter() - start) * 1000)
                status[str(resp.status_code)] += 1
                if resp.status_code == 200:
                    body = resp.json()
                    decisions[body["decision"]] += 1
                    sources["inline" if body.get("explanation") else "async (audit log)"] += 1
            except httpx.HTTPError as e:
                status[type(e).__name__] += 1

        began, tasks = time.perf_counter(), []
        for i in range(total):  # open loop: request i starts at i / rps seconds
            delay = began + i / rps - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(requests[i % len(requests)])))
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - began

    return {"requests": total, "target_rps": rps, "achieved_rps": round(total / elapsed, 2), "seconds": round(elapsed, 1),
            "latency": percentiles(lat), "status": dict(status), "decisions": dict(decisions),
            "explanations": dict(sources), "error_rate": round(1 - status.get("200", 0) / max(total, 1), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/local_demo.yaml", help="profile: tokens and the latency budget")
    ap.add_argument("--run", help="write outputs/<run>/api/loadtest.json (default: the API's active run)")
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--rps", type=float, default=5.0, help="arrival rate, requests per second")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds")
    ap.add_argument("--fresh-time", action="store_true", help="send transaction_time = now (+-15 min window)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tokens = cfg.api
    apply_token_env(tokens)
    health = httpx.get(f"{args.url.rstrip('/')}/v1/health", timeout=10).json()
    run = args.run or health.get("active_run")
    reqs = load_requests(cfg.data_dir, n=min(2000, int(args.rps * args.duration)), seed=args.seed)
    random.Random(args.seed).shuffle(reqs)
    print(f"=== load test {args.url}: {args.rps} req/s for {args.duration:.0f} s, serving {health.get('model_version')}")

    result = asyncio.run(run_load(args.url, tokens.token, reqs, args.rps, args.duration, args.fresh_time))
    budget = cfg.api.deadline_ms
    p99 = result["latency"]["p99_ms"]
    result.update(url=args.url, model_version=health.get("model_version"), active_run=health.get("active_run"),
                  at=datetime.now(timezone.utc).isoformat(), deadline_ms=budget,
                  p99_within_deadline=None if p99 is None else p99 <= budget)
    print(json.dumps(result, indent=2))
    if run:
        out = Path(cfg.output_dir) / run / "api" / "loadtest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
