"""Run the ARGUS-LM model ladder end to end (the stage order is defined in stages/__init__.py).

  python -m argus_ft.pipeline --config configs/local_demo.yaml                       # whole ladder
  python -m argus_ft.pipeline --config configs/smoke.yaml                            # tiny sizes, crash test
  python -m argus_ft.pipeline --config configs/local_demo.yaml --stages distill,quantize,calibrate
  python -m argus_ft.pipeline --config configs/local_demo.yaml --set dpo.max_pairs=200

Steps: 0) sync data from Supabase -> 1..8) run each stage -> write outputs/<run_name>/report.json after every
stage, so a partial run still leaves a full record of what finished.

Each stage runs in its own fresh process. Stages share state only through files in outputs/<run_name>/, and
a fresh process guarantees that (a) library patches made by one stage (e.g. Liger replacing model layer
classes) cannot leak into the next, (b) all GPU memory is returned between stages, and (c) a hard crash in a
stage is recorded instead of taking the runner down.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import time
import traceback

import torch
from transformers import set_seed

from .config import load_config
from .context import Ctx
from .events import EventLog
from .runs import drop_llm_scorer, seed_from
from .stages import LADDER
from .supabase_data import fetch_tables, sync


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ARGUS-LM training ladder")
    ap.add_argument("--config", required=True, help="profile YAML, e.g. configs/local_demo.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. dpo.max_pairs=200")
    ap.add_argument("--stages", help=f"comma-separated subset of: {','.join(LADDER)}")
    ap.add_argument("--no-sync", action="store_true", help="skip the Supabase download step")
    return ap.parse_args()


def _stage_worker(config: str, overrides: list[str], name: str, results) -> None:
    """Body of the per-stage process: rebuild the config and context, run the stage, send back its metrics."""
    cfg = load_config(config, overrides)
    set_seed(cfg.seed)
    # On Windows the driver silently spills past VRAM into system RAM (20x slower); cap PyTorch so it OOMs instead.
    torch.cuda.set_per_process_memory_fraction(cfg.gpu_memory_fraction)
    ctx = Ctx(cfg)
    ctx.events.stage = name  # every event this process writes is tagged with its stage
    try:
        metrics, status = LADDER[name](ctx), "ok"
    except Exception:
        metrics, status = {"error": traceback.format_exc()}, "failed"
    # the runner's own keys last: a stage's metrics can never overwrite its outcome (ok / failed)
    results.put({**metrics, "status": status, "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})


def write_report_md(run_dir) -> None:
    """Readable report next to report.json after every run (best effort: never fails the run)."""
    try:
        from .compare import run_report
        (run_dir / "report.md").write_text(run_report(run_dir), encoding="utf-8")
    except Exception:  # noqa: BLE001
        traceback.print_exc()


def run_stage(name: str, config: str, overrides: list[str]) -> dict:
    """Run one stage in a fresh process; adds status and duration, and records hard crashes."""
    start = time.time()
    spawn = mp.get_context("spawn")  # fresh interpreter: no CUDA context or library patches inherited
    results = spawn.Queue()
    proc = spawn.Process(target=_stage_worker, args=(config, overrides, name, results))
    proc.start()
    result = None
    while result is None and (proc.is_alive() or not results.empty()):  # read before join: no pipe deadlock
        try:
            result = results.get(timeout=5)
        except queue.Empty:
            pass
    proc.join()
    result = result or {"status": "failed", "error": f"stage process exited with code {proc.exitcode} before reporting"}
    return {"minutes": round((time.time() - start) / 60, 2), **result}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    stages = args.stages.split(",") if args.stages else list(cfg.stages)
    if unknown := set(stages) - set(LADDER):
        raise SystemExit(f"unknown stages: {sorted(unknown)}")

    ctx = Ctx(cfg)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"=== ARGUS ladder | run={cfg.run_name} | profile={args.config} | gpu={gpu}")
    print(f"    {' -> '.join(stages)}")

    report_path = ctx.path("report.json")
    report = json.loads(report_path.read_text()) if report_path.exists() else {"stages": {}}
    if cfg.get("reuse_from"):  # variant run: start from another run's artefacts, re-run only `stages`
        report["stages"] = seed_from(ctx.run_dir.parent / cfg.reuse_from, ctx.run_dir, stages)
        report["reused_from"] = cfg.reuse_from
    report.update({"profile": args.config, "overrides": args.set, "gpu": gpu, "config": cfg})

    # 0) data: training files from the storage bucket, feature/label tables from Postgres
    t0 = time.time()
    if not args.no_sync:
        sync(cfg.data_dir)
        fetch_tables(cfg.db_dir)

    # 1..8) the ladder
    events = EventLog(ctx.run_dir)  # run / stage boundaries for the platform UI (stages add steps + progress)
    events.emit("run_start", run_name=cfg.run_name, profile=args.config, stages=stages, overrides=args.set)
    skipped: set[str] = set()
    for i, name in enumerate(stages, 1):
        if name in skipped:
            print(f"\n=== [{i}/{len(stages)}] {name}: skipped by the week-8 gate (HYBRID)")
            events.emit("stage_skipped", stage=name, reason="the week-8 gate took the HYBRID branch")
            continue
        print(f"\n=== [{i}/{len(stages)}] {name}")
        events.emit("stage_start", stage=name, index=i, total=len(stages))
        result = report["stages"][name] = run_stage(name, args.config, args.set)
        events.emit("stage_end", stage=name, status=result["status"], minutes=result.get("minutes"),
                    peak_gpu_gb=result.get("peak_gpu_gb"), error=(result.get("error") or "")[-2000:] or None)
        if result.get("skip_stages"):  # HYBRID: no stale LLM scorer may survive in this folder or its report
            skipped |= set(result["skip_stages"])
            for stale in result["skip_stages"]:
                report["stages"].pop(stale, None)
            if removed := drop_llm_scorer(ctx.run_dir):
                print(f"    removed stale LLM-scorer artefacts: {', '.join(removed)}")
        report["total_minutes"] = round((time.time() - t0) / 60, 2)
        report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(result, indent=2)[:3000])
        if result["status"] == "failed":
            write_report_md(ctx.run_dir)
            events.emit("run_end", status="failed", total_minutes=report["total_minutes"], failed_stage=name)
            raise SystemExit(f"stage '{name}' failed - see {report_path}")

    write_report_md(ctx.run_dir)
    events.emit("run_end", status="ok", total_minutes=report.get("total_minutes", 0))
    print(f"\n=== done in {report['total_minutes']} min -> {report_path}")


if __name__ == "__main__":
    main()
