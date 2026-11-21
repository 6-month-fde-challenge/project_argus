"""Rebuild outputs/<run>/events.jsonl for a run trained before live events existed, from its console log.

    python -m argus_ft.backfill_events outputs/_logs/train_demo_hybrid.log [--force]

Only values the run itself logged are used: the stage markers (`=== [i/n] stage`), the Trainer's loss dicts with
their tqdm step, DPO's `dpo step N: loss x` lines, and per-stage minutes / peak GPU / status from report.json.
Timestamps are spread over each stage's real duration. Every event carries `"backfilled": true`.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

HEADER = re.compile(r"^=== ARGUS ladder \| run=(\S+) \| profile=(\S+)")
STAGE = re.compile(r"^=== \[(\d+)/(\d+)\] (\w+)(: skipped.*)?")
TRAINER_LOG = re.compile(r"(\d+)/(\d+) \[[^\]]*\]\s*(\{'loss'[^}]*\})")
DPO_LOG = re.compile(r"dpo step (\d+): loss ([\d.eE+-]+)")


def parse_log(text: str) -> tuple[str, str, dict[str, dict]]:
    """-> run name, profile, {stage: {"skipped": reason | None, "steps": [{step, max_steps, loss, ...}]}}."""
    run = profile = None
    stages: dict[str, dict] = {}
    current, offset, last = None, 0, 0
    for line in text.splitlines():
        if m := HEADER.match(line):
            run, profile = m.groups()
        elif m := STAGE.match(line):
            current, offset, last = m.group(3), 0, 0
            stages[current] = {"skipped": "skipped by the week-8 gate" if m.group(4) else None, "steps": []}
        elif current:
            for step, total, payload in TRAINER_LOG.findall(line):
                step, total = int(step), int(total)
                if step <= last - offset:  # a new training run inside the same stage (pilot candidates)
                    offset = last
                last = offset + step
                d = {k: float(v) for k, v in ast.literal_eval(payload).items()}
                stages[current]["steps"].append({"step": last, "max_steps": offset + total, "loss": d.get("loss"),
                                                 "learning_rate": d.get("learning_rate"), "grad_norm": d.get("grad_norm"),
                                                 "epoch": d.get("epoch")})
            for step, loss in DPO_LOG.findall(line):
                stages[current]["steps"].append({"step": int(step), "loss": float(loss)})
    if not run:
        raise SystemExit("not an ARGUS pipeline log (no '=== ARGUS ladder' header)")
    return run, profile, stages


def build_events(run: str, profile: str, parsed: dict[str, dict], report: dict, end: datetime) -> list[dict]:
    rep = report.get("stages", {})
    minutes = {s: float(rep.get(s, {}).get("minutes") or 0) for s in parsed}
    t = end - timedelta(minutes=float(report.get("total_minutes") or sum(minutes.values())))
    stamp = lambda when: when.isoformat(timespec="milliseconds").replace("+00:00", "Z")  # noqa: E731
    ev = lambda when, type_, stage=None, **data: {"ts": stamp(when), "type": type_, "stage": stage,  # noqa: E731
                                                  "data": {**data, "backfilled": True}}
    out = [ev(t, "run_start", profile=profile, run_name=run, stages=list(parsed))]
    total = len(parsed)
    for i, (stage, info) in enumerate(parsed.items(), 1):
        if info["skipped"]:
            out.append(ev(t, "stage_skipped", stage, reason=info["skipped"]))
            continue
        dur = timedelta(minutes=minutes[stage])
        out.append(ev(t, "stage_start", stage, index=i, total=total))
        n = len(info["steps"])
        for k, step in enumerate(info["steps"], 1):
            out.append(ev(t + dur * (k / (n + 1)), "train_step", stage, **{a: b for a, b in step.items() if b is not None}))
        t += dur
        r = rep.get(stage, {})
        out.append(ev(t, "stage_end", stage, status=r.get("status", "ok"), minutes=r.get("minutes"),
                      peak_gpu_gb=r.get("peak_gpu_gb"), error=r.get("error")))
    status = "failed" if any(v.get("status") == "failed" for v in rep.values()) else "ok"
    out.append(ev(t, "run_end", status=status, total_minutes=report.get("total_minutes")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", type=Path)
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing events.jsonl")
    args = ap.parse_args()

    run, profile, parsed = parse_log(args.log.read_text(encoding="utf-8", errors="replace"))
    run_dir = args.outputs / run
    target = run_dir / "events.jsonl"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} exists (use --force to overwrite)")
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text())
    end = datetime.fromtimestamp(report_path.stat().st_mtime, timezone.utc)
    events = build_events(run, profile, parsed, report, end)
    target.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    steps = sum(e["type"] == "train_step" for e in events)
    print(f"wrote {target}: {len(events)} events, {steps} loss points across {len(parsed)} stages")


if __name__ == "__main__":
    main()
