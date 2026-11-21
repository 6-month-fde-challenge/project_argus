"""Run views for the UI: list summaries, the ladder with per-stage status, loss series, comparisons."""
from __future__ import annotations

from ...core.container import Container

DEFAULT_LADDER = ["data_checks", "xgboost", "pilot", "gate", "cpt", "sft_scorer", "sft_explainer", "dpo",
                  "distill_scorer", "distill_explainer", "quantize", "calibrate", "registry"]
SECTIONS = ("data_checks", "xgboost", "pilot", "gate", "cpt", "sft_scorer", "sft_explainer", "dpo",
            "distill_scorer", "distill_explainer", "quantize", "calibrate", "registry")


class RunService:
    def __init__(self, c: Container):
        self.c = c

    # ------------------------------------------------------------ helpers

    def _ladder(self, report: dict, live: dict) -> list[str]:
        planned = (report.get("config") or {}).get("stages") or live.get("planned") or DEFAULT_LADDER
        planned = list(dict.fromkeys([*planned, *report.get("stages", {})]))  # + stages reused from another run
        return [s for s in DEFAULT_LADDER if s in planned] + [s for s in planned if s not in DEFAULT_LADDER]

    def _status(self, report: dict, live: dict, job: dict | None, ladder: list[str]) -> str:
        if job and job["status"] in ("queued", "running"):
            return job["status"]
        if live["status"] in ("running", "failed", "interrupted"):
            return live["status"]
        stages = report.get("stages", {})
        if any(v.get("status") == "failed" for v in stages.values()):
            return "failed"
        skipped = set((stages.get("gate") or {}).get("skip_stages", []))
        done = {s for s, v in stages.items() if v.get("status") == "ok"} | skipped
        if live["status"] == "succeeded" or set(ladder) <= done:
            return "succeeded"
        return "incomplete" if stages else "empty"

    def _stage_rows(self, report: dict, live: dict, ladder: list[str]) -> list[dict]:
        stages, skip = report.get("stages", {}), set((report.get("stages", {}).get("gate") or {}).get("skip_stages", []))
        rows = []
        for i, name in enumerate(ladder, 1):
            rep, lv = stages.get(name, {}), live["stages"].get(name, {})
            # live "pending" wins only while the run is live (a stage queued for a re-run); otherwise the report rules
            if lv.get("status") == "running" or (lv.get("status") == "pending" and (live.get("live") or not rep)):
                status = lv["status"]
            elif lv.get("status") == "skipped" or (name in skip and not rep):
                status = "skipped"
            elif rep.get("reused_from"):
                status = "reused"
            else:
                status = rep.get("status") or lv.get("status") or "pending"
            rows.append({"index": i, "name": name, "status": status, "minutes": rep.get("minutes", lv.get("minutes")),
                         "peak_gpu_gb": rep.get("peak_gpu_gb", lv.get("peak_gpu_gb")), "reused_from": rep.get("reused_from"),
                         "progress": lv.get("progress"), "last_step": lv.get("last_step"),
                         "error": rep.get("error") or lv.get("error")})
        return rows

    @staticmethod
    def _scorer(report: dict) -> str | None:
        q = report.get("stages", {}).get("quantize", {})
        if "scorer_fp8" in q:
            return "LLM"
        return "XGBoost" if q or report.get("stages", {}).get("calibrate") else None

    # ------------------------------------------------------------ views

    def summary(self, run: str) -> dict:
        report, live, job = self.c.runs.report(run), self.c.live.get(run), self.c.jobs.latest_for_run(run)
        ladder = self._ladder(report, live)
        rows = self._stage_rows(report, live, ladder)
        stages = report.get("stages", {})
        gate, pilot = stages.get("gate", {}), stages.get("pilot", {})
        current = next((r for r in rows if r["status"] == "running"), None)
        return {
            "name": run, "status": self._status(report, live, job, ladder),
            "profile": report.get("profile") or live.get("profile"), "reused_from": report.get("reused_from"),
            "gpu": report.get("gpu"), "total_minutes": report.get("total_minutes"),
            "stages_done": sum(r["status"] in ("ok", "reused", "skipped") for r in rows), "stages_total": len(rows),
            "current_stage": current["name"] if current else None,
            "progress": current["progress"] if current else None, "last_step": current["last_step"] if current else None,
            "gate": {k: gate.get(k) for k in ("decision", "mode", "branch", "uplift", "llm_pr_auc", "xgboost_pr_auc")} if gate else None,
            "pilot_winner": pilot.get("selected"), "scorer": self._scorer(report),
            "updated_at": self.c.runs.modified(run), "job": job,
        }

    def list(self) -> list[dict]:
        return sorted((self.summary(r) for r in self.c.runs.names()), key=lambda s: -(s["updated_at"] or 0))

    def detail(self, run: str) -> dict:
        report, live = self.c.runs.report(run), self.c.live.get(run)
        ladder = self._ladder(report, live)
        stages = report.get("stages", {})
        return {
            **self.summary(run),
            "ladder": self._stage_rows(report, live, ladder),
            "sections": {k: stages.get(k) for k in SECTIONS if k in stages},
            "config": report.get("config"), "overrides": report.get("overrides", []),
            "has_report_md": self.c.runs.report_md(run) is not None,
        }

    def loss_series(self, run: str) -> dict:
        """Per-stage training curves from train_step events, plus the offset the live SSE stream continues from."""
        events, offset = self.c.runs.events(run)
        series: dict[str, list[dict]] = {}
        for ev in events:
            if ev.get("type") == "run_start":
                for s in (ev.get("data") or {}).get("stages", []):
                    series.pop(s, None)  # a re-run of a stage replaces its old curve
            if ev.get("type") == "train_step" and ev.get("stage"):
                d = ev["data"]
                series.setdefault(ev["stage"], []).append(
                    {"step": d.get("step"), "loss": d.get("loss"), "learning_rate": d.get("learning_rate"),
                     "epoch": d.get("epoch"), "eval": d.get("eval", False), "ts": ev.get("ts")})
        return {"series": series, "offset": offset}

    def compare(self, runs: list[str]) -> list[dict]:
        out = []
        for run in runs:
            stages = self.c.runs.report(run).get("stages", {})
            out.append({"summary": self.summary(run), "calibrate": stages.get("calibrate"),
                        "quantize": stages.get("quantize"), "pilot": stages.get("pilot"), "gate": stages.get("gate"),
                        "explainers": {k: (stages.get(k) or {}).get("validation") for k in ("sft_explainer", "dpo", "distill_explainer")}})
        return out

    def is_live(self, run: str) -> bool:
        job = self.c.jobs.latest_for_run(run)
        return bool(job and job["status"] in ("queued", "running")) or self.c.live.get(run)["live"]
