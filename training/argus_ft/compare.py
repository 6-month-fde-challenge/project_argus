"""Readable reports: outputs/<run>/report.md for each run, and outputs/comparison.md side by side.

  python -m argus_ft.compare argus_local_demo argus_local_demo_llm_scorer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCORER_NAMES = {"xgboost": "XGBoost", "teacher": "LLM teacher (bf16)", "scorer_student": "LLM student (bf16)",
                "scorer_student_fp8": "LLM student (FP8)"}
EXPLAINERS = [("sft_explainer", "Teacher after SFT"), ("dpo", "Teacher after DPO"),
              ("distill_explainer", "Student (bf16)"), ("quantize", "Student (AWQ 4-bit)")]


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def _num(x, digits=3) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def scorer_rows(stages: dict) -> list[str]:
    rows = []
    for key, label in SCORER_NAMES.items():
        r = stages.get("calibrate", {}).get(key)
        if not r or "test_half" not in r:
            continue
        cap = r.get("at_review_capacity")
        rows.append(
            f"| {label} | {_num(r['test_half']['pr_auc'])} | {_num(r['test_half']['roc_auc'])} | "
            f"{r['chosen_calibrator']} ({_num(r['ece'][r['chosen_calibrator']], 4)}) | "
            + (f"{_pct(cap['recall'])} / {_pct(cap['precision'])} / {_pct(cap['rupee_recall'])} | " if cap else "– | ")
            + f"{_pct(r['at_review']['recall'])} / {_pct(r['at_review']['precision'])} | "
            f"{_pct(r['at_block']['recall'])} / {_pct(r['at_block']['precision'])} |")
    return rows


def explainer_rows(stages: dict) -> list[str]:
    rows = []
    for key, label in EXPLAINERS:
        stage = stages.get(key, {})
        v = stage.get("explainer_awq", {}).get("validation") if key == "quantize" else stage.get("validation")
        if v:
            distinct = f"{v['distinct_code_sets']} of {v['gold_distinct_code_sets']}" if "distinct_code_sets" in v else "–"
            rows.append(f"| {label} | {_pct(v['json_valid_rate'])} | {_pct(v['schema_valid_rate'])} | "
                        f"{_num(v['reason_f1'])} | {_pct(v['guard_accepted_rate'])} | {distinct} |")
    return rows


def run_report(run_dir: Path) -> str:
    report = json.loads((run_dir / "report.json").read_text())
    st = report["stages"]
    gate, pilot, q = st.get("gate", {}), st.get("pilot", {}), st.get("quantize", {})
    scorer = "LLM (FP8 scorer student)" if "scorer_fp8" in q else "XGBoost (HYBRID path)"
    if gate.get("mode", "auto") != "auto":
        scorer += f" (gate.mode={gate['mode']}: branch forced; measured decision {gate.get('decision')})"
    out = [f"# Run `{run_dir.name}`", "",
           f"- Profile: `{report['profile']}`" + (f", reusing `{report['reused_from']}`" if report.get("reused_from") else ""),
           f"- GPU: {report['gpu']}",
           f"- Pilot winner: **{pilot.get('selected', {}).get('teacher', '–')} / {pilot.get('selected', {}).get('format', '–')}**",
           f"- Week-8 gate: **{gate.get('decision', '–')}** (uplift {gate.get('uplift', '–')} vs required "
           f"{gate.get('required_uplift', '–')}); production scorer in this run: **{scorer}**", ""]
    if pilot.get("grid"):
        out += ["## Pilot (models x formats)", "", "| Pair | PR-AUC | ROC-AUC |", "|---|---|---|"]
        out += [f"| {k.replace(' | ', ' × ')} | {_num(v['pr_auc'])} | {_num(v['roc_auc'])} |"
                for k, v in sorted(pilot["grid"].items(), key=lambda kv: -(kv[1]["pr_auc"] or 0))]
        out.append("")
    out += ["## Scorers (later half of validation, calibrated)", "",
            "| Scorer | PR-AUC | ROC-AUC | Calibrator (ECE) | Top 12,000/day to REVIEW: recall / precision / rupee recall | "
            "Policy REVIEW >=0.60: recall / precision | Policy BLOCK >=0.85: recall / precision |",
            "|---|---|---|---|---|---|---|", *scorer_rows(st), "",
            "## Explainers (held-out gold cases)", "",
            "| Explainer | Valid JSON | Valid schema | Reason-code F1 | Accepted by guard | Distinct answers (gold) |",
            "|---|---|---|---|---|---|",
            *explainer_rows(st), ""]
    if q:
        out += ["## Quantisation", ""]
        if "scorer_fp8" in q:
            s = q["scorer_fp8"]
            out.append(f"- Scorer student: {s['bf16']['size_mb']} MB bf16 -> **{s['fp8']['size_mb']} MB FP8**; "
                       f"PR-AUC {_num(s['bf16']['validation']['pr_auc'])} -> {_num(s['fp8']['validation']['pr_auc'])}")
        e = q.get("explainer_awq", {})
        out += [f"- Explainer student: {e.get('bf16_size_mb')} MB bf16 -> **{e.get('awq_size_mb')} MB AWQ 4-bit** "
                f"(group size {e.get('group_size')})", ""]
    out += ["## Stages", "", "| Stage | Status | Minutes | Peak GPU (GB) |", "|---|---|---|---|"]
    out += [f"| {k} | {v.get('status')}{' (reused)' if v.get('reused_from') else ''} | {v.get('minutes')} | "
            f"{v.get('peak_gpu_gb')} |" for k, v in st.items()]
    return "\n".join(out) + "\n"


def comparison(run_dirs: list[Path]) -> str:
    out = ["# Run comparison", "", "| Run | Gate (measured) | Scorer | Pilot winner |", "|---|---|---|---|"]
    for d in run_dirs:
        st = json.loads((d / "report.json").read_text())["stages"]
        sel = st.get("pilot", {}).get("selected", {})
        out.append(f"| `{d.name}` | {st.get('gate', {}).get('decision', '–')} | "
                   f"{'LLM (FP8 student)' if 'scorer_fp8' in st.get('quantize', {}) else 'XGBoost'} | "
                   f"{sel.get('teacher', '–')} / {sel.get('format', '–')} |")
    for d in run_dirs:
        st = json.loads((d / "report.json").read_text())["stages"]
        out += ["", f"## `{d.name}` - scorers", "",
                "| Scorer | PR-AUC | ROC-AUC | Calibrator (ECE) | Top 12,000/day to REVIEW: recall / precision / rupee recall | "
                "Policy REVIEW >=0.60 | Policy BLOCK >=0.85 |", "|---|---|---|---|---|---|---|", *scorer_rows(st),
                "", f"## `{d.name}` - explainers", "",
                "| Explainer | Valid JSON | Valid schema | Reason-code F1 | Accepted by guard | Distinct answers (gold) |",
            "|---|---|---|---|---|---|",
                *explainer_rows(st)]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run names under outputs/")
    ap.add_argument("--outputs", default="outputs")
    args = ap.parse_args()
    run_dirs = [Path(args.outputs) / r for r in args.runs]
    for d in run_dirs:
        (d / "report.md").write_text(run_report(d), encoding="utf-8")
        print(f"wrote {d / 'report.md'}")
    if len(run_dirs) > 1:
        (Path(args.outputs) / "comparison.md").write_text(comparison(run_dirs), encoding="utf-8")
        print(f"wrote {Path(args.outputs) / 'comparison.md'}")


if __name__ == "__main__":
    main()
