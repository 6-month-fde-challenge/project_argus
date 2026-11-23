/** Stage report renderers, keyed by stage id (registry pattern: a new stage = one entry in SECTION_RENDERERS). */
import { useMemo, useState, type ReactNode } from "react";
import { Check, X } from "lucide-react";
import { axisStyle, baseOption, EChart, PALETTE } from "@/shared/charts/EChart";
import { REASONS } from "@/shared/lib/reasons";
import { cn, fmt } from "@/shared/lib/utils";
import { Badge, Card, CardBody, CardHeader, EmptyState, NativeSelect } from "@/shared/ui/core";
import { KeyValue } from "@/shared/ui/content";
import { useRunFile } from "../api";

type S = Record<string, any>;
const SCORER_LABEL: Record<string, string> = {
  xgboost: "XGBoost", teacher: "LLM teacher (bf16)", scorer_student: "LLM student (bf16)", scorer_student_fp8: "LLM student (FP8)",
};

function Metric({ label, value, tone }: { label: string; value: ReactNode; tone?: string }) {
  return (
    <div className="rounded-xl border border-line bg-white/[0.02] p-3">
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className={cn("mt-1 text-lg font-semibold tabular-nums", tone)}>{value}</div>
    </div>
  );
}

function Table({ head, rows }: { head: ReactNode[]; rows: ReactNode[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead><tr>{head.map((h, i) => <th key={i} className="border-b border-line px-2 py-2 text-left font-medium text-muted">{h}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i} className="hover:bg-white/[0.02]">{r.map((c, j) => <td key={j} className="border-b border-line/50 px-2 py-2 tabular-nums">{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ data checks */
function DataChecks({ s }: { s: S }) {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card><CardHeader title="Leakage tests" subtitle="Any failure stops the run before training (TR-DATA-04, DR-02)" />
        <CardBody className="space-y-1.5">
          {Object.entries(s.leakage ?? {}).map(([k, ok]) => (
            <div key={k} className="flex items-center gap-2 text-xs">
              {ok ? <Check className="h-3.5 w-3.5 text-ok" /> : <X className="h-3.5 w-3.5 text-bad" />}
              <span className={ok ? "text-ink/90" : "text-bad"}>{k.replaceAll("_", " ")}</span>
            </div>
          ))}
        </CardBody>
      </Card>
      <Card><CardHeader title="Splits and prompt size" />
        <CardBody>
          <Table head={["split", "rows", "fraud", "time span"]} rows={Object.entries(s.rows_and_fraud ?? {}).map(([k, v]: any) =>
            [k, fmt.int(v[0]), fmt.int(v[1]), `${s.time_spans?.[k]?.[0]?.slice(0, 10)} → ${s.time_spans?.[k]?.[1]?.slice(0, 10)}`])} />
          <div className="mt-3"><KeyValue data={{ ...(s.prompt_tokens ?? {}), ...(s.missing_rates ?? {}) }} /></div>
        </CardBody>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ xgboost */
function Xgboost({ s }: { s: S }) {
  const feats = Object.entries(s.top_features ?? {}) as [string, number][];
  const option = useMemo(() => ({
    ...baseOption(), grid: { left: 170, right: 20, top: 10, bottom: 20 },
    xAxis: { type: "value", ...axisStyle }, yAxis: { type: "category", data: feats.map(([k]) => k).reverse(), ...axisStyle },
    series: [{ type: "bar", data: feats.map(([, v]) => v).reverse(), itemStyle: { borderRadius: [0, 6, 6, 0], color: PALETTE[1] }, barWidth: 14 }],
  }), [feats]);
  return (
    <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
      <div className="grid grid-cols-2 gap-3">
        <Metric label="PR-AUC (validation)" value={fmt.num(s.validation?.pr_auc)} tone="text-brand-2" />
        <Metric label="ROC-AUC" value={fmt.num(s.validation?.roc_auc)} />
        <Metric label="train rows" value={fmt.int(s.train_rows)} />
        <Metric label="train fraud" value={fmt.int(s.train_fraud)} />
      </div>
      <Card><CardHeader title="Feature importance" subtitle="From the Supabase features table (FE-01..FE-10)" /><CardBody><EChart option={option} height={200} /></CardBody></Card>
    </div>
  );
}

/* ------------------------------------------------------------------ pilot */
function Pilot({ s }: { s: S }) {
  const grid = s.grid ?? {};
  const models = [...new Set(Object.keys(grid).map((k) => k.split(" | ")[0]))];
  const formats = [...new Set(Object.keys(grid).map((k) => k.split(" | ")[1]))];
  const data = models.flatMap((m, y) => formats.map((f, x) => [x, y, grid[`${m} | ${f}`]?.pr_auc ?? 0]));
  const option = useMemo(() => ({
    ...baseOption(), tooltip: { formatter: (p: any) => `${models[p.value[1]]} × ${formats[p.value[0]]}<br/>PR-AUC <b>${p.value[2].toFixed(3)}</b>` },
    grid: { left: 210, right: 30, top: 10, bottom: 40 },
    xAxis: { type: "category", data: formats, ...axisStyle }, yAxis: { type: "category", data: models.map(fmt.model), ...axisStyle },
    visualMap: { min: 0, max: Math.max(0.05, ...data.map((d) => d[2] as number)), show: false, inRange: { color: ["#111a2e", "#155e75", "#22d3ee", "#8b5cf6"] } },
    series: [{ type: "heatmap", data, label: { show: true, color: "#fff", formatter: (p: any) => p.value[2].toFixed(3) }, itemStyle: { borderColor: "#060a14", borderWidth: 3, borderRadius: 8 } }],
  }), [grid]);
  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
      <Card><CardHeader title="3 models × 2 prompt formats" subtitle="Same short LoRA run per pair; ranked on validation PR-AUC (Figure 4, FR-FE-07)" />
        <CardBody>{models.length ? <EChart option={option} height={Math.max(180, models.length * 64 + 50)} /> : <EmptyState title="No pilot grid" />}</CardBody>
      </Card>
      <div className="space-y-3">
        <Metric label="selected teacher" value={<span className="text-base">{fmt.model(s.selected?.teacher)}</span>} tone="text-brand" />
        <Metric label="selected format" value={s.selected?.format ?? "–"} />
        <Metric label="winner PR-AUC" value={fmt.num(s.selected?.pr_auc)} tone="text-ok" />
        <div className="text-[11px] text-muted">{fmt.int(s.train_rows)} training rows · {fmt.int(s.val_rows)} validation rows per pair</div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ gate */
function Gate({ s }: { s: S }) {
  const option = useMemo(() => ({
    ...baseOption(), grid: { left: 110, right: 40, top: 10, bottom: 24 },
    xAxis: { type: "value", ...axisStyle }, yAxis: { type: "category", data: ["XGBoost", "Pilot LLM"], ...axisStyle },
    series: [{ type: "bar", barWidth: 22, data: [
      { value: s.xgboost_pr_auc, itemStyle: { color: PALETTE[1], borderRadius: [0, 6, 6, 0] } },
      { value: s.llm_pr_auc, itemStyle: { color: PALETTE[0], borderRadius: [0, 6, 6, 0] } }],
      label: { show: true, position: "right", color: "#e6ebf5", formatter: (p: any) => p.value?.toFixed(3) },
      markLine: { symbol: "none", lineStyle: { color: "#fbbf24", type: "dashed" }, label: { color: "#fbbf24", formatter: "required" },
        data: [{ xAxis: (s.xgboost_pr_auc ?? 0) + (s.required_uplift ?? 0.03) }] } }],
  }), [s]);
  const llm = s.decision === "LLM";
  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
      <Card><CardHeader title="Week-8 gate: +0.03 PR-AUC over XGBoost?" subtitle={`Same ${fmt.int(s.same_rows)} validation rows · ${s.llm}`} />
        <CardBody><EChart option={option} height={150} /></CardBody></Card>
      <div className="space-y-3">
        <Metric label="measured decision" value={<Badge tone={llm ? "brand" : "violet"}>{s.decision}</Badge>} />
        <Metric label="uplift" value={`${s.uplift >= 0 ? "+" : ""}${fmt.num(s.uplift)}`} tone={llm ? "text-ok" : "text-warn"} />
        <Metric label="branch taken" value={<span className="text-base">{s.branch ?? s.decision}{s.mode && s.mode !== "auto" ? ` (mode=${s.mode})` : ""}</span>} />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ calibrate */
function Calibrate({ s }: { s: S }) {
  const scorers = Object.keys(s).filter((k) => SCORER_LABEL[k] && s[k]?.test_half);
  const [pick, setPick] = useState(scorers[scorers.length - 1] ?? "");
  const cur = s[pick];
  const sweep = useMemo(() => {
    const t = cur?.thresholds ?? [];
    return {
      ...baseOption(), grid: { left: 48, right: 56, top: 34, bottom: 36 },
      xAxis: { type: "value", name: "threshold", nameLocation: "middle", nameGap: 24, min: 0, max: 1, ...axisStyle },
      yAxis: [{ type: "value", min: 0, max: 1, ...axisStyle, axisLabel: { ...axisStyle.axisLabel, formatter: (v: number) => `${v * 100}%` } },
              { type: "log", name: "flagged / day", ...axisStyle, splitLine: { show: false } }],
      series: [
        { name: "precision", type: "line", smooth: true, data: t.map((r: any) => [r.threshold, r.precision]), color: PALETTE[2] },
        { name: "recall", type: "line", smooth: true, data: t.map((r: any) => [r.threshold, r.recall]), color: PALETTE[0] },
        { name: "flagged / day", type: "bar", yAxisIndex: 1, barWidth: 6, data: t.map((r: any) => [r.threshold, Math.max(1, r.flagged_per_day)]), color: "rgba(139,92,246,0.45)" },
      ],
    };
  }, [cur]);
  if (!scorers.length) return <EmptyState title="No calibrated scorers" />;
  return (
    <div className="space-y-4">
      <Card><CardHeader title="Scorers on the later half of validation (calibrated)" subtitle="Top 12,000 of 4.2M daily transactions sent to REVIEW (OBJ-09); policy thresholds 0.60 REVIEW / 0.85 BLOCK" />
        <CardBody>
          <Table head={["scorer", "PR-AUC", "ROC-AUC", "calibrator (ECE)", "review capacity: recall / precision / ₹ recall", "REVIEW ≥0.60", "BLOCK ≥0.85"]}
            rows={scorers.map((k) => {
              const r = s[k], cap = r.at_review_capacity;
              return [<span className="font-medium">{SCORER_LABEL[k]}</span>, fmt.num(r.test_half.pr_auc), fmt.num(r.test_half.roc_auc),
                `${r.chosen_calibrator} (${fmt.num(r.ece?.[r.chosen_calibrator], 4)})`,
                cap ? `${fmt.pct(cap.recall)} / ${fmt.pct(cap.precision)} / ${fmt.pct(cap.rupee_recall)}` : "–",
                `${fmt.pct(r.at_review?.recall)} / ${fmt.pct(r.at_review?.precision)}`, `${fmt.pct(r.at_block?.recall)} / ${fmt.pct(r.at_block?.precision)}`];
            })} />
        </CardBody>
      </Card>
      <div className="grid gap-4 xl:grid-cols-[1fr_420px]">
        <Card><CardHeader title="Threshold sweep 0.05 – 0.95 (TR-ML-04)" actions={
          <NativeSelect value={pick} onChange={(e) => setPick(e.target.value)} className="h-8 w-48 text-xs">{scorers.map((k) => <option key={k} value={k}>{SCORER_LABEL[k]}</option>)}</NativeSelect>} />
          <CardBody><EChart option={sweep} height={280} /></CardBody></Card>
        <Card><CardHeader title="Fairness at REVIEW (TR-ML-05)" subtitle="Miss rate and false-positive rate per group; gaps > 20% flagged" />
          <CardBody className="space-y-4">
            {(["age_band", "region"] as const).map((g) => (
              <Table key={g} head={[g.replace("_", " "), "n", "fraud", "miss rate", "FPR", "gap"]}
                rows={Object.entries(cur?.fairness_at_review?.[g] ?? {}).map(([k, v]: any) =>
                  [k, fmt.int(v.n), fmt.int(v.fraud), fmt.pct(v.miss_rate), fmt.pct(v.false_positive_rate, 2),
                   v.gap_over_20pct == null ? "–" : v.gap_over_20pct ? <Badge tone="warn">gap</Badge> : <Badge tone="ok">ok</Badge>])} />
            ))}
          </CardBody>
        </Card>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ explainers */
function ExplainerSamples({ run, file }: { run: string; file: string }) {
  const { data } = useRunFile(run, file);
  const samples = useMemo(() => { try { return JSON.parse(data ?? "[]"); } catch { return []; } }, [data]);
  if (!samples.length) return null;
  return (
    <div className="space-y-2">
      {samples.slice(0, 4).map((x: any, i: number) => (
        <div key={i} className="rounded-xl border border-line bg-black/20 p-3 text-xs">
          <div className="mb-1 flex flex-wrap gap-1">
            {(x.target?.reason_codes ?? []).map((c: string) => <Badge key={c} tone="neutral">gold: {REASONS[c]?.label ?? c}</Badge>)}
            {x.source && <Badge tone={x.source === "model" ? "ok" : "warn"}>{x.source}</Badge>}
          </div>
          <div className="font-mono text-[11px] text-ink/80 break-words">{x.generated}</div>
        </div>
      ))}
    </div>
  );
}

function Explainers({ sections, run }: { sections: S; run: string }) {
  const rows = [
    ["Teacher after SFT", sections.sft_explainer?.validation, "explainer/sft_samples.json"],
    [`Teacher after DPO${sections.dpo?.kept ? ` (kept: ${sections.dpo.kept})` : ""}`, sections.dpo?.validation, "explainer/dpo_samples.json"],
    ["Student (bf16)", sections.distill_explainer?.validation, "explainer_student/samples.json"],
    ["Student (AWQ 4-bit)", sections.quantize?.explainer_awq?.validation, "explainer_student/awq_samples.json"],
  ].filter(([, v]) => v) as [string, S, string][];
  const [sel, setSel] = useState(0);
  if (!rows.length) return <EmptyState title="No explainer results in this run" />;
  return (
    <div className="grid gap-4 xl:grid-cols-[1fr_440px]">
      <Card><CardHeader title="Explainer quality on held-out gold cases" subtitle="Valid JSON (OBJ-06), schema, reason-code F1, guard acceptance (TR-LLM-07), answer diversity" />
        <CardBody>
          <Table head={["explainer", "valid JSON", "valid schema", "reason F1", "guard accepted", "distinct answers"]}
            rows={rows.map(([label, v], i) => [
              <button className={cn("text-left font-medium hover:text-brand", sel === i && "text-brand")} onClick={() => setSel(i)}>{label}</button>,
              fmt.pct(v.json_valid_rate), fmt.pct(v.schema_valid_rate), fmt.num(v.reason_f1), fmt.pct(v.guard_accepted_rate),
              v.distinct_code_sets != null ? `${v.distinct_code_sets} of ${v.gold_distinct_code_sets}` : "–"])} />
        </CardBody>
      </Card>
      <Card><CardHeader title="Sample answers" subtitle={rows[sel]?.[0]} /><CardBody><ExplainerSamples run={run} file={rows[sel]?.[2]} /></CardBody></Card>
    </div>
  );
}

/* ------------------------------------------------------------------ quantise */
function Quantize({ s }: { s: S }) {
  const bars = [
    s.scorer_fp8 && ["Scorer student", s.scorer_fp8.bf16.size_mb, s.scorer_fp8.fp8.size_mb, "FP8"],
    s.explainer_awq && ["Explainer student", s.explainer_awq.bf16_size_mb, s.explainer_awq.awq_size_mb, "AWQ 4-bit"],
  ].filter(Boolean) as [string, number, number, string][];
  const option = useMemo(() => ({
    ...baseOption(), grid: { left: 130, right: 60, top: 30, bottom: 24 },
    xAxis: { type: "value", name: "MB", ...axisStyle }, yAxis: { type: "category", data: bars.map((b) => b[0]), ...axisStyle },
    series: [
      { name: "bf16", type: "bar", data: bars.map((b) => b[1]), color: "rgba(148,163,184,0.45)", barGap: "20%", label: { show: true, position: "right", color: "#8a96ad", formatter: "{c} MB" } },
      { name: "quantised", type: "bar", data: bars.map((b) => b[2]), color: PALETTE[0], label: { show: true, position: "right", color: "#e6ebf5", formatter: "{c} MB" } },
    ],
  }), [s]);
  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_360px]">
      <Card><CardHeader title="Model size before / after quantisation" subtitle="llm-compressor, compressed-tensors format (vLLM-servable)" /><CardBody><EChart option={option} height={200} /></CardBody></Card>
      <div className="space-y-3">
        {s.scorer_fp8 && <Metric label="scorer PR-AUC bf16 → FP8" value={`${fmt.num(s.scorer_fp8.bf16.validation.pr_auc)} → ${fmt.num(s.scorer_fp8.fp8.validation.pr_auc)}`} tone="text-brand" />}
        {s.explainer_awq && <Metric label="AWQ explainer valid schema" value={fmt.pct(s.explainer_awq.validation.schema_valid_rate)} tone="text-ok" />}
        {s.explainer_awq && <Metric label="AWQ group size" value={s.explainer_awq.group_size} />}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ acceptance (holdout) */
function Acceptance({ s }: { s: S }) {
  const scorers = Object.keys(s.scorers ?? {});
  const curveOption = (kind: "pr" | "roc") => ({
    ...baseOption(), grid: { left: 48, right: 16, top: 30, bottom: 40 },
    xAxis: { type: "value", min: 0, max: 1, name: kind === "pr" ? "recall" : "false-positive rate", nameLocation: "middle", nameGap: 26, ...axisStyle },
    yAxis: { type: "value", min: 0, max: 1, name: kind === "pr" ? "precision" : "recall", ...axisStyle },
    series: scorers.map((k, i) => {
      const c = s.scorers[k]?.curves?.[kind];
      const pts = c ? (kind === "pr" ? c.recall.map((x: number, j: number) => [x, c.precision[j]]) : c.fpr.map((x: number, j: number) => [x, c.tpr[j]])) : [];
      return { name: SCORER_LABEL[k] ?? k, type: "line", showSymbol: false, step: kind === "pr" ? "end" : undefined, data: pts, color: PALETTE[i] };
    }),
  });
  const pr = useMemo(() => curveOption("pr"), [s]); // eslint-disable-line react-hooks/exhaustive-deps
  const roc = useMemo(() => curveOption("roc"), [s]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!s.objectives) return <EmptyState title="No acceptance results" />;
  const go = s.verdict === "GO";
  const served = s.scorers?.[s.served];
  const thr = (t: S | null | undefined) => t ? `${fmt.pct(t.recall)} / ${fmt.pct(t.precision)} / ${fmt.pct(t.false_positive_rate, 2)} / ${fmt.int(t.flagged_per_day)}` : "–";
  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
        <Card><CardHeader title="CRD success criteria on the holdout" subtitle={`${SCORER_LABEL[s.served] ?? s.served} · ${fmt.int(s.rows)} rows (${fmt.int(s.fraud)} fraud) · frozen stage-8 calibrator · policy REVIEW 0.60 / BLOCK 0.85`} />
          <CardBody>
            <Table head={["objective", "measure", "value", "target", "result"]} rows={(s.objectives as S[]).map((c) => [
              <span className="font-medium">{c.id}</span>, <span title={c.note}>{c.name}</span>,
              c.value == null ? "–" : typeof c.value === "number" && !Number.isInteger(c.value) ? fmt.num(c.value, 4) : fmt.int(c.value),
              c.target, c.pass == null ? <Badge tone="neutral">n/a</Badge> : c.pass ? <Badge tone="ok">pass</Badge> : <Badge tone="bad">fail</Badge>])} />
          </CardBody>
        </Card>
        <div className="space-y-3">
          <Metric label="verdict" value={<Badge tone={go ? "ok" : "bad"}>{s.verdict}</Badge>} />
          <Metric label="failing" value={<span className="text-sm">{s.failing?.length ? s.failing.join(", ") : "none"}</span>} tone={go ? "text-ok" : "text-warn"} />
          <Metric label={`latency p50 / p99 (${s.latency?.scorer === "xgboost" ? "XGBoost" : "FP8 LLM"})`} value={`${fmt.num(s.latency?.p50_ms, 1)} / ${fmt.num(s.latency?.p99_ms, 1)} ms`} />
          <div className="text-[11px] text-muted">{s.latency?.where}</div>
        </div>
      </div>
      <Card><CardHeader title="Thresholds: policy vs recommended" subtitle="Recommended cut-offs are chosen on the later half of validation for the recall / precision targets, then measured here on the holdout. Apply them only through Rules & policy (two-person approval)." />
        <CardBody>
          <Table head={["scorer", "PR-AUC", "ECE", "policy REVIEW: recall / precision / FPR / per day", "policy BLOCK", "recommended cut-offs", "recommended REVIEW", "recommended BLOCK"]}
            rows={scorers.map((k) => {
              const r = s.scorers[k], rec = r.recommended ?? {};
              return [<span className="font-medium">{SCORER_LABEL[k] ?? k}{k === s.served && <Badge tone="brand" className="ml-2">served</Badge>}</span>,
                fmt.num(r.holdout?.pr_auc), fmt.num(r.ece, 4), thr(r.policy?.at_review), thr(r.policy?.at_block),
                rec.review_threshold == null ? "–" : `${fmt.num(rec.review_threshold, 4)} / ${fmt.num(rec.block_threshold, 4)}`,
                thr(rec.at_review), thr(rec.at_block)];
            })} />
          {served?.reason_f1_per_code && <div className="mt-3"><KeyValue data={Object.fromEntries(Object.entries(served.reason_f1_per_code).map(([c, v]) => [`F1 ${REASONS[c]?.label ?? c}`, fmt.num(v as number)]))} /></div>}
        </CardBody>
      </Card>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card><CardHeader title="Precision-recall on the holdout" /><CardBody><EChart option={pr} height={260} /></CardBody></Card>
        <Card><CardHeader title="ROC on the holdout" /><CardBody><EChart option={roc} height={260} /></CardBody></Card>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ generic training stage */
function Generic({ s }: { s: S }) {
  const flat = Object.fromEntries(Object.entries(s).flatMap(([k, v]) =>
    v && typeof v === "object" && !Array.isArray(v) ? Object.entries(v).filter(([, x]) => typeof x !== "object").map(([kk, x]) => [`${k}.${kk}`, x])
      : typeof v !== "object" ? [[k, v]] : []).filter(([k]) => !["status", "error"].includes(k)));
  return <Card><CardBody className="pt-5"><KeyValue data={flat as Record<string, unknown>} /></CardBody></Card>;
}

export const SECTION_RENDERERS: Record<string, (p: { s: S; sections: S; run: string }) => ReactNode> = {
  data_checks: ({ s }) => <DataChecks s={s} />,
  xgboost: ({ s }) => <Xgboost s={s} />,
  pilot: ({ s }) => <Pilot s={s} />,
  gate: ({ s }) => <Gate s={s} />,
  calibrate: ({ s }) => <Calibrate s={s} />,
  acceptance: ({ s }) => <Acceptance s={s} />,
  quantize: ({ s }) => <Quantize s={s} />,
  sft_explainer: ({ sections, run }) => <Explainers sections={sections} run={run} />,
  dpo: ({ sections, run }) => <Explainers sections={sections} run={run} />,
  distill_explainer: ({ sections, run }) => <Explainers sections={sections} run={run} />,
};

export const renderSection = (stage: string, sections: S, run: string) =>
  (SECTION_RENDERERS[stage] ?? (({ s }) => <Generic s={s} />))({ s: sections[stage] ?? {}, sections, run });

export { Acceptance, Calibrate, Explainers, Gate, Pilot, Quantize };
