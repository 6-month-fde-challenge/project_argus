import { useQuery } from "@tanstack/react-query";
import { GitCompare } from "lucide-react";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import { api } from "@/shared/api/client";
import { keys, useRuns } from "@/shared/api/queries";
import { axisStyle, baseOption, EChart, PALETTE } from "@/shared/charts/EChart";
import { stageLabel } from "@/shared/lib/stages";
import { cn, fmt } from "@/shared/lib/utils";
import { Badge, Card, CardBody, CardHeader, EmptyState, ErrorState, PageHeader, Skeleton, StatusBadge } from "@/shared/ui/core";

const SCORERS = [["xgboost", "XGBoost"], ["teacher", "LLM teacher"], ["scorer_student", "LLM student"], ["scorer_student_fp8", "LLM student FP8"]] as const;
const EXPL = [["sft_explainer", "SFT"], ["dpo", "DPO"], ["distill_explainer", "Student"]] as const;

export default function ComparePage() {
  const { data: runs } = useRuns();
  const [params, setParams] = useSearchParams();
  const initial = params.get("runs")?.split(",").filter(Boolean) ?? [];
  const [picked, setPicked] = useState<string[]>(initial);
  const chosen = picked.length ? picked : (runs ?? []).filter((r) => r.status === "succeeded").slice(0, 2).map((r) => r.name);
  const { data, error, isLoading } = useQuery({ queryKey: keys.compare(chosen), queryFn: () => api.get<any[]>(`/compare?runs=${chosen.join(",")}`), enabled: chosen.length > 0 });

  const toggle = (name: string) => {
    const next = chosen.includes(name) ? chosen.filter((n) => n !== name) : [...chosen, name].slice(-4);
    setPicked(next); setParams({ runs: next.join(",") });
  };

  const prOption = useMemo(() => !data ? null : ({
    ...baseOption(), grid: { left: 48, right: 16, top: 34, bottom: 30 },
    xAxis: { type: "category", data: SCORERS.map(([, l]) => l), ...axisStyle },
    yAxis: { type: "value", ...axisStyle, name: "PR-AUC" },
    series: data.map((d, i) => ({ name: d.summary.name, type: "bar", barGap: "15%", color: PALETTE[i],
      itemStyle: { borderRadius: [6, 6, 0, 0] }, data: SCORERS.map(([k]) => d.calibrate?.[k]?.test_half?.pr_auc ?? null) })),
  }), [data]);
  const capOption = useMemo(() => !data ? null : ({
    ...baseOption(), grid: { left: 48, right: 16, top: 34, bottom: 30 },
    xAxis: { type: "category", data: SCORERS.map(([, l]) => l), ...axisStyle },
    yAxis: { type: "value", max: 1, ...axisStyle, axisLabel: { ...axisStyle.axisLabel, formatter: (v: number) => `${v * 100}%` } },
    series: data.flatMap((d, i) => [
      { name: `${d.summary.name} · fraud recall`, type: "bar", color: PALETTE[i], itemStyle: { borderRadius: [6, 6, 0, 0] }, data: SCORERS.map(([k]) => d.calibrate?.[k]?.at_review_capacity?.recall ?? null) },
      { name: `${d.summary.name} · ₹ recall`, type: "line", symbolSize: 8, color: PALETTE[i + 4], data: SCORERS.map(([k]) => d.calibrate?.[k]?.at_review_capacity?.rupee_recall ?? null) },
    ]),
  }), [data]);

  return (
    <>
      <PageHeader eyebrow="Models" title="Compare runs" subtitle="Hybrid vs LLM scorer, or any runs side by side: scorers, recall at review capacity, gate, explainers, quantisation." />
      <Card className="mb-5"><CardBody className="flex flex-wrap gap-2 pt-4">
        {(runs ?? []).map((r) => (
          <button key={r.name} onClick={() => toggle(r.name)} className={cn("flex items-center gap-2 rounded-xl border px-3 py-1.5 text-xs transition",
            chosen.includes(r.name) ? "border-brand/50 bg-brand/10" : "border-line text-muted hover:text-ink")}>
            {r.name}<StatusBadge status={r.scorer ?? r.status} />
          </button>
        ))}
      </CardBody></Card>
      {error ? <ErrorState error={error} /> : isLoading ? <Skeleton className="h-72" /> : !data?.length ? <EmptyState title="Pick runs to compare" icon={<GitCompare className="h-5 w-5" />} /> : (
        <div className="space-y-5">
          <div className={cn("grid gap-4", data.length > 2 ? "lg:grid-cols-4" : "lg:grid-cols-2")}>
            {data.map((d, i) => (
              <Card key={d.summary.name} className="relative overflow-hidden p-5">
                <div className="absolute inset-x-0 top-0 h-1" style={{ background: PALETTE[i] }} />
                <div className="flex items-center justify-between"><span className="font-semibold">{d.summary.name}</span><StatusBadge status={d.summary.status} /></div>
                <div className="mt-3 grid grid-cols-2 gap-y-2 text-xs">
                  <span className="text-muted">production scorer</span><span className="text-right"><StatusBadge status={d.summary.scorer ?? "–"} /></span>
                  <span className="text-muted">gate measured</span><span className="text-right">{d.gate?.decision ?? "–"} ({fmt.num(d.gate?.uplift)})</span>
                  <span className="text-muted">branch</span><span className="text-right">{d.gate?.branch ?? d.gate?.decision ?? "–"}</span>
                  <span className="text-muted">teacher</span><span className="text-right">{fmt.model(d.pilot?.selected?.teacher)} · {d.pilot?.selected?.format}</span>
                  <span className="text-muted">total time</span><span className="text-right">{fmt.minutes(d.summary.total_minutes)}</span>
                </div>
              </Card>
            ))}
          </div>
          <div className="grid gap-5 xl:grid-cols-2">
            <Card><CardHeader title="PR-AUC by scorer" subtitle="Later half of validation, same rows for every scorer" /><CardBody>{prOption && <EChart option={prOption} height={300} />}</CardBody></Card>
            <Card><CardHeader title="Fraud caught at review capacity" subtitle="Top 12,000 of 4.2M transactions a day → REVIEW: share of fraud cases (bars) and fraud rupees (line)" /><CardBody>{capOption && <EChart option={capOption} height={300} />}</CardBody></Card>
          </div>
          <div className="grid gap-5 xl:grid-cols-2">
            <Card><CardHeader title="Explainers" subtitle="Reason-code F1 · guard acceptance · distinct answers" />
              <CardBody><table className="w-full text-xs"><thead><tr className="text-left text-muted"><th className="py-1.5">run</th>{EXPL.map(([, l]) => <th key={l}>{l}</th>)}<th>AWQ</th></tr></thead>
                <tbody>{data.map((d) => (
                  <tr key={d.summary.name} className="border-t border-line/60"><td className="py-2 font-medium">{d.summary.name}</td>
                    {EXPL.map(([k]) => { const v = d.explainers?.[k]; return <td key={k} className="tabular-nums">{v ? `${fmt.num(v.reason_f1)} · ${fmt.pct(v.guard_accepted_rate, 0)} · ${v.distinct_code_sets ?? "–"}` : "–"}</td>; })}
                    <td className="tabular-nums">{d.quantize?.explainer_awq ? `${fmt.num(d.quantize.explainer_awq.validation.reason_f1)} · ${d.quantize.explainer_awq.awq_size_mb} MB` : "–"}</td></tr>))}</tbody></table></CardBody></Card>
            <Card><CardHeader title="Quantisation" subtitle="Serving-format students" />
              <CardBody className="space-y-2 text-xs">{data.map((d) => (
                <div key={d.summary.name} className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-white/[0.02] px-3 py-2">
                  <span className="font-medium">{d.summary.name}</span>
                  <span className="flex flex-wrap gap-2">
                    {d.quantize?.scorer_fp8 ? <Badge tone="brand">scorer FP8 {d.quantize.scorer_fp8.fp8.size_mb} MB · PR-AUC {fmt.num(d.quantize.scorer_fp8.fp8.validation.pr_auc)}</Badge> : <Badge>no LLM scorer (XGBoost)</Badge>}
                    {d.quantize?.explainer_awq && <Badge tone="violet">explainer AWQ {d.quantize.explainer_awq.awq_size_mb} MB</Badge>}
                  </span>
                </div>))}
                <div className="pt-2 text-[11px] text-muted">{stageLabel("quantize")}: FP8 for the scorer, AWQ 4-bit for the explainer (Part 3 Figure 4).</div>
              </CardBody></Card>
          </div>
        </div>
      )}
    </>
  );
}
