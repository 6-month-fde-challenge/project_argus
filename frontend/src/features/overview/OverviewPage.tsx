import { motion } from "motion/react";
import { Activity, ArrowRight, Boxes, Cpu, FlaskConical, Rocket, ShieldCheck, Sparkles, Trophy } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";
import { useGpu, useRuns, useServices } from "@/shared/api/queries";
import type { GpuSnapshot } from "@/shared/api/types";
import { axisStyle, baseOption, EChart } from "@/shared/charts/EChart";
import { STAGE_BY_ID, stageLabel } from "@/shared/lib/stages";
import { cn, fmt, ratio } from "@/shared/lib/utils";
import { Badge, Button, Card, CardBody, CardHeader, EmptyState, PageHeader, Progress, Stat, StatusBadge } from "@/shared/ui/core";
import { RunRow } from "@/features/runs/pages/RunsPage";

/** Rolling window of GPU samples (fixed-size ring buffer: O(1) per sample, bounded memory). */
function useGpuHistory(snap: GpuSnapshot | undefined, size = 90) {
  const [hist, setHist] = useState<{ t: number; util: number | null; mem: number | null }[]>([]);
  useEffect(() => {
    const g = snap?.gpus?.[0];
    if (!g) return;
    setHist((h) => [...h.slice(-(size - 1)), { t: Date.now(), util: g.utilization_pct, mem: ratio(g.memory_used_mb, g.memory_total_mb, 100) }]);
  }, [snap, size]);
  return hist;
}

function GpuChart() {
  const { data } = useGpu(2000);
  const hist = useGpuHistory(data);
  const g = data?.gpus?.[0];
  const option = useMemo(() => ({
    ...baseOption(), grid: { left: 40, right: 16, top: 30, bottom: 24 },
    xAxis: { type: "time", ...axisStyle, splitLine: { show: false }, axisLabel: { ...axisStyle.axisLabel, formatter: "{HH}:{mm}:{ss}" } },
    yAxis: { type: "value", min: 0, max: 100, ...axisStyle, axisLabel: { ...axisStyle.axisLabel, formatter: "{value}%" } },
    series: [
      { name: "utilisation", type: "line", smooth: true, showSymbol: false, areaStyle: { opacity: 0.15 }, data: hist.map((p) => [p.t, p.util]) },
      { name: "memory", type: "line", smooth: true, showSymbol: false, areaStyle: { opacity: 0.1 }, data: hist.map((p) => [p.t, p.mem]) },
    ],
  }), [hist]);
  return (
    <Card><CardHeader title="GPU telemetry" subtitle={g ? `${g.name} · ${g.memory_used_mb} / ${g.memory_total_mb} MB · ${g.temperature_c}°C${g.power_w ? ` · ${g.power_w} W` : ""}` : "NVML not available"} icon={<Cpu className="h-4 w-4" />} />
      <CardBody>{data?.available ? <EChart option={option} height={200} /> : <EmptyState title="No GPU telemetry" hint="The platform container needs GPU access for NVML (docker compose gives it)." />}</CardBody></Card>
  );
}

function LiveRunCard() {
  const { data } = useRuns();
  const live = data?.find((r) => r.status === "running");
  if (!live) {
    return (
      <Card className="flex h-full flex-col justify-center p-6">
        <div className="flex items-center gap-3 text-muted"><Activity className="h-5 w-5" /><span className="text-sm">No training running right now</span></div>
        <Button asChild variant="primary" className="mt-4 w-fit"><Link to="/launch"><Rocket className="h-4 w-4" /> Launch a run</Link></Button>
      </Card>
    );
  }
  const p = live.progress;
  return (
    <Card className="relative h-full overflow-hidden p-6">
      <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-warn/10 blur-3xl" />
      <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-warn"><span className="h-2 w-2 animate-ping rounded-full bg-warn" /> now training</div>
      <Link to={`/runs/${live.name}`} className="mt-2 block text-xl font-semibold hover:text-brand">{live.name}</Link>
      <div className="mt-1 text-sm text-muted">{live.current_stage ? `${STAGE_BY_ID[live.current_stage]?.step} · ${stageLabel(live.current_stage)}` : "preparing…"}</div>
      <div className="mt-4 space-y-3">
        <div><div className="mb-1 flex justify-between text-[11px] text-muted"><span>ladder</span><span>{live.stages_done}/{live.stages_total}</span></div><Progress value={live.stages_done / live.stages_total} /></div>
        {p && <div><div className="mb-1 flex justify-between text-[11px] text-muted"><span>{p.label}</span><span>{p.done}/{p.total}</span></div><Progress value={p.total ? p.done / p.total : 0} tone="warn" /></div>}
        {live.last_step?.loss != null && <div className="text-sm">loss <b className="tabular-nums text-warn">{live.last_step.loss.toFixed(4)}</b> <span className="text-xs text-muted">at step {live.last_step.step}/{live.last_step.max_steps}</span></div>}
      </div>
    </Card>
  );
}

function ServiceGrid() {
  const { data } = useServices();
  const items = [
    { name: "Platform API", status: data ? "ok" : "down", detail: ":8090" },
    { name: "Fraud Model API", status: data?.scoring, detail: ":8080 scoring" },
    { name: "vLLM explainer", status: data?.explainer, detail: ":8000 AWQ" },
    { name: "MLflow", status: data?.mlflow, detail: ":5000 registry" },
    { name: "Docker", status: data?.docker === "ok" ? "ok" : "down", detail: "job runner" },
  ];
  return (
    <Card><CardHeader title="Services" icon={<ShieldCheck className="h-4 w-4" />} actions={<Button asChild size="sm" variant="ghost"><Link to="/system">System <ArrowRight className="h-3.5 w-3.5" /></Link></Button>} />
      <CardBody className="grid gap-2">
        {items.map((s) => (
          <div key={s.name} className="flex items-center justify-between rounded-lg bg-white/[0.02] px-3 py-2 text-sm">
            <span className="flex items-center gap-2"><span className={cn("h-2 w-2 rounded-full", s.status === "ok" ? "bg-ok" : "bg-bad")} />{s.name}</span>
            <span className="text-xs text-muted">{s.detail}</span>
          </div>
        ))}
      </CardBody></Card>
  );
}

export default function OverviewPage() {
  const { data: runs } = useRuns();
  const { data: gpu } = useGpu(3000);
  const best = useMemo(() => {
    const gated = (runs ?? []).filter((r) => r.gate);
    return gated.sort((a, b) => (b.gate!.llm_pr_auc ?? 0) - (a.gate!.llm_pr_auc ?? 0))[0]?.gate;
  }, [runs]);
  const g = gpu?.gpus?.[0];
  return (
    <>
      <PageHeader eyebrow="Mission control" title={<>Project <span className="text-gradient">ARGUS</span></>}
        subtitle="VKPayments' own fraud-intelligence LLM: train the ladder from Part 3, compare the hybrid and LLM-scorer designs, serve decisions through the Fraud Model API."
        actions={<Button asChild variant="primary"><Link to="/pipeline"><Sparkles className="h-4 w-4" /> Explore the pipeline</Link></Button>} />
      <div className="mb-5 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="training runs" value={runs?.length ?? 0} format={(v) => fmt.int(v)} icon={<FlaskConical className="h-4 w-4" />} hint={`${runs?.filter((r) => r.status === "succeeded").length ?? 0} completed`} />
        <Stat label="best LLM pilot PR-AUC" value={best?.llm_pr_auc} format={(v) => fmt.num(v)} tone="brand" icon={<Trophy className="h-4 w-4" />} hint={best ? `XGBoost ${fmt.num(best.xgboost_pr_auc)} on the same rows` : "no gated run yet"} />
        <Stat label="XGBoost baseline" value={best?.xgboost_pr_auc} format={(v) => fmt.num(v)} tone="violet" icon={<Boxes className="h-4 w-4" />} hint="the bar to beat (+0.03)" />
        <Stat label="GPU memory" value={ratio(g?.memory_used_mb, g?.memory_total_mb, 100)} format={(v) => (v == null ? "–" : `${v.toFixed(0)}%`)} tone="warn" icon={<Cpu className="h-4 w-4" />} hint={g ? `${g.memory_used_mb} of ${g.memory_total_mb} MB` : "–"} />
      </div>
      <div className="mb-5 grid gap-5 xl:grid-cols-[1fr_1.4fr_1fr]">
        <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}><LiveRunCard /></motion.div>
        <GpuChart />
        <ServiceGrid />
      </div>
      <Card>
        <CardHeader title="Recent runs" actions={<Button asChild size="sm" variant="ghost"><Link to="/runs">All runs <ArrowRight className="h-3.5 w-3.5" /></Link></Button>} />
        {runs?.length ? (
          <div className="overflow-x-auto"><table className="w-full text-sm"><tbody>{runs.slice(0, 6).map((r, i) => <RunRow key={r.name} r={r} i={i} />)}</tbody></table></div>
        ) : <EmptyState title="No runs yet" action={<Button asChild variant="primary"><Link to="/launch">Launch the smoke profile</Link></Button>} />}
      </Card>
      <div className="mt-3 flex justify-end gap-2 text-[11px] text-muted"><StatusBadge status="LLM" /> LLM scorer path <StatusBadge status="HYBRID" /> XGBoost scores, LLM explains <Badge>Figure 4</Badge></div>
    </>
  );
}
