import { motion } from "motion/react";
import { ArrowLeft, Cpu, FileText, Gauge, RotateCcw, Square, Timer } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { STAGE_BY_ID, stageLabel } from "@/shared/lib/stages";
import { fmt } from "@/shared/lib/utils";
import { Badge, Button, Card, CardBody, CardHeader, EmptyState, ErrorState, PageHeader, Progress, Skeleton, StatusBadge } from "@/shared/ui/core";
import { CodeView, JsonView, MarkdownView } from "@/shared/ui/content";
import { Tabs } from "@/shared/ui/overlays";
import { useArtifacts, useCancelJob, useReportMd, useRun, useRunLive, useSeries, useTailEvents } from "../api";
import { EventConsole, JobLogConsole } from "../components/Consoles";
import { LadderTimeline } from "../components/LadderTimeline";
import { LossChart } from "../components/LossChart";
import { renderSection } from "../components/sections";

const TRAINING_STAGES = ["pilot", "cpt", "sft_scorer", "sft_explainer", "dpo", "distill_scorer", "distill_explainer"];

function LiveStrip({ stage, progress, loss, step }: { stage: string; progress?: { label: string; done: number; total: number } | null; loss?: number | null; step?: string }) {
  return (
    <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }}
      className="mb-5 overflow-hidden rounded-2xl border border-warn/30 bg-gradient-to-r from-warn/[0.08] via-transparent to-brand/[0.06] p-4">
      <div className="flex flex-wrap items-center gap-6">
        <div className="flex items-center gap-3">
          <span className="relative flex h-3 w-3"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-warn opacity-60" /><span className="relative inline-flex h-3 w-3 rounded-full bg-warn" /></span>
          <div><div className="text-[10px] uppercase tracking-wider text-muted">now running</div><div className="font-semibold">{STAGE_BY_ID[stage]?.step} · {stageLabel(stage)}</div></div>
        </div>
        {progress && <div className="min-w-[240px] flex-1"><div className="mb-1 flex justify-between text-[11px] text-muted"><span>{progress.label}</span><span className="tabular-nums">{progress.done}/{progress.total}</span></div><Progress value={progress.total ? progress.done / progress.total : 0} tone="warn" /></div>}
        {loss != null && <div><div className="text-[10px] uppercase tracking-wider text-muted">loss</div><div className="text-xl font-semibold tabular-nums text-warn">{loss.toFixed(4)}</div><div className="text-[10px] text-muted">{step}</div></div>}
      </div>
    </motion.div>
  );
}

export default function RunDetailPage() {
  const { run = "" } = useParams();
  const nav = useNavigate();
  const { data: series } = useSeries(run);
  const { data, error, isLoading, refetch } = useRun(run, true);
  const live = data?.status === "running" || data?.status === "queued";
  const { feed, progress, state } = useRunLive(run, series?.offset, !!live);
  const tail = useTailEvents(run);
  const report = useReportMd(run, !!data?.has_report_md);
  const artifacts = useArtifacts(run);
  const cancel = useCancelJob();
  const [selected, setSelected] = useState<string>();

  const trained = useMemo(() => TRAINING_STAGES.filter((s) => series?.series?.[s]?.length), [series]);
  const chartStage = selected && series?.series?.[selected]?.length ? selected : data?.current_stage && series?.series?.[data.current_stage]?.length ? data.current_stage : trained[trained.length - 1];
  const events = useMemo(() => [...(tail.data?.events ?? []), ...feed], [tail.data, feed]);
  const resultStages = Object.keys(data?.sections ?? {});

  if (error) return <ErrorState error={error} retry={refetch} />;
  if (isLoading || !data) return <div className="space-y-4"><Skeleton className="h-10 w-80" /><Skeleton className="h-64" /></div>;
  const cur = data.ladder.find((r) => r.status === "running");

  return (
    <>
      <Link to="/runs" className="mb-3 inline-flex items-center gap-1 text-xs text-muted hover:text-ink"><ArrowLeft className="h-3.5 w-3.5" /> All runs</Link>
      <PageHeader eyebrow={data.profile?.replace("configs/", "")} title={<span className="flex items-center gap-3">{data.name}<StatusBadge status={data.status} /></span>}
        subtitle={<span className="flex flex-wrap items-center gap-2">
          {data.gate && <Badge tone={data.gate.decision === "LLM" ? "brand" : "violet"}>gate {data.gate.decision}{data.gate.mode && data.gate.mode !== "auto" ? ` · branch ${data.gate.branch}` : ""}</Badge>}
          {data.scorer && <Badge tone="brand">scorer: {data.scorer}</Badge>}
          {data.pilot_winner && <Badge>teacher {fmt.model(data.pilot_winner.teacher)} · {data.pilot_winner.format}</Badge>}
          {data.reused_from && <Badge tone="violet">reuses {data.reused_from}</Badge>}
          {live && <Badge tone={state === "open" ? "ok" : "warn"} dot>live stream {state}</Badge>}
        </span>}
        actions={<>
          {data.job && ["queued", "running"].includes(data.job.status) && <Button variant="danger" onClick={() => cancel.mutate(data.job!.id)} loading={cancel.isPending}><Square className="h-4 w-4" /> Cancel</Button>}
          <Button onClick={() => nav(`/launch?rerun=${data.name}`)}><RotateCcw className="h-4 w-4" /> Re-run stages</Button>
        </>} />

      {cur && <LiveStrip stage={cur.name} progress={progress[cur.name] ?? cur.progress} loss={cur.last_step?.loss} step={cur.last_step ? `step ${cur.last_step.step}/${cur.last_step.max_steps}` : undefined} />}

      <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-4">
        {[{ icon: <Gauge className="h-4 w-4" />, label: "stages done", value: `${data.stages_done} / ${data.stages_total}` },
          { icon: <Timer className="h-4 w-4" />, label: "total time", value: fmt.minutes(data.total_minutes) },
          { icon: <Cpu className="h-4 w-4" />, label: "peak GPU", value: `${Math.max(0, ...data.ladder.map((r) => r.peak_gpu_gb ?? 0)).toFixed(2)} GB` },
          { icon: <FileText className="h-4 w-4" />, label: "GPU", value: data.gpu ?? "–" },
        ].map((m) => (
          <Card key={m.label} className="flex items-center gap-3 p-4"><span className="text-brand">{m.icon}</span><div className="min-w-0"><div className="text-[10px] uppercase tracking-wider text-muted">{m.label}</div><div className="truncate text-sm font-semibold">{m.value}</div></div></Card>
        ))}
      </div>

      <Tabs tabs={[
        { id: "overview", label: "Overview", content: (
          <div className="grid gap-5 xl:grid-cols-[420px_1fr]">
            <Card><CardHeader title="Model ladder" subtitle="Part 3 Figure 4 · click a stage for its curve and results" />
              <CardBody><LadderTimeline rows={data.ladder} liveProgress={progress} selected={selected} onSelect={setSelected} /></CardBody></Card>
            <div className="space-y-5">
              <Card><CardHeader title={chartStage ? `Training curve · ${stageLabel(chartStage)}` : "Training curves"} subtitle="Loss and learning rate per optimiser step (live while training)" />
                <CardBody><LossChart series={series?.series ?? {}} stages={chartStage ? [chartStage] : []} /></CardBody></Card>
              {selected && data.sections[selected] && (
                <Card><CardHeader title={`${stageLabel(selected)} · results`} subtitle={STAGE_BY_ID[selected]?.what} /><CardBody>{renderSection(selected, data.sections, data.name)}</CardBody></Card>
              )}
            </div>
          </div>) },
        { id: "results", label: "Results", content: resultStages.length ? (
          <Tabs tabs={resultStages.map((s) => ({ id: s, label: STAGE_BY_ID[s]?.short ?? s, content: renderSection(s, data.sections, data.name) }))} />
        ) : <EmptyState title="No stage results yet" /> },
        { id: "curves", label: "All curves", content: (
          <Card><CardBody className="pt-5"><LossChart series={series?.series ?? {}} stages={trained} height={380} showLr={false} /></CardBody></Card>) },
        { id: "logs", label: "Logs", content: data.job?.container_id ? <JobLogConsole jobId={data.job.id} />
          : <EmptyState title="No container logs" hint="This run was started from the command line: see the Events tab, or `docker logs <container>` in a terminal." /> },
        { id: "events", label: "Events", badge: live ? <Badge tone="warn" dot>live</Badge> : undefined, content: <EventConsole events={events} live={!!live && state === "open"} /> },
        { id: "config", label: "Config", content: (
          <div className="grid gap-4 xl:grid-cols-2">
            <Card><CardHeader title="Resolved configuration" subtitle="Profile + extends + overrides, exactly as the pipeline ran" /><CardBody><JsonView data={data.config} maxHeight="560px" /></CardBody></Card>
            <Card><CardHeader title="Overrides" /><CardBody>{data.overrides.length ? <CodeView value={data.overrides.join("\n")} language="text" /> : <div className="text-xs text-muted">none</div>}</CardBody></Card>
          </div>) },
        { id: "artifacts", label: "Artifacts", content: (
          <Card><CardBody className="pt-4">
            <table className="w-full text-xs"><thead><tr className="text-left text-muted"><th className="py-2">artefact</th><th>files</th><th className="text-right">size</th></tr></thead>
              <tbody>{(artifacts.data ?? []).map((a) => <tr key={a.name} className="border-t border-line/60"><td className="py-2 font-mono">{a.name}{a.kind === "dir" && "/"}</td><td>{a.files}</td><td className="text-right tabular-nums">{fmt.mb(a.size_mb)}</td></tr>)}</tbody></table>
          </CardBody></Card>) },
        { id: "report", label: "Report", content: report.data ? <Card><CardBody className="pt-5"><MarkdownView source={report.data} /></CardBody></Card>
          : <EmptyState title="No report.md yet" hint="report.md is written at the end of every run (and by `python -m argus_ft.compare <run>`)." /> },
      ]} />
    </>
  );
}
