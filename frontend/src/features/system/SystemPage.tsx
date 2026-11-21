import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Cpu, Play, Power, Server, Thermometer, Zap } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { api, errorMessage } from "@/shared/api/client";
import { keys, useGpu, useJobs, useServices } from "@/shared/api/queries";
import { axisStyle, baseOption, EChart } from "@/shared/charts/EChart";
import { cn, fmt } from "@/shared/lib/utils";
import { Badge, Button, Card, CardBody, CardHeader, PageHeader, Stat, StatusBadge } from "@/shared/ui/core";

function ServiceControl({ name, label, port, status, container }: { name?: "scoring" | "explainer"; label: string; port: string; status?: string; container?: string }) {
  const qc = useQueryClient();
  const act = useMutation({
    mutationFn: (action: "start" | "stop") => api.post(`/services/${name}/${action}`),
    onSuccess: (_, a) => { toast.success(`${label}: ${a} requested`); setTimeout(() => qc.invalidateQueries({ queryKey: keys.services }), 1500); },
    onError: (e) => toast.error(errorMessage(e)),
  });
  return (
    <Card className="p-5">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3"><div className="grid h-10 w-10 place-items-center rounded-xl bg-white/5 text-brand"><Server className="h-5 w-5" /></div>
          <div><div className="font-semibold">{label}</div><div className="text-xs text-muted">{port}{container && ` · container ${container}`}</div></div></div>
        <StatusBadge status={status === "ok" ? "ok" : status ?? "down"} />
      </div>
      {name && <div className="mt-4 flex gap-2">
        <Button size="sm" onClick={() => act.mutate("start")} loading={act.isPending && act.variables === "start"} disabled={container === "running"}><Play className="h-3.5 w-3.5" /> Start</Button>
        <Button size="sm" variant="danger" onClick={() => act.mutate("stop")} loading={act.isPending && act.variables === "stop"} disabled={container !== "running"}><Power className="h-3.5 w-3.5" /> Stop</Button>
      </div>}
    </Card>
  );
}

export default function SystemPage() {
  const { data: svc } = useServices();
  const { data: gpu } = useGpu(1500);
  const { data: jobs } = useJobs();
  const g = gpu?.gpus?.[0];
  const [hist, setHist] = useState<[number, number | null, number | null][]>([]);
  useEffect(() => { if (g) setHist((h) => [...h.slice(-199), [Date.now(), g.utilization_pct, g.memory_used_mb]]); }, [gpu]); // eslint-disable-line react-hooks/exhaustive-deps
  const option = useMemo(() => ({
    ...baseOption(), grid: { left: 48, right: 56, top: 30, bottom: 28 },
    xAxis: { type: "time", ...axisStyle, splitLine: { show: false } },
    yAxis: [{ type: "value", min: 0, max: 100, name: "util %", ...axisStyle }, { type: "value", min: 0, max: g?.memory_total_mb, name: "MB", ...axisStyle, splitLine: { show: false } }],
    series: [{ name: "utilisation %", type: "line", showSymbol: false, smooth: true, areaStyle: { opacity: 0.15 }, data: hist.map((p) => [p[0], p[1]]) },
             { name: "memory MB", type: "line", yAxisIndex: 1, showSymbol: false, smooth: true, data: hist.map((p) => [p[0], p[2]]) }],
  }), [hist, g?.memory_total_mb]);
  return (
    <>
      <PageHeader eyebrow="Operate" title="System" subtitle="The 4 GB GPU is shared: training and the serving stack (Fraud Model API + vLLM) take turns. Stop serving to train, start it to score." />
      <div className="mb-5 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="GPU utilisation" value={g?.utilization_pct} format={(v) => (v == null ? "–" : `${v.toFixed(0)}%`)} icon={<Zap className="h-4 w-4" />} />
        <Stat label="GPU memory" value={g?.memory_used_mb} format={(v) => fmt.mb(v)} tone="violet" icon={<Cpu className="h-4 w-4" />} hint={g ? `of ${fmt.mb(g.memory_total_mb)}` : undefined} />
        <Stat label="temperature" value={g?.temperature_c} format={(v) => (v == null ? "–" : `${v.toFixed(0)}°C`)} tone="warn" icon={<Thermometer className="h-4 w-4" />} />
        <Stat label="active jobs" value={jobs?.filter((j) => ["queued", "running"].includes(j.status)).length ?? 0} format={(v) => fmt.int(v)} tone="ok" />
      </div>
      <div className="grid gap-5 xl:grid-cols-[1fr_1.4fr]">
        <div className="grid gap-4">
          <ServiceControl label="Platform API" port=":8090" status="ok" />
          <ServiceControl name="scoring" label="Fraud Model API" port=":8080" status={svc?.scoring} container={svc?.containers?.scoring} />
          <ServiceControl name="explainer" label="vLLM explainer" port=":8000" status={svc?.explainer} container={svc?.containers?.explainer} />
          <ServiceControl label="MLflow" port=":5000" status={svc?.mlflow} />
        </div>
        <div className="space-y-5">
          <Card><CardHeader title={g?.name ?? "GPU"} subtitle="Live NVML telemetry (1.5 s)" icon={<Cpu className="h-4 w-4" />} /><CardBody><EChart option={option} height={260} /></CardBody></Card>
          <Card><CardHeader title="Jobs" /><CardBody className="space-y-2">
            {(jobs ?? []).slice(0, 6).map((j) => (
              <div key={j.id} className={cn("flex items-center justify-between rounded-lg bg-white/[0.02] px-3 py-2 text-xs")}>
                <span className="font-medium">{j.run_name}</span><span className="text-muted">{j.profile}</span><StatusBadge status={j.status} /><span className="text-muted">{fmt.ago(j.created_at)}</span>
              </div>))}
            {!jobs?.length && <div className="text-xs text-muted">No jobs yet.</div>}
            <div className="pt-1 text-[11px] text-muted">Docker: <Badge tone={svc?.docker === "ok" ? "ok" : "bad"}>{svc?.docker === "ok" ? "connected" : "unavailable"}</Badge></div>
          </CardBody></Card>
        </div>
      </div>
    </>
  );
}
