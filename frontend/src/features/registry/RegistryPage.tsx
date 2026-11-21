import { useQuery } from "@tanstack/react-query";
import { Box, ExternalLink, Tag } from "lucide-react";
import { api } from "@/shared/api/client";
import { keys, useServices } from "@/shared/api/queries";
import { fmt } from "@/shared/lib/utils";
import { Badge, Button, Card, CardBody, CardHeader, EmptyState, ErrorState, PageHeader, Skeleton } from "@/shared/ui/core";
import { Tabs } from "@/shared/ui/overlays";

interface Model { name: string; created: number; updated: number; versions: { version: string; run_id: string; source: string; status: string; created: number }[] }
interface MlRun { run_id: string; run_name: string; status: string; start_time: number; end_time: number; tags: Record<string, string>; key_metrics: Record<string, number>; metric_count: number }

function Models() {
  const { data, error, isLoading } = useQuery({ queryKey: keys.registryModels, queryFn: () => api.get<Model[]>("/registry/models"), refetchInterval: 15000 });
  if (error) return <ErrorState error={error} />;
  if (isLoading) return <Skeleton className="h-40" />;
  if (!data?.length) return <EmptyState title="No registered models yet" hint="The registry stage of every run registers argus-scorer and argus-explainer." />;
  return (
    <div className="grid gap-5 lg:grid-cols-2">
      {data.map((m) => (
        <Card key={m.name}>
          <CardHeader title={m.name} subtitle={m.name === "argus-scorer" ? "FP8 scorer student, or XGBoost on HYBRID runs" : "AWQ 4-bit explainer student"} icon={<Box className="h-4 w-4" />}
            actions={<Badge tone="brand">{m.versions.length} versions</Badge>} />
          <CardBody className="space-y-2">
            {m.versions.slice(0, 8).map((v) => (
              <div key={v.version} className="flex items-center gap-3 rounded-xl border border-line bg-white/[0.02] px-3 py-2 text-xs">
                <span className="grid h-7 w-9 place-items-center rounded-md bg-brand/10 font-mono text-brand">v{v.version}</span>
                <span className="min-w-0 flex-1 truncate font-mono text-muted" title={v.source}>{v.source?.split("/artifacts/").pop()}</span>
                <Badge tone={v.status === "candidate" ? "warn" : "ok"}><Tag className="h-3 w-3" />{v.status}</Badge>
                <span className="text-muted">{fmt.ago(v.created)}</span>
              </div>
            ))}
          </CardBody>
        </Card>
      ))}
    </div>
  );
}

function Runs() {
  const { data, error, isLoading } = useQuery({ queryKey: keys.registryRuns, queryFn: () => api.get<MlRun[]>("/registry/runs"), refetchInterval: 15000 });
  if (error) return <ErrorState error={error} />;
  if (isLoading) return <Skeleton className="h-40" />;
  if (!data?.length) return <EmptyState title="No MLflow runs yet" />;
  return (
    <Card><div className="overflow-x-auto"><table className="w-full text-xs">
      <thead><tr className="text-left text-muted">{["run", "teacher", "scorer", "gate", "prompt format", "metrics", "logged"].map((h) => <th key={h} className="px-4 py-2.5 font-medium">{h}</th>)}</tr></thead>
      <tbody>{data.map((r) => (
        <tr key={r.run_id} className="border-t border-line/60">
          <td className="px-4 py-2.5 font-medium">{r.run_name}</td>
          <td className="px-4 py-2.5">{fmt.model(r.tags.teacher)}</td>
          <td className="px-4 py-2.5"><Badge tone={r.tags.scorer?.startsWith("LLM") ? "brand" : "violet"}>{r.tags.scorer ?? "–"}</Badge></td>
          <td className="px-4 py-2.5">{r.tags.gate_decision}</td>
          <td className="px-4 py-2.5">{r.tags.prompt_format}</td>
          <td className="px-4 py-2.5">{r.metric_count}</td>
          <td className="px-4 py-2.5 text-muted">{fmt.ago(r.start_time)}</td>
        </tr>))}</tbody></table></div></Card>
  );
}

export default function RegistryPage() {
  const { data: svc } = useServices();
  const url = svc?.mlflow_url ?? "http://localhost:5000";
  return (
    <>
      <PageHeader eyebrow="Models" title="Model registry" subtitle="MLflow tracking and registry (FR-ML-08, TR-DEP-03): every run's data snapshot, prompt format, config, metrics and serving models."
        actions={<Button asChild><a href={url} target="_blank" rel="noreferrer"><ExternalLink className="h-4 w-4" /> Open MLflow</a></Button>} />
      <Tabs tabs={[
        { id: "models", label: "Registered models", content: <Models /> },
        { id: "runs", label: "Tracked runs", content: <Runs /> },
        { id: "mlflow", label: "MLflow UI", content: svc?.mlflow === "ok"
          ? <Card className="overflow-hidden"><iframe title="MLflow" src={url} className="h-[78vh] w-full bg-white" /></Card>
          : <EmptyState title="MLflow is not running" hint="Start it with `docker compose up -d mlflow`." /> },
      ]} />
    </>
  );
}
