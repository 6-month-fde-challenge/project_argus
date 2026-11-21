import { useQuery } from "@tanstack/react-query";
import { Database, ShieldCheck } from "lucide-react";
import { useMemo } from "react";
import { api } from "@/shared/api/client";
import { keys } from "@/shared/api/queries";
import { axisStyle, baseOption, EChart, PALETTE } from "@/shared/charts/EChart";
import { fmt } from "@/shared/lib/utils";
import { Badge, Card, CardBody, CardHeader, EmptyState, ErrorState, PageHeader, Skeleton, Stat } from "@/shared/ui/core";
import { MarkdownView } from "@/shared/ui/content";

interface Overview {
  data_card: string | null; dataset_id: string; version: string; label_snapshot: string; counts: Record<string, number>;
  observed: Record<string, any>; fraud_rate: number; corpus_tokens: Record<string, number>; pii_audit: Record<string, any>; rules: Record<string, any>;
}

export default function DataPage() {
  const { data, error, isLoading } = useQuery({ queryKey: keys.data, queryFn: () => api.get<Overview>("/data/overview") });
  const splits = data?.observed?.splits ?? {};
  const splitOption = useMemo(() => ({
    ...baseOption(), grid: { left: 70, right: 20, top: 30, bottom: 28 },
    xAxis: { type: "category", data: Object.keys(splits), ...axisStyle }, yAxis: [{ type: "value", ...axisStyle }, { type: "value", ...axisStyle, splitLine: { show: false } }],
    series: [{ name: "rows", type: "bar", barWidth: 28, itemStyle: { borderRadius: [6, 6, 0, 0], color: PALETTE[0] }, data: Object.values(splits).map((s: any) => s.rows) },
             { name: "fraud", type: "line", yAxisIndex: 1, symbolSize: 9, color: PALETTE[4], data: Object.values(splits).map((s: any) => s.fraud) }],
  }), [splits]);
  const corpusOption = useMemo(() => {
    const c = Object.entries(data?.corpus_tokens ?? {}).sort((a, b) => b[1] - a[1]);
    return { ...baseOption(), grid: { left: 150, right: 30, top: 10, bottom: 24 },
      xAxis: { type: "value", ...axisStyle }, yAxis: { type: "category", data: c.map(([k]) => k.replaceAll("_", " ")).reverse(), ...axisStyle },
      series: [{ type: "bar", barWidth: 14, itemStyle: { borderRadius: [0, 6, 6, 0], color: PALETTE[1] }, data: c.map(([, v]) => v).reverse() }] };
  }, [data]);
  const channels = data?.observed?.channels ?? {};
  const channelOption = useMemo(() => ({
    ...baseOption(), tooltip: { trigger: "item" },
    series: [{ type: "pie", radius: ["50%", "78%"], itemStyle: { borderColor: "#060a14", borderWidth: 3 }, label: { color: "#e6ebf5", fontSize: 11 },
      data: Object.entries(channels).map(([name, value]) => ({ name, value })) }],
  }), [channels]);

  if (error) return <ErrorState error={error} />;
  if (isLoading || !data) return <Skeleton className="h-96" />;
  return (
    <>
      <PageHeader eyebrow="Knowledge" title="Dataset" subtitle={`Supabase: storage bucket argus-data + Postgres tables · ${data.dataset_id} v${data.version} · labels as of ${data.label_snapshot}`} />
      <div className="mb-5 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="transactions" value={data.observed?.main_transactions} format={(v) => fmt.int(v)} icon={<Database className="h-4 w-4" />} hint={`+${fmt.int(data.observed?.warmup_transactions)} warm-up`} />
        <Stat label="fraud rate" value={data.fraud_rate} format={(v) => fmt.pct(v, 2)} tone="bad" hint={`${fmt.int(data.observed?.fraud)} fraud cases`} />
        <Stat label="gold explanations" value={data.counts?.gold} format={(v) => fmt.int(v)} tone="violet" hint={`${fmt.int(data.counts?.dpo)} preference pairs`} />
        <Stat label="customers" value={data.counts?.customers} format={(v) => fmt.int(v)} tone="ok" hint={`${fmt.int(data.counts?.merchants)} merchants`} />
      </div>
      <div className="mb-5 grid gap-5 xl:grid-cols-3">
        <Card><CardHeader title="Time-based splits" subtitle="train Jan–Oct 2025 · validation Nov–Dec · holdout Jan–Feb 2026 (fenced)" /><CardBody><EChart option={splitOption} height={240} /></CardBody></Card>
        <Card><CardHeader title="Payment channels" /><CardBody><EChart option={channelOption} height={240} /></CardBody></Card>
        <Card><CardHeader title="CPT corpus (estimated tokens)" /><CardBody><EChart option={corpusOption} height={240} /></CardBody></Card>
      </div>
      <div className="grid gap-5 xl:grid-cols-[1.5fr_1fr]">
        <Card><CardHeader title="Data card" /><CardBody>{data.data_card ? <MarkdownView source={data.data_card} /> : <EmptyState title="No data card synced" />}</CardBody></Card>
        <Card><CardHeader title="PII audit and rules" icon={<ShieldCheck className="h-4 w-4" />} />
          <CardBody className="space-y-3 text-xs">
            <div className="flex items-center justify-between"><span className="text-muted">pattern audit sample</span><span>{fmt.int(data.pii_audit?.sample_size)} of {fmt.int(data.pii_audit?.population)} documents</span></div>
            <div className="flex items-center justify-between"><span className="text-muted">pattern hits</span><Badge tone={data.pii_audit?.pattern_hits?.length ? "bad" : "ok"}>{data.pii_audit?.pattern_hits?.length ?? 0}</Badge></div>
            <div className="flex items-center justify-between"><span className="text-muted">blocked devices (hard rule)</span><span>{data.rules?.blocked_devices?.length}</span></div>
            <div className="flex items-center justify-between"><span className="text-muted">allowlisted customers</span><span>{data.rules?.allowlisted_customers?.length}</span></div>
            <div className="flex items-center justify-between"><span className="text-muted">thresholds</span><span className="font-mono">{JSON.stringify(data.rules?.thresholds)}</span></div>
            <p className="pt-2 text-[11px] text-muted">{data.pii_audit?.limitations}</p>
          </CardBody></Card>
      </div>
    </>
  );
}
