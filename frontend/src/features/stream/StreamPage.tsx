import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, Pause, Play, Radio, Skull } from "lucide-react";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import { api, errorMessage } from "@/shared/api/client";
import { keys } from "@/shared/api/queries";
import { axisStyle, baseOption, EChart, PALETTE } from "@/shared/charts/EChart";
import { REASONS } from "@/shared/lib/reasons";
import { cn, fmt } from "@/shared/lib/utils";
import { Badge, Button, Card, CardBody, CardHeader, EmptyState, Input, NativeSelect, PageHeader, Stat } from "@/shared/ui/core";

/** Shapes of the stream service (stream/argus_stream/api.py), relayed by the platform under /api/v1/stream. */
interface Producer { running: boolean; rate: number; sent: number; attacks: number; source: string; error: string | null }
interface StreamState {
  stream: string; consumed: number; lag_ms: number | null; head: number; uptime_s?: number;
  scoring: {
    url: string; status: string; model_version: string | null; active_run: string | null; scored: number; errors: number;
    breaker_open: boolean; shadow_run: string | null; shadow_status: string;
  } | null;
  producer: Producer;
}
interface Group { key: string | null; n: number; review: number; block: number; fraud: number; amount: number; avg_probability: number | null }
interface Summary {
  minutes: number; bucket_s: number;
  totals: { n: number; allow: number; review: number; block: number; fail_open: number; hard_rule: number; fraud: number; amount: number; fraud_amount_blocked: number; fraud_amount_missed: number; end_to_end_ms: number };
  live_accuracy: { recall: number | null; precision: number | null; precision_at_block: number | null; false_decline_rate: number | null };
  api_latency: { n: number; p50_ms: number | null; p95_ms: number | null; p99_ms: number | null };
  timeseries: { t: number; decision: string; n: number }[];
  reasons: Record<string, number>;
  shadow: ShadowRow[];
  by_fraud_type: Group[]; by_category: Group[]; by_decision_source: Group[]; by_ring: Group[];
}
interface StreamEvent {
  id: number; ts: number; source: string; transaction_id: string; merchant_category: string; amount: number; city: string | null;
  merchant_city: string; fraud_label: number | null; fraud_type: string | null; decision: "ALLOW" | "REVIEW" | "BLOCK";
  risk_level: string; probability: number | null; decision_source: string; reason_codes: string[]; action: string;
  api_latency_ms: number | null; model_version: string | null;
}

interface ShadowRow {
  shadow_run: string; model_version: string | null; n: number; agreement: number | null; avg_latency_ms: number | null;
  live_recall: number | null; shadow_recall: number | null; live_precision_at_block: number | null; shadow_precision_at_block: number | null;
}

/** MLOPS-06 shadow: a second trained run scores the same transactions; it never decides. */
function ShadowCard({ state, rows }: { state: StreamState | undefined; rows: ShadowRow[] }) {
  const qc = useQueryClient();
  const current = state?.scoring?.shadow_run ?? null;
  const models = useQuery({ queryKey: keys.scoringModels, queryFn: () => api.get<{ active: string | null; models: { run: string; scorer: string }[] }>("/scoring/models"), retry: false });
  const [pick, setPick] = useState("");
  const set = useMutation({
    mutationFn: (run: string | null) => api.post("/stream/shadow", { run }),
    onSuccess: (_, run) => { toast.success(run ? `Shadow: ${run}` : "Shadow off"); qc.invalidateQueries({ queryKey: ["stream"] }); },
    onError: (e) => toast.error(errorMessage(e)),
  });
  const candidates = (models.data?.models ?? []).filter((m) => m.run !== models.data?.active);
  const row = rows.find((r) => r.shadow_run === current) ?? rows[0];
  return (
    <Card className="mb-5"><CardHeader title="Shadow run (MLOPS-06)" subtitle="A second trained run scores every transaction too, for a live comparison with the model that decides. It never changes a decision." />
      <CardBody>
        <div className="mb-4 flex flex-wrap items-center gap-2 text-xs">
          <span className="text-muted">live:</span><Badge tone="brand">{state?.scoring?.active_run ?? "–"}</Badge>
          <span className="ml-2 text-muted">shadow:</span>{current ? <Badge tone="violet">{current} · {state?.scoring?.shadow_status}</Badge> : <Badge>off</Badge>}
          <NativeSelect value={pick} onChange={(e) => setPick(e.target.value)} className="ml-auto h-8 w-56 text-xs">
            <option value="">{candidates.length ? "choose a run to shadow" : "no other servable run"}</option>
            {candidates.map((m) => <option key={m.run} value={m.run}>{m.run} ({m.scorer})</option>)}
          </NativeSelect>
          <Button size="sm" disabled={!pick} loading={set.isPending} onClick={() => set.mutate(pick)}>Shadow it</Button>
          {current && <Button size="sm" variant="danger" onClick={() => set.mutate(null)}>Stop shadow</Button>}
        </div>
        {row ? (
          <table className="w-full text-xs">
            <thead><tr>{["", "live model", `shadow · ${row.shadow_run}`].map((h) => <th key={h} className="border-b border-line px-2 py-1.5 text-left font-medium text-muted">{h}</th>)}</tr></thead>
            <tbody>
              {([["recall (REVIEW + BLOCK)", fmt.pct(row.live_recall), fmt.pct(row.shadow_recall)],
                 ["precision at BLOCK", fmt.pct(row.live_precision_at_block), fmt.pct(row.shadow_precision_at_block)],
                 ["same decision", fmt.pct(row.agreement), `${fmt.int(row.n)} transactions scored by both`],
                 ["model version", state?.scoring?.model_version ?? "–", row.model_version ?? "–"]] as const).map(([k, a, b]) => (
                <tr key={k}><td className="border-b border-line/50 px-2 py-1.5 text-muted">{k}</td>
                  <td className="border-b border-line/50 px-2 py-1.5 tabular-nums">{a}</td><td className="border-b border-line/50 px-2 py-1.5 tabular-nums">{b}</td></tr>))}
            </tbody>
          </table>
        ) : <div className="text-xs text-muted">No shadow comparison in this window yet.</div>}
      </CardBody>
    </Card>
  );
}

const FRAUD_TYPES =["account_takeover", "card_testing", "mule_account", "impossible_travel"];
const DECISION_TONE = { ALLOW: "ok", REVIEW: "warn", BLOCK: "bad" } as const;
const DECISION_COLOR = { ALLOW: "#34d399", REVIEW: "#fbbf24", BLOCK: "#fb7185" };

function Stage({ label, detail, ok }: { label: string; detail?: string | null; ok: boolean | null }) {
  return (
    <div className="min-w-0 flex-1 rounded-xl border border-line bg-white/[0.02] px-3 py-2">
      <div className="flex items-center gap-2 text-xs font-medium">
        <span className={cn("h-2 w-2 shrink-0 rounded-full", ok == null ? "bg-muted" : ok ? "bg-ok animate-pulse-soft" : "bg-bad")} />
        <span className="truncate">{label}</span>
      </div>
      {detail && <div className="mt-0.5 truncate text-[10px] text-muted" title={detail}>{detail}</div>}
    </div>
  );
}

function GroupTable({ rows, label }: { rows: Group[]; label: string }) {
  return (
    <table className="w-full text-xs">
      <thead><tr>{[label, "n", "review", "block", "fraud (truth)"].map((h) => <th key={h} className="border-b border-line px-2 py-1.5 text-left font-medium text-muted">{h}</th>)}</tr></thead>
      <tbody>{rows.map((r) => (
        <tr key={r.key ?? "none"} className="hover:bg-white/[0.02]">
          <td className="border-b border-line/50 px-2 py-1.5">{r.key ?? "–"}</td>
          {[r.n, r.review, r.block, r.fraud].map((v, i) => <td key={i} className="border-b border-line/50 px-2 py-1.5 tabular-nums">{fmt.int(v)}</td>)}
        </tr>))}
      </tbody>
    </table>
  );
}

export default function StreamPage() {
  const qc = useQueryClient();
  const [minutes, setMinutes] = useState(60);
  const [rate, setRate] = useState("2");
  const [attack, setAttack] = useState("");
  const state = useQuery({ queryKey: keys.streamState, queryFn: () => api.get<StreamState>("/stream/state"), refetchInterval: 2000, retry: false });
  const summary = useQuery({ queryKey: keys.streamSummary(minutes), queryFn: () => api.get<Summary>(`/stream/summary?minutes=${minutes}`), refetchInterval: 3000, retry: false, enabled: state.isSuccess });
  const feed = useQuery({ queryKey: keys.streamEvents, queryFn: () => api.get<{ events: StreamEvent[] }>("/stream/events?limit=25"), refetchInterval: 2000, retry: false, enabled: state.isSuccess });

  const refresh = () => qc.invalidateQueries({ queryKey: ["stream"] });
  const traffic = useMutation({
    mutationFn: (action: "start" | "stop") => api.post("/stream/producer", { action, rate: Number(rate) || undefined }),
    onSuccess: (_, a) => { toast.success(a === "start" ? `Traffic on: ${rate} tx/s` : "Traffic off"); refresh(); },
    onError: (e) => toast.error(errorMessage(e)),
  });
  const inject = useMutation({
    mutationFn: () => api.post<{ sent: number; fraud_type: string }>("/stream/inject", { n: 20, fraud_type: attack || undefined }),
    onSuccess: (r) => { toast.success(`Attack wave: ${r.sent} × ${r.fraud_type.replaceAll("_", " ")}`); refresh(); },
    onError: (e) => toast.error(errorMessage(e)),
  });

  const s = summary.data;
  const timeline = useMemo(() => {
    const ts = [...new Set((s?.timeseries ?? []).map((p) => p.t))].sort();
    return {
      ...baseOption(), grid: { left: 40, right: 16, top: 30, bottom: 28 },
      xAxis: { type: "category", data: ts.map((t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })), ...axisStyle },
      yAxis: { type: "value", ...axisStyle },
      series: (["ALLOW", "REVIEW", "BLOCK"] as const).map((d) => ({
        name: d, type: "bar", stack: "d", color: DECISION_COLOR[d],
        data: ts.map((t) => s?.timeseries.find((p) => p.t === t && p.decision === d)?.n ?? 0),
      })),
    };
  }, [s]);
  const reasons = useMemo(() => {
    const entries = Object.entries(s?.reasons ?? {}).sort((a, b) => a[1] - b[1]);
    return {
      ...baseOption(), grid: { left: 120, right: 30, top: 10, bottom: 20 },
      xAxis: { type: "value", ...axisStyle }, yAxis: { type: "category", data: entries.map(([k]) => REASONS[k]?.label ?? k), ...axisStyle },
      series: [{ type: "bar", data: entries.map(([, v]) => v), barWidth: 12, itemStyle: { color: PALETTE[1], borderRadius: [0, 6, 6, 0] } }],
    };
  }, [s]);

  if (state.isError) {
    return (
      <>
        <PageHeader eyebrow="Serve" title="Live stream" />
        <EmptyState icon={<Radio className="h-6 w-6" />} title="The stream service is not reachable" hint={<>
          Locally: <code>.\argus.ps1 stream</code> (Kinesis emulator + stream scorer; needs the serving stack). On AWS: set
          <code> ARGUS_STREAM_URL</code> and <code>ARGUS_STREAM_CONTROL_KEY</code> for the platform (printed by <code>python deploy/ops.py platform-env</code>),
          then restart the platform. A 401 means the platform's <code>ARGUS_STREAM_CONTROL_KEY</code> differs from the stream's key.
          <div className="mt-2 text-bad/80">{errorMessage(state.error)}</div></>} />
      </>
    );
  }
  if (state.data?.stream === "disabled") {
    return (
      <>
        <PageHeader eyebrow="Serve" title="Live stream" />
        <EmptyState icon={<Radio className="h-6 w-6" />} title="Stream processing is off" hint={<>
          The stream service answers, but runs without Kinesis (<code>KINESIS_ENABLED</code> is not 1), so there is no
          traffic to control or score. Locally: <code>.\argus.ps1 stream</code>; on AWS the service sets it.</>} />
      </>
    );
  }
  const st = state.data, p = st?.producer, sc = st?.scoring, t = s?.totals, acc = s?.live_accuracy;
  return (
    <>
      <PageHeader eyebrow="Serve" title="Live stream"
        subtitle="Transactions flow through Amazon Kinesis to the Fraud Model API; every decision becomes an action on the decisions stream (CRD 10.4). If the API cannot answer, the fail-open rule decides (FR-API-01)."
        actions={<NativeSelect value={minutes} onChange={(e) => setMinutes(Number(e.target.value))} className="h-9 w-36 text-xs">
          {[15, 60, 360, 1440].map((m) => <option key={m} value={m}>last {m >= 60 ? `${m / 60} h` : `${m} min`}</option>)}</NativeSelect>} />

      <Card className="mb-5 p-4">
        <div className="flex flex-wrap items-center gap-2">
          <Stage label="Producer" detail={p ? `${p.running ? `${p.rate} tx/s` : "off"} · ${fmt.int(p.sent)} sent · ${p.source}` : null} ok={p ? p.running && !p.error : null} />
          <ArrowRight className="h-4 w-4 shrink-0 text-muted" />
          <Stage label="Kinesis · transactions" detail={st ? `${st.stream}${st.lag_ms != null ? ` · lag ${fmt.int(st.lag_ms)} ms` : ""}` : null} ok={st ? st.stream === "connected" : null} />
          <ArrowRight className="h-4 w-4 shrink-0 text-muted" />
          <Stage label="Fraud Model API" detail={sc ? `${sc.status}${sc.model_version ? ` · ${sc.model_version}` : ""}${sc.breaker_open ? " · breaker open (fail-open)" : ""}` : null} ok={sc ? sc.status === "ok" && !sc.breaker_open : null} />
          <ArrowRight className="h-4 w-4 shrink-0 text-muted" />
          <Stage label="Decision → action" detail={st ? `${fmt.int(st.consumed)} scored since start` : null} ok={st ? st.consumed > 0 : null} />
          <ArrowRight className="h-4 w-4 shrink-0 text-muted" />
          <Stage label="Kinesis · decisions" detail="→ S3 audit (Firehose), SNS for BLOCK" ok={st ? st.stream === "connected" : null} />
        </div>
        <div className="mt-4 flex flex-wrap items-end gap-2 border-t border-line pt-4">
          <div><div className="mb-1 text-[10px] uppercase tracking-wider text-muted">rate (tx/s)</div>
            <Input value={rate} onChange={(e) => setRate(e.target.value)} className="h-9 w-24" inputMode="decimal" /></div>
          {p?.running
            ? <Button variant="danger" onClick={() => traffic.mutate("stop")} loading={traffic.isPending}><Pause className="h-4 w-4" /> Stop traffic</Button>
            : <Button variant="primary" onClick={() => traffic.mutate("start")} loading={traffic.isPending}><Play className="h-4 w-4" /> Start traffic</Button>}
          <div className="ml-4"><div className="mb-1 text-[10px] uppercase tracking-wider text-muted">attack pattern</div>
            <NativeSelect value={attack} onChange={(e) => setAttack(e.target.value)} className="h-9 w-48 text-xs">
              <option value="">a random fraud ring</option>{FRAUD_TYPES.map((f) => <option key={f} value={f}>{f.replaceAll("_", " ")}</option>)}</NativeSelect></div>
          <Button onClick={() => inject.mutate()} loading={inject.isPending}><Skull className="h-4 w-4" /> Inject attack wave</Button>
          {p?.error && <Badge tone="bad">{p.error}</Badge>}
        </div>
      </Card>

      <ShadowCard state={st} rows={s?.shadow ?? []} />

      <div className="mb-5 grid grid-cols-2 gap-4 lg:grid-cols-6">
        <Stat label="transactions" value={t?.n} format={(v) => fmt.int(v)} />
        <Stat label="blocked" value={t?.block} format={(v) => fmt.int(v)} tone="bad" />
        <Stat label="held for review" value={t?.review} format={(v) => fmt.int(v)} tone="warn" />
        <Stat label="fail-open" value={t?.fail_open} format={(v) => fmt.int(v)} tone="violet" hint="API unavailable" />
        <Stat label="live recall" value={acc?.recall} format={(v) => fmt.pct(v)} tone="ok" hint={acc?.precision_at_block != null ? `precision at BLOCK ${fmt.pct(acc.precision_at_block)}` : "vs ground truth"} />
        <Stat label="API p99" value={s?.api_latency.p99_ms} format={(v) => (v == null ? "–" : `${fmt.int(v)} ms`)} hint={s?.api_latency.p50_ms != null ? `p50 ${fmt.int(s.api_latency.p50_ms)} ms` : undefined} />
      </div>

      <div className="mb-5 grid gap-5 xl:grid-cols-[1.4fr_1fr]">
        <Card><CardHeader title="Decisions over time" subtitle={`per ${s ? s.bucket_s / 60 : 1} min`} /><CardBody><EChart option={timeline} height={240} /></CardBody></Card>
        <Card><CardHeader title="Reason codes on REVIEW / BLOCK" /><CardBody>{Object.keys(s?.reasons ?? {}).length ? <EChart option={reasons} height={240} /> : <EmptyState title="No flagged transactions yet" />}</CardBody></Card>
      </div>

      <div className="mb-5 grid gap-5 xl:grid-cols-[1.6fr_1fr]">
        <Card><CardHeader title="Live feed" subtitle="Newest first · ground truth comes from the producer, never sent to the API" />
          <CardBody className="overflow-x-auto">
            {feed.data?.events.length ? (
              <table className="w-full text-xs">
                <thead><tr>{["time", "transaction", "amount", "category", "route", "p", "decision", "action", "truth", "ms"].map((h) =>
                  <th key={h} className="border-b border-line px-2 py-1.5 text-left font-medium text-muted">{h}</th>)}</tr></thead>
                <tbody>{feed.data.events.map((e) => (
                  <tr key={e.id} className="hover:bg-white/[0.02]">
                    <td className="border-b border-line/50 px-2 py-1.5 text-muted">{new Date(e.ts * 1000).toLocaleTimeString()}</td>
                    <td className="border-b border-line/50 px-2 py-1.5 font-mono text-[11px]" title={e.reason_codes.join(", ")}>{e.transaction_id}</td>
                    <td className="border-b border-line/50 px-2 py-1.5 tabular-nums">₹{fmt.int(e.amount)}</td>
                    <td className="border-b border-line/50 px-2 py-1.5">{e.merchant_category}</td>
                    <td className="border-b border-line/50 px-2 py-1.5 text-muted">{e.city ?? "?"} → {e.merchant_city}</td>
                    <td className="border-b border-line/50 px-2 py-1.5 tabular-nums">{fmt.num(e.probability)}</td>
                    <td className="border-b border-line/50 px-2 py-1.5"><Badge tone={DECISION_TONE[e.decision]}>{e.decision}</Badge>{e.decision_source !== "MODEL" && <span className="ml-1 text-[10px] text-muted">{e.decision_source}</span>}</td>
                    <td className="border-b border-line/50 px-2 py-1.5 text-[11px]">{e.action}</td>
                    <td className="border-b border-line/50 px-2 py-1.5">{e.fraud_label ? <Badge tone="bad">{(e.fraud_type ?? "fraud").replaceAll("_", " ")}</Badge> : <span className="text-muted">genuine</span>}</td>
                    <td className="border-b border-line/50 px-2 py-1.5 tabular-nums text-muted">{e.api_latency_ms == null ? "–" : fmt.int(e.api_latency_ms)}</td>
                  </tr>))}
                </tbody>
              </table>
            ) : <EmptyState title="No transactions yet" hint="Start traffic above, or inject an attack wave." />}
          </CardBody>
        </Card>
        <div className="space-y-5">
          <Card><CardHeader title="By fraud pattern" subtitle="Did each pattern get caught?" /><CardBody><GroupTable rows={s?.by_fraud_type ?? []} label="pattern" /></CardBody></Card>
          <Card><CardHeader title="By decision source" subtitle="MODEL · HARD_RULE · FAIL_OPEN · AUTH_ERROR" /><CardBody><GroupTable rows={s?.by_decision_source ?? []} label="source" /></CardBody></Card>
          <Card><CardHeader title="Rupees" /><CardBody className="space-y-1 text-xs">
            <div className="flex justify-between"><span className="text-muted">fraud blocked</span><span className="tabular-nums text-ok">₹{fmt.int(t?.fraud_amount_blocked)}</span></div>
            <div className="flex justify-between"><span className="text-muted">fraud approved (missed)</span><span className="tabular-nums text-bad">₹{fmt.int(t?.fraud_amount_missed)}</span></div>
            <div className="flex justify-between"><span className="text-muted">genuine blocked (false declines)</span><span className="tabular-nums">{fmt.pct(acc?.false_decline_rate, 2)}</span></div>
            <div className="flex justify-between"><span className="text-muted">Kinesis → decision, average</span><span className="tabular-nums">{fmt.int(t?.end_to_end_ms)} ms</span></div>
          </CardBody></Card>
        </div>
      </div>
    </>
  );
}
