import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import { CheckCircle2, Rocket, Send, ShieldAlert, ShieldCheck, ShieldX, Zap } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { api, errorMessage } from "@/shared/api/client";
import { keys, useServices } from "@/shared/api/queries";
import type { Approval, SampleRequest, ScoreResponse } from "@/shared/api/types";
import { baseOption, EChart } from "@/shared/charts/EChart";
import { DECISION_POLICY, REASONS } from "@/shared/lib/reasons";
import { cn, fmt } from "@/shared/lib/utils";
import { Badge, Button, Card, CardBody, CardHeader, EmptyState, ErrorState, PageHeader, StatusBadge } from "@/shared/ui/core";
import { CodeView, JsonView } from "@/shared/ui/content";
import { ApprovalFields, approvalReady, emptyApproval, isApprovalError } from "@/shared/ui/governance";
import { Dialog, Tip } from "@/shared/ui/overlays";

interface Deployments {
  active: string | null; shadow?: string | null;
  models: { run: string; scorer: string; loaded: boolean; active: boolean; shadow?: boolean; forced?: boolean; forced_reason?: string | null }[];
}
interface Decision extends ScoreResponse { logged_at: string; explanation_source: string | null; feature_source: string | null }

const DECISION_STYLE = {
  ALLOW: { icon: ShieldCheck, cls: "from-ok/25 to-ok/5 border-ok/40 text-ok" },
  REVIEW: { icon: ShieldAlert, cls: "from-warn/25 to-warn/5 border-warn/40 text-warn" },
  BLOCK: { icon: ShieldX, cls: "from-bad/25 to-bad/5 border-bad/40 text-bad" },
};

function ProbabilityGauge({ p }: { p: number }) {
  const option = useMemo(() => ({
    ...baseOption(),
    series: [{
      type: "gauge", startAngle: 200, endAngle: -20, min: 0, max: 1, radius: "100%", center: ["50%", "62%"],
      axisLine: { lineStyle: { width: 16, color: [[0.3, "#34d399"], [0.6, "#a3e635"], [0.85, "#fbbf24"], [1, "#fb7185"]] } },
      pointer: { length: "62%", width: 5, itemStyle: { color: "#e6ebf5" } }, anchor: { show: true, size: 10, itemStyle: { color: "#e6ebf5" } },
      axisTick: { show: false }, splitLine: { length: 16, lineStyle: { color: "#060a14", width: 3 } },
      axisLabel: { color: "#8a96ad", distance: 22, fontSize: 10, formatter: (v: number) => ([0, 0.3, 0.6, 0.85, 1].includes(+v.toFixed(2)) ? v.toFixed(2) : "") },
      detail: { valueAnimation: true, offsetCenter: [0, "30%"], fontSize: 26, fontWeight: 600, color: "#e6ebf5", formatter: (v: number) => v.toFixed(4) },
      data: [{ value: p }],
    }],
  }), [p]);
  return <EChart option={option} height={210} />;
}

function ResponseCard({ status, body }: { status: number; body: any }) {
  if (status !== 200) {
    return (
      <Card className="border-bad/40 p-5">
        <div className="mb-3 flex items-center gap-2"><Badge tone="bad">HTTP {status}</Badge><span className="text-sm font-medium">{body?.error}</span></div>
        <JsonView data={body} />
        <div className="mt-3 text-xs text-muted">{status === 400 ? "FR-API-02: invalid requests return 400 with the field." : status === 503 ? "FR-API-01: the Payment API applies fail-open (approve < ₹10,000, rest to REVIEW)." : null}</div>
      </Card>
    );
  }
  const r = body as ScoreResponse;
  const style = DECISION_STYLE[r.decision];
  const Icon = style.icon;
  return (
    <motion.div initial={{ opacity: 0, y: 10, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ type: "spring", stiffness: 160, damping: 18 }} className="space-y-4">
      <div className={cn("flex items-center justify-between rounded-2xl border bg-gradient-to-br p-5", style.cls)}>
        <div className="flex items-center gap-4"><Icon className="h-10 w-10" /><div><div className="text-3xl font-bold tracking-tight">{r.decision}</div><div className="text-xs text-ink/70">risk {r.risk_level} · {r.decision_source}</div></div></div>
        <div className="text-right text-xs text-ink/70"><div className="flex items-center justify-end gap-1"><Zap className="h-3.5 w-3.5" />{r.latency_ms} ms</div><div className="mt-1 font-mono">{r.model_version}</div></div>
      </div>
      {r.fraud_probability != null && <Card className="p-2"><ProbabilityGauge p={r.fraud_probability} /></Card>}
      <Card className="p-4">
        <div className="mb-2 text-xs font-medium text-muted">reason codes (head ≥ 0.5 and feature flag true · LLD 2.3)</div>
        <div className="flex flex-wrap gap-2">{r.reason_codes.length ? r.reason_codes.map((c) => (
          <Tip key={c} content={REASONS[c]?.meaning ?? c}><span><Badge tone="brand">{c} · {REASONS[c]?.label}</Badge></span></Tip>)) : <span className="text-xs text-muted">none</span>}</div>
        <div className="mt-4 mb-1 text-xs font-medium text-muted">explanation</div>
        <p className="text-sm leading-relaxed">{r.explanation ?? <span className="text-muted">produced asynchronously (ALLOW, FR-API-05) - see the decision in the audit log below</span>}</p>
      </Card>
    </motion.div>
  );
}

function DeploymentsCard() {
  const qc = useQueryClient();
  const { data, error } = useQuery({ queryKey: keys.scoringModels, queryFn: () => api.get<Deployments>("/scoring/models"), refetchInterval: 10000, retry: false });
  const { data: svc } = useServices();
  const production = svc?.serving_mode === "production";
  const [pending, setPending] = useState<string | null>(null);   // run waiting for two-person approval
  const [approval, setApproval] = useState<Approval>(emptyApproval);
  const activate = useMutation({
    mutationFn: ({ run, withApproval }: { run: string; withApproval?: boolean }) =>
      api.post(`/scoring/models/${run}/activate`, withApproval ? { approval } : {}),
    onSuccess: (_, v) => { toast.success(`Deployed ${v.run}`); setPending(null); setApproval(emptyApproval); qc.invalidateQueries({ queryKey: keys.scoringModels }); qc.invalidateQueries({ queryKey: keys.services }); },
    onError: (e, v) => { if (isApprovalError(e)) setPending(v.run); else toast.error(errorMessage(e)); },
  });
  const start = (run: string) => (production ? setPending(run) : activate.mutate({ run }));
  return (
    <Card><CardHeader title="Deployments" subtitle="Activate a trained run's scorer (LLD 4.4 rollback); switching is instant once loaded" icon={<Rocket className="h-4 w-4" />} />
      <CardBody className="space-y-2">
        {error && <div className="text-xs text-bad">Scoring API not reachable: start the serving stack (System page).</div>}
        {data?.models.length === 0 && <div className="text-xs text-muted">No servable runs yet.</div>}
        {data?.models.map((m) => (
          <div key={m.run} className={cn("flex items-center gap-3 rounded-xl border px-3 py-2", m.active ? "border-brand/40 bg-brand/[0.06]" : "border-line")}>
            <div className="min-w-0 flex-1"><div className="truncate text-sm font-medium">{m.run}</div><div className="text-[11px] text-muted">{m.scorer === "llm" ? "FP8 LLM scorer" : "XGBoost (HYBRID)"}{m.loaded && " · warm"}{m.shadow && " · shadow"}</div></div>
            {m.forced && <Tip content={`Not a measured run: ${m.forced_reason ?? "the week-8 branch was forced"}. A demo, never promotable: a production profile refuses to activate it.`}><span><Badge tone="warn">demo · forced</Badge></span></Tip>}
            {m.active ? <Badge tone="ok"><CheckCircle2 className="h-3 w-3" /> live</Badge>
              : <Button size="sm" onClick={() => start(m.run)} loading={activate.isPending && activate.variables?.run === m.run}>Activate</Button>}
          </div>
        ))}
      </CardBody>
      {pending && (
        <Dialog open onOpenChange={(o) => !o && setPending(null)} title={`Activate ${pending}`} description="Production mode: switching the live model changes what every payment meets. The API refuses demo runs here regardless.">
          <div className="space-y-3">
            <ApprovalFields value={approval} onChange={setApproval} />
            <div className="flex justify-end gap-2"><Button variant="ghost" onClick={() => setPending(null)}>Cancel</Button>
              <Button variant="primary" onClick={() => activate.mutate({ run: pending, withApproval: true })} loading={activate.isPending} disabled={!approvalReady(approval)}>Activate</Button></div>
          </div>
        </Dialog>)}
    </Card>
  );
}

function RecentDecisions() {
  const { data } = useQuery({ queryKey: keys.decisions, queryFn: () => api.get<Decision[]>("/scoring/admin/decisions"), refetchInterval: 4000, retry: false });
  const [open, setOpen] = useState<string | null>(null);
  const detail = useQuery({ queryKey: ["scoring", "decision", open], queryFn: () => api.get<any>(`/scoring/admin/decisions/${open}`), enabled: !!open });
  return (
    <Card><CardHeader title="Recent decisions" subtitle="Audit log (FR-API-03) · click for the support lookup with the async explanation" />
      <div className="max-h-[380px] overflow-auto"><table className="w-full text-xs">
        <thead className="sticky top-0 bg-surface"><tr className="text-left text-muted">{["time", "transaction", "p", "decision", "codes", "explanation", "model", "ms"].map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}</tr></thead>
        <tbody><AnimatePresence initial={false}>{(data ?? []).map((d) => (
          <motion.tr key={d.logged_at + d.transaction_id} layout initial={{ opacity: 0, backgroundColor: "rgba(34,211,238,0.15)" }} animate={{ opacity: 1, backgroundColor: "rgba(0,0,0,0)" }}
            className="cursor-pointer border-t border-line/60 hover:bg-white/[0.03]" onClick={() => setOpen(d.transaction_id)}>
            <td className="px-3 py-2 text-muted">{d.logged_at?.slice(11, 19)}</td><td className="px-3 py-2 font-mono">{fmt.short(d.transaction_id, 18)}</td>
            <td className="px-3 py-2 tabular-nums">{fmt.num(d.fraud_probability, 4)}</td><td className="px-3 py-2"><StatusBadge status={d.decision} /></td>
            <td className="px-3 py-2">{d.reason_codes?.map((c) => REASONS[c]?.label ?? c).join(", ") || "–"}</td>
            <td className="max-w-[220px] truncate px-3 py-2 text-muted">{d.explanation_source ?? "pending"}</td>
            <td className="px-3 py-2 font-mono text-[10px]">{fmt.short(d.model_version, 26)}</td><td className="px-3 py-2 tabular-nums">{d.latency_ms}</td>
          </motion.tr>))}</AnimatePresence></tbody></table>
        {!data?.length && <EmptyState title="No decisions yet" hint="Send a request above." />}
      </div>
      <Dialog open={!!open} onOpenChange={(o) => !o && setOpen(null)} title={`Support lookup · ${open}`} description="GET /admin/v1/decisions/{id} - decision, features used, prompt hash, explanation" wide>
        {detail.data ? <JsonView data={detail.data} maxHeight="60vh" /> : detail.error ? <ErrorState error={detail.error} /> : <div className="text-xs text-muted">Loading…</div>}
      </Dialog>
    </Card>
  );
}

export default function ScoringPage() {
  const qc = useQueryClient();
  const samples = useQuery({ queryKey: keys.samples, queryFn: () => api.get<SampleRequest[]>("/scoring/samples") });
  const [label, setLabel] = useState<string>("");
  const [text, setText] = useState("");
  const [result, setResult] = useState<{ status: number; body: any } | null>(null);
  useEffect(() => {
    if (samples.data?.length && !text) { setLabel(samples.data[0].label); setText(JSON.stringify(samples.data[0].request, null, 2)); }
  }, [samples.data, text]);
  const send = useMutation({
    mutationFn: async () => {
      let body: unknown;
      try { body = JSON.parse(text); } catch { throw new Error("The request is not valid JSON"); }
      return api.raw("POST", "/scoring/score", body);
    },
    onSuccess: (r) => { setResult(r); qc.invalidateQueries({ queryKey: keys.decisions }); },
    onError: (e) => toast.error(errorMessage(e)),
  });
  const pick = (s: SampleRequest) => { setLabel(s.label); setText(JSON.stringify(s.request, null, 2)); setResult(null); };

  return (
    <>
      <PageHeader eyebrow="Serve" title="Scoring playground" subtitle="POST /v1/fraud/score exactly as the Payment API calls it (CRD 10.2 → 10.3): hard rules, features, scorer, calibration, policy, reason codes, explanation."
        actions={<Button variant="primary" size="lg" onClick={() => send.mutate()} loading={send.isPending}><Send className="h-4 w-4" /> Score transaction</Button>} />
      <div className="grid gap-5 xl:grid-cols-[300px_1fr_1fr]">
        <div className="space-y-5">
          <DeploymentsCard />
          <Card><CardHeader title="Sample requests" />
            <CardBody className="space-y-1.5">{samples.data?.map((s) => (
              <button key={s.label} onClick={() => pick(s)} className={cn("w-full rounded-xl border px-3 py-2 text-left text-xs transition", label === s.label ? "border-brand/40 bg-brand/[0.06]" : "border-line hover:bg-white/[0.03]")}>
                <div className="font-medium">{s.label}</div><div className="mt-0.5 text-[10px] text-muted">{s.note}</div></button>))}
            </CardBody></Card>
        </div>
        <Card><CardHeader title="Request" subtitle="Editable JSON · validated by the API (LLD 4.1)" />
          <CardBody><CodeView value={text} onChange={setText} language="json" height="560px" maxHeight="560px" /></CardBody></Card>
        <div className="space-y-4">
          {result ? <ResponseCard status={result.status} body={result.body} /> : (
            <Card className="p-5"><div className="mb-3 text-sm font-medium">Decision policy (CRD 10.4)</div>
              <table className="w-full text-xs"><tbody>{DECISION_POLICY.map((p) => (
                <tr key={p.range} className="border-t border-line/60"><td className="py-2 font-mono">{p.range}</td><td>{p.risk}</td><td><StatusBadge status={p.decision} /></td><td className="text-muted">{p.action}</td></tr>))}</tbody></table>
              <div className="mt-3 text-[11px] text-muted">Hard rules run first: a blocklisted device always BLOCKs (FR-RE-02). Allowlisted customers are capped at REVIEW (FR-RE-03).</div>
            </Card>)}
        </div>
      </div>
      <div className="mt-5"><RecentDecisions /></div>
    </>
  );
}
