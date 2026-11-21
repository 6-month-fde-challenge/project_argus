import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "motion/react";
import { History, ListX, Plus, Scale, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { api, errorMessage } from "@/shared/api/client";
import { keys } from "@/shared/api/queries";
import { fmt } from "@/shared/lib/utils";
import { Badge, Button, Card, CardBody, CardHeader, EmptyState, ErrorState, Input, Label, NativeSelect, PageHeader } from "@/shared/ui/core";

interface Thresholds { medium: number; review: number; block: number }
interface ThresholdState { thresholds: Thresholds; mcc_thresholds: Record<string, Thresholds>; rules_version: number; threshold_version: number }
const MCCS = ["education", "electronics", "food", "gaming", "gift_cards", "grocery", "healthcare", "other", "quick_commerce", "travel"];

function ThresholdBar({ t }: { t: Thresholds }) {
  const seg = [["LOW · ALLOW", 0, t.medium, "bg-ok/50"], ["MEDIUM · ALLOW", t.medium, t.review, "bg-lime-400/50"],
    ["HIGH · REVIEW", t.review, t.block, "bg-warn/60"], ["CRITICAL · BLOCK", t.block, 1, "bg-bad/60"]] as const;
  return (
    <div className="flex h-9 overflow-hidden rounded-xl border border-line text-[10px] font-medium">
      {seg.map(([label, a, b, cls]) => (
        <motion.div key={label} layout className={`${cls} grid place-items-center overflow-hidden whitespace-nowrap px-1 text-ink`} style={{ width: `${(b - a) * 100}%` }}>{b - a > 0.08 ? label : ""}</motion.div>
      ))}
    </div>
  );
}

function ThresholdEditor({ state }: { state: ThresholdState }) {
  const qc = useQueryClient();
  const [mcc, setMcc] = useState("");
  const base = mcc ? state.mcc_thresholds[mcc] ?? state.thresholds : state.thresholds;
  const [t, setT] = useState<Thresholds>(base);
  const [who, setWho] = useState({ requested_by: "karthik", approved_by: "", reason: "" });
  useEffect(() => setT(base), [mcc, state]); // eslint-disable-line react-hooks/exhaustive-deps
  const save = useMutation({
    mutationFn: async () => {
      const r = await api.raw("POST", "/scoring/admin/thresholds", { ...t, mcc: mcc || null, ...who });
      if (r.status !== 200) throw new Error((r.body as any)?.message ?? `HTTP ${r.status}`);
      return r.body;
    },
    onSuccess: () => { toast.success(`Thresholds ${mcc ? `for ${mcc} ` : ""}approved`); qc.invalidateQueries({ queryKey: keys.thresholds }); qc.invalidateQueries({ queryKey: keys.changelog }); },
    onError: (e) => toast.error(errorMessage(e)),
  });
  return (
    <Card><CardHeader title="Decision thresholds" subtitle="Configurable without deployment (FR-RE-01), per merchant category (FR-RE-04), two-person approval (LLD 4.4)" icon={<Scale className="h-4 w-4" />}
      actions={<Badge tone="info">threshold v{state.threshold_version}</Badge>} />
      <CardBody className="space-y-5">
        <div><Label>scope</Label><NativeSelect value={mcc} onChange={(e) => setMcc(e.target.value)}><option value="">global</option>{MCCS.map((m) => <option key={m} value={m}>{m}{state.mcc_thresholds[m] ? " (override)" : ""}</option>)}</NativeSelect></div>
        <ThresholdBar t={t} />
        <div className="grid gap-4 sm:grid-cols-3">
          {(["medium", "review", "block"] as const).map((k) => (
            <div key={k}><Label>{k} ≥ <b className="text-ink">{t[k].toFixed(2)}</b></Label>
              <input type="range" min={0.01} max={0.99} step={0.01} value={t[k]} onChange={(e) => setT({ ...t, [k]: +e.target.value })} className="w-full accent-cyan-400" /></div>
          ))}
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          <div><Label>requested by</Label><Input value={who.requested_by} onChange={(e) => setWho({ ...who, requested_by: e.target.value })} /></div>
          <div><Label>approved by (different person)</Label><Input value={who.approved_by} onChange={(e) => setWho({ ...who, approved_by: e.target.value })} placeholder="ananya" /></div>
          <div><Label>reason</Label><Input value={who.reason} onChange={(e) => setWho({ ...who, reason: e.target.value })} placeholder="why this change" /></div>
        </div>
        <Button variant="primary" onClick={() => save.mutate()} loading={save.isPending} disabled={!who.approved_by || !who.reason}>Approve and apply</Button>
      </CardBody></Card>
  );
}

function ListCard({ name, title, hint }: { name: string; title: string; hint: string }) {
  const qc = useQueryClient();
  const { data } = useQuery({ queryKey: keys.list(name), queryFn: () => api.get<Record<string, string[]>>(`/scoring/admin/lists/${name}`), retry: false });
  const [item, setItem] = useState("");
  const change = useMutation({
    mutationFn: ({ value, add }: { value: string; add: boolean }) =>
      (add ? api.post : api.del)(`/scoring/admin/lists/${name}`, { item: value, by: "karthik", reason: add ? "added from the ARGUS console" : "removed from the ARGUS console" }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: keys.list(name) }); qc.invalidateQueries({ queryKey: keys.changelog }); setItem(""); },
    onError: (e) => toast.error(errorMessage(e)),
  });
  const items = data?.[name] ?? [];
  return (
    <Card><CardHeader title={title} subtitle={hint} icon={<ListX className="h-4 w-4" />} actions={<Badge>{items.length}</Badge>} />
      <CardBody className="space-y-3">
        <div className="flex gap-2"><Input value={item} onChange={(e) => setItem(e.target.value)} placeholder={name === "blocked_devices" ? "D_…" : "C_…"} />
          <Button onClick={() => change.mutate({ value: item, add: true })} disabled={!item}><Plus className="h-4 w-4" /></Button></div>
        <div className="max-h-56 space-y-1 overflow-auto">{items.map((x) => (
          <div key={x} className="flex items-center justify-between rounded-lg bg-white/[0.02] px-3 py-1.5 font-mono text-xs">{x}
            <button className="text-muted hover:text-bad" aria-label={`remove ${x}`} onClick={() => change.mutate({ value: x, add: false })}><Trash2 className="h-3.5 w-3.5" /></button></div>))}</div>
      </CardBody></Card>
  );
}

function Changelog() {
  const { data } = useQuery({ queryKey: keys.changelog, queryFn: () => api.get<any[]>("/scoring/admin/changelog"), refetchInterval: 10000, retry: false });
  const rows = [...(data ?? [])].reverse();
  return (
    <Card><CardHeader title="Change log" subtitle="Every threshold, list and deployment change with who and why (FR-RE-01)" icon={<History className="h-4 w-4" />} />
      <CardBody>{rows.length ? (
        <ol className="relative space-y-3 border-l border-line pl-5">{rows.map((c, i) => (
          <li key={i} className="relative"><span className="absolute -left-[25px] top-1 h-2.5 w-2.5 rounded-full bg-gradient-to-br from-brand to-brand-2" />
            <div className="text-xs"><b>{c.change}</b>{c.mcc && ` · ${c.mcc}`}{c.item && ` · ${c.item}`}{c.run && ` · ${c.run}`}</div>
            <div className="text-[11px] text-muted">{c.requested_by ?? c.by}{c.approved_by && ` → approved by ${c.approved_by}`} · {c.reason} · {fmt.ago(c.at)}</div></li>))}</ol>
      ) : <EmptyState title="No changes yet" />}</CardBody></Card>
  );
}

export default function RulesPage() {
  const { data, error } = useQuery({ queryKey: keys.thresholds, queryFn: () => api.get<ThresholdState>("/scoring/admin/thresholds"), retry: false });
  return (
    <>
      <PageHeader eyebrow="Serve" title="Rules & policy" subtitle="The model advises, the policy decides (Part 3): thresholds, hard rules and lists that turn a probability into ALLOW, REVIEW or BLOCK." />
      {error ? <ErrorState error={error} /> : (
        <div className="grid gap-5 xl:grid-cols-[1.3fr_1fr]">
          <div className="space-y-5">{data && <ThresholdEditor state={data} />}<Changelog /></div>
          <div className="space-y-5">
            <ListCard name="blocked_devices" title="Blocked devices" hint="Hard rule: always BLOCK, before the model runs (FR-RE-02)" />
            <ListCard name="allowlisted_customers" title="Allowlisted customers" hint="Capped at REVIEW instead of BLOCK (FR-RE-03)" />
          </div>
        </div>)}
    </>
  );
}
