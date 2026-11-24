import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CirclePlay, ListChecks, Square, Terminal, Wrench } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import { toast } from "sonner";
import { api, errorMessage } from "@/shared/api/client";
import { keys } from "@/shared/api/queries";
import type { Approval, OpsAction, OpsActionDef, OpsCatalog, OpsParam } from "@/shared/api/types";
import { cn, fmt } from "@/shared/lib/utils";
import { Badge, Button, Card, CardBody, CardHeader, EmptyState, ErrorState, Input, Label, NativeSelect, PageHeader, Skeleton, StatusBadge } from "@/shared/ui/core";
import { AgentNote, ApprovalFields, approvalReady, CommandLine, emptyApproval, ServingModeBadge } from "@/shared/ui/governance";
import { Dialog } from "@/shared/ui/overlays";

type Params = Record<string, unknown>;
const ACTIVE = ["queued", "running"];

function initialParams(a: OpsActionDef, cat: OpsCatalog): Params {
  const out: Params = {};
  for (const [name, p] of Object.entries(a.params)) {
    if (p.default !== undefined) out[name] = p.default;
    else if (p.type === "run" && p.required) out[name] = cat.defaults.run ?? cat.runs[0] ?? "";
    else if (p.type === "serving_config") out[name] = cat.defaults.config ?? "";
    else if (p.type === "vm_profile") out[name] = cat.defaults.profile ?? "";
    else if (p.type === "bool") out[name] = false;
    else if (p.type === "services" || p.type === "runs") out[name] = [];
  }
  return out;
}

function ParamInput({ p, value, onChange, cat }: { p: OpsParam; value: unknown; onChange: (v: unknown) => void; cat: OpsCatalog }) {
  const list = (opts: string[]) => {
    const cur = (value as string[]) ?? [];
    return (
      <div className="flex flex-wrap gap-2">{opts.map((o) => (
        <label key={o} className={cn("flex cursor-pointer items-center gap-1.5 rounded-lg border px-2 py-1 text-xs", cur.includes(o) ? "border-brand/50 bg-brand/10 text-ink" : "border-line text-muted")}>
          <input type="checkbox" className="accent-cyan-400" checked={cur.includes(o)} onChange={(e) => onChange(e.target.checked ? [...cur, o] : cur.filter((x) => x !== o))} />{o}</label>))}</div>
    );
  };
  switch (p.type) {
    case "bool":
      return <label className="flex items-center gap-2 text-xs"><input type="checkbox" className="accent-cyan-400" checked={!!value} onChange={(e) => onChange(e.target.checked)} />{p.label}</label>;
    case "int":
      return <Input type="number" min={p.min} max={p.max} value={String(value ?? "")} onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))} />;
    case "run":
      return <NativeSelect value={String(value ?? "")} onChange={(e) => onChange(e.target.value || null)}>{!p.required && <option value="">(none)</option>}{cat.runs.map((r) => <option key={r} value={r}>{r}</option>)}</NativeSelect>;
    case "runs":
      return list(cat.runs);
    case "services":
      return list(p.options ?? []);
    case "profile":
      return <NativeSelect value={String(value ?? "")} onChange={(e) => onChange(e.target.value)}><option value="">choose…</option>{cat.profiles.map((r) => <option key={r} value={r}>{r}</option>)}</NativeSelect>;
    case "vm_profile":
      return (
        <NativeSelect value={String(value ?? "")} onChange={(e) => onChange(e.target.value)}>
          <option value="">choose…</option>
          <option value="prod">prod (prod_a100: the only profile for live traffic)</option>
          <option value="demo">demo (forced branches: demonstrations only)</option>
          <option value="smoke">smoke (tiny sizes: health run)</option>
          {cat.profiles.filter((x) => !["base", "smoke"].includes(x)).map((r) => <option key={r} value={r}>{r}</option>)}
        </NativeSelect>);
    case "serving_config":
      return <NativeSelect value={String(value ?? "")} onChange={(e) => onChange(e.target.value)}>{cat.profiles.filter((x) => x !== "base").map((r) => <option key={r} value={`configs/${r}.yaml`}>configs/{r}.yaml</option>)}</NativeSelect>;
    default:
      return <Input value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} />;
  }
}

function RunDialog({ action, cat, onClose, onQueued }: { action: OpsActionDef; cat: OpsCatalog; onClose: () => void; onQueued: (a: OpsAction) => void }) {
  const [params, setParams] = useState<Params>(() => initialParams(action, cat));
  const [approval, setApproval] = useState<Approval>(emptyApproval);
  const os = cat.agent.os;
  const { data: preview, error: previewError } = useQuery({
    queryKey: ["ops", "preview", action.name, params], retry: false,
    queryFn: () => api.post<{ commands: { linux: string | null; windows: string | null } }>("/ops/preview", { action: action.name, params }),
  });
  const run = useMutation({
    mutationFn: () => api.post<OpsAction>("/ops/actions", { action: action.name, params, approval: action.approval_needed ? approval : undefined }),
    onSuccess: (a) => { toast.success(`${a.label} queued`); onQueued(a); onClose(); },
    onError: (e) => toast.error(errorMessage(e)),
  });
  const unsupported = os && !action[os];
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()} title={action.label} description={action.description} wide>
      <div className="space-y-4">
        {Object.entries(action.params).map(([name, p]) => (
          <div key={name}>{p.type !== "bool" && <Label>{p.label}{p.required && " *"}{p.type === "int" && p.min !== undefined ? ` (${p.min}-${p.max})` : ""}</Label>}
            <ParamInput p={p} value={params[name]} onChange={(v) => setParams({ ...params, [name]: v })} cat={cat} /></div>))}
        {!Object.keys(action.params).length && <div className="text-xs text-muted">No parameters.</div>}
        {previewError ? <div className="text-xs text-bad">{errorMessage(previewError)}</div> : (
          os ? <CommandLine command={preview?.commands[os] ?? (unsupported ? `not available on this ${os} machine` : "…")} label={`What the agent runs on ${cat.agent.host ?? os} (and what you can run yourself)`} />
            : <div className="space-y-2"><CommandLine command={preview?.commands.linux} label="GPU VM (Linux)" /><CommandLine command={preview?.commands.windows} label="Windows PC" /></div>)}
        {action.approval_needed && <ApprovalFields value={approval} onChange={setApproval} why="production mode: this changes what live traffic meets" />}
        {!cat.agent.online && <div className="text-xs text-warn">The operations agent is not running: the request waits in the queue until it starts (or run the command above yourself).</div>}
        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant="primary" onClick={() => run.mutate()} loading={run.isPending}
            disabled={!!previewError || !!unsupported || (action.approval_needed && !approvalReady(approval))}><CirclePlay className="h-4 w-4" /> Run</Button>
        </div>
      </div>
    </Dialog>
  );
}

function LogView({ id }: { id: string }) {
  const qc = useQueryClient();
  const [text, setText] = useState("");
  const offset = useRef(0);
  const box = useRef<HTMLPreElement>(null);
  const { data: action } = useQuery({ queryKey: keys.opsAction(id), queryFn: () => api.get<OpsAction>(`/ops/actions/${id}`), refetchInterval: (q) => (ACTIVE.includes(q.state.data?.status ?? "queued") ? 2000 : false) });
  useEffect(() => { setText(""); offset.current = 0; }, [id]);
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const r = await api.get<{ text: string; offset: number; status: string }>(`/ops/actions/${id}/log?offset=${offset.current}`);
        if (stop) return;
        if (r.text) { offset.current = r.offset; setText((t) => (t + r.text).slice(-400_000)); }
        if (ACTIVE.includes(r.status) || r.text) setTimeout(tick, 1500);
      } catch { if (!stop) setTimeout(tick, 4000); }
    };
    tick();
    return () => { stop = true; };
  }, [id]);
  useEffect(() => { box.current?.scrollTo({ top: box.current.scrollHeight }); }, [text]);
  const cancel = useMutation({
    mutationFn: () => api.post(`/ops/actions/${id}/cancel`),
    onSuccess: () => { toast.success("Cancel requested"); qc.invalidateQueries({ queryKey: keys.opsAction(id) }); },
    onError: (e) => toast.error(errorMessage(e)),
  });
  return (
    <Card><CardHeader title={action ? action.label : "…"} subtitle={action && <>{action.id} · {fmt.ago(action.created)}{action.requested_by && ` · ${action.requested_by} → ${action.approved_by}`}{action.message && ` · ${action.message}`}</>}
      icon={<Terminal className="h-4 w-4" />} actions={action && <div className="flex items-center gap-2"><StatusBadge status={action.status} />
        {ACTIVE.includes(action.status) && <Button size="sm" variant="danger" onClick={() => cancel.mutate()} loading={cancel.isPending} disabled={action.cancel_requested}><Square className="h-3 w-3" /> {action.cancel_requested ? "cancelling" : "cancel"}</Button>}</div>} />
      <CardBody className="space-y-3">
        <CommandLine command={action?.command ?? (action?.agent_os ? action.commands[action.agent_os as "linux" | "windows"] : action?.commands.linux)} label="Command" />
        <pre ref={box} className="max-h-[32rem] min-h-40 overflow-auto rounded-lg bg-black/40 p-3 font-mono text-[11px] leading-relaxed text-ink/90 whitespace-pre-wrap">
          {text || (action?.status === "queued" ? "Waiting for the operations agent…" : "No output yet.")}</pre>
      </CardBody></Card>
  );
}

export default function OperationsPage() {
  const qc = useQueryClient();
  const [params, setParams] = useSearchParams();
  const selected = params.get("action");
  const [open, setOpen] = useState<OpsActionDef | null>(null);
  const { data: cat, error, isLoading } = useQuery({ queryKey: keys.opsCatalog, queryFn: () => api.get<OpsCatalog>("/ops/catalog"), refetchInterval: 10000 });
  const { data: history } = useQuery({ queryKey: keys.opsActions, queryFn: () => api.get<OpsAction[]>("/ops/actions?limit=40"), refetchInterval: 3000 });
  const groups = useMemo(() => {
    const g: Record<string, OpsActionDef[]> = {};
    for (const a of cat?.actions ?? []) (g[a.group] ??= []).push(a);
    return Object.entries(g);
  }, [cat]);
  const select = (id: string) => setParams({ action: id });
  useEffect(() => { if (!selected && history?.length) setParams({ action: history[0].id }, { replace: true }); }, [history, selected, setParams]);
  const os = cat?.agent.os;
  return (
    <>
      <PageHeader eyebrow="Operate" title="Operations"
        subtitle="The GPU VM's scripts (deploy/hyperstack) and argus.ps1, run from here through the operations agent on the host. Every action shows its command: the same command works by hand."
        actions={<div className="flex items-center gap-2"><ServingModeBadge mode={cat?.serving.mode} demoModel={cat?.serving.active_run_forced} reason={cat?.serving.active_run_forced_reason} /><AgentNote agent={cat?.agent} /></div>} />
      {error ? <ErrorState error={error} /> : isLoading || !cat ? <Skeleton className="h-96" /> : (
        <div className="grid gap-5 xl:grid-cols-[1fr_1.3fr]">
          <div className="space-y-5">
            {groups.map(([group, actions]) => (
              <Card key={group}><CardHeader title={group} icon={<Wrench className="h-4 w-4" />} />
                <CardBody className="space-y-2">{actions.map((a) => {
                  const unsupported = os && !a[os];
                  return (
                    <div key={a.name} className="flex items-start gap-3 rounded-xl border border-line px-3 py-2">
                      <div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-2 text-sm font-medium">{a.label}
                        {a.approval_needed && <Badge tone="warn">approval</Badge>}{!a.windows && <Badge>Linux VM</Badge>}</div>
                        <div className="text-[11px] text-muted">{a.description}</div></div>
                      <Button size="sm" onClick={() => setOpen(a)} disabled={!!unsupported}><CirclePlay className="h-3.5 w-3.5" /> Run</Button>
                    </div>);
                })}</CardBody></Card>))}
          </div>
          <div className="space-y-5">
            {selected ? <LogView id={selected} /> : <Card className="p-6"><EmptyState title="No action yet" hint="Run one on the left: its live output appears here." /></Card>}
            <Card><CardHeader title="History" icon={<ListChecks className="h-4 w-4" />} />
              <CardBody className="max-h-96 space-y-1 overflow-auto">
                {(history ?? []).map((h) => (
                  <button key={h.id} onClick={() => select(h.id)} className={cn("flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-xs hover:bg-white/[0.04]", h.id === selected && "bg-brand/[0.08]")}>
                    <span className="min-w-0 flex-1 truncate font-medium">{h.label}<span className="ml-2 text-muted">{Object.entries(h.params).filter(([, v]) => v !== null && v !== false && !(Array.isArray(v) && !v.length)).map(([k, v]) => `${k}=${Array.isArray(v) ? v.join(",") : v}`).join(" ")}</span></span>
                    <StatusBadge status={h.status} /><span className="w-16 text-right text-muted">{fmt.ago(h.created)}</span>
                  </button>))}
                {!history?.length && <div className="text-xs text-muted">Nothing run yet.</div>}
              </CardBody></Card>
          </div>
        </div>)}
      {open && cat && <RunDialog action={open} cat={cat} onClose={() => setOpen(null)} onQueued={(a) => { qc.invalidateQueries({ queryKey: keys.opsActions }); select(a.id); }} />}
    </>
  );
}
