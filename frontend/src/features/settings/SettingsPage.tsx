import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { History, KeyRound, Lock, RotateCcw, Save, Settings2, Sparkles, TriangleAlert, Zap } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router";
import { toast } from "sonner";
import { api, errorMessage } from "@/shared/api/client";
import { keys } from "@/shared/api/queries";
import type { Approval, ChangeEntry, OpsAction, SettingItem, SettingsResponse } from "@/shared/api/types";
import { cn, fmt } from "@/shared/lib/utils";
import { Badge, Button, Card, CardBody, CardHeader, EmptyState, ErrorState, Input, Label, NativeSelect, PageHeader, Skeleton } from "@/shared/ui/core";
import { AgentNote, ApprovalFields, approvalReady, CommandLine, emptyApproval, ServingModeBadge } from "@/shared/ui/governance";
import { Tip } from "@/shared/ui/overlays";

const GENERATE = "@generate";
type Draft = Record<string, string | null>;   // key -> new value, null = remove (back to the default)

interface SaveResult { saved: string[]; restart: string[]; action: OpsAction | null; commands: { linux: string | null; windows: string | null }; message?: string }

function Field({ item, draft, setDraft, production }: { item: SettingItem; draft: Draft; setDraft: (d: Draft) => void; production: boolean }) {
  const has = item.key in draft;
  const value = has ? draft[item.key] : item.value;
  const set = (v: string | null) => setDraft({ ...draft, [item.key]: v });
  const undo = () => { const d = { ...draft }; delete d[item.key]; setDraft(d); };
  let input;
  if (item.locked) {
    input = <div className="flex items-center gap-2 text-xs text-muted"><Lock className="h-3.5 w-3.5" /><code className="text-ink">{item.value ?? item.default ?? "–"}</code></div>;
  } else if (item.secret) {
    const generating = draft[item.key] === GENERATE;
    input = (
      <div className="flex gap-2">
        <Input type="password" autoComplete="new-password" value={generating || !has || draft[item.key] === null ? "" : (draft[item.key] ?? "")}
          onChange={(e) => set(e.target.value)} placeholder={generating ? "a new random value will be generated" : item.is_set ? "set (enter a new value to replace it)" : "not set"} disabled={generating} />
        {item.type === "token" && <Button size="sm" variant={generating ? "primary" : "secondary"} onClick={() => (generating ? undo() : set(GENERATE))}><Sparkles className="h-3.5 w-3.5" /> {generating ? "will generate" : "generate"}</Button>}
      </div>
    );
  } else if (item.options.length) {
    input = (
      <NativeSelect value={value ?? ""} onChange={(e) => set(e.target.value || null)}>
        <option value="">(not set{item.default ? `: ${item.default}` : ""})</option>
        {item.options.map((o) => <option key={o} value={o}>{o}{item.option_modes?.[o] ? ` · ${item.option_modes[o]}` : ""}</option>)}
      </NativeSelect>
    );
  } else {
    input = <Input value={value ?? ""} onChange={(e) => set(e.target.value)} placeholder={item.default ? `default ${item.default}` : "not set"} inputMode={["port", "int", "float"].includes(item.type) ? "decimal" : undefined} />;
  }
  return (
    <div className={cn("grid gap-2 border-b border-line/60 py-3 last:border-0 lg:grid-cols-[1fr_1.2fr]", has && "bg-brand/[0.03]")}>
      <div>
        <div className="flex flex-wrap items-center gap-2"><Label className="mb-0">{item.label}</Label><code className="text-[10px] text-muted">{item.key}</code>
          {item.approval && production && <Badge tone="warn">approval</Badge>}
          {item.restart.length > 0 && <Tip content={`Applying restarts: ${item.restart.join(", ")}`}><span><Badge>restarts {item.restart.join(", ")}</Badge></span></Tip>}
        </div>
        <div className="mt-1 text-[11px] text-muted">{item.help}</div>
        {item.locked && <div className="mt-1 text-[11px] text-muted/80">Locked: {item.locked}</div>}
        {item.note && <div className="mt-1 text-[11px] text-info">{item.note}</div>}
        {item.warn && <div className="mt-1 flex items-center gap-1 text-[11px] text-bad"><TriangleAlert className="h-3 w-3" />{item.warn}</div>}
      </div>
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">{input}</div>
        {!item.locked && (has
          ? <Button size="icon" variant="ghost" onClick={undo} aria-label="undo"><RotateCcw className="h-4 w-4" /></Button>
          : item.is_set && <Tip content="Remove from .env (back to the default)"><Button size="icon" variant="ghost" onClick={() => set(null)} aria-label="reset"><RotateCcw className="h-4 w-4 opacity-50" /></Button></Tip>)}
      </div>
    </div>
  );
}

function ChangeLogCard() {
  const { data } = useQuery({ queryKey: keys.changes(), queryFn: () => api.get<ChangeEntry[]>("/settings/changes?limit=60"), refetchInterval: 10000 });
  return (
    <Card><CardHeader title="Change log" subtitle="Everything changed through the UI: settings, profiles, operations, activation, AWS (training/outputs/_platform/changes.jsonl)" icon={<History className="h-4 w-4" />} />
      <CardBody>{data?.length ? (
        <ol className="relative max-h-[28rem] space-y-3 overflow-auto border-l border-line pl-5">{data.map((c, i) => (
          <li key={i} className="relative"><span className="absolute -left-[25px] top-1 h-2.5 w-2.5 rounded-full bg-gradient-to-br from-brand to-brand-2" />
            <div className="text-xs"><Badge className="mr-2">{c.kind}</Badge><b>{c.summary}</b></div>
            <div className="text-[11px] text-muted">{c.requested_by ? `${c.requested_by}${c.approved_by ? ` → approved by ${c.approved_by}` : ""} · ${c.reason ?? ""} · ` : ""}{fmt.ago(c.at)}</div></li>))}</ol>
      ) : <EmptyState title="No changes yet" />}</CardBody></Card>
  );
}

export default function SettingsPage() {
  const qc = useQueryClient();
  const { data, error, isLoading } = useQuery({ queryKey: keys.settings, queryFn: () => api.get<SettingsResponse>("/settings"), refetchInterval: 15000 });
  const [draft, setDraft] = useState<Draft>({});
  const [approval, setApproval] = useState<Approval>(emptyApproval);
  const [result, setResult] = useState<SaveResult | null>(null);
  const items = useMemo(() => Object.fromEntries((data?.groups ?? []).flatMap((g) => g.items.map((i) => [i.key, i]))), [data]);
  const changed = Object.keys(draft).filter((k) => {
    const item = items[k];
    if (!item) return false;
    if (item.secret) return draft[k] === null ? item.is_set : !!draft[k];
    return (draft[k] ?? null) !== (item.value ?? null);
  });
  const restart = [...new Set(changed.flatMap((k) => items[k]?.restart ?? []))].sort();
  const production = !!data?.serving.production;
  const needsApproval = production && changed.some((k) => items[k]?.approval);
  const save = useMutation({
    mutationFn: (apply: boolean) => api.put<SaveResult>("/settings", {
      changes: Object.fromEntries(changed.map((k) => [k, draft[k] === "" ? null : draft[k]])), apply,
      approval: needsApproval ? approval : undefined,
    }),
    onSuccess: (r) => {
      setResult(r); setDraft({}); setApproval(emptyApproval);
      toast.success(r.saved.length ? `Saved ${r.saved.join(", ")}${r.action ? " · applying" : ""}` : r.message ?? "Nothing changed");
      qc.invalidateQueries({ queryKey: keys.settings }); qc.invalidateQueries({ queryKey: keys.changes() });
    },
    onError: (e) => toast.error(errorMessage(e)),
  });
  const applyNow = useMutation({
    mutationFn: (services: string[]) => api.post<OpsAction>("/settings/apply", { services }),
    onSuccess: (a) => { toast.success("Apply queued"); setResult((r) => (r ? { ...r, action: a } : r)); },
    onError: (e) => toast.error(errorMessage(e)),
  });
  const os = data?.agent.os ?? "linux";
  return (
    <>
      <PageHeader eyebrow="Operate" title="Settings"
        subtitle={<>The repository's <code>.env</code>: the same file docker compose, the GPU VM scripts, argus.ps1 and deploy/ops.py read. Change it here or by hand; both stay in step. Saving writes the file; applying restarts what reads it.</>}
        actions={<div className="flex items-center gap-2"><ServingModeBadge mode={data?.serving.mode} demoModel={data?.serving.active_run_forced} reason={data?.serving.active_run_forced_reason} /><AgentNote agent={data?.agent} /></div>} />
      {error ? <ErrorState error={error} /> : isLoading ? <Skeleton className="h-96" /> : data && (
        <div className="grid gap-5 2xl:grid-cols-[1.6fr_1fr]">
          <div className="space-y-5 pb-28">
            {data.serving.mismatch && (
              <div className="flex items-start gap-2 rounded-xl border border-warn/30 bg-warn/10 p-3 text-xs text-warn"><TriangleAlert className="h-4 w-4 shrink-0" />
                The running API is in {data.serving.live_mode} mode but .env says {data.serving.configured.mode} ({data.serving.configured.config}): apply the scoring service to align them.</div>)}
            {result && result.saved.length > 0 && (
              <Card className="border-brand/30 p-4">
                <div className="mb-2 text-sm font-medium">Saved {result.saved.join(", ")}</div>
                {result.restart.length ? (<>
                  <div className="mb-2 text-xs text-muted">These read the new values when they restart: <b className="text-ink">{result.restart.join(", ")}</b>.
                    {result.action ? <> Apply is queued (<Link className="text-brand hover:underline" to={`/operations?action=${result.action.id}`}>follow it</Link>).</> : " Apply now, or run the command yourself."}</div>
                  {!result.action && <Button size="sm" variant="primary" onClick={() => applyNow.mutate(result.restart)} loading={applyNow.isPending}><Zap className="h-3.5 w-3.5" /> Apply now</Button>}
                  <CommandLine className="mt-3" command={os === "windows" ? result.commands.windows : result.commands.linux} />
                </>) : <div className="text-xs text-muted">Nothing needs a restart: scripts and the platform read it on their next use.</div>}
              </Card>)}
            {data.groups.map((g) => (
              <Card key={g.name}><CardHeader title={g.name} icon={g.name.startsWith("Secrets") ? <KeyRound className="h-4 w-4" /> : <Settings2 className="h-4 w-4" />} />
                <CardBody className="py-1">{g.items.map((i) => <Field key={i.key} item={i} draft={draft} setDraft={setDraft} production={production} />)}</CardBody></Card>))}
            {data.unknown_keys.length > 0 && <div className="text-[11px] text-muted">Other keys in .env, kept as they are: {data.unknown_keys.join(", ")}</div>}
          </div>
          <div className="space-y-5"><ChangeLogCard /></div>
        </div>)}
      {changed.length > 0 && (
        <div className="fixed inset-x-0 bottom-0 z-30 border-t border-line bg-bg/90 px-6 py-3 backdrop-blur-xl md:left-60">
          <div className="mx-auto flex max-w-[1600px] flex-col gap-3">
            {needsApproval && <ApprovalFields value={approval} onChange={setApproval} why="these settings change what live traffic meets" />}
            <div className="flex flex-wrap items-center gap-3">
              <div className="text-xs text-muted"><b className="text-ink">{changed.length}</b> change{changed.length > 1 && "s"}: {changed.join(", ")}{restart.length > 0 && <> · restarts <b className="text-ink">{restart.join(", ")}</b></>}</div>
              <div className="ml-auto flex gap-2">
                <Button variant="ghost" onClick={() => setDraft({})}>Discard</Button>
                <Button onClick={() => save.mutate(false)} loading={save.isPending && save.variables === false} disabled={needsApproval && !approvalReady(approval)}><Save className="h-4 w-4" /> Save</Button>
                {restart.length > 0 && <Button variant="primary" onClick={() => save.mutate(true)} loading={save.isPending && save.variables === true} disabled={needsApproval && !approvalReady(approval)}><Zap className="h-4 w-4" /> Save and apply</Button>}
              </div>
            </div>
          </div>
        </div>)}
    </>
  );
}
