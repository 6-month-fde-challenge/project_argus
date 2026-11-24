/** Shared pieces of the governance UI: two-person approval, the serving-mode badge, "run it yourself" commands. */
import { Check, Copy, ShieldAlert, ShieldCheck, Terminal } from "lucide-react";
import { useState } from "react";
import { ApiError } from "@/shared/api/client";
import type { AgentStatus, Approval } from "@/shared/api/types";
import { cn } from "@/shared/lib/utils";
import { Badge, Input, Label } from "./core";
import { Tip } from "./overlays";

export const emptyApproval: Approval = { requested_by: "", approved_by: "", reason: "" };

/** One spelling per person (trimmed, spaces collapsed, case-folded): the platform applies the same rule. */
const person = (name: string) => name.trim().split(/\s+/).join(" ").toLowerCase();

export const approvalReady = (a: Approval) =>
  !!(a.requested_by.trim() && a.approved_by.trim() && a.reason.trim()) && person(a.requested_by) !== person(a.approved_by);

/** The platform answers 428 APPROVAL_REQUIRED when a change needs two people in production mode. */
export const isApprovalError = (e: unknown) => e instanceof ApiError && e.status === 428;

export function ApprovalFields({ value, onChange, why, className }: {
  value: Approval; onChange: (a: Approval) => void; why?: string; className?: string;
}) {
  const same = value.requested_by.trim() && value.approved_by.trim() && person(value.requested_by) === person(value.approved_by);
  return (
    <div className={cn("rounded-xl border border-warn/30 bg-warn/[0.06] p-3", className)}>
      <div className="mb-2 flex items-center gap-2 text-xs text-warn"><ShieldAlert className="h-3.5 w-3.5" />
        Two-person approval{why ? `: ${why}` : ""}</div>
      <div className="grid gap-2 sm:grid-cols-3">
        <div><Label>requested by</Label><Input value={value.requested_by} onChange={(e) => onChange({ ...value, requested_by: e.target.value })} placeholder="your name" /></div>
        <div><Label>approved by (another person)</Label><Input value={value.approved_by} onChange={(e) => onChange({ ...value, approved_by: e.target.value })} placeholder="approver" /></div>
        <div><Label>reason</Label><Input value={value.reason} onChange={(e) => onChange({ ...value, reason: e.target.value })} placeholder="why" /></div>
      </div>
      {same && <div className="mt-2 text-[11px] text-bad">The approver must be a different person.</div>}
      <div className="mt-2 text-[11px] text-muted">Recorded in the change log. This UI has no login: access is the SSH tunnel, the names are the audit trail.</div>
    </div>
  );
}

export function ServingModeBadge({ mode, demoModel, reason }: { mode?: string | null; demoModel?: boolean; reason?: string | null }) {
  if (!mode || mode === "unknown") return <Badge>serving mode unknown</Badge>;
  if (mode === "production" && !demoModel)
    return <Tip content="The Fraud Model API refuses demo (forced / unmeasured) runs: safe behind live traffic."><span><Badge tone="ok"><ShieldCheck className="h-3 w-3" /> production</Badge></span></Tip>;
  return (
    <Tip content={reason ?? (mode === "demo" ? "Demo mode: the API also serves demo runs. Fine for demonstrations, never behind live traffic." : "The live run is a demo run.")}>
      <span><Badge tone="warn"><ShieldAlert className="h-3 w-3" /> {mode === "demo" ? "demo mode" : "demo model live"}</Badge></span>
    </Tip>
  );
}

/** The same thing without the UI: what the agent runs, ready to copy. */
export function CommandLine({ command, label = "Run it yourself", className }: { command: string | null | undefined; label?: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  if (!command) return null;
  const copy = async () => {
    try { await navigator.clipboard.writeText(command); setCopied(true); setTimeout(() => setCopied(false), 1500); } catch { /* clipboard blocked */ }
  };
  return (
    <div className={cn("group flex items-start gap-2 rounded-lg border border-line bg-black/30 px-3 py-2 font-mono text-[11px] text-muted", className)}>
      <Terminal className="mt-0.5 h-3.5 w-3.5 shrink-0 text-brand" />
      <div className="min-w-0 flex-1"><div className="mb-0.5 font-sans text-[10px] uppercase tracking-wider text-muted/70">{label}</div>
        <code className="break-all text-ink/90">{command}</code></div>
      <button type="button" onClick={copy} aria-label="copy command" className="shrink-0 text-muted hover:text-ink">
        {copied ? <Check className="h-3.5 w-3.5 text-ok" /> : <Copy className="h-3.5 w-3.5" />}
      </button>
    </div>
  );
}

/** Whether the operations agent (deploy/agent.py) is running on the host, and how to start it. */
export function AgentNote({ agent }: { agent?: AgentStatus }) {
  if (agent?.online) return <Badge tone="ok" dot>agent on {agent.host} ({agent.os})</Badge>;
  return (
    <Tip content={<>The operations agent runs what this page and Operations ask for on the host. Start it:<br />GPU VM: <code>sudo systemctl start argus-agent</code> (setup_vm.sh installs it)<br />Windows: <code>.\argus.ps1 agent start</code><br />Requests wait in the queue until it runs; you can also run the shown commands yourself.</>}>
      <span><Badge tone="warn">agent offline</Badge></span>
    </Tip>
  );
}
