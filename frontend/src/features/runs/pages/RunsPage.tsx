import { motion } from "motion/react";
import { Plus, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";
import { useRuns } from "@/shared/api/queries";
import type { RunSummary } from "@/shared/api/types";
import { stageLabel } from "@/shared/lib/stages";
import { fmt } from "@/shared/lib/utils";
import { Badge, Button, Card, EmptyState, ErrorState, Input, PageHeader, Progress, Skeleton, StatusBadge } from "@/shared/ui/core";

const FILTERS = ["all", "running", "succeeded", "failed", "incomplete"] as const;

export function RunRow({ r, i }: { r: RunSummary; i: number }) {
  const nav = useNavigate();
  const pct = r.stages_total ? r.stages_done / r.stages_total : 0;
  return (
    <motion.tr initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.03 }}
      onClick={() => nav(`/runs/${r.name}`)} className="cursor-pointer border-b border-line/60 transition hover:bg-white/[0.03]">
      <td className="px-4 py-3">
        <div className="font-medium">{r.name}</div>
        <div className="text-[11px] text-muted">{r.profile?.replace("configs/", "")}{r.reused_from && ` · reuses ${r.reused_from}`}</div>
      </td>
      <td className="px-4 py-3"><StatusBadge status={r.status} /></td>
      <td className="w-64 px-4 py-3">
        <div className="flex items-center gap-2">
          <Progress value={pct} tone={r.status === "failed" ? "bad" : r.status === "running" ? "warn" : "ok"} />
          <span className="shrink-0 text-[11px] text-muted tabular-nums">{r.stages_done}/{r.stages_total}</span>
        </div>
        {r.current_stage && <div className="mt-1 truncate text-[11px] text-warn">{stageLabel(r.current_stage)}{r.progress && ` · ${r.progress.label} ${r.progress.done}/${r.progress.total}`}</div>}
      </td>
      <td className="px-4 py-3">{r.gate ? (
        <div className="flex flex-col items-start gap-1">
          <Badge tone={r.gate.decision === "LLM" ? "brand" : "violet"}>measured {r.gate.decision} {r.gate.uplift >= 0 ? "+" : ""}{fmt.num(r.gate.uplift)}</Badge>
          {r.gate.forced
            ? <Badge tone="warn">ran {r.gate.branch ?? r.gate.decision} · demo (forced)</Badge>
            : <Badge tone="ok">ran {r.gate.branch ?? r.gate.decision} · as measured</Badge>}
        </div>) : "–"}</td>
      <td className="px-4 py-3">{r.scorer ? <StatusBadge status={r.scorer} /> : "–"}</td>
      <td className="px-4 py-3 text-xs">{r.pilot_winner ? `${fmt.model(r.pilot_winner.teacher)} · ${r.pilot_winner.format}` : "–"}</td>
      <td className="px-4 py-3 text-right text-xs tabular-nums text-muted">{fmt.minutes(r.total_minutes)}</td>
      <td className="px-4 py-3 text-right text-xs text-muted">{fmt.ago(r.updated_at)}</td>
    </motion.tr>
  );
}

export default function RunsPage() {
  const { data, isLoading, error, refetch } = useRuns();
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>("all");
  const rows = useMemo(() => (data ?? []).filter((r) =>
    (filter === "all" || r.status === filter) && (!q || r.name.includes(q.toLowerCase()) || r.profile?.includes(q))), [data, q, filter]);

  return (
    <>
      <PageHeader eyebrow="Operate" title="Training runs" subtitle="Every run of the ARGUS-LM ladder, started from this UI or from the command line. Live runs update in place."
        actions={<Button asChild variant="primary"><Link to="/launch"><Plus className="h-4 w-4" /> New run</Link></Button>} />
      <Card>
        <div className="flex flex-wrap items-center gap-3 border-b border-line p-4">
          <div className="relative w-64"><Search className="absolute left-3 top-2.5 h-4 w-4 text-muted" /><Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search runs…" className="pl-9" /></div>
          <div className="flex gap-1">{FILTERS.map((f) => (
            <Button key={f} size="sm" variant={filter === f ? "secondary" : "ghost"} onClick={() => setFilter(f)}>{f}</Button>))}</div>
        </div>
        {error ? <div className="p-4"><ErrorState error={error} retry={refetch} /></div>
          : isLoading ? <div className="space-y-3 p-4">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-10" />)}</div>
          : rows.length === 0 ? <EmptyState title="No runs yet" hint="Launch the smoke profile to see the whole ladder run end to end in about 20 minutes."
              action={<Button asChild variant="primary"><Link to="/launch">Launch a run</Link></Button>} />
          : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead><tr className="text-left text-[11px] uppercase tracking-wider text-muted">
                  {["run", "status", "progress", "week-8 gate", "scorer", "pilot winner", "duration", "updated"].map((h) =>
                    <th key={h} className={`px-4 py-2.5 font-medium ${["duration", "updated"].includes(h) ? "text-right" : ""}`}>{h}</th>)}
                </tr></thead>
                <tbody>{rows.map((r, i) => <RunRow key={r.name} r={r} i={i} />)}</tbody>
              </table>
            </div>
          )}
      </Card>
    </>
  );
}
