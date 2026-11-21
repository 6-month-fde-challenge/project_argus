import { motion } from "motion/react";
import { CheckCircle2, CircleDashed, Copy, Loader2, MinusCircle, XCircle } from "lucide-react";
import type { LadderRow, Progress } from "@/shared/api/types";
import { STAGE_BY_ID } from "@/shared/lib/stages";
import { cn, fmt } from "@/shared/lib/utils";
import { Badge, Progress as Bar } from "@/shared/ui/core";
import { Tip } from "@/shared/ui/overlays";

const ICON = {
  ok: <CheckCircle2 className="h-4 w-4 text-ok" />,
  failed: <XCircle className="h-4 w-4 text-bad" />,
  running: <Loader2 className="h-4 w-4 animate-spin text-warn" />,
  skipped: <MinusCircle className="h-4 w-4 text-muted" />,
  reused: <Copy className="h-4 w-4 text-brand-2" />,
  pending: <CircleDashed className="h-4 w-4 text-muted/60" />,
};

/** Vertical stage timeline: status, duration, peak GPU, live progress and last loss for every ladder stage. */
export function LadderTimeline({ rows, liveProgress, selected, onSelect }: {
  rows: LadderRow[]; liveProgress?: Record<string, Progress>; selected?: string; onSelect?: (stage: string) => void;
}) {
  return (
    <ol className="relative space-y-1">
      {rows.map((r, i) => {
        const meta = STAGE_BY_ID[r.name];
        const p = liveProgress?.[r.name] ?? r.progress;
        const running = r.status === "running";
        return (
          <motion.li key={r.name} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.02 }}>
            <button onClick={() => onSelect?.(r.name)}
              className={cn("group w-full rounded-xl border px-3 py-2.5 text-left transition",
                selected === r.name ? "border-brand/40 bg-brand/[0.06]" : "border-transparent hover:bg-white/[0.03]",
                running && "border-warn/30 bg-warn/[0.05]")}>
              <div className="flex items-center gap-3">
                <span className="grid h-7 w-9 shrink-0 place-items-center rounded-md bg-white/5 font-mono text-[11px] text-muted">{meta?.step ?? r.index}</span>
                {ICON[r.status as keyof typeof ICON] ?? ICON.pending}
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className={cn("truncate text-sm", r.status === "pending" || r.status === "skipped" ? "text-muted" : "text-ink")}>
                      {meta?.label ?? r.name}
                    </span>
                    {r.status === "reused" && <Badge tone="violet">reused</Badge>}
                    {r.status === "skipped" && <Badge>skipped</Badge>}
                  </div>
                  {running && p && (
                    <div className="mt-1.5 flex items-center gap-2">
                      <Bar value={p.total ? p.done / p.total : 0} tone="warn" className="flex-1" />
                      <span className="shrink-0 text-[10px] text-muted tabular-nums">{p.label} {p.done}/{p.total}</span>
                    </div>
                  )}
                  {running && r.last_step?.loss != null && (
                    <div className="mt-1 text-[10px] text-muted">step {r.last_step.step}/{r.last_step.max_steps} · loss <b className="text-ink tabular-nums">{fmt.num(r.last_step.loss, 4)}</b></div>
                  )}
                </div>
                <div className="flex shrink-0 items-center gap-3 text-[11px] text-muted tabular-nums">
                  {r.peak_gpu_gb != null && r.peak_gpu_gb > 0 && <Tip content="peak GPU memory"><span>{r.peak_gpu_gb.toFixed(2)} GB</span></Tip>}
                  <span className="w-12 text-right">{fmt.minutes(r.minutes)}</span>
                </div>
              </div>
              {r.error && <div className="mt-2 line-clamp-2 rounded-md bg-bad/10 px-2 py-1 font-mono text-[10px] text-bad">{r.error.split("\n").filter(Boolean).slice(-1)[0]}</div>}
            </button>
          </motion.li>
        );
      })}
    </ol>
  );
}
