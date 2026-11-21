import { useEffect, useRef, useState } from "react";
import { Terminal } from "lucide-react";
import type { TrainEvent } from "@/shared/api/types";
import { useEventStream } from "@/shared/api/useEventStream";
import { stageLabel } from "@/shared/lib/stages";
import { cn } from "@/shared/lib/utils";
import { Badge } from "@/shared/ui/core";

const TYPE_TONE: Record<string, string> = {
  run_start: "text-brand", run_end: "text-brand", stage_start: "text-info", stage_end: "text-ok",
  stage_skipped: "text-muted", train_step: "text-ink/80", progress: "text-muted", metric: "text-brand-2",
};

function describe(e: TrainEvent): string {
  const d = e.data;
  switch (e.type) {
    case "run_start": return `run started · ${d.stages?.length} stages · ${d.profile}`;
    case "run_end": return `run ${d.status} in ${d.total_minutes} min`;
    case "stage_start": return `${stageLabel(e.stage!)} started (${d.index}/${d.total})`;
    case "stage_end": return `${stageLabel(e.stage!)} ${d.status} in ${d.minutes} min · peak GPU ${d.peak_gpu_gb} GB`;
    case "stage_skipped": return `${stageLabel(e.stage!)} skipped: ${d.reason}`;
    case "train_step": return `step ${d.step}/${d.max_steps}  loss ${d.loss?.toFixed?.(4) ?? d.loss}  lr ${d.learning_rate?.toExponential?.(2) ?? "–"}`;
    case "progress": return `${d.label} ${d.done}/${d.total}`;
    case "metric": return `${d.name} = ${typeof d.value === "number" ? d.value.toFixed(3) : d.value}`;
    default: return JSON.stringify(d);
  }
}

function useAutoScroll(dep: unknown) {
  const ref = useRef<HTMLDivElement>(null);
  const [stick, setStick] = useState(true);
  useEffect(() => { if (stick && ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [dep, stick]);
  const onScroll = () => {
    const el = ref.current;
    if (el) setStick(el.scrollHeight - el.scrollTop - el.clientHeight < 40);
  };
  return { ref, onScroll };
}

/** Structured training events (from events.jsonl): works for UI- and CLI-started runs alike. */
export function EventConsole({ events, live }: { events: TrainEvent[]; live?: boolean }) {
  const { ref, onScroll } = useAutoScroll(events.length);
  const shown = events.filter((e) => e.type !== "progress" || e.data.done === e.data.total);
  return (
    <div ref={ref} onScroll={onScroll} className="h-[420px] overflow-auto rounded-xl border border-line bg-black/40 p-3 font-mono text-[11px] leading-relaxed">
      {shown.length === 0 && <div className="text-muted">No events yet.</div>}
      {shown.map((e, i) => (
        <div key={i} className="flex gap-3">
          <span className="shrink-0 text-muted/60">{e.ts?.slice(11, 19)}</span>
          {e.stage && <span className="w-24 shrink-0 truncate text-brand-2/80">{e.stage}</span>}
          <span className={cn("min-w-0", TYPE_TONE[e.type] ?? "text-ink")}>{describe(e)}</span>
        </div>
      ))}
      {live && <div className="mt-1 text-warn animate-pulse-soft">▍ listening…</div>}
    </div>
  );
}

/** Raw container output of a platform job (stdout/stderr), streamed over SSE. */
export function JobLogConsole({ jobId }: { jobId: string }) {
  const [lines, setLines] = useState<string[]>([]);
  const state = useEventStream(`/jobs/${jobId}/logs/stream?tail=400`, {
    log: (d: { line: string }) => setLines((l) => [...l, d.line].slice(-2000)),
  });
  const { ref, onScroll } = useAutoScroll(lines.length);
  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-xs text-muted">
        <Terminal className="h-3.5 w-3.5" /> container logs <Badge tone={state === "open" ? "ok" : state === "ended" ? "neutral" : "warn"} dot={state === "open"}>{state}</Badge>
      </div>
      <div ref={ref} onScroll={onScroll} className="h-[420px] overflow-auto rounded-xl border border-line bg-black/50 p-3 font-mono text-[11px] leading-relaxed text-ink/85">
        {lines.map((l, i) => <div key={i} className="whitespace-pre-wrap break-all">{l}</div>)}
        {lines.length === 0 && <div className="text-muted">Waiting for output…</div>}
      </div>
    </div>
  );
}
