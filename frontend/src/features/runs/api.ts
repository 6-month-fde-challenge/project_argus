import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";
import { api, errorMessage } from "@/shared/api/client";
import { keys } from "@/shared/api/queries";
import type { Job, LossPoint, Progress, RunDetail, TrainEvent } from "@/shared/api/types";
import { useEventStream } from "@/shared/api/useEventStream";

export interface SeriesResponse { series: Record<string, LossPoint[]>; offset: number }

export const useRun = (name: string, live = false) =>
  useQuery({ queryKey: keys.run(name), queryFn: () => api.get<RunDetail>(`/runs/${name}`), enabled: !!name, refetchInterval: live ? 10000 : 30000 });

export const useSeries = (name: string) =>
  useQuery({ queryKey: keys.series(name), queryFn: () => api.get<SeriesResponse>(`/runs/${name}/series`), staleTime: Infinity });

export const useArtifacts = (name: string) =>
  useQuery({ queryKey: keys.artifacts(name), queryFn: () => api.get<{ name: string; kind: string; files: number; size_mb: number }[]>(`/runs/${name}/artifacts`) });

export const useReportMd = (name: string, enabled: boolean) =>
  useQuery({ queryKey: keys.report(name), queryFn: () => api.text(`/runs/${name}/report.md`), enabled });

export const useTailEvents = (name: string) =>
  useQuery({ queryKey: ["runs", name, "tail"], queryFn: () => api.get<{ events: TrainEvent[] }>(`/runs/${name}/events?tail=150`), staleTime: Infinity });

export const useRunFile = (name: string, path: string, enabled = true) =>
  useQuery({ queryKey: ["runs", name, "file", path], queryFn: () => api.text(`/runs/${name}/files/${path}`), enabled, retry: false });

export function useCancelJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.post<Job>(`/jobs/${id}/cancel`),
    onSuccess: (j) => { toast.success(`Cancelled ${j.run_name}`); qc.invalidateQueries({ queryKey: keys.jobs }); qc.invalidateQueries({ queryKey: keys.runs }); },
    onError: (e) => toast.error(errorMessage(e)),
  });
}

/**
 * Live view of a run: subscribes to the run's SSE event stream from the offset the series was read at, appends
 * train_step points straight into the cached series, keeps per-stage progress, and refreshes the run detail
 * (throttled) on stage boundaries. One source of truth; no polling while training.
 */
export function useRunLive(name: string, offset: number | undefined, enabled: boolean) {
  const qc = useQueryClient();
  const [feed, setFeed] = useState<TrainEvent[]>([]);
  const [progress, setProgress] = useState<Record<string, Progress>>({});
  const lastRefresh = useRef(0);

  const refresh = useCallback(() => {
    const now = Date.now();
    if (now - lastRefresh.current < 1500) return;
    lastRefresh.current = now;
    qc.invalidateQueries({ queryKey: keys.run(name) });
    qc.invalidateQueries({ queryKey: keys.runs });
  }, [qc, name]);

  const onEvents = useCallback((batch: TrainEvent[]) => {
    const steps = batch.filter((e) => e.type === "train_step" && e.stage);
    if (steps.length) {
      qc.setQueryData<SeriesResponse>(keys.series(name), (old) => {
        const series = { ...(old?.series ?? {}) };
        for (const e of steps) {
          const d = e.data;
          series[e.stage!] = [...(series[e.stage!] ?? []), { step: d.step, loss: d.loss, learning_rate: d.learning_rate, epoch: d.epoch, eval: d.eval, ts: e.ts }];
        }
        return { series, offset: old?.offset ?? 0 };
      });
    }
    const prog = batch.filter((e) => e.type === "progress" && e.stage);
    if (prog.length) {
      setProgress((p) => {
        const next = { ...p };
        for (const e of prog) next[e.stage!] = { label: e.data.label, done: e.data.done, total: e.data.total, ts: e.ts };
        return next;
      });
    }
    if (batch.some((e) => ["stage_start", "stage_end", "stage_skipped", "run_end"].includes(e.type))) refresh();
    setFeed((f) => [...f, ...batch].slice(-400));
  }, [qc, name, refresh]);

  const state = useEventStream(enabled && offset != null ? `/runs/${name}/events/stream?after=${offset}` : null,
    { events: onEvents, end: () => refresh() });
  return { feed, progress, state };
}
