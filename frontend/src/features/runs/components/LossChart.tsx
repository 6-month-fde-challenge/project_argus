import { useMemo } from "react";
import type { LossPoint } from "@/shared/api/types";
import { axisStyle, baseOption, EChart, PALETTE } from "@/shared/charts/EChart";
import { stageLabel } from "@/shared/lib/stages";
import { EmptyState } from "@/shared/ui/core";

/** Training loss (and learning rate) for one or more stages; updates in place as live points arrive. */
export function LossChart({ series, stages, height = 300, showLr = true }: {
  series: Record<string, LossPoint[]>; stages: string[]; height?: number; showLr?: boolean;
}) {
  const option = useMemo(() => {
    const visible = stages.filter((s) => series[s]?.length);
    const lossSeries = visible.map((s, i) => ({
      name: `${stageLabel(s)} loss`, type: "line", smooth: true, showSymbol: false, yAxisIndex: 0,
      lineStyle: { width: 2.2 }, color: PALETTE[i % PALETTE.length],
      areaStyle: visible.length === 1 ? { opacity: 0.12 } : undefined,
      data: series[s].filter((p) => !p.eval && p.loss != null).map((p) => [p.step, p.loss]),
    }));
    const lrSeries = showLr && visible.length === 1 ? [{
      name: "learning rate", type: "line", smooth: true, showSymbol: false, yAxisIndex: 1,
      lineStyle: { width: 1.2, type: "dashed" }, color: "#8a96ad",
      data: series[visible[0]].filter((p) => p.learning_rate != null).map((p) => [p.step, p.learning_rate]),
    }] : [];
    return {
      ...baseOption(),
      grid: { left: 52, right: lrSeries.length ? 64 : 20, top: 34, bottom: 40 },
      xAxis: { type: "value", name: "step", nameLocation: "middle", nameGap: 26, ...axisStyle, splitLine: { show: false } },
      yAxis: [
        { type: "value", name: "loss", scale: true, ...axisStyle },
        ...(lrSeries.length ? [{ type: "value", name: "lr", scale: true, ...axisStyle, splitLine: { show: false },
          axisLabel: { ...axisStyle.axisLabel, formatter: (v: number) => v.toExponential(0) } }] : []),
      ],
      dataZoom: [{ type: "inside" }],
      series: [...lossSeries, ...lrSeries],
    };
  }, [series, stages, showLr]);

  if (!stages.some((s) => series[s]?.length)) {
    return <EmptyState title="No training steps yet" hint="Loss curves appear here as soon as a training stage logs its first steps." />;
  }
  return <EChart option={option} height={height} />;
}
