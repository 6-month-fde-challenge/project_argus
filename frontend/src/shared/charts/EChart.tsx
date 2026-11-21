/** ECharts wrapper: tree-shaken modules, resize-aware, merges option updates (smooth live charts). */
import { BarChart, GaugeChart, HeatmapChart, LineChart, PieChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent, GridComponent, LegendComponent, MarkLineComponent, TitleComponent, TooltipComponent, VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useRef } from "react";
import { cn } from "@/shared/lib/utils";

echarts.use([LineChart, BarChart, GaugeChart, HeatmapChart, PieChart, ScatterChart, GridComponent, TooltipComponent, LegendComponent,
  DataZoomComponent, VisualMapComponent, MarkLineComponent, TitleComponent, CanvasRenderer]);

export type EChartsOption = echarts.EChartsCoreOption;

export const PALETTE = ["#22d3ee", "#8b5cf6", "#34d399", "#fbbf24", "#fb7185", "#60a5fa", "#f472b6", "#a3e635"];

export const axisStyle = {
  axisLine: { lineStyle: { color: "rgba(148,163,184,0.25)" } },
  axisLabel: { color: "#8a96ad", fontSize: 11 },
  splitLine: { lineStyle: { color: "rgba(148,163,184,0.08)" } },
  nameTextStyle: { color: "#8a96ad", fontSize: 11 },
};

export const baseOption = (): EChartsOption => ({
  backgroundColor: "transparent",
  color: PALETTE,
  textStyle: { fontFamily: "Inter, Segoe UI, system-ui, sans-serif", color: "#e6ebf5" },
  grid: { left: 48, right: 20, top: 30, bottom: 36, containLabel: false },
  tooltip: {
    trigger: "axis", backgroundColor: "rgba(12,18,34,0.95)", borderColor: "rgba(148,163,184,0.2)",
    textStyle: { color: "#e6ebf5", fontSize: 12 },
  },
  legend: { textStyle: { color: "#8a96ad", fontSize: 11 }, top: 0, right: 0, icon: "roundRect", itemWidth: 10, itemHeight: 6 },
  animationDuration: 600,
  animationDurationUpdate: 300,
});

export function EChart({ option, height = 280, className, onClick }: {
  option: EChartsOption; height?: number | string; className?: string; onClick?: (params: any) => void;
}) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!el.current) return;
    const c = echarts.init(el.current, undefined, { renderer: "canvas" });
    chart.current = c;
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(el.current);
    return () => { ro.disconnect(); c.dispose(); chart.current = null; };
  }, []);

  useEffect(() => { chart.current?.setOption(option, { notMerge: false, lazyUpdate: true }); }, [option]);

  useEffect(() => {
    const c = chart.current;
    if (!c || !onClick) return;
    c.on("click", onClick);
    return () => { c.off("click", onClick); };
  }, [onClick]);

  return <div ref={el} className={cn("w-full", className)} style={{ height }} />;
}
