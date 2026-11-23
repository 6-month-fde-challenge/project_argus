import { Background, Handle, MarkerType, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { motion } from "motion/react";
import { ArrowRight, BookOpen } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router";
import { useRuns } from "@/shared/api/queries";
import type { LadderRow } from "@/shared/api/types";
import { STAGE_BY_ID, STAGES, type StageMeta } from "@/shared/lib/stages";
import { cn, fmt } from "@/shared/lib/utils";
import { Badge, Button, Card, CardBody, CardHeader, NativeSelect, PageHeader, StatusBadge } from "@/shared/ui/core";
import { useRun } from "@/features/runs/api";

const GROUP_COLOR: Record<StageMeta["group"], string> = {
  foundation: "from-sky-500/25 to-sky-500/5 border-sky-400/40",
  decide: "from-violet-500/25 to-violet-500/5 border-violet-400/40",
  adapt: "from-fuchsia-500/20 to-fuchsia-500/5 border-fuchsia-400/40",
  compress: "from-cyan-500/20 to-cyan-500/5 border-cyan-400/40",
  ship: "from-emerald-500/20 to-emerald-500/5 border-emerald-400/40",
};
const STATUS_RING: Record<string, string> = {
  ok: "ring-2 ring-ok/70", running: "ring-2 ring-warn animate-pulse-soft", failed: "ring-2 ring-bad/80", skipped: "opacity-40", reused: "ring-2 ring-brand-2/60",
};

type StageData = { meta: StageMeta; row?: LadderRow; selected: boolean };

function StageNode({ data }: NodeProps<Node<StageData>>) {
  const { meta, row, selected } = data;
  return (
    <div className={cn("w-[190px] rounded-2xl border bg-gradient-to-br px-4 py-3 shadow-xl backdrop-blur transition", GROUP_COLOR[meta.group],
      row && STATUS_RING[row.status], selected && "scale-[1.04] shadow-brand/30")}>
      <Handle type="target" position={Position.Left} id="l" className="!bg-transparent !border-0" />
      <Handle type="target" position={Position.Top} id="t" className="!bg-transparent !border-0" />
      <Handle type="source" position={Position.Right} id="r" className="!bg-transparent !border-0" />
      <Handle type="source" position={Position.Bottom} id="b" className="!bg-transparent !border-0" />
      <Handle type="target" position={Position.Right} id="rt" className="!bg-transparent !border-0" />
      <Handle type="source" position={Position.Left} id="ls" className="!bg-transparent !border-0" />
      <div className="flex items-center justify-between"><span className="font-mono text-[11px] text-ink/70">{meta.step}</span>{row && <StatusBadge status={row.status} />}</div>
      <div className="mt-1 text-[13px] font-semibold leading-tight text-ink">{meta.label}</div>
      {row?.minutes != null && <div className="mt-1 text-[10px] text-ink/60">{fmt.minutes(row.minutes)}{row.peak_gpu_gb ? ` · ${row.peak_gpu_gb.toFixed(1)} GB` : ""}</div>}
    </div>
  );
}

function GateNode({ data }: NodeProps<Node<StageData & { decision?: string; uplift?: number }>>) {
  return (
    <div className={cn("relative grid h-[130px] w-[130px] place-items-center", data.selected && "scale-105")}>
      <div className={cn("absolute inset-3 rotate-45 rounded-2xl border border-amber-400/60 bg-gradient-to-br from-amber-400/25 to-amber-400/5 shadow-xl",
        data.row && STATUS_RING[data.row.status])} />
      <Handle type="target" position={Position.Left} id="l" className="!bg-transparent !border-0" />
      <Handle type="source" position={Position.Right} id="r" className="!bg-transparent !border-0" />
      <Handle type="source" position={Position.Bottom} id="b" className="!bg-transparent !border-0" />
      <div className="relative text-center"><div className="text-[13px] font-semibold">Week 8</div><div className="text-[10px] text-ink/70">+0.03 PR-AUC?</div>
        {data.decision && <div className="mt-1"><Badge tone={data.decision === "LLM" ? "brand" : "violet"}>{data.decision}</Badge></div>}</div>
    </div>
  );
}

function HybridNode() {
  return (
    <div className="w-[170px] rounded-2xl border border-stone-400/40 bg-gradient-to-br from-stone-400/20 to-stone-400/5 px-4 py-3">
      <Handle type="target" position={Position.Left} id="l" className="!bg-transparent !border-0" />
      <div className="text-[13px] font-semibold">Hybrid</div><div className="text-[10px] text-ink/70">XGBoost scores, LLM explains</div>
    </div>
  );
}

const NODE_TYPES = { stage: StageNode, gate: GateNode, hybrid: HybridNode };
const X = [0, 250, 500, 760, 960];
const POS: Record<string, [number, number]> = {
  data_checks: [X[0], 0], xgboost: [X[1], 0], pilot: [X[2], 0], gate: [X[3], -25], hybrid: [1010, 20],
  cpt: [X[3] - 30, 220], sft_scorer: [X[2], 170], sft_explainer: [X[2], 275], dpo: [X[1], 220],
  distill_scorer: [X[0], 420], distill_explainer: [X[0], 525], quantize: [X[1], 470], calibrate: [X[2], 470], acceptance: [X[3] - 30, 470], registry: [X[4] + 30, 470],
};
const EDGES: [string, string, string, string, string?][] = [
  ["data_checks", "xgboost", "r", "l"], ["xgboost", "pilot", "r", "l"], ["pilot", "gate", "r", "l"],
  ["gate", "hybrid", "r", "l", "no"], ["gate", "cpt", "b", "t", "yes"],
  ["cpt", "sft_scorer", "ls", "rt"], ["cpt", "sft_explainer", "ls", "rt"],
  ["sft_scorer", "dpo", "ls", "rt"], ["sft_explainer", "dpo", "ls", "rt"],
  ["dpo", "distill_scorer", "b", "t"], ["dpo", "distill_explainer", "b", "t"],
  ["distill_scorer", "quantize", "r", "l"], ["distill_explainer", "quantize", "r", "l"],
  ["quantize", "calibrate", "r", "l"], ["calibrate", "acceptance", "r", "l"], ["acceptance", "registry", "r", "l"],
];

export default function PipelinePage() {
  const { data: runs } = useRuns();
  const [runName, setRunName] = useState<string>("");
  const active = runName || runs?.[0]?.name || "";
  const { data: run } = useRun(active, !!active);
  const [selected, setSelected] = useState<string>("pilot");
  const rows = useMemo(() => Object.fromEntries((run?.ladder ?? []).map((r) => [r.name, r])), [run]);
  const gate = run?.sections?.gate;

  const nodes: Node[] = useMemo(() => [
    ...STAGES.map((meta) => ({
      id: meta.id, type: meta.id === "gate" ? "gate" : "stage", position: { x: POS[meta.id][0], y: POS[meta.id][1] },
      data: { meta, row: rows[meta.id], selected: selected === meta.id, decision: gate?.decision, uplift: gate?.uplift },
    })),
    { id: "hybrid", type: "hybrid", position: { x: POS.hybrid[0], y: POS.hybrid[1] }, data: {} },
  ], [rows, selected, gate]);

  const edges: Edge[] = useMemo(() => EDGES.map(([s, t, sh, th, label]) => {
    const done = rows[s]?.status === "ok" || rows[s]?.status === "reused";
    const taken = label ? (gate?.branch ?? gate?.decision) === (label === "no" ? "HYBRID" : "LLM") : true;
    return {
      id: `${s}-${t}`, source: s, target: t, sourceHandle: sh, targetHandle: th, label, animated: done && taken,
      style: { stroke: label === "no" ? "#fb7185" : done ? "#22d3ee" : "rgba(148,163,184,0.35)", strokeWidth: done ? 2.2 : 1.5, opacity: taken ? 1 : 0.5 },
      labelStyle: { fill: "#e6ebf5", fontSize: 11 }, labelBgStyle: { fill: "#0c1222" },
      markerEnd: { type: MarkerType.ArrowClosed, color: label === "no" ? "#fb7185" : done ? "#22d3ee" : "#64748b" },
      type: "smoothstep",
    };
  }), [rows, gate]);

  const meta = STAGE_BY_ID[selected];
  const row = rows[selected];
  return (
    <>
      <PageHeader eyebrow="Part 3 · Figure 4" title="Earn the LLM: baseline first, then four stages"
        subtitle="The ARGUS-LM model ladder as it runs in this platform. Pick a run to light up each stage with its live status; click a stage for what it does and where the client documents require it."
        actions={<NativeSelect value={active} onChange={(e) => setRunName(e.target.value)} className="w-64">{runs?.map((r) => <option key={r.name} value={r.name}>{r.name} ({r.status})</option>)}</NativeSelect>} />
      <div className="grid gap-5 xl:grid-cols-[1fr_360px]">
        <Card className="grid-bg h-[680px] overflow-hidden">
          <ReactFlow nodes={nodes} edges={edges} nodeTypes={NODE_TYPES} fitView fitViewOptions={{ padding: 0.15 }} proOptions={{ hideAttribution: true }}
            nodesDraggable={false} nodesConnectable={false} onNodeClick={(_, n) => n.id !== "hybrid" && setSelected(n.id)} minZoom={0.4} maxZoom={1.6}>
            <Background color="rgba(148,163,184,0.08)" gap={28} />
          </ReactFlow>
        </Card>
        <motion.div key={selected} initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }}>
          <Card className="h-full">
            <CardHeader title={<span>{meta?.step} · {meta?.label}</span>} subtitle={<span className="flex items-center gap-1"><BookOpen className="h-3 w-3" />{meta?.doc}</span>} />
            <CardBody className="space-y-4">
              <p className="text-sm leading-relaxed text-ink/85">{meta?.what}</p>
              <div className="flex flex-wrap gap-2">
                <Badge tone="info">{meta?.group}</Badge>
                <Badge tone={meta?.branch === "llm-scorer" ? "brand" : meta?.branch === "explainer" ? "violet" : "neutral"}>
                  {meta?.branch === "both" ? "both branches" : meta?.branch === "llm-scorer" ? "LLM-scorer branch only" : "explainer branch"}</Badge>
              </div>
              {row && <div className="rounded-xl border border-line bg-white/[0.02] p-3 text-xs">
                <div className="mb-2 flex items-center justify-between"><span className="text-muted">in {run?.name}</span><StatusBadge status={row.status} /></div>
                <div className="grid grid-cols-2 gap-2"><span className="text-muted">duration</span><span className="text-right tabular-nums">{fmt.minutes(row.minutes)}</span>
                  <span className="text-muted">peak GPU</span><span className="text-right tabular-nums">{row.peak_gpu_gb != null ? `${row.peak_gpu_gb.toFixed(2)} GB` : "–"}</span></div>
              </div>}
              {selected === "gate" && gate && <div className="rounded-xl border border-amber-400/30 bg-amber-400/10 p-3 text-xs">
                Pilot LLM <b>{fmt.num(gate.llm_pr_auc)}</b> vs XGBoost <b>{fmt.num(gate.xgboost_pr_auc)}</b> → uplift {fmt.num(gate.uplift)} (needs +{gate.required_uplift}) → <b>{gate.decision}</b>{gate.mode && gate.mode !== "auto" && <>; branch forced to <b>{gate.branch}</b> (gate.mode={gate.mode})</>}
              </div>}
              {run && <Button asChild variant="primary" className="w-full"><Link to={`/runs/${run.name}`}>Open run results <ArrowRight className="h-4 w-4" /></Link></Button>}
            </CardBody>
          </Card>
        </motion.div>
      </div>
    </>
  );
}
