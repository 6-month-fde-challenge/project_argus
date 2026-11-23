import { AnimatePresence, motion } from "motion/react";
import { Activity, Cpu, ExternalLink } from "lucide-react";
import { NavLink, Outlet, useLocation } from "react-router";
import { useGpu, useServices } from "@/shared/api/queries";
import { cn, ratio } from "@/shared/lib/utils";
import { Tip } from "@/shared/ui/overlays";
import { FEATURES } from "./features";
import type { NavSection } from "./feature";

const SECTIONS: NavSection[] = ["Operate", "Models", "Serve", "Knowledge"];

function Sidebar() {
  const items = FEATURES.flatMap((f) => (f.nav ? [f.nav] : [])).sort((a, b) => a.order - b.order);
  return (
    <aside className="glass sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-line md:flex">
      <div className="flex items-center gap-3 px-5 py-5">
        <img src="/argus.svg" alt="" className="h-9 w-9" />
        <div>
          <div className="text-sm font-bold tracking-[0.2em] text-gradient">ARGUS</div>
          <div className="text-[10px] uppercase tracking-wider text-muted">Fraud intelligence LM</div>
        </div>
      </div>
      <nav className="flex-1 overflow-y-auto px-3 pb-4">
        {SECTIONS.map((section) => {
          const sectionItems = items.filter((i) => i.section === section);
          if (!sectionItems.length) return null;
          return (
            <div key={section} className="mb-4">
              <div className="px-3 pb-1.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-muted/70">{section}</div>
              {sectionItems.map(({ label, icon: Icon, path, end }) => (
                <NavLink key={path} to={path} end={end}
                  className={({ isActive }) => cn(
                    "group relative mb-0.5 flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition",
                    isActive ? "bg-white/[0.08] text-ink" : "text-muted hover:bg-white/[0.04] hover:text-ink",
                  )}>
                  {({ isActive }) => (
                    <>
                      {isActive && <motion.span layoutId="nav-active" className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-gradient-to-b from-brand to-brand-2" />}
                      <Icon className={cn("h-4 w-4", isActive && "text-brand")} />
                      {label}
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          );
        })}
      </nav>
      <div className="border-t border-line px-5 py-3 text-[10px] text-muted">VKPayments · Project ARGUS · local demo</div>
    </aside>
  );
}

function ServicePill({ name, status, href }: { name: string; status?: string; href?: string }) {
  const tone = status === "ok" ? "bg-ok" : status === "down" ? "bg-bad" : "bg-warn";
  const body = (
    <span className="flex items-center gap-1.5 rounded-full border border-line bg-white/[0.03] px-2.5 py-1 text-[11px] text-muted hover:text-ink">
      <span className={cn("h-1.5 w-1.5 rounded-full", tone, status === "ok" && "animate-pulse-soft")} />
      {name}
      {href && <ExternalLink className="h-3 w-3" />}
    </span>
  );
  return href ? <a href={href} target="_blank" rel="noreferrer">{body}</a> : body;
}

function TopBar() {
  const { data: svc } = useServices();
  const { data: gpu } = useGpu(3000);
  const g = gpu?.gpus?.[0];
  const memPct = ratio(g?.memory_used_mb, g?.memory_total_mb) ?? 0;
  return (
    <header className="sticky top-0 z-20 flex h-14 items-center justify-between gap-4 border-b border-line bg-bg/70 px-6 backdrop-blur-xl">
      <div className="flex items-center gap-2 text-xs text-muted">
        <Activity className="h-3.5 w-3.5 text-brand" />
        {svc?.gpu_jobs?.length ? <span>Training <b className="text-ink">{svc.gpu_jobs.join(", ")}</b></span> : <span>No training running</span>}
      </div>
      <div className="flex items-center gap-2">
        <ServicePill name="platform" status={svc ? "ok" : "down"} />
        <ServicePill name="scoring" status={svc?.scoring} />
        <ServicePill name="vLLM" status={svc?.explainer} />
        <ServicePill name="MLflow" status={svc?.mlflow} href={svc?.mlflow === "ok" ? svc.mlflow_url : undefined} />
        {svc?.stream === "ok" && <ServicePill name="stream" status="ok" />}
        {g && (
          <Tip content={`${g.name}: ${g.memory_used_mb} / ${g.memory_total_mb} MB, ${g.utilization_pct}% util, ${g.temperature_c}°C`}>
            <span className="ml-2 flex items-center gap-2 rounded-full border border-line bg-white/[0.03] px-2.5 py-1 text-[11px] text-muted">
              <Cpu className="h-3.5 w-3.5 text-brand-2" />
              <span className="relative h-1.5 w-16 overflow-hidden rounded-full bg-white/10">
                <span className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-brand to-brand-2 transition-all" style={{ width: `${memPct * 100}%` }} />
              </span>
              {g.utilization_pct ?? "–"}%
            </span>
          </Tip>
        )}
      </div>
    </header>
  );
}

export function AppShell() {
  const location = useLocation();
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar />
        <main className="mx-auto w-full max-w-[1600px] flex-1 px-6 py-6">
          <AnimatePresence mode="wait">
            <motion.div key={location.pathname.split("/")[1]} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }} transition={{ duration: 0.22, ease: "easeOut" }}>
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
