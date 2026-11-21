import { useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";
import { FileText, Image, Maximize2 } from "lucide-react";
import { useState } from "react";
import { api, API_BASE } from "@/shared/api/client";
import { keys } from "@/shared/api/queries";
import { cn } from "@/shared/lib/utils";
import { Card, CardBody, EmptyState, ErrorState, PageHeader, Skeleton } from "@/shared/ui/core";
import { Dialog, Tabs } from "@/shared/ui/overlays";

interface Doc { name: string; title: string; size_kb: number }
interface Diagram { name: string; title: string; media_type: string }

function Documents() {
  const list = useQuery({ queryKey: keys.documents, queryFn: () => api.get<Doc[]>("/docs/documents") });
  const [active, setActive] = useState<string>();
  const current = active ?? list.data?.[0]?.name;
  const html = useQuery({ queryKey: keys.document(current ?? ""), queryFn: () => api.text(`/docs/documents/${encodeURIComponent(current!)}`), enabled: !!current });
  if (list.error) return <ErrorState error={list.error} />;
  if (!list.data?.length) return <EmptyState title="No client documents mounted" />;
  return (
    <div className="grid gap-5 xl:grid-cols-[300px_1fr]">
      <Card><CardBody className="space-y-1 pt-4">{list.data.map((d) => (
        <button key={d.name} onClick={() => setActive(d.name)} className={cn("flex w-full items-start gap-2 rounded-xl px-3 py-2 text-left text-xs transition",
          current === d.name ? "bg-brand/10 text-ink" : "text-muted hover:bg-white/[0.04] hover:text-ink")}>
          <FileText className="mt-0.5 h-3.5 w-3.5 shrink-0" /><span>{d.title}</span></button>))}</CardBody></Card>
      <Card><CardBody className="pt-5">{html.isLoading ? <Skeleton className="h-96" />
        : <motion.div key={current} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="prose-argus max-h-[75vh] overflow-auto pr-2" dangerouslySetInnerHTML={{ __html: html.data ?? "" }} />}
      </CardBody></Card>
    </div>
  );
}

function Diagrams() {
  const { data, error } = useQuery({ queryKey: keys.diagrams, queryFn: () => api.get<Diagram[]>("/docs/diagrams") });
  const [open, setOpen] = useState<Diagram | null>(null);
  if (error) return <ErrorState error={error} />;
  if (!data?.length) return <EmptyState title="No diagrams mounted" />;
  return (
    <>
      <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">{data.map((d, i) => (
        <motion.button key={d.name} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }} whileHover={{ y: -3 }}
          onClick={() => setOpen(d)} className="glass group overflow-hidden rounded-2xl text-left">
          <div className="h-52 overflow-hidden bg-white p-3"><img src={`${API_BASE}/docs/diagrams/${d.name}`} alt={d.title} className="h-full w-full object-contain" loading="lazy" /></div>
          <div className="flex items-center justify-between px-4 py-3 text-sm"><span>{d.title}</span><Maximize2 className="h-4 w-4 text-muted group-hover:text-brand" /></div>
        </motion.button>))}</div>
      <Dialog open={!!open} onOpenChange={(o) => !o && setOpen(null)} title={open?.title} wide>
        {open && <div className="rounded-xl bg-white p-4"><img src={`${API_BASE}/docs/diagrams/${open.name}`} alt={open.title} className="mx-auto max-h-[70vh]" /></div>}
      </Dialog>
    </>
  );
}

export default function DocsPage() {
  return (
    <>
      <PageHeader eyebrow="Knowledge" title="Client documents & diagrams" subtitle="The CRD, architecture, capacity plan, HLD and LLD, rendered from the client's own .docx files, plus the solution diagrams from Part 3." />
      <Tabs tabs={[
        { id: "diagrams", label: <span className="flex items-center gap-1.5"><Image className="h-3.5 w-3.5" /> Diagrams</span>, content: <Diagrams /> },
        { id: "documents", label: <span className="flex items-center gap-1.5"><FileText className="h-3.5 w-3.5" /> Documents</span>, content: <Documents /> },
      ]} />
    </>
  );
}
