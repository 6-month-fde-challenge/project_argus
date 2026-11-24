import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "motion/react";
import { Eye, Rocket, Save, Square, Terminal } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { toast } from "sonner";
import { api, errorMessage } from "@/shared/api/client";
import { keys, useJobs, useRuns, useServices } from "@/shared/api/queries";
import type { Job, ProfileDetail, ProfileSummary } from "@/shared/api/types";
import { STAGES } from "@/shared/lib/stages";
import { cn, fmt } from "@/shared/lib/utils";
import { Badge, Button, Card, CardBody, CardHeader, Input, Label, NativeSelect, PageHeader, Skeleton, StatusBadge } from "@/shared/ui/core";
import { CodeView, JsonView } from "@/shared/ui/content";
import { Dialog, Tabs } from "@/shared/ui/overlays";
import { diffOverrides, getPath, GROUPS } from "./params";
import { useCancelJob } from "@/features/runs/api";

interface Preview { run_name: string; command: string[]; config: Record<string, any> }

function ProfileCard({ p, active, onClick }: { p: ProfileSummary; active: boolean; onClick: () => void }) {
  return (
    <motion.button whileHover={{ y: -2 }} onClick={onClick}
      className={cn("glass w-full rounded-2xl p-4 text-left transition", active ? "ring-2 ring-brand/60" : "hover:border-white/20")}>
      <div className="flex items-center justify-between"><span className="font-semibold">{p.name}</span>
        {p.gate_mode && p.gate_mode !== "auto" && <Badge tone={p.gate_mode === "llm" ? "brand" : "violet"}>{p.gate_mode}</Badge>}</div>
      <p className="mt-1 line-clamp-2 text-xs text-muted">{p.description}</p>
      {p.error && <p className="mt-1 line-clamp-2 text-xs text-bad">{p.error}</p>}
      <div className="mt-2 flex flex-wrap gap-1 text-[10px] text-muted">
        <span>{p.stages.length} stages</span>{p.extends && <span>· extends {p.extends}</span>}{p.reuse_from && <span>· reuses {p.reuse_from}</span>}
      </div>
    </motion.button>
  );
}

function JobQueue() {
  const { data } = useJobs();
  const cancel = useCancelJob();
  const nav = useNavigate();
  const jobs = (data ?? []).slice(0, 8);
  return (
    <Card><CardHeader title="Job queue" subtitle="One GPU job at a time (FIFO); waits while serving holds the GPU" icon={<Terminal className="h-4 w-4" />} />
      <CardBody className="space-y-2">
        {jobs.length === 0 && <div className="text-xs text-muted">No jobs yet.</div>}
        {jobs.map((j: Job) => (
          <div key={j.id} className="flex items-center gap-3 rounded-xl border border-line bg-white/[0.02] px-3 py-2">
            <button className="min-w-0 flex-1 text-left" onClick={() => nav(`/runs/${j.run_name}`)}>
              <div className="truncate text-sm font-medium">{j.run_name}</div>
              <div className="truncate text-[11px] text-muted">{j.profile} · {fmt.ago(j.created_at)}{j.message && ` · ${j.message}`}</div>
            </button>
            <StatusBadge status={j.status} />
            {["queued", "running"].includes(j.status) && <Button size="icon" variant="ghost" aria-label="Cancel job" onClick={() => cancel.mutate(j.id)}><Square className="h-3.5 w-3.5" /></Button>}
          </div>
        ))}
      </CardBody>
    </Card>
  );
}

/** These parameter changes as a reusable profile: extends the chosen one, lists only what differs (Profiles page). */
function SaveAsProfile({ parent, overrides, onClose, onSaved }: { parent: string; overrides: string[]; onClose: () => void; onSaved: (name: string) => void }) {
  const [name, setName] = useState(`${parent}_custom`);
  const [description, setDescription] = useState("");
  const save = useMutation({
    mutationFn: () => api.post<{ name: string }>("/profiles", { name, extends: parent, overrides, description }),
    onSuccess: (p) => { toast.success(`Profile ${p.name} saved (training/configs/${p.name}.yaml)`); onSaved(p.name); onClose(); },
    onError: (e) => toast.error(errorMessage(e)),
  });
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()} title="Save as profile" description={`A new training/configs/<name>.yaml that extends ${parent} with these ${overrides.length} change(s). Editable later on the Profiles page.`}>
      <div className="space-y-3">
        <div><Label>name</Label><Input value={name} onChange={(e) => setName(e.target.value.toLowerCase())} /></div>
        <div><Label>description</Label><Input value={description} onChange={(e) => setDescription(e.target.value)} placeholder="what it is for" /></div>
        <CodeView value={overrides.join("\n")} language="text" maxHeight="200px" />
        <div className="flex justify-end gap-2"><Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant="primary" onClick={() => save.mutate()} loading={save.isPending} disabled={!/^[a-z0-9][a-z0-9_]{1,40}$/.test(name)}><Save className="h-4 w-4" /> Save</Button></div>
      </div>
    </Dialog>
  );
}

export default function LaunchPage() {
  const [params] = useSearchParams();
  const rerun = params.get("rerun");
  const nav = useNavigate();
  const qc = useQueryClient();
  const profiles = useQuery({ queryKey: keys.profiles, queryFn: () => api.get<ProfileSummary[]>("/profiles") });
  const runs = useRuns();
  const svc = useServices();
  const rerunRun = runs.data?.find((r) => r.name === rerun);

  const [profile, setProfile] = useState<string>("smoke");
  const [runName, setRunName] = useState("");
  const [stages, setStages] = useState<string[] | null>(null);
  const [reuseFrom, setReuseFrom] = useState("");
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [extra, setExtra] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [saveAs, setSaveAs] = useState(false);

  useEffect(() => {
    if (rerunRun) {
      setProfile(rerunRun.profile?.replace("configs/", "").replace(".yaml", "") ?? "smoke");
      setRunName(rerunRun.name);
    }
  }, [rerunRun]);

  const detail = useQuery({ queryKey: keys.profile(profile), queryFn: () => api.get<ProfileDetail>(`/profiles/${profile}`), enabled: !!profile });
  const resolved = detail.data?.resolved;
  useEffect(() => { setEdits({}); setStages(null); setReuseFrom(""); }, [profile]);

  const overrides = useMemo(() => [...diffOverrides(resolved, edits), ...extra.split("\n").map((s) => s.trim()).filter(Boolean)], [resolved, edits, extra]);
  const body = {
    profile, run_name: runName || undefined, overrides, stages: stages ?? undefined, reuse_from: reuseFrom || undefined,
    allow_existing: !!rerun,
  };

  const previewM = useMutation({ mutationFn: () => api.post<Preview>("/jobs/preview", body), onSuccess: setPreview, onError: (e) => toast.error(errorMessage(e)) });
  const launchM = useMutation({
    mutationFn: () => api.post<Job>("/jobs", body),
    onSuccess: (j) => {
      toast.success(`Queued ${j.run_name}`, { description: j.message ?? "It starts as soon as the GPU is free." });
      qc.invalidateQueries({ queryKey: keys.jobs }); qc.invalidateQueries({ queryKey: keys.runs });
      nav(`/runs/${j.run_name}`);
    },
    onError: (e) => toast.error(errorMessage(e)),
  });

  const planned: string[] = stages ?? resolved?.stages ?? [];
  const toggleStage = (id: string) => setStages((cur) => {
    const base = cur ?? resolved?.stages ?? [];
    return base.includes(id) ? base.filter((s: string) => s !== id) : STAGES.map((s) => s.id).filter((s) => s === id || base.includes(s));
  });
  const servingHoldsGpu = svc.data && Object.values(svc.data.containers ?? {}).some((s) => s === "running");

  return (
    <>
      <PageHeader eyebrow="Operate" title={rerun ? `Re-run stages of ${rerun}` : "Launch a training run"}
        subtitle="Pick a profile, tune its parameters, choose the stages, preview the exact command, launch. The run appears live on its run page." />
      {servingHoldsGpu && <div className="mb-4 rounded-xl border border-warn/30 bg-warn/10 px-4 py-2 text-xs text-warn">The serving stack holds the GPU: a launched job will wait in the queue until serving is stopped (System page).</div>}
      <div className="grid gap-5 xl:grid-cols-[1fr_380px]">
        <div className="space-y-5">
          <div className="grid gap-3 md:grid-cols-3">
            {profiles.isLoading ? Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-24" />)
              : profiles.data?.map((p) => <ProfileCard key={p.name} p={p} active={p.name === profile} onClick={() => setProfile(p.name)} />)}
          </div>
          <Card>
            <CardHeader title="Run" subtitle="Name, reuse and stages" />
            <CardBody className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div><Label>run name</Label><Input value={runName} onChange={(e) => setRunName(e.target.value.toLowerCase())} placeholder={resolved?.run_name ?? "run name"} /></div>
                <div><Label>reuse artefacts from (optional)</Label>
                  <NativeSelect value={reuseFrom} onChange={(e) => setReuseFrom(e.target.value)}>
                    <option value="">{resolved?.reuse_from ? `profile default: ${resolved.reuse_from}` : "— none —"}</option>
                    {runs.data?.map((r) => <option key={r.name} value={r.name}>{r.name}</option>)}
                  </NativeSelect></div>
              </div>
              <div><Label>stages ({planned.length})</Label>
                <div className="flex flex-wrap gap-1.5">{STAGES.map((s) => {
                  const on = planned.includes(s.id);
                  return <button key={s.id} onClick={() => toggleStage(s.id)} className={cn("rounded-lg border px-2.5 py-1 text-xs transition",
                    on ? "border-brand/40 bg-brand/10 text-ink" : "border-line text-muted hover:text-ink")}><span className="mr-1 font-mono text-[10px] opacity-60">{s.step}</span>{s.short}</button>;
                })}</div></div>
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="Parameters" subtitle="Changed values become dotted overrides (--set key=value); everything else comes from the profile" />
            <CardBody>
              {!resolved ? <Skeleton className="h-40" /> : (
                <Tabs tabs={[
                  { id: "form", label: "Editor", content: (
                    <div className="space-y-5">{GROUPS.map((g) => (
                      <div key={g.id}><div className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">{g.label}</div>
                        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{g.params.filter((p) => getPath(resolved, p.key) !== undefined || p.key === "eval.max_rows").map((p) => {
                          const current = edits[p.key] ?? String(getPath(resolved, p.key) ?? "");
                          const changed = edits[p.key] !== undefined && edits[p.key] !== String(getPath(resolved, p.key) ?? "");
                          return (
                            <div key={p.key}><Label className={cn(changed && "text-brand")}>{p.label}</Label>
                              {p.kind === "select" ? (
                                <NativeSelect value={current} onChange={(e) => setEdits({ ...edits, [p.key]: e.target.value })}>{p.options!.map((o) => <option key={o}>{o}</option>)}</NativeSelect>
                              ) : <Input type={p.kind === "number" ? "number" : "text"} step={p.step} value={current} onChange={(e) => setEdits({ ...edits, [p.key]: e.target.value })} className={cn(changed && "border-brand/60")} />}
                            </div>);
                        })}</div></div>))}
                      <div><Label>extra overrides (one key=value per line)</Label><textarea value={extra} onChange={(e) => setExtra(e.target.value)} rows={3}
                        className="w-full rounded-lg border border-line bg-black/20 p-2 font-mono text-xs outline-none focus:border-brand/60" placeholder="lora.r=32" /></div>
                    </div>) },
                  { id: "yaml", label: "Profile YAML", content: <CodeView value={detail.data?.yaml ?? ""} language="yaml" maxHeight="480px" /> },
                  { id: "resolved", label: "Resolved config", content: <JsonView data={resolved} maxHeight="480px" /> },
                ]} />
              )}
            </CardBody>
          </Card>
          <div className="flex flex-wrap items-center gap-3">
            <Button size="lg" onClick={() => previewM.mutate()} loading={previewM.isPending}><Eye className="h-4 w-4" /> Preview</Button>
            <Button size="lg" variant="primary" onClick={() => launchM.mutate()} loading={launchM.isPending}><Rocket className="h-4 w-4" /> Launch</Button>
            {overrides.length > 0 && <Button size="lg" variant="ghost" onClick={() => setSaveAs(true)}><Save className="h-4 w-4" /> Save as profile</Button>}
            {overrides.length > 0 && <span className="text-xs text-muted">{overrides.length} override{overrides.length > 1 && "s"}: <code className="text-brand">{overrides.join("  ")}</code></span>}
          </div>
        </div>
        <div className="space-y-5"><JobQueue /></div>
      </div>
      {saveAs && <SaveAsProfile parent={profile} overrides={overrides} onClose={() => setSaveAs(false)}
        onSaved={(name) => { qc.invalidateQueries({ queryKey: keys.profiles }); setProfile(name); }} />}
      <Dialog open={!!preview} onOpenChange={(o) => !o && setPreview(null)} title={`Preview · ${preview?.run_name}`} description="Exactly what the job container will run" wide>
        {preview && <div className="space-y-4">
          <CodeView value={preview.command.join(" \\\n    ")} language="text" />
          <JsonView data={preview.config} maxHeight="380px" />
          <div className="flex justify-end"><Button variant="primary" onClick={() => { setPreview(null); launchM.mutate(); }}><Rocket className="h-4 w-4" /> Launch</Button></div>
        </div>}
      </Dialog>
    </>
  );
}
