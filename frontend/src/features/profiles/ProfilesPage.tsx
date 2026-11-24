import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, FilePlus2, FileSliders, Save, Trash2, TriangleAlert } from "lucide-react";
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router";
import { toast } from "sonner";
import { api, errorMessage } from "@/shared/api/client";
import { keys } from "@/shared/api/queries";
import type { ProfileDetail, ProfileSummary } from "@/shared/api/types";
import { cn } from "@/shared/lib/utils";
import { Badge, Button, Card, CardBody, CardHeader, EmptyState, ErrorState, Input, Label, NativeSelect, PageHeader, Skeleton } from "@/shared/ui/core";
import { CodeView, JsonView } from "@/shared/ui/content";
import { Dialog } from "@/shared/ui/overlays";

function ProfileBadges({ p }: { p: ProfileSummary }) {
  return (
    <div className="flex flex-wrap gap-1">
      {p.error && <Badge tone="bad">broken</Badge>}
      {p.gate_mode && p.gate_mode !== "auto" ? <Badge tone="warn">demo · gate {p.gate_mode}</Badge> : p.gate_mode && <Badge tone="ok">measured gate</Badge>}
      {p.serving_mode === "production" ? <Badge tone="ok">serves production</Badge> : p.serving_mode && <Badge>serves demo runs</Badge>}
      {p.built_in && <Badge>shipped</Badge>}
    </div>
  );
}

function NewProfileDialog({ profiles, onClose, onCreated }: { profiles: ProfileSummary[]; onClose: () => void; onCreated: (name: string) => void }) {
  const [name, setName] = useState("");
  const [parent, setParent] = useState(profiles.find((p) => p.name === "prod_a100") ? "prod_a100" : "base");
  const [description, setDescription] = useState("");
  const create = useMutation({
    mutationFn: () => api.put<ProfileSummary>(`/profiles/${name}`, {
      yaml: `# ${description.trim() || `Based on ${parent}: only what differs is listed here`}\nextends: ${parent}.yaml\nrun_name: argus_${name}\n`,
      reason: "created on the Profiles page",
    }),
    onSuccess: () => { toast.success(`Profile ${name} created`); onCreated(name); onClose(); },
    onError: (e) => toast.error(errorMessage(e)),
  });
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()} title="New profile" description="A new training/configs/<name>.yaml that extends another profile and overrides only what differs.">
      <div className="space-y-3">
        <div><Label>name</Label><Input value={name} onChange={(e) => setName(e.target.value.toLowerCase())} placeholder="e.g. prod_a100_longer" /></div>
        <div><Label>extends</Label><NativeSelect value={parent} onChange={(e) => setParent(e.target.value)}>{profiles.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}</NativeSelect></div>
        <div><Label>description (first comment line)</Label><Input value={description} onChange={(e) => setDescription(e.target.value)} /></div>
        <div className="flex justify-end gap-2"><Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant="primary" onClick={() => create.mutate()} loading={create.isPending} disabled={!/^[a-z0-9][a-z0-9_]{1,40}$/.test(name)}><FilePlus2 className="h-4 w-4" /> Create</Button></div>
      </div>
    </Dialog>
  );
}

function Editor({ name, onDeleted }: { name: string; onDeleted: () => void }) {
  const qc = useQueryClient();
  const { data, error } = useQuery({ queryKey: keys.profile(name), queryFn: () => api.get<ProfileDetail>(`/profiles/${name}`) });
  const [text, setText] = useState("");
  const [reason, setReason] = useState("");
  const [resolved, setResolved] = useState<Record<string, unknown> | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => { if (data) { setText(data.yaml); setResolved(data.resolved); setProblem(null); } }, [data]);
  const dirty = !!data && text !== data.yaml;
  const validate = useMutation({
    mutationFn: () => api.post<{ resolved: Record<string, unknown> }>(`/profiles/${name}/validate`, { yaml: text }),
    onSuccess: (r) => { setResolved(r.resolved); setProblem(null); toast.success("Valid: resolved below"); },
    onError: (e) => setProblem(errorMessage(e)),
  });
  const save = useMutation({
    mutationFn: () => api.put(`/profiles/${name}`, { yaml: text, reason }),
    onSuccess: () => { toast.success(`Saved ${name}.yaml`); setReason(""); qc.invalidateQueries({ queryKey: keys.profiles }); qc.invalidateQueries({ queryKey: keys.profile(name) }); },
    onError: (e) => setProblem(errorMessage(e)),
  });
  const remove = useMutation({
    mutationFn: () => api.del(`/profiles/${name}`),
    onSuccess: () => { toast.success(`Deleted ${name}`); qc.invalidateQueries({ queryKey: keys.profiles }); onDeleted(); },
    onError: (e) => toast.error(errorMessage(e)),
  });
  if (error) return <ErrorState error={error} />;
  if (!data) return <Skeleton className="h-96" />;
  return (
    <div className="space-y-5">
      <Card><CardHeader title={`${name}.yaml`} subtitle={<>training/configs/{name}.yaml · {data.extends ? `extends ${data.extends}` : "the root every profile extends"} · tracked by git (a change shows in <code>git status</code>)</>}
        icon={<FileSliders className="h-4 w-4" />} actions={<ProfileBadges p={data} />} />
        <CardBody className="space-y-3">
          {name === "base" && <div className="flex items-center gap-2 rounded-lg border border-warn/30 bg-warn/10 p-2 text-xs text-warn"><TriangleAlert className="h-4 w-4" />Every profile inherits base.yaml: a change here changes all of them (each one is re-checked before saving).</div>}
          <CodeView value={text} language="yaml" onChange={(v) => { setText(v); setProblem(null); }} maxHeight="560px" />
          {problem && <div className="rounded-lg border border-bad/30 bg-bad/10 p-2 text-xs text-bad">{problem}</div>}
          <div className="flex flex-wrap items-end gap-2">
            <div className="min-w-56 flex-1"><Label>reason (change log)</Label><Input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="why this change" /></div>
            <Button onClick={() => validate.mutate()} loading={validate.isPending}><CheckCircle2 className="h-4 w-4" /> Validate</Button>
            <Button variant="primary" onClick={() => save.mutate()} loading={save.isPending} disabled={!dirty}><Save className="h-4 w-4" /> Save</Button>
            {!data.built_in && <Button variant="danger" onClick={() => { if (window.confirm(`Delete training/configs/${name}.yaml?`)) remove.mutate(); }} loading={remove.isPending}><Trash2 className="h-4 w-4" /> Delete</Button>}
          </div>
          <div className="text-[11px] text-muted">By hand: edit training/configs/{name}.yaml; the trainer, the Launch page and the VM scripts read the same file. Per-run changes without a new profile: Launch page → parameters.</div>
        </CardBody></Card>
      {resolved && <Card><CardHeader title="Resolved configuration" subtitle="What a run of this profile gets: extends chain merged, exactly as the trainer resolves it" /><CardBody><JsonView data={resolved} maxHeight="420px" /></CardBody></Card>}
    </div>
  );
}

export default function ProfilesPage() {
  const [params, setParams] = useSearchParams();
  const [creating, setCreating] = useState(false);
  const { data, error, isLoading } = useQuery({ queryKey: keys.profiles, queryFn: () => api.get<ProfileSummary[]>("/profiles") });
  const selected = params.get("profile") ?? data?.find((p) => p.name === "prod_a100")?.name ?? data?.[0]?.name;
  return (
    <>
      <PageHeader eyebrow="Models" title="Training profiles"
        subtitle="training/configs/*.yaml: models, LoRA, every stage's settings, the gate, the serving api: section. Edit here or by hand; validation resolves them exactly like the trainer."
        actions={<Button variant="primary" onClick={() => setCreating(true)}><FilePlus2 className="h-4 w-4" /> New profile</Button>} />
      {error ? <ErrorState error={error} /> : isLoading || !data ? <Skeleton className="h-96" /> : (
        <div className="grid gap-5 xl:grid-cols-[22rem_1fr]">
          <Card><CardBody className="space-y-1 p-2">{data.map((p) => (
            <button key={p.name} onClick={() => setParams({ profile: p.name })} className={cn("w-full rounded-xl px-3 py-2 text-left hover:bg-white/[0.04]", p.name === selected && "bg-brand/[0.08]")}>
              <div className="flex items-center justify-between gap-2"><span className="text-sm font-medium">{p.name}</span>{p.extends && <span className="text-[10px] text-muted">→ {p.extends.replace(".yaml", "")}</span>}</div>
              <div className="mb-1 truncate text-[11px] text-muted">{p.description}</div><ProfileBadges p={p} />
            </button>))}</CardBody></Card>
          {selected ? <Editor key={selected} name={selected} onDeleted={() => setParams({})} /> : <EmptyState title="No profiles" />}
        </div>)}
      {creating && data && <NewProfileDialog profiles={data} onClose={() => setCreating(false)} onCreated={(n) => setParams({ profile: n })} />}
    </>
  );
}
