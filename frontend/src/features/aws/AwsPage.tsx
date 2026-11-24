import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Cloud, CloudUpload, ExternalLink, GitBranch, RefreshCw, Trash2, TriangleAlert } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router";
import { toast } from "sonner";
import { api, errorMessage } from "@/shared/api/client";
import { keys } from "@/shared/api/queries";
import type { Approval, AwsStatus, WorkflowRun } from "@/shared/api/types";
import { fmt } from "@/shared/lib/utils";
import { Badge, Button, Card, CardBody, CardHeader, ErrorState, Input, Label, PageHeader, Skeleton, StatusBadge } from "@/shared/ui/core";
import { ApprovalFields, approvalReady, CommandLine, emptyApproval, ServingModeBadge } from "@/shared/ui/governance";

function Runs({ title, runs }: { title: string; runs?: WorkflowRun[] }) {
  return (
    <div><div className="mb-1 text-xs font-medium">{title}</div>
      {runs?.length ? runs.map((r) => (
        <a key={r.id} href={r.html_url} target="_blank" rel="noreferrer" className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-[11px] hover:bg-white/[0.04]">
          <StatusBadge status={r.status === "completed" ? (r.conclusion === "success" ? "succeeded" : r.conclusion ?? "failed") : r.status === "in_progress" ? "running" : "queued"} />
          <span className="min-w-0 flex-1 truncate">{r.display_title} <span className="text-muted">· {r.event} · {r.head_branch} {r.sha}</span></span>
          <span className="text-muted">{fmt.ago(r.created_at)}</span><ExternalLink className="h-3 w-3 text-muted" /></a>))
        : <div className="px-2 text-[11px] text-muted">No runs yet.</div>}</div>
  );
}

export default function AwsPage() {
  const qc = useQueryClient();
  const { data, error, isLoading, refetch, isFetching } = useQuery({ queryKey: keys.aws, queryFn: () => api.get<AwsStatus>("/aws"), refetchInterval: 20000 });
  const [approval, setApproval] = useState<Approval>(emptyApproval);
  const [allowDemo, setAllowDemo] = useState(false);
  const [confirm, setConfirm] = useState("");
  const done = (msg: string) => { toast.success(msg); setApproval(emptyApproval); setTimeout(() => qc.invalidateQueries({ queryKey: keys.aws }), 4000); };
  const sync = useMutation({ mutationFn: () => api.post("/aws/sync", { approval }), onSuccess: () => done("Settings synced to GitHub"), onError: (e) => toast.error(errorMessage(e)) });
  const deploy = useMutation({ mutationFn: () => api.post("/aws/deploy", { approval, allow_demo: allowDemo }), onSuccess: () => done("Deploy started (deploy.yml)"), onError: (e) => toast.error(errorMessage(e)) });
  const destroy = useMutation({ mutationFn: () => api.post("/aws/destroy", { approval, confirm }), onSuccess: () => { setConfirm(""); done("Destroy started (destroy.yml)"); }, onError: (e) => toast.error(errorMessage(e)) });
  const s = data?.serving;
  const demoLive = !!s && (s.live_mode !== "production" || s.active_run_forced);
  const ready = approvalReady(approval);
  return (
    <>
      <PageHeader eyebrow="Serve" title="AWS deployment"
        subtitle="The real-time stream on AWS (Kinesis, Fargate, S3 audit, CloudWatch) is deployed by the repository's GitHub Actions through the OIDC role: no AWS key is stored here. Models stay on this GPU VM; AWS calls its Fraud Model API."
        actions={<div className="flex items-center gap-2"><ServingModeBadge mode={s?.mode} demoModel={s?.active_run_forced} reason={s?.active_run_forced_reason} />
          <Button size="sm" variant="ghost" onClick={() => refetch()} loading={isFetching}><RefreshCw className="h-3.5 w-3.5" /></Button></div>} />
      {error ? <ErrorState error={error} /> : isLoading || !data ? <Skeleton className="h-96" /> : (
        <div className="grid gap-5 xl:grid-cols-[1.1fr_1fr]">
          <div className="space-y-5">
            <Card><CardHeader title="GitHub" icon={<GitBranch className="h-4 w-4" />} actions={data.github.ok ? <Badge tone="ok">{data.github.repo}</Badge> : <Badge tone="bad">not connected</Badge>} />
              <CardBody className="space-y-2 text-xs">
                {!data.github.ok && <div className="text-bad">{data.github.error}</div>}
                {!data.github.ok && <div className="text-muted">Set <b>ARGUS_GITHUB_TOKEN</b> (and the repository if it is not the git remote) on the <Link className="text-brand hover:underline" to="/settings">Settings page</Link>: a fine-grained token for this repository with Actions, Variables and Secrets read/write.</div>}
                {data.github.ok && !data.bootstrapped && (
                  <div className="rounded-lg border border-warn/30 bg-warn/10 p-3 text-warn">Not bootstrapped: run once from your PC (it needs AWS credentials: Terraform state bucket + the GitHub deploy role). After that, everything else works from here.
                    <CommandLine className="mt-2" command="python deploy/ops.py up" label="On your PC" /></div>)}
                {data.github.ok && data.bootstrapped && <div className="text-muted">Bootstrapped: stack <b className="text-ink">{data.stack}</b>, role {data.variables.AWS_ROLE_ARN}, region {data.variables.AWS_REGION ?? "ap-south-1"}.</div>}
                {data.github.ok && <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 pt-1 text-[11px]">{["SCORING_HOST", "ALLOWED_CIDRS", "ALERT_EMAIL", "ALLOW_DEMO"].map((k) => (
                  <div key={k} className="contents"><span className="text-muted">{k}</span><code className="truncate">{data.variables[k] ?? "–"}</code></div>))}</div>}
              </CardBody></Card>

            <Card><CardHeader title="Deploy" icon={<CloudUpload className="h-4 w-4" />} subtitle="Every action here needs two named people and a reason: it changes production." />
              <CardBody className="space-y-4">
                {demoLive && (
                  <div className="flex items-start gap-2 rounded-lg border border-bad/30 bg-bad/10 p-3 text-xs text-bad"><TriangleAlert className="h-4 w-4 shrink-0" />
                    <div>Live traffic would meet a demo setup: {s?.live_mode !== "production" ? `the API runs in ${s?.live_mode ?? "an unknown"} mode` : `the active run ${s?.active_run} is a demo run (${s?.active_run_forced_reason})`}. Serve a measured run with the production profile first (Settings → Serving profile, Operations → Serve a run). The deploy is refused unless you allow a demo explicitly.</div></div>)}
                <ApprovalFields value={approval} onChange={setApproval} why="AWS changes production" />
                <div className="flex flex-wrap items-center gap-2">
                  <Button onClick={() => sync.mutate()} loading={sync.isPending} disabled={!data.github.ok || (!!s?.production && !ready)}><RefreshCw className="h-4 w-4" /> Sync settings to GitHub</Button>
                  <Button variant="primary" onClick={() => deploy.mutate()} loading={deploy.isPending} disabled={!data.github.ok || !data.bootstrapped || !ready || (demoLive && !allowDemo)}><Cloud className="h-4 w-4" /> Deploy</Button>
                  <label className="flex items-center gap-2 text-xs text-muted"><input type="checkbox" className="accent-amber-400" checked={allowDemo} onChange={(e) => setAllowDemo(e.target.checked)} />allow a demo setup (this deploy only; never for real payments)</label>
                </div>
                <div className="text-[11px] text-muted">Sync writes SCORING_HOST, ALLOWED_CIDRS, ALERT_EMAIL, TF_VARS and the SCORING_TOKEN secret from this machine's .env (TF_VAR_* and the tokens on the Settings page). Deploy runs deploy.yml on main: tests → stream image → terraform apply → ECS rollout.</div>
                <CommandLine command="python deploy/ops.py deploy" label="Same from your PC (with AWS credentials)" />
              </CardBody></Card>

            <Card className="border-bad/20"><CardHeader title="Destroy" icon={<Trash2 className="h-4 w-4" />} subtitle="Removes everything this project created in AWS (only what is in its Terraform state), incl. the Kinesis streams and the S3 audit archive." />
              <CardBody className="flex flex-wrap items-end gap-2">
                <div><Label>type {data.stack ?? "<project>-<environment>"} to confirm</Label><Input value={confirm} onChange={(e) => setConfirm(e.target.value)} placeholder={data.stack} /></div>
                <Button variant="danger" onClick={() => destroy.mutate()} loading={destroy.isPending} disabled={!data.bootstrapped || !ready || confirm !== data.stack}><Trash2 className="h-4 w-4" /> Destroy</Button>
              </CardBody></Card>
          </div>
          <div className="space-y-5">
            <Card><CardHeader title="Workflow runs" subtitle="GitHub Actions (newest first)" />
              <CardBody className="space-y-4"><Runs title="deploy.yml" runs={data.runs.deploy} /><Runs title="destroy.yml" runs={data.runs.destroy} /><Runs title="ci.yml" runs={data.runs.ci} /></CardBody></Card>
            <Card><CardHeader title="Stream the platform uses" subtitle="Settings → ARGUS_STREAM_URL / ARGUS_STREAM_CONTROL_KEY (python deploy/ops.py platform-env prints the AWS values)" />
              <CardBody className="flex items-center gap-2 text-xs"><code className="min-w-0 flex-1 truncate">{data.stream.url}</code>
                <Badge>{data.stream.remote ? "remote (AWS)" : "local"}</Badge><StatusBadge status={data.stream.up ? "ok" : "down"} /></CardBody></Card>
          </div>
        </div>)}
    </>
  );
}
