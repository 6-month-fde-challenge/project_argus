import { createBrowserRouter, Link, useRouteError } from "react-router";
import { Button, EmptyState, ErrorState } from "@/shared/ui/core";
import { AppShell } from "./AppShell";
import { FEATURES } from "./features";

function NotFound() {
  return (
    <EmptyState title="Page not found" hint="This page does not exist in Project ARGUS."
      action={<Button asChild variant="primary"><Link to="/">Back to overview</Link></Button>} />
  );
}

/** A page bundle that no longer exists: the UI was redeployed while this tab was open (its file names changed). */
export function isStaleBundle(error: unknown): boolean {
  const msg = error instanceof Error ? `${error.name} ${error.message}` : String(error ?? "");
  return /dynamically imported module|Importing a module script failed|error loading dynamically imported|ChunkLoadError|Unable to preload CSS/i.test(msg);
}

const RELOADED_AT = "argus.reloaded-for-new-build";

/** Reload once for a new build; never loop (a second failure within a minute shows the error instead). */
export function reloadOnceForNewBuild(storage: Pick<Storage, "getItem" | "setItem"> = sessionStorage, now = Date.now()): boolean {
  try {
    if (now - Number(storage.getItem(RELOADED_AT) ?? 0) < 60_000) return false;
    storage.setItem(RELOADED_AT, String(now));
  } catch {
    return false; // storage blocked: show the message, the user reloads
  }
  window.location.reload();
  return true;
}

/** Any error inside a page: the shell and the menu stay, the page area says what happened (never a blank page). */
function PageError() {
  const error = useRouteError();
  if (isStaleBundle(error) && reloadOnceForNewBuild()) {
    return <EmptyState title="Loading the new version of ARGUS…" />;
  }
  return (
    <div className="space-y-4">
      <ErrorState error={isStaleBundle(error) ? new Error("ARGUS was updated while this tab was open.") : error} />
      <Button variant="primary" onClick={() => window.location.reload()}>Reload the page</Button>
    </div>
  );
}

/** Routes are assembled from the feature registry: the router itself never changes when a feature is added. */
export const router = createBrowserRouter([
  {
    path: "/", element: <AppShell />,
    // pathless layout route: every page error (incl. a stale bundle after a redeploy) is caught inside the shell
    children: [{ errorElement: <PageError />, children: [...FEATURES.flatMap((f) => f.routes), { path: "*", element: <NotFound /> }] }],
  },
]);
