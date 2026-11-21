import { createBrowserRouter, Link } from "react-router";
import { Button, EmptyState } from "@/shared/ui/core";
import { AppShell } from "./AppShell";
import { FEATURES } from "./features";

function NotFound() {
  return (
    <EmptyState title="Page not found" hint="This page does not exist in Project ARGUS."
      action={<Button asChild variant="primary"><Link to="/">Back to overview</Link></Button>} />
  );
}

/** Routes are assembled from the feature registry: the router itself never changes when a feature is added. */
export const router = createBrowserRouter([
  { path: "/", element: <AppShell />, children: [...FEATURES.flatMap((f) => f.routes), { path: "*", element: <NotFound /> }] },
]);
