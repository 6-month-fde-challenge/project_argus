import type { LucideIcon } from "lucide-react";
import type { RouteObject } from "react-router";

/**
 * A feature plugs into the app with one object: its routes and (optionally) its sidebar entry.
 * The shell never imports feature internals, so adding a feature never edits existing ones (open/closed).
 */
export type NavSection = "Operate" | "Models" | "Serve" | "Knowledge";

export interface FeatureDef {
  id: string;
  nav?: { label: string; icon: LucideIcon; path: string; section: NavSection; order: number; end?: boolean };
  routes: RouteObject[];
}

/** Route helper: lazy-load a page component (one chunk per feature page). */
export const lazyPage = (load: () => Promise<{ default: React.ComponentType }>) => ({
  lazy: async () => ({ Component: (await load()).default }),
});
