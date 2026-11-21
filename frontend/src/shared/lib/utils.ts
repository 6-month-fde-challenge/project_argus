import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { formatDistanceToNowStrict } from "date-fns";

export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs));

export const fmt = {
  num: (v: number | null | undefined, digits = 3) => (v == null || Number.isNaN(v) ? "–" : v.toFixed(digits)),
  int: (v: number | null | undefined) => (v == null ? "–" : Math.round(v).toLocaleString("en-IN")),
  pct: (v: number | null | undefined, digits = 1) => (v == null ? "–" : `${(v * 100).toFixed(digits)}%`),
  minutes: (m: number | null | undefined) => {
    if (m == null) return "–";
    if (m < 1) return `${Math.round(m * 60)}s`;
    const h = Math.floor(m / 60);
    return h ? `${h}h ${Math.round(m % 60)}m` : `${m.toFixed(m < 10 ? 1 : 0)}m`;
  },
  mb: (v: number | null | undefined) => (v == null ? "–" : v >= 1000 ? `${(v / 1000).toFixed(2)} GB` : `${v.toFixed(0)} MB`),
  ago: (ts: number | string | null | undefined) => {
    if (ts == null) return "–";
    const d = typeof ts === "number" ? new Date(ts < 1e12 ? ts * 1000 : ts) : new Date(ts);
    return Number.isNaN(d.getTime()) ? "–" : `${formatDistanceToNowStrict(d)} ago`;
  },
  short: (s: string | null | undefined, n = 28) => (!s ? "–" : s.length > n ? `${s.slice(0, n - 1)}…` : s),
  model: (name: string | null | undefined) => (name ? name.split("/").pop()! : "–"),
};

/** Stable sort by a numeric key, descending; returns a new array. */
export const sortDesc = <T,>(xs: T[], key: (x: T) => number | null | undefined) =>
  [...xs].sort((a, b) => (key(b) ?? -Infinity) - (key(a) ?? -Infinity));

/** a / b, or null when either side is unknown (or b is 0): for telemetry that may be partially missing. */
export const ratio = (a: number | null | undefined, b: number | null | undefined, scale = 1) =>
  a == null || !b ? null : (a / b) * scale;
