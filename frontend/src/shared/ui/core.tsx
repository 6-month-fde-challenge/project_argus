/** Core design-system primitives. Variants via class-variance-authority; everything themable through tokens. */
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { motion } from "motion/react";
import { AlertTriangle, Inbox, Loader2 } from "lucide-react";
import { forwardRef, useEffect, useState, type ComponentProps, type ReactNode } from "react";
import { cn } from "@/shared/lib/utils";

/* ----------------------------------------------------------------- Button */
const button = cva(
  "inline-flex items-center justify-center gap-2 rounded-lg text-sm font-medium transition-all disabled:opacity-40 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60 active:scale-[0.98]",
  {
    variants: {
      variant: {
        primary: "bg-gradient-to-r from-brand to-brand-2 text-bg shadow-lg shadow-brand/20 hover:brightness-110",
        secondary: "bg-white/[0.06] text-ink border border-line hover:bg-white/[0.1]",
        ghost: "text-muted hover:text-ink hover:bg-white/[0.06]",
        danger: "bg-bad/15 text-bad border border-bad/30 hover:bg-bad/25",
      },
      size: { sm: "h-8 px-3 text-xs", md: "h-9 px-4", lg: "h-11 px-6 text-base", icon: "h-9 w-9" },
    },
    defaultVariants: { variant: "secondary", size: "md" },
  },
);

export interface ButtonProps extends ComponentProps<"button">, VariantProps<typeof button> {
  asChild?: boolean;
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild, loading, children, disabled, ...props }, ref) => {
    const cls = cn(button({ variant, size }), className);
    // Slot merges onto exactly one child element, so the loading spinner exists only on a real <button>
    if (asChild) return <Slot ref={ref} className={cls} {...props}>{children}</Slot>;
    return (
      <button ref={ref} className={cls} disabled={disabled || loading} {...props}>
        {loading && <Loader2 className="h-4 w-4 animate-spin" />}
        {children}
      </button>
    );
  },
);
Button.displayName = "Button";

/* ----------------------------------------------------------------- Card */
export const Card = ({ className, ...props }: ComponentProps<"div">) => (
  <div className={cn("glass rounded-2xl shadow-xl shadow-black/20", className)} {...props} />
);

export function CardHeader({ title, subtitle, icon, actions, className }: {
  title: ReactNode; subtitle?: ReactNode; icon?: ReactNode; actions?: ReactNode; className?: string;
}) {
  return (
    <div className={cn("flex items-start justify-between gap-4 px-5 pt-4 pb-3", className)}>
      <div className="flex items-start gap-3 min-w-0">
        {icon && <div className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-white/5 text-brand">{icon}</div>}
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-ink truncate">{title}</h3>
          {subtitle && <p className="text-xs text-muted mt-0.5">{subtitle}</p>}
        </div>
      </div>
      {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
    </div>
  );
}

export const CardBody = ({ className, ...props }: ComponentProps<"div">) => (
  <div className={cn("px-5 pb-5", className)} {...props} />
);

/* ----------------------------------------------------------------- Badge */
const badge = cva("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-medium whitespace-nowrap", {
  variants: {
    tone: {
      neutral: "border-line bg-white/5 text-muted",
      ok: "border-ok/30 bg-ok/10 text-ok",
      warn: "border-warn/30 bg-warn/10 text-warn",
      bad: "border-bad/30 bg-bad/10 text-bad",
      info: "border-info/30 bg-info/10 text-info",
      brand: "border-brand/30 bg-brand/10 text-brand",
      violet: "border-brand-2/30 bg-brand-2/10 text-brand-2",
    },
  },
  defaultVariants: { tone: "neutral" },
});
export type Tone = NonNullable<VariantProps<typeof badge>["tone"]>;

export const Badge = ({ tone, className, children, dot }: { tone?: Tone; className?: string; children: ReactNode; dot?: boolean }) => (
  <span className={cn(badge({ tone }), className)}>
    {dot && <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse-soft" />}
    {children}
  </span>
);

const STATUS_TONE: Record<string, Tone> = {
  ok: "ok", succeeded: "ok", running: "warn", queued: "info", pending: "neutral", failed: "bad", interrupted: "bad",
  cancelled: "neutral", skipped: "neutral", reused: "violet", incomplete: "warn", empty: "neutral", down: "bad",
  degraded: "warn", ALLOW: "ok", REVIEW: "warn", BLOCK: "bad", LLM: "brand", XGBoost: "violet", HYBRID: "violet",
};
export const StatusBadge = ({ status, className }: { status: string; className?: string }) => (
  <Badge tone={STATUS_TONE[status] ?? "neutral"} dot={status === "running" || status === "queued"} className={className}>
    {status}
  </Badge>
);

/* ----------------------------------------------------------------- Progress */
export function Progress({ value, className, tone = "brand", indeterminate }: {
  value: number; className?: string; tone?: "brand" | "ok" | "warn" | "bad"; indeterminate?: boolean;
}) {
  const color = { brand: "from-brand to-brand-2", ok: "from-ok to-emerald-300", warn: "from-warn to-orange-300", bad: "from-bad to-rose-300" }[tone];
  return (
    <div className={cn("h-1.5 w-full overflow-hidden rounded-full bg-white/[0.06]", className)}>
      {indeterminate ? (
        <div className={cn("h-full w-1/3 rounded-full bg-gradient-to-r animate-shimmer", color)} />
      ) : (
        <motion.div className={cn("h-full rounded-full bg-gradient-to-r", color)} initial={false}
          animate={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} transition={{ type: "spring", stiffness: 80, damping: 20 }} />
      )}
    </div>
  );
}

/* ----------------------------------------------------------------- Stat (animated number) */
export function useCountUp(target: number | null | undefined, ms = 700) {
  const [v, setV] = useState(0);
  useEffect(() => {
    if (target == null || Number.isNaN(target)) return;
    const start = performance.now();
    const from = 0;
    let raf = 0;
    const tick = (t: number) => {
      const k = Math.min(1, (t - start) / ms);
      setV(from + (target - from) * (1 - Math.pow(1 - k, 3)));
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return target == null ? null : v;
}

export function Stat({ label, value, format, hint, icon, tone = "brand" }: {
  label: string; value: number | null | undefined; format: (v: number | null) => string; hint?: ReactNode;
  icon?: ReactNode; tone?: "brand" | "ok" | "warn" | "bad" | "violet";
}) {
  const v = useCountUp(value);
  const color = { brand: "text-brand", ok: "text-ok", warn: "text-warn", bad: "text-bad", violet: "text-brand-2" }[tone];
  return (
    <Card className="p-5 relative overflow-hidden">
      <div className="absolute -right-6 -top-6 h-24 w-24 rounded-full bg-gradient-to-br from-brand/10 to-brand-2/10 blur-2xl" />
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-wider text-muted">{label}</span>
        {icon && <span className={cn("opacity-80", color)}>{icon}</span>}
      </div>
      <div className={cn("mt-2 text-3xl font-semibold tabular-nums", color)}>{format(v)}</div>
      {hint && <div className="mt-1 text-xs text-muted">{hint}</div>}
    </Card>
  );
}

/* ----------------------------------------------------------------- states */
export const Skeleton = ({ className }: { className?: string }) => <div className={cn("skeleton h-4", className)} />;

export function EmptyState({ title, hint, action, icon }: { title: string; hint?: ReactNode; action?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-14 text-center">
      <div className="grid h-12 w-12 place-items-center rounded-2xl bg-white/5 text-muted">{icon ?? <Inbox className="h-5 w-5" />}</div>
      <div className="text-sm font-medium">{title}</div>
      {hint && <div className="max-w-md text-xs text-muted">{hint}</div>}
      {action}
    </div>
  );
}

export function ErrorState({ error, retry }: { error: unknown; retry?: () => void }) {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div className="flex items-start gap-3 rounded-xl border border-bad/30 bg-bad/10 p-4 text-sm">
      <AlertTriangle className="h-4 w-4 mt-0.5 text-bad shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="font-medium text-bad">Something went wrong</div>
        <div className="text-xs text-ink/80 mt-1 break-words">{msg}</div>
      </div>
      {retry && <Button size="sm" onClick={retry}>Retry</Button>}
    </div>
  );
}

/* ----------------------------------------------------------------- page header */
export function PageHeader({ title, subtitle, actions, eyebrow }: { title: ReactNode; subtitle?: ReactNode; actions?: ReactNode; eyebrow?: ReactNode }) {
  return (
    <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}
      className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        {eyebrow && <div className="mb-1 text-xs font-medium uppercase tracking-[0.18em] text-brand">{eyebrow}</div>}
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-sm text-muted">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </motion.div>
  );
}

/* ----------------------------------------------------------------- form controls */
export const Input = forwardRef<HTMLInputElement, ComponentProps<"input">>(({ className, ...props }, ref) => (
  <input ref={ref} className={cn("h-9 w-full rounded-lg border border-line bg-black/20 px-3 text-sm outline-none transition focus:border-brand/60 focus:ring-2 focus:ring-brand/20 placeholder:text-muted/60", className)} {...props} />
));
Input.displayName = "Input";

export const NativeSelect = ({ className, ...props }: ComponentProps<"select">) => (
  <select className={cn("h-9 w-full rounded-lg border border-line bg-surface-2 px-3 text-sm outline-none focus:border-brand/60", className)} {...props} />
);

export const Label = ({ className, ...props }: ComponentProps<"label">) => (
  <label className={cn("mb-1.5 block text-xs font-medium text-muted", className)} {...props} />
);

export const Kbd = ({ children }: { children: ReactNode }) => (
  <kbd className="rounded border border-line bg-white/5 px-1.5 py-0.5 font-mono text-[10px] text-muted">{children}</kbd>
);
