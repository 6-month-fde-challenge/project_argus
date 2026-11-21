/** Radix-based overlays and navigation: Tabs, Dialog, Tooltip - accessible by default (focus, keyboard, ARIA). */
import * as DialogPrimitive from "@radix-ui/react-dialog";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { X } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/shared/lib/utils";

export function Tabs({ tabs, value, onValueChange, className }: {
  tabs: { id: string; label: ReactNode; content: ReactNode; badge?: ReactNode }[];
  value?: string; onValueChange?: (v: string) => void; className?: string;
}) {
  return (
    <TabsPrimitive.Root defaultValue={tabs[0]?.id} value={value} onValueChange={onValueChange} className={className}>
      <TabsPrimitive.List className="mb-4 flex flex-wrap gap-1 rounded-xl border border-line bg-black/20 p-1">
        {tabs.map((t) => (
          <TabsPrimitive.Trigger key={t.id} value={t.id}
            className="flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium text-muted transition data-[state=active]:bg-white/10 data-[state=active]:text-ink hover:text-ink">
            {t.label}{t.badge}
          </TabsPrimitive.Trigger>
        ))}
      </TabsPrimitive.List>
      {tabs.map((t) => (
        <TabsPrimitive.Content key={t.id} value={t.id} className="outline-none data-[state=active]:animate-[fadeIn_.2s_ease]">
          {t.content}
        </TabsPrimitive.Content>
      ))}
    </TabsPrimitive.Root>
  );
}

export function Dialog({ open, onOpenChange, title, description, children, wide }: {
  open: boolean; onOpenChange: (o: boolean) => void; title: ReactNode; description?: ReactNode; children: ReactNode; wide?: boolean;
}) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm" />
        <DialogPrimitive.Content className={cn("glass fixed left-1/2 top-1/2 z-50 max-h-[85vh] w-[92vw] -translate-x-1/2 -translate-y-1/2 overflow-auto rounded-2xl p-6 shadow-2xl", wide ? "max-w-5xl" : "max-w-lg")}>
          <div className="mb-4 flex items-start justify-between gap-4">
            <div>
              <DialogPrimitive.Title className="text-base font-semibold">{title}</DialogPrimitive.Title>
              {description && <DialogPrimitive.Description className="mt-1 text-xs text-muted">{description}</DialogPrimitive.Description>}
            </div>
            <DialogPrimitive.Close className="rounded-lg p-1 text-muted hover:bg-white/10 hover:text-ink" aria-label="Close">
              <X className="h-4 w-4" />
            </DialogPrimitive.Close>
          </div>
          {children}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

export const TooltipProvider = TooltipPrimitive.Provider;

export function Tip({ content, children, side = "top" }: { content: ReactNode; children: ReactNode; side?: "top" | "bottom" | "left" | "right" }) {
  return (
    <TooltipPrimitive.Root delayDuration={150}>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content side={side} sideOffset={6}
          className="z-50 max-w-xs rounded-lg border border-line bg-surface-2 px-3 py-2 text-xs text-ink shadow-xl">
          {content}
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  );
}
