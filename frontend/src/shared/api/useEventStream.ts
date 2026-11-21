import { useEffect, useRef, useState } from "react";
import { API_BASE } from "./client";

export type StreamState = "connecting" | "open" | "ended" | "error";

/**
 * Subscribe to a Server-Sent Events endpoint. EventSource reconnects on its own and sends Last-Event-ID (the
 * platform uses a byte offset), so a dropped connection resumes exactly. Handlers are kept in a ref so callers
 * can pass inline functions without re-subscribing on every render.
 */
export function useEventStream(
  path: string | null,
  handlers: Record<string, (data: any) => void>,
): StreamState {
  const [state, setState] = useState<StreamState>("connecting");
  const ref = useRef(handlers);
  ref.current = handlers;

  useEffect(() => {
    if (!path) return;
    setState("connecting");
    const es = new EventSource(`${API_BASE}${path}`);
    es.onopen = () => setState("open");
    es.onerror = () => setState(es.readyState === EventSource.CLOSED ? "error" : "connecting");
    const names = Object.keys(ref.current);
    const listeners = names.map((name) => {
      const fn = (e: MessageEvent) => {
        if (name === "end") {
          setState("ended");
          es.close();
        }
        try { ref.current[name]?.(JSON.parse(e.data)); } catch { ref.current[name]?.(e.data); }
      };
      es.addEventListener(name, fn as EventListener);
      return [name, fn] as const;
    });
    if (!names.includes("end")) {
      es.addEventListener("end", () => { setState("ended"); es.close(); });
    }
    return () => {
      listeners.forEach(([n, fn]) => es.removeEventListener(n, fn as EventListener));
      es.close();
    };
  }, [path]);

  return state;
}
