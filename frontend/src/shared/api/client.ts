/** Typed HTTP client for the ARGUS platform API (same origin: nginx proxies /api to the platform). */
export const API_BASE = "/api/v1";

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public body?: unknown) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    signal,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  const data = text ? safeJson(text) : null;
  if (!res.ok) {
    const d = (data ?? {}) as { error?: string; message?: string; field?: string };
    throw new ApiError(res.status, d.error ?? `HTTP_${res.status}`, d.message ?? (d.field ? `${d.field}: invalid` : res.statusText), data);
  }
  return data as T;
}

function safeJson(text: string): unknown {
  try { return JSON.parse(text); } catch { return text; }
}

export const api = {
  get: <T,>(path: string, signal?: AbortSignal) => request<T>("GET", path, undefined, signal),
  post: <T,>(path: string, body?: unknown) => request<T>("POST", path, body ?? {}),
  put: <T,>(path: string, body?: unknown) => request<T>("PUT", path, body ?? {}),
  del: <T,>(path: string, body?: unknown) => request<T>("DELETE", path, body),
  text: async (path: string) => {
    const res = await fetch(`${API_BASE}${path}`);
    if (!res.ok) throw new ApiError(res.status, `HTTP_${res.status}`, await res.text());
    return res.text();
  },
  /** For relaying calls whose non-2xx body is meaningful (e.g. a scoring 400 shown as-is). */
  raw: async (method: string, path: string, body?: unknown) => {
    const res = await fetch(`${API_BASE}${path}`, {
      method, headers: { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body),
    });
    const text = await res.text();
    return { status: res.status, body: text ? safeJson(text) : null };
  },
};

export const errorMessage = (e: unknown) =>
  e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e);
