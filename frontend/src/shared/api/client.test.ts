import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, errorMessage } from "./client";

const reply = (status: number, body: unknown) =>
  vi.fn().mockResolvedValue(new Response(body === null ? "" : typeof body === "string" ? body : JSON.stringify(body), { status }));

const failure = (p: Promise<unknown>) => p.then(() => { throw new Error("expected a failure"); }, (e) => e as ApiError);

afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("prefixes /api/v1 and parses JSON", async () => {
    const f = reply(200, { ok: 1 });
    vi.stubGlobal("fetch", f);
    await expect(api.get("/runs")).resolves.toEqual({ ok: 1 });
    expect(f.mock.calls[0][0]).toBe("/api/v1/runs");
  });

  it("sends JSON bodies on POST", async () => {
    const f = reply(201, { id: "j1" });
    vi.stubGlobal("fetch", f);
    await api.post("/jobs", { profile: "smoke" });
    const init = f.mock.calls[0][1];
    expect(init.method).toBe("POST");
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(init.body)).toEqual({ profile: "smoke" });
  });

  it("turns platform errors into ApiError with code and message", async () => {
    vi.stubGlobal("fetch", reply(409, { error: "CONFLICT", message: "a job for this run is already queued" }));
    const err = await failure(api.post("/jobs", {}));
    expect(err).toBeInstanceOf(ApiError);
    expect([err.status, err.code, err.message]).toEqual([409, "CONFLICT", "a job for this run is already queued"]);
    expect(errorMessage(err)).toBe("a job for this run is already queued");
  });

  it("falls back to HTTP_<status> for non-JSON and empty error bodies", async () => {
    vi.stubGlobal("fetch", reply(502, "Bad Gateway"));
    const err = await failure(api.get("/x"));
    expect(err.code).toBe("HTTP_502");
    vi.stubGlobal("fetch", reply(500, null));
    expect((await failure(api.get("/x"))).code).toBe("HTTP_500");
  });

  it("returns status and body as-is from raw()", async () => {
    vi.stubGlobal("fetch", reply(400, { error: "INVALID_REQUEST", field: "transaction_amount" }));
    await expect(api.raw("POST", "/scoring/score", {})).resolves.toEqual({
      status: 400, body: { error: "INVALID_REQUEST", field: "transaction_amount" },
    });
  });

  it("errorMessage handles any thrown value", () => {
    expect(errorMessage(new Error("boom"))).toBe("boom");
    expect(errorMessage("plain")).toBe("plain");
  });
});
