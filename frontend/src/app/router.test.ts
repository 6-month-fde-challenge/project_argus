import { describe, expect, it, vi } from "vitest";
import { isStaleBundle, reloadOnceForNewBuild } from "./router";

describe("a page bundle missing after a redeploy", () => {
  it("is recognised in every browser's wording, and other errors are not", () => {
    expect(isStaleBundle(new TypeError("Failed to fetch dynamically imported module: http://x/assets/RulesPage-a1.js"))).toBe(true); // Chrome
    expect(isStaleBundle(new TypeError("Importing a module script failed."))).toBe(true); // Safari
    expect(isStaleBundle(new TypeError("error loading dynamically imported module"))).toBe(true); // Firefox
    expect(isStaleBundle(new Error("Cannot read properties of undefined"))).toBe(false);
    expect(isStaleBundle(undefined)).toBe(false);
  });

  it("reloads once, never in a loop", () => {
    const reload = vi.fn();
    vi.stubGlobal("location", { ...window.location, reload });
    const store = new Map<string, string>();
    const storage = { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => void store.set(k, v) };
    expect(reloadOnceForNewBuild(storage, 1_000_000)).toBe(true);
    expect(reloadOnceForNewBuild(storage, 1_030_000)).toBe(false); // failed again right after: show the error instead
    expect(reloadOnceForNewBuild(storage, 1_100_000)).toBe(true);  // a later redeploy reloads again
    expect(reload).toHaveBeenCalledTimes(2);
    vi.unstubAllGlobals();
  });
});
