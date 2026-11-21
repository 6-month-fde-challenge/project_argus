import { describe, expect, it } from "vitest";
import { GROUPS, diffOverrides, getPath } from "./params";

const resolved = { max_length: 384, gate: { mode: "auto", min_uplift: 0.03 }, sft_scorer: { lr: 0.0002, method: "qlora" }, eval: { max_rows: null } };

describe("getPath", () => {
  it("reads dotted paths and tolerates missing branches", () => {
    expect(getPath(resolved, "gate.mode")).toBe("auto");
    expect(getPath(resolved, "max_length")).toBe(384);
    expect(getPath(resolved, "nope.deeper.still")).toBeUndefined();
    expect(getPath(null, "a")).toBeUndefined();
  });
});

describe("diffOverrides", () => {
  it("emits only values that actually changed", () => {
    expect(diffOverrides(resolved, { "gate.mode": "llm", max_length: "384", "sft_scorer.lr": "0.0001" }))
      .toEqual(["gate.mode=llm", "sft_scorer.lr=0.0001"]);
  });

  it("ignores cleared fields and treats null originals as empty", () => {
    expect(diffOverrides(resolved, { "gate.mode": "", "eval.max_rows": "" })).toEqual([]);
    expect(diffOverrides(resolved, { "eval.max_rows": "5000" })).toEqual(["eval.max_rows=5000"]);
  });

  it("allows keys the profile does not define yet", () => {
    expect(diffOverrides(resolved, { "dpo.beta": "0.2" })).toEqual(["dpo.beta=0.2"]);
  });
});

describe("GROUPS", () => {
  it("has unique, dotted-or-top-level keys and method selects with lora/qlora", () => {
    const keys = GROUPS.flatMap((g) => g.params.map((p) => p.key));
    expect(new Set(keys).size).toBe(keys.length);
    const methods = GROUPS.flatMap((g) => g.params).filter((p) => p.key.endsWith(".method"));
    expect(methods.length).toBeGreaterThan(0);
    for (const m of methods) expect(m.options).toEqual(["lora", "qlora"]);
    expect(GROUPS.find((g) => g.id === "general")!.params.find((p) => p.key === "gate.mode")!.options).toEqual(["auto", "hybrid", "llm"]);
  });
});
