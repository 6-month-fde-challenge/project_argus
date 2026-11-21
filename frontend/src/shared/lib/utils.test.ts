import { describe, expect, it } from "vitest";
import { fmt, ratio, sortDesc } from "./utils";
import { STAGES, stageLabel } from "./stages";

describe("fmt", () => {
  it("renders missing values as a dash", () => {
    for (const f of [fmt.num, fmt.int, fmt.pct, fmt.minutes, fmt.mb, fmt.ago, fmt.short, fmt.model]) {
      expect(f(null as never)).toBe("–");
      expect(f(undefined as never)).toBe("–");
    }
    expect(fmt.num(Number.NaN)).toBe("–");
  });

  it("formats numbers, percentages and sizes", () => {
    expect(fmt.num(0.38312)).toBe("0.383");
    expect(fmt.num(0.5, 1)).toBe("0.5");
    expect(fmt.pct(0.143)).toBe("14.3%");
    expect(fmt.pct(0)).toBe("0.0%");
    expect(fmt.int(110000)).toBe("1,10,000");                    // Indian grouping
    expect(fmt.mb(512)).toBe("512 MB");
    expect(fmt.mb(4096)).toBe("4.10 GB");
  });

  it("formats durations across seconds, minutes and hours", () => {
    expect(fmt.minutes(0.5)).toBe("30s");
    expect(fmt.minutes(4.25)).toBe("4.3m");
    expect(fmt.minutes(37)).toBe("37m");
    expect(fmt.minutes(125)).toBe("2h 5m");
  });

  it("handles epoch seconds, epoch ms, ISO strings and garbage", () => {
    const now = Date.now();
    expect(fmt.ago(now / 1000 - 120)).toMatch(/2 minutes ago/);
    expect(fmt.ago(now - 120_000)).toMatch(/2 minutes ago/);
    expect(fmt.ago(new Date(now - 3_600_000).toISOString())).toMatch(/1 hour ago/);
    expect(fmt.ago("not a date")).toBe("–");
  });

  it("shortens strings and model ids", () => {
    expect(fmt.short("abcdef", 4)).toBe("abc…");
    expect(fmt.short("abc", 4)).toBe("abc");
    expect(fmt.short("")).toBe("–");
    expect(fmt.model("Qwen/Qwen2.5-0.5B")).toBe("Qwen2.5-0.5B");
    expect(fmt.model("local-model")).toBe("local-model");
  });
});

describe("sortDesc", () => {
  it("sorts descending, puts nulls last and does not mutate", () => {
    const xs = [{ v: 1 }, { v: null }, { v: 3 }, { v: 2 }];
    const out = sortDesc(xs, (x) => x.v);
    expect(out.map((x) => x.v)).toEqual([3, 2, 1, null]);
    expect(xs.map((x) => x.v)).toEqual([1, null, 3, 2]);
  });
});

describe("stages", () => {
  it("has unique ids and readable labels, falling back to the id", () => {
    const ids = STAGES.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toContain("xgboost");
    expect(stageLabel("unknown_stage")).toBe("unknown_stage");
  });
});

describe("ratio", () => {
  it("divides, scales and returns null for unknown or zero denominators", () => {
    expect(ratio(1, 4)).toBe(0.25);
    expect(ratio(1, 4, 100)).toBe(25);
    expect(ratio(0, 4, 100)).toBe(0);
    expect(ratio(null, 4)).toBeNull();
    expect(ratio(1, null)).toBeNull();
    expect(ratio(1, 0)).toBeNull();
  });
});
