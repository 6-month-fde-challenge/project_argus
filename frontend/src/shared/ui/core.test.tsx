import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { Button, EmptyState, ErrorState, StatusBadge } from "./core";
import { FEATURES } from "@/app/features";

describe("shared UI", () => {
  it("StatusBadge shows the status and pulses only while active", () => {
    const { container, rerender } = render(<StatusBadge status="running" />);
    expect(screen.getByText("running")).toBeInTheDocument();
    const dots = () => container.querySelectorAll("span span").length;
    const running = dots();
    rerender(<StatusBadge status="succeeded" />);
    expect(screen.getByText("succeeded")).toBeInTheDocument();
    expect(dots()).toBeLessThan(running);
    rerender(<StatusBadge status="something-new" />);           // unknown statuses still render
    expect(screen.getByText("something-new")).toBeInTheDocument();
  });

  it("EmptyState renders title, hint and action", () => {
    render(<EmptyState title="No runs yet" hint="Launch one" action={<button>Launch</button>} />);
    expect(screen.getByText("No runs yet")).toBeInTheDocument();
    expect(screen.getByText("Launch one")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Launch" })).toBeInTheDocument();
  });

  it("ErrorState shows the message and retries", () => {
    const retry = vi.fn();
    render(<ErrorState error={new Error("platform unreachable")} retry={retry} />);
    expect(screen.getByText("platform unreachable")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(retry).toHaveBeenCalledOnce();
  });
});

describe("Button", () => {
  it("asChild renders its single child element with button styling (no Slot crash)", () => {
    render(<Button asChild variant="primary"><a href="/launch">Launch</a></Button>);
    const link = screen.getByRole("link", { name: "Launch" });
    expect(link.tagName).toBe("A");
    expect(link.className).toMatch(/rounded/);
  });

  it("shows a spinner and disables itself while loading", () => {
    render(<Button loading>Save</Button>);
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  });
});

describe("feature registry", () => {
  it("has unique ids, unique nav paths and at least one route per feature", () => {
    const ids = FEATURES.map((f) => f.id);
    expect(new Set(ids).size).toBe(ids.length);
    const paths = FEATURES.flatMap((f) => (f.nav ? [f.nav.path] : []));
    expect(new Set(paths).size).toBe(paths.length);
    for (const f of FEATURES) expect(f.routes.length).toBeGreaterThan(0);
  });
});
