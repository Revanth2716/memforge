import { describe, expect, it } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../src/App";
import { installFetch, renderAt, healthyRoutes } from "./helpers";
import { memoryPage, metrics, traceList, evaluation, emptySearch } from "./fixtures";

const ROUTES: { name: string; path: string; heading: RegExp }[] = [
  { name: "Dashboard", path: "/", heading: /^Dashboard$/ },
  { name: "Memories", path: "/memories", heading: /^Memories$/ },
  { name: "Retrieval Lab", path: "/retrieval", heading: /^Retrieval Lab$/ },
  { name: "Context Compiler", path: "/context", heading: /^Context Compiler$/ },
  { name: "Agent Playground", path: "/agent", heading: /^Agent Playground$/ },
  { name: "Traces", path: "/traces", heading: /^Traces$/ },
  { name: "Evaluation", path: "/evaluation", heading: /^Evaluation$/ },
  { name: "Metrics", path: "/metrics", heading: /^Metrics$/ },
];

/** The page heading is the level-1 heading in the header. */
async function pageHeading(name: RegExp) {
  return screen.findByRole("heading", { level: 1, name });
}

function allRoutes() {
  return [
    { method: "POST", path: "/memories/search", body: emptySearch },
    { path: "/memories", body: memoryPage },
    { path: "/health", body: { status: "ok", service: "memforge", version: "1.0.0", storage: "sqlite" } },
    { path: "/metrics", body: metrics },
    { path: "/traces", body: traceList },
    { path: "/evaluations/latest", body: evaluation },
    ...healthyRoutes().filter((route) => route.path === "/health"),
  ];
}

describe("navigation", () => {
  it("renders every one of the eight primary areas from its own route", async () => {
    installFetch(allRoutes());
    for (const route of ROUTES) {
      const view = renderAt(<App />, route.path);
      expect(await pageHeading(route.heading)).toBeDefined();
      view.unmount();
    }
  });

  it("opens on the dashboard home with the control-plane framing", async () => {
    installFetch(allRoutes());
    renderAt(<App />, "/");
    expect(await screen.findByRole("heading", { name: /MemForge control plane/ })).toBeDefined();
    expect(screen.getByText(/is a/i)).toBeDefined();
    expect(screen.getByText(/visualisation layer/i)).toBeDefined();
  });

  it("moves between pages through the sidebar, marking the current page", async () => {
    const user = userEvent.setup();
    installFetch(allRoutes());
    renderAt(<App />, "/");

    await user.click(screen.getByRole("link", { name: /Retrieval Lab/ }));
    expect(await pageHeading(/Retrieval Lab/)).toBeDefined();
    expect(screen.getByRole("link", { name: /Retrieval Lab/ })).toHaveAttribute(
      "aria-current",
      "page",
    );

    await user.click(screen.getByRole("link", { name: /Context Compiler/ }));
    expect(await pageHeading(/Context Compiler/)).toBeDefined();

    await user.click(screen.getByRole("link", { name: /Evaluation/ }));
    expect(await screen.findByText(/Deterministic MemForge evaluation/)).toBeDefined();

    await waitFor(() =>
      expect(screen.getByRole("link", { name: /Evaluation/ })).toHaveAttribute("aria-current", "page"),
    );
  });

  it("offers a not-found page for an unknown route", async () => {
    installFetch(allRoutes());
    renderAt(<App />, "/does-not-exist");
    expect(await screen.findByRole("heading", { name: /No such page/ })).toBeDefined();
    expect(screen.getByRole("link", { name: /Go to the dashboard/ })).toBeDefined();
  });

  it("shows the API connection state in the sidebar", async () => {
    installFetch(allRoutes());
    renderAt(<App />, "/");
    expect(await screen.findByText(/API ok · v1\.0\.0/)).toBeDefined();
  });

  it("reports an unreachable API in the sidebar instead of failing silently", async () => {
    installFetch([{ path: "/health", networkError: "Failed to fetch" }]);
    renderAt(<App />, "/");
    expect(await screen.findByText(/API unreachable/)).toBeDefined();
  });

  it("exposes a theme toggle that flips the document theme", async () => {
    const user = userEvent.setup();
    installFetch(allRoutes());
    renderAt(<App />, "/");
    const toggle = screen.getByRole("button", { name: /Switch to light theme/ });
    await user.click(toggle);
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(screen.getByRole("button", { name: /Switch to dark theme/ })).toBeDefined();
  });
});
