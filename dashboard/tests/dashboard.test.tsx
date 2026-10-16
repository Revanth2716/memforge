import { describe, expect, it } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { DashboardPage } from "../src/pages/DashboardPage";
import { installFetch, renderAt, healthyRoutes } from "./helpers";
import {
  emptyTraceList,
  evaluation,
  memoryPage,
  metrics,
  traceList,
  traces,
} from "./fixtures";

const base = () => [
  { path: "/memories?limit=1&offset=0", body: { ...memoryPage, items: [], limit: 1 } },
  { path: "/memories", body: memoryPage },
  { path: "/traces", body: traceList },
  { path: "/evaluations/latest", body: evaluation },
  { path: "/metrics", body: metrics },
];

describe("dashboard home", () => {
  it("renders the real memory counts, one request per status", async () => {
    const fetch = installFetch(base());
    renderAt(<DashboardPage />, "/");
    expect(await screen.findByText("Active memories")).toBeDefined();
    await waitFor(() => expect(screen.getByText("Forgotten")).toBeDefined());
    // ACTIVE, ARCHIVED and DELETED each get their own counted request.
    const statuses = fetch.calls
      .map((call) => call.path)
      .filter((path) => /status=/.test(path))
      .map((path) => path.match(/status=([A-Z]+)/)?.[1]);
    expect(statuses.sort()).toEqual(["ACTIVE", "ARCHIVED", "DELETED"]);
  });

  it("renders the real operational counters from /metrics", async () => {
    installFetch(base());
    renderAt(<DashboardPage />, "/");
    expect(await screen.findByText("Retrieval operations")).toBeDefined();
    await waitFor(() => expect(screen.getByText("Context compilations")).toBeDefined());
    // metrics.operations["memory.retrieval"] === 1 and ["context.compile"] === 1
    expect(screen.getAllByText("1").length).toBeGreaterThan(0);
    expect(screen.getByText("provider calls", { exact: false })).toBeDefined();
  });

  it("renders the measured evaluation score, not a curated one", async () => {
    installFetch(base());
    renderAt(<DashboardPage />, "/");
    expect((await screen.findAllByText("0.991")).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/memory-core@v1/).length).toBeGreaterThan(0);
    expect((await screen.findAllByText("passed")).length).toBeGreaterThan(0);
  });

  it("lists the latest traces with a link to each", async () => {
    installFetch(base());
    renderAt(<DashboardPage />, "/");
    const link = await screen.findByRole("link", {
      name: new RegExp(traces[0]!.trace_id.slice(0, 10)),
    });
    expect(link.getAttribute("href")).toBe(`/traces/${traces[0]!.trace_id}`);
  });

  it("lists recent memory activity from the memory endpoint", async () => {
    installFetch(base());
    renderAt(<DashboardPage />, "/");
    expect(await screen.findByRole("link", { name: memoryPage.items[0]!.content })).toBeDefined();
  });

  it("offers the 30-60 second demo path as quick actions", async () => {
    installFetch(base());
    renderAt(<DashboardPage />, "/");
    expect(await screen.findByRole("link", { name: /Run a retrieval/ })).toBeDefined();
    expect(screen.getByRole("link", { name: /Compile a context/ })).toBeDefined();
    expect(screen.getByRole("link", { name: /Ask the agent/ })).toBeDefined();
    expect(screen.getByRole("link", { name: /Open a trace/ })).toBeDefined();
    expect(screen.getByRole("link", { name: /Read the evaluation/ })).toBeDefined();
    expect(screen.getByRole("link", { name: /Inspect metrics/ })).toBeDefined();
  });

  it("names the eight differentiators the product ships", async () => {
    installFetch(base());
    renderAt(<DashboardPage />, "/");
    for (const title of [
      "MEMORY",
      "INTELLIGENCE",
      "RETRIEVAL",
      "CONTEXT",
      "AGENT",
      "MCP",
      "EVALUATION",
      "OBSERVABILITY",
    ]) {
      expect(await screen.findByText(title)).toBeDefined();
    }
  });

  it("shows an explicit not-available state when /metrics is missing", async () => {
    installFetch([
      { path: "/memories?limit=1&offset=0", body: { ...memoryPage, items: [], limit: 1 } },
      { path: "/memories", body: memoryPage },
      { path: "/traces", body: traceList },
      { path: "/evaluations/latest", body: evaluation },
    ]);
    renderAt(<DashboardPage />, "/");
    await waitFor(() =>
      expect(screen.getAllByText(/endpoint not available|not available/i).length).toBeGreaterThan(0),
    );
    // The rest of the page still renders.
    expect(screen.getByText("Active memories")).toBeDefined();
    expect((await screen.findAllByText("0.991")).length).toBeGreaterThan(0);
  });

  it("shows a loading state while the evaluation is still running", async () => {
    installFetch([
      { path: "/memories?limit=1&offset=0", body: { ...memoryPage, items: [], limit: 1 } },
      { path: "/memories", body: memoryPage },
      { path: "/traces", body: traceList },
      { path: "/metrics", body: metrics },
      { path: "/evaluations/latest", pending: true },
    ]);
    renderAt(<DashboardPage />, "/");
    await waitFor(() =>
      expect(screen.getAllByText(/running dataset/i).length).toBeGreaterThan(0),
    );
  });

  it("reports an unreachable API on the memory cards without blanking the page", async () => {
    installFetch([
      { path: "/memories", networkError: "Failed to fetch" },
      ...healthyRoutes(),
    ]);
    renderAt(<DashboardPage />, "/");
    await waitFor(() =>
      expect(
        screen.getAllByText(/the memory counter endpoint did not answer/i).length,
      ).toBeGreaterThan(0),
    );
    expect(screen.getByText("30–60 second demo path")).toBeDefined();
  });

  it("shows an empty state when there is no trace and no memory", async () => {
    installFetch([
      { path: "/memories?limit=1&offset=0", body: { items: [], total: 0, limit: 1, offset: 0 } },
      { path: "/memories", body: { items: [], total: 0, limit: 25, offset: 0 } },
      { path: "/traces", body: emptyTraceList },
      { path: "/evaluations/latest", body: evaluation },
      { path: "/metrics", body: metrics },
    ]);
    renderAt(<DashboardPage />, "/");
    expect(await screen.findByText("No traces recorded yet")).toBeDefined();
    expect(screen.getByText("No memories stored yet")).toBeDefined();
  });

  it("reports a malformed metrics payload instead of rendering invented counters", async () => {
    installFetch([
      { path: "/memories?limit=1&offset=0", body: { ...memoryPage, items: [], limit: 1 } },
      { path: "/memories", body: memoryPage },
      { path: "/traces", body: traceList },
      { path: "/evaluations/latest", body: evaluation },
      { path: "/metrics", body: { ...metrics, provider: { calls: "many" } } },
    ]);
    renderAt(<DashboardPage />, "/");
    await waitFor(() => expect(screen.getAllByText(/malformed API response/i).length).toBeGreaterThan(0));
    expect(document.body.textContent).toContain("not available");
  });

  it("keeps the dashboard self-describing about where its data comes from", async () => {
    installFetch(base());
    renderAt(<DashboardPage />, "/");
    await screen.findByText("Active memories");
    expect(screen.getAllByText(/GET \/memories\?status=ACTIVE/i).length).toBe(1);
  });
});
