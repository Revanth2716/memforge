import { describe, expect, it } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { EvaluationPage } from "../src/pages/EvaluationPage";
import { MetricsPage } from "../src/pages/MetricsPage";
import { installFetch, renderAt } from "./helpers";
import {
  emptyTraceList,
  evaluation,
  failingEvaluation,
  metrics,
  traceList,
} from "./fixtures";

describe("evaluation", () => {
  it("shows the dataset, the case counts and the measured score", async () => {
    installFetch([{ path: "/evaluations/latest", body: evaluation }]);
    renderAt(<EvaluationPage />, "/evaluation");

    expect((await screen.findAllByText("memory-core@v1")).length).toBeGreaterThan(0);
    expect(screen.getByText("version v1 · threshold 1.00")).toBeDefined();
    expect(screen.getAllByText("36").length).toBeGreaterThan(0);
    expect(screen.getAllByText("0.991").length).toBeGreaterThan(0);
    expect(screen.getAllByText("0.983").length).toBeGreaterThan(0);
    expect(screen.getAllByText("1.000").length).toBeGreaterThan(0);
  });

  it("labels the gate result and the threshold it is gated on", async () => {
    installFetch([{ path: "/evaluations/latest", body: evaluation }]);
    renderAt(<EvaluationPage />, "/evaluation");
    expect((await screen.findAllByText("passed")).length).toBeGreaterThan(0);
    expect(screen.getByText(/the threshold gates the case pass rate, not the blended score/)).toBeDefined();
  });

  it("shows every one of the twelve metrics with its own counts", async () => {
    installFetch([{ path: "/evaluations/latest", body: evaluation }]);
    renderAt(<EvaluationPage />, "/evaluation");
    await screen.findByText("retrieval_recall");

    const names = [
      "retrieval_recall",
      "retrieval_precision",
      "top_k_hit_rate",
      "duplicate_suppression_accuracy",
      "conflict_resolution_accuracy",
      "stale_memory_avoidance",
      "context_budget_compliance",
      "context_selection_accuracy",
      "context_provenance_completeness",
      "history_preservation",
      "memory_write_policy_compliance",
      "recall_relevance",
    ];
    for (const name of names) {
      expect(screen.getByText(name)).toBeDefined();
    }
    expect(screen.getAllByText("23/23").length).toBeGreaterThan(0);
    expect(screen.getAllByText("13/13").length).toBeGreaterThan(0);
  });

  it("prints the real precision score, never a rounded-up 1.0", async () => {
    installFetch([{ path: "/evaluations/latest", body: evaluation }]);
    renderAt(<EvaluationPage />, "/evaluation");
    await screen.findByText("retrieval_precision");
    // 23/29 = 0.793
    expect(screen.getAllByText("23/29").length).toBeGreaterThan(0);
    expect(screen.getAllByText("0.793").length).toBeGreaterThan(0);
    const definition = screen.getByText(/share of returned memories that were declared relevant/);
    expect(definition).toBeDefined();
  });

  it("shows the per-category pass counts", async () => {
    installFetch([{ path: "/evaluations/latest", body: evaluation }]);
    renderAt(<EvaluationPage />, "/evaluation");
    for (const category of ["agent", "conflict", "context", "duplicate", "lifecycle", "retrieval"]) {
      expect((await screen.findAllByText(category)).length).toBeGreaterThan(0);
    }
  });

  it("shows an explicit empty state when no case failed", async () => {
    installFetch([{ path: "/evaluations/latest", body: evaluation }]);
    renderAt(<EvaluationPage />, "/evaluation");
    expect(await screen.findByText(/No failing case in this run/)).toBeDefined();
  });

  it("shows failing cases in full when a later run fails", async () => {
    installFetch([{ path: "/evaluations/latest", body: failingEvaluation }]);
    renderAt(<EvaluationPage />, "/evaluation");

    expect((await screen.findAllByText("failed")).length).toBeGreaterThan(0);
    expect(screen.getByText("context-budget-sweep")).toBeDefined();
    expect(screen.getByText(/12\/13 compilations respected the budget/)).toBeDefined();
    expect(screen.getAllByText(/12\/13/).length).toBeGreaterThan(0);
  });

  it("labels the page as the deterministic MemForge evaluation", async () => {
    installFetch([{ path: "/evaluations/latest", body: evaluation }]);
    renderAt(<EvaluationPage />, "/evaluation");
    expect(await screen.findByRole("heading", { name: /Deterministic MemForge evaluation/ })).toBeDefined();
    expect(screen.getByText(/No judge model, no reference answers, no network/)).toBeDefined();
  });

  it("carries the limitations note the evaluation ships with", async () => {
    installFetch([{ path: "/evaluations/latest", body: evaluation }]);
    renderAt(<EvaluationPage />, "/evaluation");
    expect(
      await screen.findByText(/measures the supplied fixtures/),
    ).toBeDefined();
    expect(screen.getByText(/not proof of universal AI memory quality/)).toBeDefined();
  });

  it("points at the CLI that re-runs the same benchmark", async () => {
    installFetch([{ path: "/evaluations/latest", body: evaluation }]);
    renderAt(<EvaluationPage />, "/evaluation");
    const panel = (await screen.findByText("Re-check from the command line")).closest("section")!;
    expect(panel.textContent).toContain("python -m app.evaluation.run");
    expect(panel.textContent).toContain("memory-core@v1");
  });

  it("reports an API failure and an unavailable endpoint", async () => {
    installFetch([]);
    renderAt(<EvaluationPage />, "/evaluation");
    expect(await screen.findByText("Endpoint not available")).toBeDefined();
    expect(screen.getAllByText(/not part of this deployment/).length).toBeGreaterThan(0);
  });

  it("reports a malformed evaluation payload", async () => {
    installFetch([
      { path: "/evaluations/latest", body: { ...evaluation, overall: "0.991" } },
    ]);
    renderAt(<EvaluationPage />, "/evaluation");
    expect(await screen.findByText("Malformed API response")).toBeDefined();
    expect(screen.queryByText("memory-core@v1")).toBeNull();
  });

  it("shows a loading state while the dataset runs", async () => {
    installFetch([{ path: "/evaluations/latest", pending: true }]);
    renderAt(<EvaluationPage />, "/evaluation");
    await waitFor(() => expect(screen.getAllByText("Loading…").length).toBeGreaterThan(0));
  });
});

describe("metrics", () => {
  it("renders the real aggregate counters with the population they came from", async () => {
    installFetch([
      { path: "/metrics", body: metrics },
      { path: "/traces", body: traceList },
    ]);
    renderAt(<MetricsPage />, "/metrics");

    expect(await screen.findByText("Events recorded")).toBeDefined();
    expect(screen.getAllByText("13").length).toBeGreaterThan(0);
    expect(screen.getByText("the population every counter below is derived from")).toBeDefined();
    expect(screen.getByText("Failed traces")).toBeDefined();
    expect(screen.getByText("Usage reported")).toBeDefined();
  });

  it("groups the counters by capability", async () => {
    installFetch([
      { path: "/metrics", body: metrics },
      { path: "/traces", body: traceList },
    ]);
    renderAt(<MetricsPage />, "/metrics");

    await screen.findByText("Events recorded");
    for (const group of ["Memory", "Retrieval", "Context", "Agent", "MCP", "Provider"]) {
      expect(screen.getByText(group)).toBeDefined();
    }
    expect(screen.getAllByText("writes_skipped").length).toBeGreaterThan(0);
    expect(screen.getByText("mcp_calls")).toBeDefined();
    expect(screen.getByText("calls")).toBeDefined();
  });

  it("reports usage as zero, not as an estimate, when no provider sent any", async () => {
    installFetch([
      { path: "/metrics", body: metrics },
      { path: "/traces", body: traceList },
    ]);
    renderAt(<MetricsPage />, "/metrics");
    await screen.findByText("Events recorded");
    expect(
      screen.getByText("provider calls that actually sent a usage block — never derived"),
    ).toBeDefined();
    expect(screen.getAllByText("0").length).toBeGreaterThan(0);
  });

  it("does not fabricate a chart when the API has no time series", async () => {
    installFetch([
      { path: "/metrics", body: metrics },
      { path: "/traces", body: traceList },
    ]);
    renderAt(<MetricsPage />, "/metrics");
    await screen.findByText("Events recorded");
    expect(screen.getByText(/Why there is no chart\./)).toBeDefined();
    expect(document.querySelector("svg")).toBeNull();
    expect(document.querySelector("canvas")).toBeNull();
    expect(screen.getByText(/Latency, where the backend measured it/)).toBeDefined();
  });

  it("shows an empty state when no trace has been recorded", async () => {
    installFetch([
      { path: "/metrics", body: metrics },
      { path: "/traces", body: emptyTraceList },
    ]);
    renderAt(<MetricsPage />, "/metrics");
    expect(await screen.findByText("No trace recorded yet")).toBeDefined();
  });

  it("reports an unavailable metrics endpoint", async () => {
    installFetch([]);
    renderAt(<MetricsPage />, "/metrics");
    expect(await screen.findByText("Endpoint not available")).toBeDefined();
  });

  it("reports a malformed metrics payload", async () => {
    installFetch([{ path: "/metrics", body: { ...metrics, operations: [] } }]);
    renderAt(<MetricsPage />, "/metrics");
    expect(await screen.findByText("Malformed API response")).toBeDefined();
  });

  it("shows a loading state while the counters load", async () => {
    installFetch([{ path: "/metrics", pending: true }]);
    renderAt(<MetricsPage />, "/metrics");
    await waitFor(() => expect(screen.getAllByText("Loading…").length).toBeGreaterThan(0));
  });
});