import { describe, expect, it } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RetrievalPage } from "../src/pages/RetrievalPage";
import { installFetch, renderAt } from "./helpers";
import { emptySearch, IDS, search } from "./fixtures";

const SEARCH = { method: "POST", path: "/memories/search", body: search };

describe("retrieval lab", () => {
  it("starts with no result and explains what to do", () => {
    installFetch([]);
    renderAt(<RetrievalPage />, "/retrieval");
    expect(screen.getByText("No query run yet")).toBeDefined();
    expect(screen.getByRole("button", { name: /Run retrieval/ })).toBeDefined();
  });

  it("runs the real search endpoint with the values the user entered", async () => {
    const user = userEvent.setup();
    const fetch = installFetch([SEARCH]);
    renderAt(<RetrievalPage />, "/retrieval");

    await user.clear(screen.getByLabelText("Query"));
    await user.type(screen.getByLabelText("Query"), "python backend");
    await user.clear(screen.getByLabelText("Top-k"));
    await user.type(screen.getByLabelText("Top-k"), "3");
    await user.selectOptions(screen.getByLabelText("Retrieval mode"), "hybrid");
    await user.clear(screen.getByLabelText("Min confidence"));
    await user.type(screen.getByLabelText("Min confidence"), "0.4");
    await user.selectOptions(screen.getByLabelText("Memory type"), "FACT");
    await user.click(screen.getByRole("button", { name: /Run retrieval/ }));

    await waitFor(() => expect(fetch.calls.length).toBeGreaterThan(0));
    expect(fetch.calls[0]).toMatchObject({
      method: "POST",
      path: "/memories/search",
      body: {
        query: "python backend",
        top_k: 3,
        retrieval_mode: "hybrid",
        min_confidence: 0.4,
        memory_type: "FACT",
        status: "ACTIVE",
      },
    });
  });

  it("renders rank #1 with its score, relevance, importance, confidence, freshness and priority", async () => {
    const user = userEvent.setup();
    installFetch([SEARCH]);
    renderAt(<RetrievalPage />, "/retrieval");
    await user.click(screen.getByRole("button", { name: /Run retrieval/ }));

    expect(await screen.findByText("Why rank #1 ranked first")).toBeDefined();
    const top = search.results[0]!;
    expect(screen.getByText("#1")).toBeDefined();
    expect(screen.getAllByText("0.699").length).toBeGreaterThan(0);
    expect(screen.getAllByText(top.content).length).toBeGreaterThan(0);
    // Every figure the spec requires, rendered from the response.
    expect(screen.getAllByText("Relevance").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Effective priority").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Freshness").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Confidence (stored)").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Importance (stored)").length).toBeGreaterThan(0);
    // The footer repeats the whole score line for each result.
    expect(screen.getAllByText("effective priority").length).toBe(3);
  });

  it("makes the ranking explanation visible, factor by factor", async () => {
    const user = userEvent.setup();
    installFetch([SEARCH]);
    renderAt(<RetrievalPage />, "/retrieval");
    await user.click(screen.getByRole("button", { name: /Run retrieval/ }));

    expect((await screen.findAllByText(/Why it ranked/)).length).toBeGreaterThan(0);
    for (const factor of ["relevance", "effective_priority", "confidence", "freshness", "type_preference"]) {
      expect(screen.getAllByText(factor).length).toBeGreaterThan(0);
    }
    expect(
      screen.getAllByText(
        /M2 effective priority 0.8520 \(importance 0.8100, confidence 0.95, decay applied\)/,
      ).length,
    ).toBe(3);
    expect(screen.getAllByText(/stored confidence 0.95/).length).toBeGreaterThan(0);
    expect(
      screen.getAllByText(
        /final 0.6992 for query 'python backend language memforge' = relevance=0.41/,
      ).length,
    ).toBeGreaterThan(0);
  });

  it("shows the query-level provenance the service reported", async () => {
    const user = userEvent.setup();
    installFetch([SEARCH]);
    renderAt(<RetrievalPage />, "/retrieval");
    await user.click(screen.getByRole("button", { name: /Run retrieval/ }));

    expect(await screen.findByText("Query provenance")).toBeDefined();
    expect(screen.getAllByText("Retrieval mode").length).toBeGreaterThan(0);
    expect(screen.getByText("Semantic availability")).toBeDefined();
    expect(screen.getByText("unavailable")).toBeDefined();
    expect(screen.getByText("Candidates considered")).toBeDefined();
    expect(screen.getByText("Returned")).toBeDefined();
    expect(screen.getByText("Suppressed duplicates")).toBeDefined();
    expect(screen.getByText("32.1 ms")).toBeDefined();
    expect(screen.getByText(/measured at runtime, not a benchmark/)).toBeDefined();
  });

  it("reports a degradation when the service says it fell back", async () => {
    const user = userEvent.setup();
    installFetch([
      {
        method: "POST",
        path: "/memories/search",
        body: {
          ...search,
          requested_mode: "hybrid",
          retrieval_mode: "lexical",
          semantic_available: false,
          degraded_reason: "no embedding provider is configured",
        },
      },
    ]);
    renderAt(<RetrievalPage />, "/retrieval");
    await user.selectOptions(screen.getByLabelText("Retrieval mode"), "hybrid");
    await user.click(screen.getByRole("button", { name: /Run retrieval/ }));
    expect(await screen.findByText(/no embedding provider is configured/)).toBeDefined();
  });

  it("lists the suppressed candidates with the reason", async () => {
    const user = userEvent.setup();
    installFetch([SEARCH]);
    renderAt(<RetrievalPage />, "/retrieval");
    await user.click(screen.getByRole("button", { name: /Run retrieval/ }));

    expect(await screen.findByText("Suppressed candidates")).toBeDefined();
    expect(screen.getAllByText(/which ranks higher in this result set/).length).toBe(2);
    expect(screen.getByText(/never a deletion/)).toBeDefined();
  });

  it("links the top result to the memory it came from", async () => {
    const user = userEvent.setup();
    installFetch([SEARCH]);
    renderAt(<RetrievalPage />, "/retrieval");
    await user.click(screen.getByRole("button", { name: /Run retrieval/ }));
    expect(await screen.findByText("Why rank #1 ranked first")).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Open this memory" }).getAttribute("href"),
    ).toBe(`/memories/${IDS.PROFILE}`);
  });

  it("shows an empty state when nothing passed the filters", async () => {
    const user = userEvent.setup();
    installFetch([{ method: "POST", path: "/memories/search", body: emptySearch }]);
    renderAt(<RetrievalPage />, "/retrieval");
    await user.click(screen.getByRole("button", { name: /Run retrieval/ }));
    expect(await screen.findByText("No memory passed the filters")).toBeDefined();
  });

  it("shows the backend's validation error when a request is rejected", async () => {
    const user = userEvent.setup();
    installFetch([
      {
        method: "POST",
        path: "/memories/search",
        status: 400,
        body: { error: { code: "RETRIEVAL_VALIDATION_ERROR", message: "query must not be empty" } },
      },
    ]);
    renderAt(<RetrievalPage />, "/retrieval");
    await user.click(screen.getByRole("button", { name: /Run retrieval/ }));
    expect(await screen.findByText("RETRIEVAL_VALIDATION_ERROR")).toBeDefined();
    expect(screen.getByText(/query must not be empty/)).toBeDefined();
  });

  it("reports a malformed retrieval payload instead of rendering a ranking", async () => {
    const user = userEvent.setup();
    installFetch([
      {
        method: "POST",
        path: "/memories/search",
        body: { ...search, results: [{ ...search.results[0]!, final_score: "high" }] },
      },
    ]);
    renderAt(<RetrievalPage />, "/retrieval");
    await user.click(screen.getByRole("button", { name: /Run retrieval/ }));
    expect(await screen.findByText("Malformed API response")).toBeDefined();
    expect(screen.queryByText("Why rank #1 ranked first")).toBeNull();
  });

  it("does not fire a search while the request is still running", async () => {
    const user = userEvent.setup();
    installFetch([{ method: "POST", path: "/memories/search", pending: true }]);
    renderAt(<RetrievalPage />, "/retrieval");
    await user.click(screen.getByRole("button", { name: /Run retrieval/ }));
    await waitFor(() => expect(screen.getByText("Running the retrieval service…")).toBeDefined());
  });
});