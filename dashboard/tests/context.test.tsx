import { describe, expect, it } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ContextPage } from "../src/pages/ContextPage";
import { installFetch, renderAt } from "./helpers";
import { compile, IDS } from "./fixtures";
import { truncateMiddle } from "../src/lib/format";

const COMPILE = { method: "POST", path: "/context/compile", body: compile };

describe("context compiler", () => {
  it("starts with no compilation and a deliberately small default budget", () => {
    installFetch([]);
    renderAt(<ContextPage />, "/context");
    expect(screen.getByText("No context compiled yet")).toBeDefined();
    expect(screen.getByLabelText("Token budget")).toHaveValue(160);
  });

  it("sends the real compile request with the entered budget and filters", async () => {
    const user = userEvent.setup();
    const fetch = installFetch([COMPILE]);
    renderAt(<ContextPage />, "/context");

    await user.clear(screen.getByLabelText("Token budget"));
    await user.type(screen.getByLabelText("Token budget"), "120");
    await user.selectOptions(screen.getByLabelText("Preferred type"), "FACT");
    await user.click(screen.getByRole("button", { name: /Compile context/ }));

    await waitFor(() => expect(fetch.calls.length).toBeGreaterThan(0));
    expect(fetch.calls[0]).toMatchObject({
      method: "POST",
      path: "/context/compile",
      body: {
        token_budget: 120,
        preferred_memory_types: ["FACT"],
        retrieval_mode: "lexical",
        top_k: 20,
      },
    });
  });

  it("shows the hard budget guarantee from the backend's own figures", async () => {
    const user = userEvent.setup();
    installFetch([COMPILE]);
    renderAt(<ContextPage />, "/context");
    await user.click(screen.getByRole("button", { name: /Compile context/ }));

    expect(await screen.findByText("within budget")).toBeDefined();
    // 58 of 160 estimated tokens, 102 remaining, utilisation 0.3625.
    expect(screen.getAllByText("58").length).toBeGreaterThan(0);
    expect(screen.getByText("/ 160 estimated tokens")).toBeDefined();
    expect(screen.getAllByText("102").length).toBeGreaterThan(0);
    expect(screen.getByText("0.362")).toBeDefined();
    expect(
      screen.getByText(/hard guarantee: estimated tokens ≤ token budget/),
    ).toBeDefined();
    expect(screen.getByText("memforge-char-word/v1")).toBeDefined();
  });

  it("shows the pipeline: candidates → selected → suppressed → final context", async () => {
    const user = userEvent.setup();
    installFetch([COMPILE]);
    renderAt(<ContextPage />, "/context");
    await user.click(screen.getByRole("button", { name: /Compile context/ }));

    expect(await screen.findByText("Candidate memories")).toBeDefined();
    expect(screen.getByText("Selected")).toBeDefined();
    expect(screen.getByText("Suppressed")).toBeDefined();
    expect(screen.getByText("Final context")).toBeDefined();
    const flow = document.querySelector(".flow")!;
    // 5 considered, 4 selected, 1 suppressed, 58 estimated tokens.
    expect(flow.textContent).toContain("5");
    expect(flow.textContent).toContain("1 duplicate");
    expect(flow.textContent).toContain("0 conflict");
    expect(flow.textContent).toContain("0 budget");
  });

  it("lists every selection decision with its reason and remaining budget", async () => {
    const user = userEvent.setup();
    installFetch([COMPILE]);
    renderAt(<ContextPage />, "/context");
    await user.click(screen.getByRole("button", { name: /Compile context/ }));

    expect(await screen.findByText("Selection reasons")).toBeDefined();
    expect(screen.getAllByText("selected").length).toBeGreaterThan(0);
    expect(
      screen.getByText(/highest utility \(0.7103\) that fits the remaining budget of 160 tokens/),
    ).toBeDefined();
    // One row per candidate: 5.
    const traceTable = screen.getByText("Selection reasons").closest("section")!;
    expect(traceTable.querySelectorAll("tbody tr")).toHaveLength(5);
  });

  it("shows duplicate suppression with the memory that beat it", async () => {
    const user = userEvent.setup();
    installFetch([COMPILE]);
    renderAt(<ContextPage />, "/context");
    await user.click(screen.getByRole("button", { name: /Compile context/ }));

    expect(await screen.findByText(/Suppressed candidates/)).toBeDefined();
    expect(screen.getAllByText("duplicate_of").length).toBe(3);
    expect(screen.getAllByText(truncateMiddle(IDS.PROFILE, 18)).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/represents this duplicate group in the context/).length).toBe(2);
  });

  it("shows conflict suppression when the backend suppressed one", async () => {
    const user = userEvent.setup();
    installFetch([
      {
        method: "POST",
        path: "/context/compile",
        body: {
          ...compile,
          status: "ARCHIVED",
          suppressed_candidates: [
            {
              memory_id: IDS.FACT_PYTHON,
              memory_type: "FACT",
              selected: false,
              reason: "conflict_superseded",
              detail: `superseded by ${IDS.FACT_TS}, which won the recorded conflict resolution`,
              utility_score: 0.3,
              value_score: 0.4,
              estimated_tokens: 10,
              remaining_tokens: 160,
              related_memory_id: IDS.FACT_TS,
            },
          ],
          selection_trace: [
            ...compile.selection_trace.slice(0, 4),
            {
              memory_id: IDS.FACT_PYTHON,
              memory_type: "FACT",
              selected: false,
              reason: "conflict_superseded",
              detail: `superseded by ${IDS.FACT_TS}, which won the recorded conflict resolution`,
              utility_score: 0.3,
              value_score: 0.4,
              estimated_tokens: 10,
              remaining_tokens: 160,
              related_memory_id: IDS.FACT_TS,
            },
          ],
          explanation: {
            ...compile.explanation,
            selection_counts: { ...compile.explanation.selection_counts, selected: 4, duplicate_of: 0, conflict_superseded: 1 },
          },
        },
      },
    ]);
    renderAt(<ContextPage />, "/context");
    await user.click(screen.getByRole("button", { name: /Compile context/ }));
    expect((await screen.findAllByText("conflict_superseded")).length).toBeGreaterThan(0);
    expect(
      screen.getAllByText(/which won the recorded conflict resolution/).length,
    ).toBeGreaterThan(0);
  });

  it("shows the rendered context in its own panel, separated from the analysis", async () => {
    const user = userEvent.setup();
    installFetch([COMPILE]);
    renderAt(<ContextPage />, "/context");
    await user.click(screen.getByRole("button", { name: /Compile context/ }));

    expect(await screen.findByText("Final rendered context")).toBeDefined();
    const panel = document.querySelector(".code-panel")!;
    expect(panel.textContent).toContain("PROFILE");
    expect(panel.textContent).toContain("User prefers Python for backend work.");
    expect(screen.getByText("context_text · 58 estimated tokens")).toBeDefined();
  });

  it("shows the estimator, utility formula and ordering policy the backend reported", async () => {
    const user = userEvent.setup();
    installFetch([COMPILE]);
    renderAt(<ContextPage />, "/context");
    await user.click(screen.getByRole("button", { name: /Compile context/ }));

    expect(await screen.findByText("Provenance and explanation")).toBeDefined();
    expect(screen.getByText(/memforge-char-word\/v1: estimated = max\(1, ceil/)).toBeDefined();
    expect(screen.getByText(/utility = value \* \(1 - 0.35/)).toBeDefined();
    expect(screen.getByText("section-then-utility/v1")).toBeDefined();
    expect(screen.getByText("relevance 0.40")).toBeDefined();
  });

  it("repeats the backend's own caveats rather than paraphrasing them", async () => {
    const user = userEvent.setup();
    installFetch([COMPILE]);
    renderAt(<ContextPage />, "/context");
    await user.click(screen.getByRole("button", { name: /Compile context/ }));
    expect(
      await screen.findByText(/not provider tokenisations and not billing data/),
    ).toBeDefined();
    expect(screen.getByText(/content is never paraphrased or summarised/)).toBeDefined();
  });

  it("does not recompute a token count of its own", async () => {
    const user = userEvent.setup();
    installFetch([
      {
        method: "POST",
        path: "/context/compile",
        body: { ...compile, estimated_tokens: 41, remaining_tokens: 119, utilization: 0.25625 },
      },
    ]);
    renderAt(<ContextPage />, "/context");
    await user.click(screen.getByRole("button", { name: /Compile context/ }));
    expect((await screen.findAllByText("41")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("119").length).toBeGreaterThan(0);
    expect(screen.getByText("0.256")).toBeDefined();
  });

  it("shows a budget violation when the backend reports one", async () => {
    const user = userEvent.setup();
    installFetch([
      {
        method: "POST",
        path: "/context/compile",
        body: { ...compile, estimated_tokens: 161, remaining_tokens: -1, within_budget: false },
      },
    ]);
    renderAt(<ContextPage />, "/context");
    await user.click(screen.getByRole("button", { name: /Compile context/ }));
    expect(await screen.findByText("over budget")).toBeDefined();
  });

  it("shows the backend's validation error for an impossible budget", async () => {
    const user = userEvent.setup();
    installFetch([
      {
        method: "POST",
        path: "/context/compile",
        status: 400,
        body: { error: { code: "CONTEXT_VALIDATION_ERROR", message: "token budget exceeds the maximum" } },
      },
    ]);
    renderAt(<ContextPage />, "/context");
    await user.click(screen.getByRole("button", { name: /Compile context/ }));
    expect(await screen.findByText("CONTEXT_VALIDATION_ERROR")).toBeDefined();
  });

  it("reports a malformed compile payload instead of rendering a budget", async () => {
    const user = userEvent.setup();
    installFetch([
      {
        method: "POST",
        path: "/context/compile",
        body: { ...compile, estimated_tokens: "58 tokens" },
      },
    ]);
    renderAt(<ContextPage />, "/context");
    await user.click(screen.getByRole("button", { name: /Compile context/ }));
    expect(await screen.findByText("Malformed API response")).toBeDefined();
    expect(screen.queryByText("within budget")).toBeNull();
  });

  it("shows a running state while the compiler works", async () => {
    const user = userEvent.setup();
    installFetch([{ method: "POST", path: "/context/compile", pending: true }]);
    renderAt(<ContextPage />, "/context");
    await user.click(screen.getByRole("button", { name: /Compile context/ }));
    await waitFor(() =>
      expect(screen.getByText("Retrieving, scoring and packing…")).toBeDefined(),
    );
  });
});