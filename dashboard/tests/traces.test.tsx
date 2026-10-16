import { describe, expect, it } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TracesPage } from "../src/pages/TracesPage";
import { TraceDetailPage } from "../src/pages/TraceDetailPage";
import { installFetch, renderAt, renderWithParams } from "./helpers";
import { emptyTraceList, IDS, traceDetail, traceEvents, traceList } from "./fixtures";

const traceRoutes = () => [
  { path: "/traces", body: traceList },
  { path: new RegExp(`^/traces/${IDS.TRACE}$`), body: traceDetail },
  { path: new RegExp(`^/traces/${IDS.TRACE}/events$`), body: traceEvents },
];

describe("trace list", () => {
  it("renders recent traces with kind, operation, start time and attributes", async () => {
    installFetch(traceRoutes());
    renderAt(<TracesPage />, "/traces");

    const link = await screen.findByRole("link", { name: IDS.TRACE });
    expect(link.getAttribute("href")).toBe(`/traces/${IDS.TRACE}`);
    expect(screen.getAllByText("agent_request").length).toBeGreaterThan(0);
    expect(screen.getByText("agent.respond")).toBeDefined();
    expect(screen.getAllByText(/model=mock-echo-v1/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/newest first/).length).toBeGreaterThan(0);
  });

  it("filters by kind on the backend query string", async () => {
    const user = userEvent.setup();
    const fetch = installFetch(traceRoutes());
    renderAt(<TracesPage />, "/traces");
    await screen.findByRole("link", { name: IDS.TRACE });

    await user.selectOptions(screen.getByLabelText("Kind"), "mcp_request");
    await waitFor(() => expect(fetch.calls.at(-1)!.path).toContain("kind=mcp_request"));

    await user.selectOptions(screen.getByLabelText("Limit"), "50");
    await waitFor(() => expect(fetch.calls.at(-1)!.path).toContain("limit=50"));
  });

  it("shows an empty state when nothing has been recorded", async () => {
    installFetch([{ path: "/traces", body: emptyTraceList }]);
    renderAt(<TracesPage />, "/traces");
    expect(await screen.findByText("No trace has been recorded yet")).toBeDefined();
    expect(screen.getByText(/Run an agent turn, a memory write or a retrieval/)).toBeDefined();
  });

  it("states what a trace holds and what it never holds", async () => {
    installFetch(traceRoutes());
    renderAt(<TracesPage />, "/traces");
    await screen.findByRole("link", { name: IDS.TRACE });
    expect(screen.getByText("RECORDED")).toBeDefined();
    expect(screen.getByText("NEVER RECORDED")).toBeDefined();
    expect(screen.getByText(/chain-of-thought/)).toBeDefined();
  });

  it("reports an API failure", async () => {
    installFetch([
      { path: "/traces", status: 500, body: { error: { code: "OBSERVABILITY_ERROR", message: "corrupt row" } } },
    ]);
    renderAt(<TracesPage />, "/traces");
    expect(await screen.findByText("OBSERVABILITY_ERROR")).toBeDefined();
  });
});

describe("trace detail", () => {
  it("shows the trace id prominently and links back to the list", async () => {
    installFetch(traceRoutes());
    renderWithParams(<TraceDetailPage />, `/traces/${IDS.TRACE}`, "/traces/:traceId");

    const heading = await screen.findByRole("heading", { name: IDS.TRACE });
    expect(heading).toBeDefined();
    expect(screen.getByRole("link", { name: /All traces/ })).toBeDefined();
  });

  it("renders the summary counters the backend derived", async () => {
    installFetch(traceRoutes());
    renderWithParams(<TraceDetailPage />, `/traces/${IDS.TRACE}`, "/traces/:traceId");

    expect(await screen.findByText("succeeded")).toBeDefined();
    expect(screen.getAllByText("skipped").length).toBeGreaterThan(0);
    expect(screen.getByText("75.3 ms")).toBeDefined();
    expect(screen.getByText(/measured at runtime, not a benchmark/)).toBeDefined();
    expect(screen.getAllByText("44.0 ms").length).toBeGreaterThan(0);
  });

  it("draws the lifecycle rail in the recorded order", async () => {
    installFetch(traceRoutes());
    renderWithParams(<TraceDetailPage />, `/traces/${IDS.TRACE}`, "/traces/:traceId");

    const rail = await waitFor(() => {
      const element = document.querySelector(".rail");
      expect(element).not.toBeNull();
      return element!;
    });
    const stages = [...rail.querySelectorAll(".rail__step")].map((node) => node.textContent);
    expect(stages).toEqual([
      "Agent Request",
      "Retrieval",
      "Context Compilation",
      "LLM Call",
      "Memory Gate",
      "Candidate Extraction",
      "Memory Write",
      "Response",
    ]);
  });

  it("renders one node per event, nested under its parent", async () => {
    installFetch(traceRoutes());
    renderWithParams(<TraceDetailPage />, `/traces/${IDS.TRACE}`, "/traces/:traceId");

    await waitFor(() =>
      expect(document.querySelectorAll(".timeline__node")).toHaveLength(traceEvents.length),
    );
    const nested = [...document.querySelectorAll(".timeline__node")].filter(
      (node) => Number(node.getAttribute("data-depth")) > 0,
    );
    // retrieval_complete, context_compile_complete and llm_response are nested.
    expect(nested.length).toBe(3);
  });

  it("shows operation, status, duration, sequence and parent for every event", async () => {
    installFetch(traceRoutes());
    renderWithParams(<TraceDetailPage />, `/traces/${IDS.TRACE}`, "/traces/:traceId");

    await waitFor(() => expect(document.querySelectorAll(".timeline__node").length).toBeGreaterThan(0));
    expect(screen.getByText("Retrieval complete")).toBeDefined();
    expect(screen.getAllByText("memory.retrieval").length).toBeGreaterThan(0);
    expect(screen.getAllByText("31.1 ms").length).toBeGreaterThan(0);
    expect(screen.getByText("#2")).toBeDefined();
    expect(screen.getAllByText(/parent/).length).toBeGreaterThan(0);
    expect(screen.getAllByText("ok").length).toBeGreaterThan(0);
    expect(screen.getAllByText("skipped").length).toBeGreaterThan(0);
  });

  it("shows the safe metadata and the memory ids each event touched", async () => {
    installFetch(traceRoutes());
    renderWithParams(<TraceDetailPage />, `/traces/${IDS.TRACE}`, "/traces/:traceId");

    await waitFor(() => expect(screen.getByText("Retrieval complete")).toBeDefined());
    expect(screen.getByText("Estimated tokens")).toBeDefined();
    expect(screen.getByText("58")).toBeDefined();
    expect(screen.getAllByText(/^mock · mock-echo-v1$/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/e71e8d69/).length).toBeGreaterThan(0);
  });

  it("never renders a credential, a prompt or an answer", async () => {
    installFetch([
      ...traceRoutes(),
      {
        path: new RegExp(`^/traces/${IDS.TRACE}/events$`),
        body: traceEvents.map((event, index) =>
          index === 0
            ? {
                ...event,
                metadata: {
                  ...event.metadata,
                  // A hostile payload: the filter must still drop these.
                  api_key: "sk-should-never-render",
                  authorization: "Bearer secret-token",
                  prompt: "system prompt text",
                  answer: "the answer text",
                  chain_of_thought: "step 1 then step 2",
                },
              }
            : event,
        ),
      },
    ]);
    renderWithParams(<TraceDetailPage />, `/traces/${IDS.TRACE}`, "/traces/:traceId");

    await waitFor(() => expect(screen.getByText("Request")).toBeDefined());
    const text = document.body.textContent ?? "";
    expect(text).not.toContain("sk-should-never-render");
    expect(text).not.toContain("Bearer secret-token");
    expect(text).not.toContain("system prompt text");
    expect(text).not.toContain("the answer text");
    expect(text).not.toContain("step 1 then step 2");
    // The safe fields of the same event still render.
    expect(text).toContain("Provider");
  });

  it("shows an empty state for a trace with no events", async () => {
    installFetch([
      { path: "/traces", body: traceList },
      { path: new RegExp(`^/traces/${IDS.TRACE}$`), body: { ...traceDetail, event_count: 0 } },
      { path: new RegExp(`^/traces/${IDS.TRACE}/events$`), body: [] },
    ]);
    renderWithParams(<TraceDetailPage />, `/traces/${IDS.TRACE}`, "/traces/:traceId");
    expect((await screen.findAllByText("This trace recorded no event")).length).toBeGreaterThan(0);
  });

  it("reports an unknown trace id with the backend's envelope", async () => {
    installFetch([
      {
        path: /^\/traces\/.+$/,
        status: 404,
        body: { error: { code: "TRACE_NOT_FOUND", message: "no trace with id 'nope'" } },
      },
    ]);
    renderWithParams(<TraceDetailPage />, "/traces/nope", "/traces/:traceId");
    expect((await screen.findAllByText("TRACE_NOT_FOUND")).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/no trace with id/).length).toBeGreaterThan(0);
  });

  it("reports a malformed event list instead of drawing a timeline", async () => {
    installFetch([
      { path: "/traces", body: traceList },
      { path: new RegExp(`^/traces/${IDS.TRACE}$`), body: traceDetail },
      {
        path: new RegExp(`^/traces/${IDS.TRACE}/events$`),
        body: [{ ...traceEvents[0]!, status: "probably-fine" }],
      },
    ]);
    renderWithParams(<TraceDetailPage />, `/traces/${IDS.TRACE}`, "/traces/:traceId");
    expect((await screen.findAllByText("Malformed API response")).length).toBeGreaterThan(0);
    expect(document.querySelectorAll(".timeline__node")).toHaveLength(0);
  });

  it("shows a loading state while the events are in flight", async () => {
    installFetch([
      { path: "/traces", body: traceList },
      { path: new RegExp(`^/traces/${IDS.TRACE}$`), body: traceDetail },
      { path: new RegExp(`^/traces/${IDS.TRACE}/events$`), pending: true },
    ]);
    renderWithParams(<TraceDetailPage />, `/traces/${IDS.TRACE}`, "/traces/:traceId");
    await waitFor(() => expect(screen.getAllByText("Loading…").length).toBeGreaterThan(0));
  });
});