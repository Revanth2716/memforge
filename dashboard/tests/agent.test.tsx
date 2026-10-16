import { describe, expect, it } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AgentPage } from "../src/pages/AgentPage";
import { installFetch, renderAt } from "./helpers";
import { agent, agentContextOnly, agentCustomProvider, agentSmallTalk, IDS } from "./fixtures";
import { TRACE_HEADER } from "../src/api/client";

const RESPOND = (body: unknown = agent) => ({
  method: "POST",
  path: "/agent/respond",
  body,
  headers: { [TRACE_HEADER]: IDS.TRACE },
});

describe("agent playground", () => {
  it("starts with no turn and explains what the default request does", () => {
    installFetch([]);
    renderAt(<AgentPage />, "/agent");
    expect(screen.getByText("No turn run yet")).toBeDefined();
    expect(screen.getByRole("button", { name: /Run agent turn/ })).toBeDefined();
  });

  it("posts the real agent request, with storage off by default", async () => {
    const user = userEvent.setup();
    const fetch = installFetch([RESPOND()]);
    renderAt(<AgentPage />, "/agent");

    await user.clear(screen.getByLabelText("User request"));
    await user.type(screen.getByLabelText("User request"), "What language?");
    await user.clear(screen.getByLabelText("Context token budget"));
    await user.type(screen.getByLabelText("Context token budget"), "250");
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));

    await waitFor(() => expect(fetch.calls.length).toBeGreaterThan(0));
    expect(fetch.calls[0]).toMatchObject({
      method: "POST",
      path: "/agent/respond",
      body: {
        request: "What language?",
        token_budget: 250,
        top_k: 10,
        retrieval_mode: "lexical",
        use_memory: true,
        context_only: false,
        extract_candidates: true,
        store_candidates: false,
        source: "agent:dashboard",
      },
    });
  });

  it("renders the memory gate result with its reason", async () => {
    const user = userEvent.setup();
    installFetch([RESPOND()]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));

    expect(await screen.findByText("Memory gate")).toBeDefined();
    expect(screen.getAllByText("used").length).toBeGreaterThan(0);
    expect(screen.getByText("the request may depend on stored memory")).toBeDefined();
  });

  it("shows a skipped gate for small talk and compiles no context", async () => {
    const user = userEvent.setup();
    installFetch([RESPOND(agentSmallTalk)]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByRole("button", { name: /Try small talk/ }));
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));

    expect(await screen.findByText("skipped")).toBeDefined();
    expect(
      screen.getByText(/memory was not used for this turn, so no context was compiled/),
    ).toBeDefined();
  });

  it("renders the provider, model and answer exactly as the API returned them", async () => {
    const user = userEvent.setup();
    installFetch([RESPOND()]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));

    expect(await screen.findByText("mock")).toBeDefined();
    expect(screen.getByText("model mock-echo-v1")).toBeDefined();
    expect(screen.getByText(/\[mock provider: deterministic offline stand-in, no model was called\]/)).toBeDefined();
    expect(document.body.textContent).toContain("MemForge");
  });

  it("reports the configured custom provider when the backend uses one", async () => {
    const user = userEvent.setup();
    installFetch([RESPOND(agentCustomProvider)]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));

    expect(await screen.findByText("custom")).toBeDefined();
    expect(screen.getByText("model memforge-oss-120b")).toBeDefined();
    // Usage is echoed only when the provider sent it.
    expect(screen.getByText("1,292 tokens")).toBeDefined();
    expect(screen.getByText("reported by provider")).toBeDefined();
  });

  it("says the mock reported no usage instead of inventing one", async () => {
    const user = userEvent.setup();
    installFetch([RESPOND()]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));

    expect(await screen.findByText("Usage reported")).toBeDefined();
    expect(screen.getByText("none")).toBeDefined();
    expect(screen.getByText(/the provider sent no usage block; none is derived/)).toBeDefined();
  });

  it("renders the recalled memories with their sections and utility", async () => {
    const user = userEvent.setup();
    installFetch([RESPOND()]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));

    expect((await screen.findAllByText("Recalled memories")).length).toBeGreaterThan(0);
    expect(screen.getByText("1. PROFILE")).toBeDefined();
    expect(screen.getByText("4. RECENT_EVENTS")).toBeDefined();
    expect(screen.getByText(/10 tokens · retrieval 0.699/)).toBeDefined();
  });

  it("renders the compiled context and its budget, from the backend's figures", async () => {
    const user = userEvent.setup();
    installFetch([RESPOND()]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));

    expect(await screen.findByText("Compiled context")).toBeDefined();
    expect(screen.getByText("/ 300 estimated tokens")).toBeDefined();
    expect(screen.getAllByText("58").length).toBeGreaterThan(0);
    expect(screen.getByText("242")).toBeDefined();
    const panel = document.querySelector(".code-panel")!;
    expect(panel.textContent).toContain("User prefers Python for backend work.");
  });

  it("shows the candidate and states that nothing was stored", async () => {
    const user = userEvent.setup();
    installFetch([RESPOND()]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));

    expect(await screen.findByText("MemForge uses Python and FastAPI.")).toBeDefined();
    expect(screen.getByText(/explicit 'remember:' instruction in the request/)).toBeDefined();
    expect(screen.getByText("Candidates stored")).toBeDefined();
    expect(screen.getByText(/storage is opt-in/)).toBeDefined();
  });

  it("stores candidates when the operator asks for it", async () => {
    const user = userEvent.setup();
    installFetch([
      RESPOND({ ...agent, stored_memory_ids: [IDS.FACT_PYTHON] }),
    ]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByLabelText(/store candidates/));
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));

    expect(await screen.findByText("Stored")).toBeDefined();
    expect(screen.getAllByText(IDS.FACT_PYTHON).length).toBeGreaterThan(0);
  });

  it("renders the operational operation record, not a chain of thought", async () => {
    const user = userEvent.setup();
    installFetch([RESPOND()]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));

    expect(await screen.findByText("Operations")).toBeDefined();
    expect(screen.getByText("compile_context")).toBeDefined();
    expect(screen.getByText("respond")).toBeDefined();
    expect(screen.getByText("extract_candidates")).toBeDefined();
    expect(screen.getAllByText(/3 recalled/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/58 est\. context tokens/).length).toBeGreaterThan(0);
    // The messages array MemForge returns is never rendered.
    expect(document.body.textContent).not.toContain("You are answering with compiled memory context.");
  });

  it("states plainly that no chain-of-thought is recorded", async () => {
    const user = userEvent.setup();
    installFetch([RESPOND()]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));
    expect(
      await screen.findByText(/no chain-of-thought is recorded or rendered anywhere/),
    ).toBeDefined();
  });

  it("promotes the trace id and links to its timeline", async () => {
    const user = userEvent.setup();
    installFetch([RESPOND()]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));

    expect((await screen.findAllByText("Trace id")).length).toBeGreaterThan(0);
    const link = screen.getByRole("link", { name: "Open the trace timeline" });
    expect(link.getAttribute("href")).toBe(`/traces/${IDS.TRACE}`);
  });

  it("shows no trace id when the response carried none", async () => {
    const user = userEvent.setup();
    installFetch([{ method: "POST", path: "/agent/respond", body: agent }]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));
    expect((await screen.findAllByText(/no trace id was returned/)).length).toBeGreaterThan(0);
    expect(screen.queryByRole("link", { name: "Open the trace timeline" })).toBeNull();
  });

  it("makes a context-only turn explicit instead of showing a blank answer", async () => {
    const user = userEvent.setup();
    installFetch([RESPOND(agentContextOnly)]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByLabelText(/context only/));
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));

    expect(await screen.findByText("No answer was produced")).toBeDefined();
    expect(screen.getByText(/context-only run, no provider call/)).toBeDefined();
  });

  it("surfaces a provider failure as an explicit error, not a fabricated answer", async () => {
    const user = userEvent.setup();
    installFetch([
      {
        method: "POST",
        path: "/agent/respond",
        status: 504,
        body: {
          error: { code: "LLM_TIMEOUT", message: "custom API timeout after 30.0s", details: { kind: "timeout" } },
        },
      },
    ]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));

    expect(await screen.findByText("LLM_TIMEOUT")).toBeDefined();
    expect(screen.getByText(/custom API timeout after 30.0s/)).toBeDefined();
    expect(document.body.textContent).not.toContain("Answer\n");
  });

  it("reports a malformed agent payload", async () => {
    const user = userEvent.setup();
    installFetch([RESPOND({ ...agent, context_estimated_tokens: "58" })]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));
    expect(await screen.findByText("Malformed API response")).toBeDefined();
  });

  it("shows a running state while the turn executes", async () => {
    const user = userEvent.setup();
    installFetch([{ method: "POST", path: "/agent/respond", pending: true }]);
    renderAt(<AgentPage />, "/agent");
    await user.click(screen.getByRole("button", { name: /Run agent turn/ }));
    await waitFor(() =>
      expect(screen.getByText("Gate → recall → compile → provider…")).toBeDefined(),
    );
  });
});