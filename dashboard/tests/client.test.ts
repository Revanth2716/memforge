import { describe, expect, it } from "vitest";
import { API_BASE, ApiError, TRACE_HEADER, api } from "../src/api/client";
import { MalformedResponseError } from "../src/api/parse";
import { installFetch } from "./helpers";
import {
  agent,
  conflicts,
  duplicates,
  evaluation,
  history,
  intelligence,
  memoryPage,
  metrics,
  relationships,
  search,
  traceEvents,
  traceList,
} from "./fixtures";

const ok = [{ path: "/health", body: { status: "ok", service: "memforge", version: "1.0.0", storage: "sqlite" } }];

describe("api base url", () => {
  it("defaults to a same-origin relative prefix", () => {
    expect(API_BASE).toBe("/api/v1");
  });
});

describe("api health", () => {
  it("calls /health outside the api prefix", async () => {
    const fetch = installFetch(ok);
    await api.health();
    expect(fetch.calls[0]).toMatchObject({ method: "GET", path: "/health" });
  });
});

describe("api listMemories", () => {
  it("puts status and memory_type on the query string", async () => {
    const fetch = installFetch([{ path: "/memories", body: memoryPage }]);
    await api.listMemories({ status: "ARCHIVED", memory_type: "FACT", limit: 25, offset: 50 });
    expect(fetch.calls[0]!.path).toBe("/memories?limit=25&offset=50&status=ARCHIVED&memory_type=FACT");
  });

  it("omits filters that were not supplied", async () => {
    const fetch = installFetch([{ path: "/memories", body: memoryPage }]);
    await api.listMemories({ limit: 10, offset: 0 });
    expect(fetch.calls[0]!.path).toBe("/memories?limit=10&offset=0");
  });
});

describe("api error handling", () => {
  it("surfaces the backend error envelope with its code", async () => {
    installFetch([
      {
        path: "/memories",
        status: 404,
        body: { error: { code: "MEMORY_NOT_FOUND", message: "no memory with id 'x'" } },
      },
    ]);
    await expect(api.listMemories({ limit: 1, offset: 0 })).rejects.toMatchObject({
      name: "ApiError",
      kind: "not_found",
      status: 404,
      code: "MEMORY_NOT_FOUND",
      message: "no memory with id 'x'",
    });
  });

  it("maps a bare 404 to an unavailable endpoint", async () => {
    installFetch([]);
    await expect(api.getMetrics()).rejects.toMatchObject({ kind: "unavailable", status: 404 });
  });

  it("maps a 422 to a validation failure", async () => {
    installFetch([
      {
        method: "POST",
        path: "/memories/search",
        status: 422,
        body: { error: { code: "VALIDATION_ERROR", message: "Request validation failed" } },
      },
    ]);
    await expect(api.search({ query: "x", top_k: 1, retrieval_mode: "lexical" })).rejects.toMatchObject(
      { kind: "validation", status: 422 },
    );
  });

  it("reports an unreachable API as its own failure kind", async () => {
    installFetch([{ path: "/metrics", networkError: "Failed to fetch" }]);
    const error = await api.getMetrics().catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).kind).toBe("unreachable");
    expect((error as ApiError).message).toContain("Cannot reach the MemForge API");
  });

  it("rejects a response that is not JSON at all", async () => {
    installFetch([{ path: "/metrics", body: "<html>502 Bad Gateway</html>" }]);
    await expect(api.getMetrics()).rejects.toThrow(MalformedResponseError);
  });

  it("rejects a response whose JSON does not match the contract", async () => {
    installFetch([{ path: "/metrics", body: { ...metrics, event_count: "many" } }]);
    await expect(api.getMetrics()).rejects.toThrow(/malformed API response at event_count/);
  });

  it("rejects a 500 with an envelope but no usable message", async () => {
    installFetch([{ path: "/metrics", status: 500, body: { error: { code: "INTERNAL_ERROR" } } }]);
    const error = (await api.getMetrics().catch((caught: unknown) => caught)) as ApiError;
    expect(error.kind).toBe("http");
    expect(error.message).toContain("HTTP 500");
  });
});

describe("api request construction", () => {
  it("sends no authorization header and no credentials, ever", async () => {
    installFetch([{ path: "/memories", body: memoryPage }]);
    await api.listMemories({ limit: 1, offset: 0 });
    const init = (globalThis.fetch as unknown as { mock: { calls: [string, RequestInit][] } }).mock
      .calls[0]![1];
    const headers = init.headers as Record<string, string>;
    expect(Object.keys(headers).map((key) => key.toLowerCase())).not.toContain("authorization");
    expect(headers).not.toHaveProperty("x-api-key");
    expect(init.credentials).toBe("omit");
  });

  it("serialises POST bodies as JSON", async () => {
    const fetch = installFetch([{ method: "POST", path: "/memories/search", body: search }]);
    await api.search({
      query: "python",
      top_k: 3,
      retrieval_mode: "hybrid",
      min_confidence: 0.1,
    });
    expect(fetch.calls[0]).toMatchObject({
      method: "POST",
      body: {
        query: "python",
        top_k: 3,
        retrieval_mode: "hybrid",
        min_confidence: 0.1,
      },
    });
  });

  it("returns the trace id header from an agent turn", async () => {
    installFetch([
      { method: "POST", path: "/agent/respond", body: agent, headers: { [TRACE_HEADER]: agent.traceId ?? "" } },
    ]);
    const result = await api.respond({
      request: "hi",
      token_budget: 100,
      top_k: 5,
      retrieval_mode: "lexical",
      use_memory: true,
      context_only: false,
      extract_candidates: false,
      store_candidates: false,
      source: "agent:test",
    });
    expect(result.traceId).toBe(agent.traceId);
  });

  it("returns a null trace id when the header is absent", async () => {
    installFetch([{ method: "POST", path: "/agent/respond", body: agent }]);
    const result = await api.respond({
      request: "hi",
      token_budget: 100,
      top_k: 5,
      retrieval_mode: "lexical",
      use_memory: true,
      context_only: false,
      extract_candidates: false,
      store_candidates: false,
      source: "agent:test",
    });
    expect(result.traceId).toBeNull();
  });
});

describe("api read endpoints", () => {
  it("reads a trace list, a trace and its events", async () => {
    const fetch = installFetch([
      { path: "/traces", body: traceList },
      { path: /^\/traces\/[^/]+$/, body: { trace: traceList.traces[0], summary: { trace_id: traceList.traces[0]!.trace_id, kind: "agent_request", operation: "agent.respond", started_at: traceList.traces[0]!.started_at, event_count: 13, operations: {}, durations_ms: {}, total_duration_ms: 0, errors: [], memory_ids: [], succeeded: true }, event_count: 13 } },
      { path: /^\/traces\/[^/]+\/events$/, body: traceEvents },
    ]);
    await api.listTraces(25, "agent_request");
    expect(fetch.calls[0]!.path).toBe("/traces?limit=25&kind=agent_request");
    await api.getTrace("abc");
    await api.getTraceEvents("abc");
    expect(fetch.summary()).toEqual([
      "GET /traces?limit=25&kind=agent_request",
      "GET /traces/abc",
      "GET /traces/abc/events",
    ]);
  });

  it("reads the latest evaluation", async () => {
    installFetch([{ path: "/evaluations/latest", body: evaluation }]);
    const result = await api.getLatestEvaluation();
    expect(result.reference).toBe("memory-core@v1");
    expect(result.overall).toBe(0.99138);
  });

  it("reads a single memory, its history, intelligence, duplicates, conflicts and relationships", async () => {
    const fetch = installFetch([
      { path: /^\/memories\/[^/]+$/, body: memoryPage.items[0] },
      { path: /^\/memories\/[^/]+\/history$/, body: history },
      { path: /^\/memories\/[^/]+\/intelligence$/, body: intelligence },
      { path: /^\/memories\/[^/]+\/duplicates$/, body: duplicates },
      { path: /^\/memories\/[^/]+\/conflicts$/, body: conflicts },
      { path: /^\/memories\/[^/]+\/relationships$/, body: relationships },
    ]);
    await api.getMemory("m1");
    await api.getHistory("m1");
    await api.getIntelligence("m1");
    await api.getDuplicates("m1");
    await api.getConflicts("m1");
    await api.getRelationships("m1");
    expect(fetch.summary()).toEqual([
      "GET /memories/m1",
      "GET /memories/m1/history",
      "GET /memories/m1/intelligence",
      "GET /memories/m1/duplicates",
      "GET /memories/m1/conflicts",
      "GET /memories/m1/relationships",
    ]);
  });

  it("posts a conflict resolution", async () => {
    const fetch = installFetch([
    {
      method: "POST",
      path: /\/resolve$/,
      body: {
        winner_memory_id: "a",
        loser_memory_id: "b",
        decided_by: "policy",
        explanation: "x",
        steps: [],
        loser_archived: true,
        conflict: {
          target_memory_id: "b",
          attribute: "a",
          source_value: "1",
          target_value: "2",
          value_similarity: 0,
          source_content: "s",
          target_content: "t",
          explanation: "e",
        },
      },
    },
  ]);
    await api.resolveConflict("left", "right");
    expect(fetch.summary()).toEqual(["POST /memories/left/conflicts/right/resolve"]);
  });
});
