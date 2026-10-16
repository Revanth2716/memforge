import { describe, expect, it } from "vitest";
import {
  REDACTED,
  isDeniedKey,
  looksLikeSecret,
  safeEntries,
  safeMapping,
  safeValue,
} from "../src/lib/redact";
import { buildTraceTree, flattenTree, eventCaption, AGENT_LIFECYCLE } from "../src/lib/traceTree";
import { traceEvents, traceDetail } from "./fixtures";

describe("redaction", () => {
  it("drops credential-shaped keys whatever their value", () => {
    for (const key of ["api_key", "Authorization", "api-key", "token", "secret", "password", "prompt", "answer", "content", "chain_of_thought"]) {
      expect(isDeniedKey(key)).toBe(true);
    }
    expect(isDeniedKey("estimated_tokens")).toBe(false);
    expect(isDeniedKey("memory_ids")).toBe(false);
  });

  it("redacts a credential-shaped value under a benign key", () => {
    expect(looksLikeSecret("sk-abcdef123456")).toBe(true);
    expect(looksLikeSecret("Authorization: Bearer xyz")).toBe(true);
    expect(looksLikeSecret("memforge-char-word/v1")).toBe(false);
    expect(safeValue("Bearer abcdef")).toBe(REDACTED);
    expect(safeValue("lexical")).toBe("lexical");
    expect(safeValue(58)).toBe("58");
    expect(safeValue(null)).toBe("—");
  });

  it("renders safe mappings and drops the rest", () => {
    const safe = safeMapping({ api_key: "sk-1", estimated_tokens: 58, provider: "mock" });
    expect(safe).toEqual({ estimated_tokens: 58, provider: "mock" });
  });

  it("returns only renderable entries", () => {
    const entries = safeEntries({
      estimated_tokens: 58,
      api_key: "sk-1",
      provider: "mock",
      usage_reported: false,
    });
    expect(entries).toEqual([
      ["estimated_tokens", "58"],
      ["provider", "mock"],
      ["usage_reported", "false"],
    ]);
    expect(safeEntries(undefined)).toEqual([]);
  });
});

describe("trace tree", () => {
  it("nests events under their parent_id and keeps the recorded sequence", () => {
    const nodes = buildTraceTree(traceEvents);
    // The retrieval completion points at its opening event.
    const retrievalStart = nodes
      .flatMap((node) => [node, ...node.children])
      .find((node) => node.event.kind === "retrieval_start");
    expect(retrievalStart?.children.map((child) => child.event.kind)).toEqual([
      "retrieval_complete",
    ]);
  });

  it("renders children after their parent, depth first", () => {
    const ordered = flattenTree(buildTraceTree(traceEvents));
    expect(ordered.map((node) => node.event.sequence)).toEqual(
      [...traceEvents].sort((left, right) => left.sequence - right.sequence).map((event) => event.sequence),
    );
  });

  it("assigns increasing depth", () => {
    const nodes = buildTraceTree(traceEvents);
    const nested = nodes.flatMap((node) => node.children);
    expect(nested.every((node) => node.depth === 1)).toBe(true);
  });

  it("treats an event whose parent is missing as a root, so no row is dropped", () => {
    const orphan = {
      ...traceEvents[0]!,
      event_id: "orphan",
      parent_id: "does-not-exist",
    };
    const nodes = buildTraceTree([orphan]);
    expect(nodes).toHaveLength(1);
    expect(nodes[0]!.event.event_id).toBe("orphan");
  });

  it("captions an event kind", () => {
    expect(eventCaption("retrieval_complete")).toBe("Retrieval complete");
    expect(eventCaption("llm_request")).toBe("Llm request");
  });

  it("documents the lifecycle order the backend emits", () => {
    expect(AGENT_LIFECYCLE.map((stage) => stage.operation)).toEqual([
      "agent.request",
      "agent.memory_gate",
      "memory.retrieval",
      "context.compile",
      "llm.generate",
      "agent.memory_candidate",
      "memory.write",
      "agent.response",
    ]);
  });

  it("derives the recorded operations from the fixture summary", () => {
    expect(Object.keys(traceDetail.summary.operations)).toContain("memory.retrieval");
  });
});
