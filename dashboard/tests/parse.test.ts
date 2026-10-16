import { describe, expect, it } from "vitest";
import { MalformedResponseError, numberRecord, parseResponse } from "../src/api/parse";
import {
  agentShape,
  compileShape,
  evaluationShape,
  memoryPageShape,
  searchShape,
  traceEventShape,
  metricsShape,
} from "../src/api/client";
import {
  evaluation,
  memoryPage,
  metrics,
  search,
  traceEvents,
  compile,
  agent,
} from "./fixtures";

describe("response-shape validation", () => {
  it("accepts the real memory page payload", () => {
    expect(() => parseResponse(memoryPage, memoryPageShape, "GET /memories")).not.toThrow();
  });

  it("accepts the real retrieval, context, metrics, evaluation and agent payloads", () => {
    expect(() => parseResponse(search, searchShape, "search")).not.toThrow();
    expect(() => parseResponse(compile, compileShape, "compile")).not.toThrow();
    expect(() => parseResponse(metrics, metricsShape, "metrics")).not.toThrow();
    expect(() => parseResponse(evaluation, evaluationShape, "evaluation")).not.toThrow();
    traceEvents.forEach((event) =>
      expect(() => parseResponse(event, traceEventShape, "event")).not.toThrow(),
    );
    expect(() => parseResponse(agent, agentShape, "agent")).not.toThrow();
  });

  it("names the exact JSON path of a type violation", () => {
    const broken = {
      ...memoryPage,
      items: [{ ...memoryPage.items[0]!, confidence: "high" }],
    };
    expect(() => parseResponse(broken, memoryPageShape, "GET /memories")).toThrow(
      /GET \/memories: malformed API response at items\[0\]\.confidence \(expected number\)/,
    );
  });

  it("rejects a wrong enum value instead of guessing a category", () => {
    const broken = { ...memoryPage, items: [{ ...memoryPage.items[0]!, memory_type: "MEME" }] };
    expect(() => parseResponse(broken, memoryPageShape, "GET /memories")).toThrow(MalformedResponseError);
  });

  it("rejects a missing required field", () => {
    const { total: _total, ...broken } = memoryPage;
    expect(() => parseResponse(broken, memoryPageShape, "GET /memories")).toThrow(/expected number/);
  });

  it("accepts null for a nullable field and rejects an out-of-place null", () => {
    const withNull = { ...memoryPage, items: [{ ...memoryPage.items[0]!, source: null }] };
    expect(() => parseResponse(withNull, memoryPageShape, "GET /memories")).not.toThrow();
    expect(() => parseResponse({ a: null }, { object: { a: "string" } }, "x")).toThrow(
      /malformed API response at a/,
    );
  });

  it("walks arrays and nested objects", () => {
    const broken = {
      ...search,
      results: [{ ...search.results[0]!, factors: "not-a-list" }],
    };
    expect(() => parseResponse(broken, searchShape, "search")).toThrow(
      /malformed API response at results\[0\]\.factors/,
    );
  });

  it("rejects a non-finite number", () => {
    const broken = { ...metrics, event_count: Number.NaN };
    expect(() => parseResponse(broken, metricsShape, "metrics")).toThrow(/event_count/);
  });

  it("rejects a scalar where a record of numbers is declared", () => {
    expect(() => parseResponse({ traces: "no" }, { object: { traces: numberRecord } }, "x")).toThrow(
      /expected object of number/,
    );
  });

  it("reports the root when the whole payload is the wrong type", () => {
    expect(() => parseResponse([], memoryPageShape, "GET /memories")).toThrow(/<root>/);
  });
});
