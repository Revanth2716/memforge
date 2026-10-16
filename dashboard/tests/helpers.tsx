/**
 * Test helpers.
 *
 * Tests stub **`fetch`**, not the API client: every test therefore exercises the
 * real request construction, response parsing, error-envelope handling and
 * malformed-payload detection in `src/api/client.ts`, as well as the page that
 * renders the result.
 *
 * The stubs return shape-faithful transcriptions of real MemForge responses
 * (see `fixtures.ts`). The real application never uses this module.
 */

import { afterEach, beforeEach, vi } from "vitest";
import type { ReactElement } from "react";
import { render, type RenderResult } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { clearResourceCache } from "../src/hooks/useResource";

export interface FetchCall {
  method: string;
  path: string;
  body: unknown;
}

/** A route the test fetch router can answer. */
export interface Route {
  /** HTTP method; defaults to `GET`. POST routes must name it explicitly. */
  method?: string;
  /** Path relative to the API prefix, e.g. `/memories`. A regex matches too. */
  path: string | RegExp;
  /** Response status; defaults to 200. */
  status?: number;
  /** JSON body. `undefined` produces an empty body. */
  body?: unknown;
  /** Extra response headers. */
  headers?: Record<string, string>;
  /** Never settles, for loading-state assertions. */
  pending?: boolean;
  /** Rejected by `fetch` itself, for unreachable-API assertions. */
  networkError?: string;
}

/** A minimal stand-in for `Response`: the client only reads these four fields. */
interface FakeResponse {
  ok: boolean;
  status: number;
  headers: { get(name: string): string | null };
  text(): Promise<string>;
}

function toResponse(spec: {
  status?: number;
  body?: unknown;
  headers?: Record<string, string>;
}): FakeResponse {
  const status = spec.status ?? 200;
  const raw =
    spec.body === undefined
      ? ""
      : typeof spec.body === "string"
        ? spec.body
        : JSON.stringify(spec.body);
  const headers = new Map<string, string>(
    Object.entries({
      "content-type": "application/json",
      ...(spec.headers ?? {}),
    }).map(([key, value]) => [key.toLowerCase(), value]),
  );
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (name) => headers.get(name.toLowerCase()) ?? null },
    text: async () => raw,
  };
}

export interface InstalledFetch {
  calls: FetchCall[];
  /** Requests grouped by method + path prefix, e.g. `POST /memories/search`. */
  summary(): string[];
}

/**
 * Replace `globalThis.fetch` with a router over `routes`.
 *
 * Any request without a matching route returns `404 {"error":{"code":"HTTP_ERROR"}}`,
 * which the client reports as an *unavailable endpoint*.
 */
export function installFetch(routes: readonly Route[] = []): InstalledFetch {
  const calls: FetchCall[] = [];
  const matchers = routes.map((route) => ({
    route,
    method: (route.method ?? "GET").toUpperCase(),
  }));

  globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : String(input);
    const parsed = new URL(url, "http://dashboard.test");
    const path = `${parsed.pathname}${parsed.search}`.replace(/^\/api\/v1/, "");
    const method = (init?.method ?? "GET").toUpperCase();
    calls.push({
      method,
      path,
      body: init?.body === undefined ? undefined : JSON.parse(String(init.body)),
    });

    const match = matchers.find(
      (candidate) =>
        candidate.method === method &&
        (typeof candidate.route.path === "string"
          ? path === candidate.route.path || path.startsWith(`${candidate.route.path}?`)
          : candidate.route.path.test(path)),
    );

    if (match === undefined) {
      return toResponse({
        status: 404,
        body: { error: { code: "HTTP_ERROR", message: `no route for ${method} ${path}` } },
      }) as unknown as Response;
    }

    if (match.route.networkError !== undefined) {
      throw new TypeError(match.route.networkError);
    }
    if (match.route.pending === true) {
      return new Promise<Response>(() => {
        /* never settles on purpose */
      });
    }
    return toResponse(match.route) as unknown as Response;
  }) as unknown as typeof fetch;

  return {
    calls,
    summary: () => calls.map((call) => `${call.method} ${call.path}`),
  };
}

/** Routes for a fully healthy backend, for pages that only need data present. */
export const healthyRoutes = (
  overrides: readonly Route[] = [],
): Route[] => [
  { path: "/health", body: { status: "ok", service: "memforge", version: "1.0.0", storage: "sqlite" } },
  { path: "/memories/search", body: emptySearchBody },
  { path: "/metrics", body: emptyMetricsBody },
  { path: "/traces", body: { count: 0, traces: [] } },
  { path: "/evaluations/latest", body: emptyEvaluationBody },
  { path: "/memories", body: emptyMemoryPageBody },
  ...overrides,
];

const emptySearchBody = { query: "", top_k: 0, requested_mode: "lexical", retrieval_mode: "lexical", semantic_available: false, semantic_provider: null, semantic_model: null, degraded_reason: null, candidates_considered: 0, candidates_returned: 0, candidates_suppressed: 0, latency_ms: 0, results: [], suppressed: [] };
const emptyMetricsBody = { traces: {}, operations: {}, statuses: {}, memory: {}, context: {}, provider: {}, usage_reported: 0, failed_traces: 0, event_count: 0 };
const emptyEvaluationBody = { dataset: "memory-core", version: "v1", reference: "memory-core@v1", total_cases: 0, passed_cases: 0, failed_cases: 0, case_score: 0, metric_score: 0, overall: 0, threshold: 1, passed: false, categories: {}, metrics: [], cases: [] };
const emptyMemoryPageBody = { items: [], total: 0, limit: 25, offset: 0 };

/** Render a page inside a router, at `route`. */
export function renderAt(ui: ReactElement, route = "/"): RenderResult {
  return render(<MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>);
}

/**
 * Render a page that reads a route parameter (`/memories/:memoryId`), so
 * `useParams` sees it.
 */
export function renderWithParams(
  ui: ReactElement,
  route: string,
  pattern: string,
): RenderResult {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <Routes>
        <Route path={pattern} element={ui} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  clearResourceCache();
});

afterEach(() => {
  clearResourceCache();
  vi.restoreAllMocks();
});
