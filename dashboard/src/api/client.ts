/**
 * The typed MemForge API client.
 *
 * One place holds: the base URL, request construction, response parsing, error
 * handling and the shapes the dashboard accepts. Every page goes through
 * `api.*`; no component calls `fetch`.
 *
 * **No credential ever appears here.** MemForge's API needs no authentication,
 * and building a key/token header would be a leak inside a browser bundle. The
 * only configuration accepted is `VITE_MEMFORGE_API_BASE`, a URL.
 */

import type {
  AgentBody,
  AgentRespondResponse,
  CompileBody,
  ConflictResolution,
  Conflicts,
  ContextCompileResponse,
  Duplicates,
  Evaluation,
  HealthResponse,
  Intelligence,
  Memory,
  MemoryHistory,
  MemoryListParams,
  MemoryPage,
  Metrics,
  Relationships,
  RetrievalSearchResponse,
  SearchBody,
  TraceDetail,
  TraceEvent,
  TraceListResponse,
} from "./types";
import {
  MalformedResponseError,
  nullableNumber,
  nullableString,
  numberRecord,
  optional,
  parseResponse,
  type Shape,
} from "./parse";

/** Default base: same-origin, so the build works behind any proxy. */
const DEFAULT_BASE = "/api/v1";

function resolveBase(): string {
  const configured = import.meta.env.VITE_MEMFORGE_API_BASE;
  if (typeof configured === "string" && configured.trim().length > 0) {
    return configured.trim().replace(/\/+$/, "");
  }
  return DEFAULT_BASE;
}

export const API_BASE: string = resolveBase();

/**
 * Origin (or empty string for same-origin) with the ``/api/v1`` suffix removed,
 * because ``GET /health`` is deliberately served outside the API prefix.
 */
export const ROOT_BASE: string = API_BASE.replace(/\/api\/v1\/?$/, "");

/** How a request failed, in the dashboard's own vocabulary. */
export type ApiFailureKind =
  | "unreachable"
  | "http"
  | "validation"
  | "not_found"
  | "unavailable"
  | "malformed"
  | "aborted";

/** A failed API call, carrying the backend's error envelope when there was one. */
export class ApiError extends Error {
  readonly kind: ApiFailureKind;
  readonly status: number;
  readonly code: string;
  readonly details: unknown;

  constructor(kind: ApiFailureKind, status: number, message: string, code = "", details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.code = code;
    this.details = details;
  }

  /**
   * The endpoint exists in the backend contract but this deployment does not
   * expose it (a bare 404/405 rather than a typed MemForge error).
   */
  get isUnavailable(): boolean {
    return this.kind === "unavailable";
  }
}

/** Response header carrying the id of the trace an agent turn produced. */
export const TRACE_HEADER = "X-MemForge-Trace-Id";

interface RawResponse {
  status: number;
  headers: Headers;
  body: unknown;
}

export interface RequestOptions {
  signal?: AbortSignal;
  /** Wall-clock budget for the request, in milliseconds (default 30s). */
  timeoutMs?: number;
}

const DEFAULT_TIMEOUT_MS = 30_000;

async function request(
  base: string,
  method: "GET" | "POST" | "PATCH" | "DELETE",
  path: string,
  body?: unknown,
  options: RequestOptions = {},
): Promise<RawResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const onAbort = () => controller.abort();
  options.signal?.addEventListener("abort", onAbort);
  const headers: Record<string, string> = { accept: "application/json" };
  if (body !== undefined) {
    headers["content-type"] = "application/json";
  }

  let response: Response;
  try {
    response = await fetch(`${base}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
      // Credentials are never sent: the API is unauthenticated by design, and a
      // cookie-bearing request would only widen the dashboard's blast radius.
      credentials: "omit",
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new ApiError("unreachable", 0, `Cannot reach the MemForge API at ${API_BASE}. ${detail}`);
  } finally {
    clearTimeout(timeout);
    options.signal?.removeEventListener("abort", onAbort);
  }

  const text = await response.text();
  let parsed: unknown;
  if (text.trim().length > 0) {
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new MalformedResponseError(`${method} ${path}`, "", "a JSON body");
    }
  }

  if (!response.ok) {
    throw toApiError(method, path, response.status, parsed);
  }
  return { status: response.status, headers: response.headers, body: parsed };
}

function toApiError(method: string, path: string, status: number, parsed: unknown): ApiError {
  const envelope =
    parsed !== null && typeof parsed === "object" && "error" in parsed
      ? ((parsed as { error: unknown }).error as Record<string, unknown>)
      : null;
  const code = typeof envelope?.code === "string" ? envelope.code : "";
  const message =
    typeof envelope?.message === "string" && envelope.message.length > 0
      ? envelope.message
      : `${method} ${path} failed with HTTP ${status}`;
  if (status === 404 || status === 405) {
    // A typed MemForge 404 (MEMORY_NOT_FOUND, TRACE_NOT_FOUND, ...) is a real
    // answer; a bare 404/405 means the deployment does not expose this route.
    const typed = code.length > 0 && code !== "HTTP_ERROR";
    return new ApiError(typed ? "not_found" : "unavailable", status, message, code, envelope?.details);
  }
  if (status === 422) {
    return new ApiError("validation", status, message, code, envelope?.details);
  }
  return new ApiError("http", status, message, code, envelope?.details);
}

async function read<T>(
  base: string,
  method: "GET" | "POST" | "PATCH" | "DELETE",
  path: string,
  body: unknown,
  shape: Shape,
  label: string,
): Promise<T> {
  const response = await request(base, method, path, body);
  return parseResponse<T>(response.body, shape, label);
}

// -- response shapes ----------------------------------------------------------
// Each one mirrors the Pydantic response model of the same name in
// `app/api/schemas.py`. Only the fields the dashboard renders are required.
//
// Exported so `tests/parse.test.ts` can assert the *declared* contract against
// real captured payloads: a dashboard that silently accepts a drifted backend
// is a dashboard that renders the wrong thing confidently.

export const memoryTypeShape: Shape = {
  enum: ["PROFILE", "FACT", "PREFERENCE", "EVENT", "TASK", "OBSERVATION"],
};
export const statusShape: Shape = { enum: ["ACTIVE", "ARCHIVED", "DELETED"] };

export const healthShape: Shape = {
  object: { status: "string", service: "string", version: "string", storage: "string" },
};

export const memoryShape: Shape = {
  object: {
    id: "string",
    content: "string",
    memory_type: memoryTypeShape,
    source: nullableString,
    confidence: "number",
    importance: "number",
    status: statusShape,
    created_at: "string",
    updated_at: "string",
    version: "number",
  },
};

export const memoryPageShape: Shape = {
  object: { items: { array: memoryShape }, total: "number", limit: "number", offset: "number" },
};

export const historyShape: Shape = {
  object: {
    memory_id: "string",
    current_version: "number",
    total: "number",
    items: {
      array: {
        object: {
          memory_id: "string",
          version: "number",
          change: "string",
          content: "string",
          memory_type: memoryTypeShape,
          source: nullableString,
          confidence: "number",
          importance: "number",
          status: statusShape,
          recorded_at: "string",
        },
      },
    },
  },
};

export const intelligenceShape: Shape = {
  object: {
    memory_id: "string",
    memory_type: memoryTypeShape,
    status: statusShape,
    version: "number",
    confidence: "number",
    stored_importance: "number",
    importance_score: "number",
    importance_factors: {
      array: {
        object: {
          name: "string",
          weight: "number",
          raw_value: "number",
          contribution: "number",
          explanation: "string",
        },
      },
    },
    importance_explanation: "string",
    effective_priority: "number",
    base_priority: "number",
    decay_multiplier: "number",
    age_days: "number",
    priority_explanation: "string",
    access_count: "number",
    last_accessed_at: nullableString,
    decay_half_life_days: "number",
    similarity_threshold: "number",
    computed_at: "string",
  },
};

export const duplicateShape: Shape = {
  object: {
    memory_id: "string",
    similarity: "number",
    content: "string",
    memory_type: memoryTypeShape,
    shared_tokens: { array: "string" },
    explanation: "string",
  },
};

export const duplicatesShape: Shape = {
  object: { memory_id: "string", threshold: "number", total: "number", items: { array: duplicateShape } },
};

export const conflictShape: Shape = {
  object: {
    target_memory_id: "string",
    attribute: "string",
    source_value: "string",
    target_value: "string",
    value_similarity: "number",
    source_content: "string",
    target_content: "string",
    explanation: "string",
  },
};

export const conflictsShape: Shape = {
  object: { memory_id: "string", total: "number", items: { array: conflictShape } },
};

export const resolutionShape: Shape = {
  object: {
    winner_memory_id: "string",
    loser_memory_id: "string",
    decided_by: "string",
    explanation: "string",
    steps: {
      array: {
        object: {
          factor: "string",
          left_value: "string",
          right_value: "string",
          winner_id: "string",
        },
      },
    },
    loser_archived: "boolean",
    conflict: conflictShape,
  },
};

export const relationshipsShape: Shape = {
  object: {
    memory_id: "string",
    total: "number",
    items: {
      array: {
        object: {
          source_id: "string",
          target_id: "string",
          relationship: "string",
          similarity: nullableNumber,
          created_at: "string",
        },
      },
    },
  },
};

export const conflictInfoShape: Shape = {
  object: {
    target_memory_id: "string",
    relationship: "string",
    attribute: "string",
    source_value: "string",
    target_value: "string",
    resolution: "string",
    resolution_reason: nullableString,
  },
};

export const rankingFactorShape: Shape = {
  object: {
    name: "string",
    weight: "number",
    value: "number",
    contribution: "number",
    explanation: "string",
  },
};

export const retrievalResultShape: Shape = {
  object: {
    memory_id: "string",
    content: "string",
    memory_type: memoryTypeShape,
    status: statusShape,
    confidence: "number",
    importance: "number",
    effective_priority: "number",
    freshness: "number",
    lexical_score: "number",
    semantic_score: nullableNumber,
    relevance: "number",
    final_score: "number",
    factors: { array: rankingFactorShape },
    explanation: "string",
    duplicate_of: nullableString,
    superseded_by: nullableString,
    conflicts: { array: conflictInfoShape },
  },
};

export const searchShape: Shape = {
  object: {
    query: "string",
    top_k: "number",
    requested_mode: "string",
    retrieval_mode: "string",
    semantic_available: "boolean",
    semantic_provider: nullableString,
    semantic_model: nullableString,
    degraded_reason: nullableString,
    candidates_considered: "number",
    candidates_returned: "number",
    candidates_suppressed: "number",
    latency_ms: "number",
    results: { array: retrievalResultShape },
    suppressed: { array: { object: { memory_id: "string", final_score: "number", reason: "string" } } },
  },
};

export const compressionShape: Shape = {
  object: {
    original_text: "string",
    compressed_text: "string",
    applied: { array: "string" },
    original_tokens: "number",
    compressed_tokens: "number",
    tokens_saved: "number",
    truncated: "boolean",
    retained_segments: { array: "number" },
  },
};

export const contextBlockShape: Shape = {
  object: {
    memory_id: "string",
    memory_type: memoryTypeShape,
    status: statusShape,
    source: nullableString,
    confidence: "number",
    importance: "number",
    content: "string",
    rendered_content: "string",
    section: "string",
    section_rank: "number",
    order: "number",
    relevance: "number",
    retrieval_score: "number",
    utility_score: "number",
    estimated_tokens: "number",
    compression: compressionShape,
    conflicts: { array: conflictInfoShape },
  },
};

export const contextSelectionShape: Shape = {
  object: {
    memory_id: "string",
    memory_type: memoryTypeShape,
    selected: "boolean",
    reason: "string",
    detail: "string",
    utility_score: "number",
    value_score: "number",
    estimated_tokens: "number",
    remaining_tokens: "number",
    related_memory_id: nullableString,
  },
};

export const contextExplanationShape: Shape = {
  object: {
    estimator: "string",
    estimator_formula: "string",
    utility_formula: "string",
    utility_weights: numberRecord,
    ordering_policy: "string",
    section_order: { array: "string" },
    selection_counts: numberRecord,
    compressed_blocks: "number",
    tokens_saved: "number",
    notes: { array: "string" },
  },
};

export const compileShape: Shape = {
  object: {
    query: "string",
    token_budget: "number",
    estimated_tokens: "number",
    remaining_tokens: "number",
    within_budget: "boolean",
    utilization: "number",
    candidates_considered: "number",
    selected_count: "number",
    suppressed_count: "number",
    requested_retrieval_mode: "string",
    retrieval_mode: "string",
    semantic_available: "boolean",
    degraded_reason: nullableString,
    context_text: "string",
    blocks: { array: contextBlockShape },
    selection_trace: { array: contextSelectionShape },
    suppressed_candidates: { array: contextSelectionShape },
    explanation: contextExplanationShape,
  },
};

export const usageShape: Shape = {
  object: {
    prompt_tokens: nullableNumber,
    completion_tokens: nullableNumber,
    total_tokens: nullableNumber,
    reported_by: "string",
  },
};

export const agentShape: Shape = {
  object: {
    request: "string",
    answer: "string",
    memory_used: "boolean",
    memory_gate: { record: "string" },
    provider: "string",
    model: "string",
    usage: { nullable: usageShape },
    context_text: "string",
    context_estimated_tokens: "number",
    context_token_budget: "number",
    context_blocks: {
      array: {
        object: {
          memory_id: "string",
          memory_type: memoryTypeShape,
          section: "string",
          order: "number",
          utility_score: "number",
          retrieval_score: "number",
          estimated_tokens: "number",
        },
      },
    },
    candidate_memories: {
      array: {
        object: {
          content: "string",
          memory_type: memoryTypeShape,
          confidence: "number",
          importance: "number",
          reason: "string",
          rejection: nullableString,
        },
      },
    },
    stored_memory_ids: { array: "string" },
    operations: {
      array: {
        object: {
          operation: "string",
          success: "boolean",
          latency_ms: "number",
          provider: nullableString,
          model: nullableString,
          usage: { nullable: usageShape },
          context_estimated_tokens: optional({ nullable: "number" }),
          memories_recalled: optional({ nullable: "number" }),
          memories_stored: optional({ nullable: "number" }),
          error_kind: nullableString,
        },
      },
    },
    messages: { array: { record: "string" } },
  },
};

export const traceShape: Shape = {
  object: {
    trace_id: "string",
    kind: "string",
    operation: "string",
    started_at: "string",
    attributes: { record: "unknown" },
  },
};

export const traceListShape: Shape = {
  object: { count: "number", traces: { array: traceShape } },
};

export const traceEventShape: Shape = {
  object: {
    event_id: "string",
    trace_id: "string",
    sequence: "number",
    kind: "string",
    operation: "string",
    status: { enum: ["ok", "skipped", "error"] },
    timestamp: "string",
    parent_id: nullableString,
    duration_ms: nullableNumber,
    metadata: { record: "unknown" },
    detail: nullableString,
  },
};

export const traceDetailShape: Shape = {
  object: {
    trace: traceShape,
    summary: {
      object: {
        trace_id: "string",
        kind: "string",
        operation: "string",
        started_at: "string",
        event_count: "number",
        operations: { record: "string" },
        durations_ms: numberRecord,
        total_duration_ms: "number",
        errors: { array: "string" },
        memory_ids: { array: "string" },
        succeeded: "boolean",
      },
    },
    event_count: "number",
  },
};

export const metricsShape: Shape = {
  object: {
    traces: numberRecord,
    operations: numberRecord,
    statuses: numberRecord,
    memory: numberRecord,
    context: numberRecord,
    provider: numberRecord,
    usage_reported: "number",
    failed_traces: "number",
    event_count: "number",
  },
};

export const evaluationShape: Shape = {
  object: {
    dataset: "string",
    version: "string",
    reference: "string",
    total_cases: "number",
    passed_cases: "number",
    failed_cases: "number",
    case_score: "number",
    metric_score: "number",
    overall: "number",
    threshold: "number",
    passed: "boolean",
    categories: { record: { object: { passed: "number", total: "number" } } },
    metrics: {
      array: {
        object: {
          name: "string",
          definition: "string",
          numerator: "number",
          denominator: "number",
          score: "number",
          failures: { array: "string" },
        },
      },
    },
    cases: {
      array: {
        object: {
          case_id: "string",
          category: "string",
          passed: "boolean",
          failures: { array: "string" },
          numerators: numberRecord,
          denominators: numberRecord,
          measurements: numberRecord,
        },
      },
    },
  },
};

// -- the client ---------------------------------------------------------------

/** Every endpoint the dashboard consumes, exactly as the backend defines them. */
export const api = {
  health(): Promise<HealthResponse> {
    return read(ROOT_BASE, "GET", "/health", undefined, healthShape, "GET /health");
  },

  listMemories(params: MemoryListParams): Promise<MemoryPage> {
    const query = new URLSearchParams({ limit: String(params.limit), offset: String(params.offset) });
    if (params.status) {
      query.set("status", params.status);
    }
    if (params.memory_type) {
      query.set("memory_type", params.memory_type);
    }
    return read(
      API_BASE,
      "GET",
      `/memories?${query.toString()}`,
      undefined,
      memoryPageShape,
      "GET /memories",
    );
  },

  getMemory(id: string): Promise<Memory> {
    return read(
      API_BASE,
      "GET",
      `/memories/${encodeURIComponent(id)}`,
      undefined,
      memoryShape,
      "GET /memories/{id}",
    );
  },

  getHistory(id: string): Promise<MemoryHistory> {
    return read(
      API_BASE,
      "GET",
      `/memories/${encodeURIComponent(id)}/history`,
      undefined,
      historyShape,
      "GET /memories/{id}/history",
    );
  },

  getIntelligence(id: string): Promise<Intelligence> {
    return read(
      API_BASE,
      "GET",
      `/memories/${encodeURIComponent(id)}/intelligence`,
      undefined,
      intelligenceShape,
      "GET /memories/{id}/intelligence",
    );
  },

  getDuplicates(id: string): Promise<Duplicates> {
    return read(
      API_BASE,
      "GET",
      `/memories/${encodeURIComponent(id)}/duplicates`,
      undefined,
      duplicatesShape,
      "GET /memories/{id}/duplicates",
    );
  },

  getConflicts(id: string): Promise<Conflicts> {
    return read(
      API_BASE,
      "GET",
      `/memories/${encodeURIComponent(id)}/conflicts`,
      undefined,
      conflictsShape,
      "GET /memories/{id}/conflicts",
    );
  },

  getRelationships(id: string): Promise<Relationships> {
    return read(
      API_BASE,
      "GET",
      `/memories/${encodeURIComponent(id)}/relationships`,
      undefined,
      relationshipsShape,
      "GET /memories/{id}/relationships",
    );
  },

  resolveConflict(id: string, targetId: string): Promise<ConflictResolution> {
    return read(
      API_BASE,
      "POST",
      `/memories/${encodeURIComponent(id)}/conflicts/${encodeURIComponent(targetId)}/resolve`,
      undefined,
      resolutionShape,
      "POST /memories/{id}/conflicts/{target_id}/resolve",
    );
  },

  search(body: SearchBody): Promise<RetrievalSearchResponse> {
    return read(API_BASE, "POST", "/memories/search", body, searchShape, "POST /memories/search");
  },

  compile(body: CompileBody): Promise<ContextCompileResponse> {
    return read(API_BASE, "POST", "/context/compile", body, compileShape, "POST /context/compile");
  },

  async respond(body: AgentBody): Promise<AgentRespondResponse & { traceId: string | null }> {
    const response = await request(API_BASE, "POST", "/agent/respond", body);
    const parsed = parseResponse<AgentRespondResponse>(response.body, agentShape, "POST /agent/respond");
    return { ...parsed, traceId: response.headers.get(TRACE_HEADER) };
  },

  listTraces(limit: number, kind?: string): Promise<TraceListResponse> {
    const query = new URLSearchParams({ limit: String(limit) });
    if (kind) {
      query.set("kind", kind);
    }
    return read(
      API_BASE,
      "GET",
      `/traces?${query.toString()}`,
      undefined,
      traceListShape,
      "GET /traces",
    );
  },

  getTrace(traceId: string): Promise<TraceDetail> {
    return read(
      API_BASE,
      "GET",
      `/traces/${encodeURIComponent(traceId)}`,
      undefined,
      traceDetailShape,
      "GET /traces/{trace_id}",
    );
  },

  getTraceEvents(traceId: string): Promise<TraceEvent[]> {
    return read(
      API_BASE,
      "GET",
      `/traces/${encodeURIComponent(traceId)}/events`,
      undefined,
      { array: traceEventShape },
      "GET /traces/{trace_id}/events",
    );
  },

  getMetrics(): Promise<Metrics> {
    return read(API_BASE, "GET", "/metrics", undefined, metricsShape, "GET /metrics");
  },

  getLatestEvaluation(): Promise<Evaluation> {
    return read(
      API_BASE,
      "GET",
      "/evaluations/latest",
      undefined,
      evaluationShape,
      "GET /evaluations/latest",
    );
  },
};
