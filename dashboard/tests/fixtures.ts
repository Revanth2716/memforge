/**
 * Deterministic API fixtures for the dashboard's frontend tests.
 *
 * These are **shape-faithful transcriptions of real MemForge responses**: the
 * corpus, the retrieval explanation, the budget accounting, the trace event
 * sequence and the evaluation figures were captured from a live uvicorn run of
 * `app.main:app` with the default configuration. Nothing here is invented
 * except the timestamps, which are pinned so the tests are deterministic.
 *
 * They exist **only** inside `dashboard/tests/`. The real application never
 * reads them: in the running dashboard every value comes from the backend.
 */

import type {
  AgentRespondResponse,
  ContextCompileResponse,
  Evaluation,
  Memory,
  MemoryHistory,
  Intelligence,
  Duplicates,
  Conflicts,
  Relationships,
  MemoryPage,
  Metrics,
  RetrievalSearchResponse,
  Trace,
  TraceDetail,
  TraceEvent,
  TraceListResponse,
} from "../src/api/types";

const PROFILE = "e71e8d69-89b7-4c13-b40f-f53d78ce1b56";
const NEAR_DUPLICATE = "9835dfc5-4849-4ead-8702-b9dc8c8afed8";
const FACT_PYTHON = "bf47c5bb-92cb-4e74-a463-883b643493b4";
const FACT_TS = "26ee9be7-8fec-4949-b18e-64a74c1a5f37";
const EVENT = "7cd91b15-c516-4b77-80a2-d45a8c9b6b1b";
const KUBERNETES = "562e0c78-b45a-4911-98d4-32da50ef7b65";

export const IDS = {
  PROFILE,
  NEAR_DUPLICATE,
  FACT_PYTHON,
  FACT_TS,
  EVENT,
  KUBERNETES,
  TRACE: "trace-47c62f17-84dd-46e1-b95f-9bd955ffe5d3",
};

export const health = {
  status: "ok",
  service: "memforge",
  version: "1.0.0",
  storage: "sqlite",
};

function memory(
  id: string,
  content: string,
  memory_type: Memory["memory_type"],
  confidence: number,
  importance: number,
  source: string | null,
  version: number,
): Memory {
  return {
    id,
    content,
    memory_type,
    source,
    confidence,
    importance,
    status: "ACTIVE",
    created_at: "2026-10-14T07:04:57.062175Z",
    updated_at: "2026-10-14T07:04:57.062175Z",
    version,
  };
}

export const memories: Memory[] = [
  memory(PROFILE, "User prefers Python for backend work.", "PROFILE", 0.95, 0.9, "user:demo", 1),
  memory(NEAR_DUPLICATE, "The user prefers python for backend development work", "OBSERVATION", 0.5, 0.3, "agent:note", 1),
  memory(FACT_PYTHON, "preferred backend language = Python", "FACT", 0.7, 0.5, null, 1),
  memory(FACT_TS, "preferred backend language = TypeScript", "FACT", 0.5, 0.9, null, 1),
  memory(EVENT, "User recently started the MemForge context compiler.", "EVENT", 0.9, 0.7, "agent:planner", 1),
  memory(KUBERNETES, "The staging cluster runs on Kubernetes with three worker nodes.", "OBSERVATION", 0.5, 0.2, "agent:note", 1),
];

export const memoryPage: MemoryPage = {
  items: memories,
  total: 6,
  limit: 25,
  offset: 0,
};

export const emptyMemoryPage: MemoryPage = { items: [], total: 0, limit: 25, offset: 0 };

/** A real v1 CREATED → v2 UPDATED → v3 FORGOTTEN history, captured live. */
export const history: MemoryHistory = {
  memory_id: PROFILE,
  current_version: 3,
  total: 3,
  items: [
    {
      memory_id: PROFILE,
      version: 1,
      change: "CREATED",
      content: "User prefers Python for backend work.",
      memory_type: "PROFILE",
      source: "user:demo",
      confidence: 0.95,
      importance: 0.9,
      status: "ACTIVE",
      recorded_at: "2026-10-14T07:04:57.062175Z",
    },
    {
      memory_id: PROFILE,
      version: 2,
      change: "UPDATED",
      content: "User prefers Python for backend work.",
      memory_type: "PROFILE",
      source: "user:revanth",
      confidence: 0.95,
      importance: 1.0,
      status: "ACTIVE",
      recorded_at: "2026-10-14T07:09:44.074367Z",
    },
    {
      memory_id: PROFILE,
      version: 3,
      change: "FORGOTTEN",
      content: "User prefers Python for backend work.",
      memory_type: "PROFILE",
      source: "user:revanth",
      confidence: 0.95,
      importance: 1.0,
      status: "DELETED",
      recorded_at: "2026-10-14T07:10:14.803998Z",
    },
  ],
};

export const intelligence: Intelligence = {
  memory_id: PROFILE,
  memory_type: "PROFILE",
  status: "ACTIVE",
  version: 1,
  confidence: 0.95,
  stored_importance: 0.9,
  importance_score: 0.809999,
  importance_factors: [
    {
      name: "memory_type",
      weight: 0.2,
      raw_value: 1.0,
      contribution: 0.2,
      explanation: "PROFILE has type weight 1.00",
    },
    {
      name: "explicit_importance",
      weight: 0.4,
      raw_value: 0.9,
      contribution: 0.36,
      explanation: "caller supplied importance 0.90",
    },
    {
      name: "recurrence",
      weight: 0.15,
      raw_value: 0.666667,
      contribution: 0.1,
      explanation: "2 near-duplicate active memories (saturation 3)",
    },
    {
      name: "recency",
      weight: 0.15,
      raw_value: 0.999995,
      contribution: 0.149999,
      explanation: "updated 0.00 days ago, half-life 30 days",
    },
    {
      name: "access_frequency",
      weight: 0.1,
      raw_value: 0.0,
      contribution: 0.0,
      explanation: "0 recorded access(es) (saturation 5)",
    },
  ],
  importance_explanation:
    "importance 0.8100 = 0.20*memory_type=0.2000 + 0.40*explicit_importance=0.3600 + 0.15*recurrence=0.1000 + 0.15*recency=0.1500 + 0.10*access_frequency=0.0000",
  effective_priority: 0.851995,
  base_priority: 0.851999,
  decay_multiplier: 0.999995,
  age_days: 0.00023,
  priority_explanation:
    "priority 0.8520 = (0.70*importance 0.8100 + 0.30*confidence 0.95) * decay 1.0000 after 0.00 days (half-life 30 days)",
  access_count: 0,
  last_accessed_at: null,
  decay_half_life_days: 30.0,
  similarity_threshold: 0.6,
  computed_at: "2026-10-14T07:05:16.901984Z",
};

export const duplicates: Duplicates = {
  memory_id: PROFILE,
  threshold: 0.6,
  total: 2,
  items: [
    {
      memory_id: NEAR_DUPLICATE,
      similarity: 0.909091,
      content: "The user prefers python for backend development work",
      memory_type: "OBSERVATION",
      shared_tokens: ["backend", "prefer", "python", "user", "work"],
      explanation:
        "Dice=0.9091 over 5 shared token(s) (backend, prefer, python, user, work); left=['backend', 'prefer', 'python', 'user', 'work'], right=['backend', 'development', 'prefer', 'python', 'user', 'work']",
    },
    {
      memory_id: FACT_PYTHON,
      similarity: 0.666667,
      content: "preferred backend language = Python",
      memory_type: "FACT",
      shared_tokens: ["backend", "prefer", "python"],
      explanation: "Dice=0.6667 over 3 shared token(s) (backend, prefer, python)",
    },
  ],
};

export const conflicts: Conflicts = {
  memory_id: FACT_TS,
  total: 1,
  items: [
    {
      target_memory_id: FACT_PYTHON,
      attribute: "prefer backend language",
      source_value: "typescript",
      target_value: "python",
      value_similarity: 0.0,
      source_content: "preferred backend language = TypeScript",
      target_content: "preferred backend language = Python",
      explanation:
        "both memories state 'prefer backend language' but with different values ('typescript' vs 'python'); value similarity 0.00 < 0.50",
    },
  ],
};

export const emptyConflicts: Conflicts = { memory_id: PROFILE, total: 0, items: [] };

export const relationships: Relationships = {
  memory_id: PROFILE,
  total: 2,
  items: [
    {
      source_id: PROFILE,
      target_id: NEAR_DUPLICATE,
      relationship: "DUPLICATE_OF",
      similarity: 0.909091,
      created_at: "2026-10-14T07:05:16.911430Z",
    },
    {
      source_id: PROFILE,
      target_id: FACT_PYTHON,
      relationship: "DUPLICATE_OF",
      similarity: 0.666667,
      created_at: "2026-10-14T07:05:16.920418Z",
    },
  ],
};

const rankedFactors = [
  {
    name: "relevance",
    weight: 0.45,
    value: 0.413955,
    contribution: 0.18628,
    explanation: "relevance 0.4140 = lexical coverage only",
  },
  {
    name: "effective_priority",
    weight: 0.2,
    value: 0.851995,
    contribution: 0.170399,
    explanation: "M2 effective priority 0.8520 (importance 0.8100, confidence 0.95, decay applied)",
  },
  {
    name: "confidence",
    weight: 0.15,
    value: 0.95,
    contribution: 0.1425,
    explanation: "stored confidence 0.95",
  },
  {
    name: "freshness",
    weight: 0.1,
    value: 0.999995,
    contribution: 0.1,
    explanation: "decay factor 1.0000 (half-life 30 days)",
  },
  {
    name: "type_preference",
    weight: 0.1,
    value: 1.0,
    contribution: 0.1,
    explanation: "PROFILE has retrieval type weight 1.00",
  },
];

export const search: RetrievalSearchResponse = {
  query: "python backend language memforge",
  top_k: 5,
  requested_mode: "lexical",
  retrieval_mode: "lexical",
  semantic_available: false,
  semantic_provider: null,
  semantic_model: null,
  degraded_reason: null,
  candidates_considered: 5,
  candidates_returned: 3,
  candidates_suppressed: 2,
  latency_ms: 32.107,
  results: [
    {
      memory_id: PROFILE,
      content: "User prefers Python for backend work.",
      memory_type: "PROFILE",
      status: "ACTIVE",
      confidence: 0.95,
      importance: 0.9,
      effective_priority: 0.851995,
      freshness: 0.999995,
      lexical_score: 0.413955,
      semantic_score: null,
      relevance: 0.413955,
      final_score: 0.699178,
      factors: rankedFactors,
      explanation:
        "final 0.6992 for query 'python backend language memforge' = relevance=0.41 (x0.45 -> 0.1863) + effective_priority=0.85 (x0.20 -> 0.1704) + confidence=0.95 (x0.15 -> 0.1425) + freshness=1.00 (x0.10 -> 0.1000) + type_preference=1.00 (x0.10 -> 0.1000)",
      duplicate_of: NEAR_DUPLICATE,
      superseded_by: null,
      conflicts: [],
    },
    {
      memory_id: EVENT,
      content: "User recently started the MemForge context compiler.",
      memory_type: "EVENT",
      status: "ACTIVE",
      confidence: 0.9,
      importance: 0.7,
      effective_priority: 0.640996,
      freshness: 0.999995,
      lexical_score: 0.322,
      semantic_score: null,
      relevance: 0.322,
      final_score: 0.568099,
      factors: rankedFactors.map((factor) => ({ ...factor })),
      explanation:
        "final 0.5681 for query 'python backend language memforge' = relevance=0.32 (x0.45 -> 0.1449) + effective_priority=0.64 (x0.20 -> 0.1282) + confidence=0.90 (x0.15 -> 0.1350) + freshness=1.00 (x0.10 -> 0.1000) + type_preference=0.60 (x0.10 -> 0.0600)",
      duplicate_of: null,
      superseded_by: null,
      conflicts: [],
    },
    {
      memory_id: FACT_TS,
      content: "preferred backend language = TypeScript",
      memory_type: "FACT",
      status: "ACTIVE",
      confidence: 0.5,
      importance: 0.9,
      effective_priority: 0.618996,
      freshness: 0.999995,
      lexical_score: 0.455075,
      semantic_score: null,
      relevance: 0.455075,
      final_score: 0.445187,
      factors: rankedFactors.map((factor) => ({ ...factor })),
      explanation:
        "final 0.4452 for query 'python backend language memforge' = relevance=0.46 (x0.45 -> 0.2048) + effective_priority=0.62 (x0.20 -> 0.1238) + confidence=0.50 (x0.15 -> 0.0750) + freshness=1.00 (x0.10 -> 0.1000) + type_preference=0.90 (x0.10 -> 0.0900) | conflict penalty x0.25 (0.50 penalised)",
      duplicate_of: null,
      superseded_by: null,
      conflicts: [
        {
          target_memory_id: FACT_PYTHON,
          relationship: "CONFLICTS_WITH",
          attribute: "prefer backend language",
          source_value: "typescript",
          target_value: "python",
          resolution: "unresolved",
          resolution_reason: null,
        },
      ],
    },
  ],
  suppressed: [
    {
      memory_id: FACT_PYTHON,
      final_score: 0.162112,
      reason: `duplicate_of ${PROFILE}, which ranks higher in this result set`,
    },
    {
      memory_id: NEAR_DUPLICATE,
      final_score: 0.145344,
      reason: `duplicate_of ${PROFILE}, which ranks higher in this result set`,
    },
  ],
};

export const emptySearch: RetrievalSearchResponse = {
  query: "nothing matches this",
  top_k: 5,
  requested_mode: "lexical",
  retrieval_mode: "lexical",
  semantic_available: false,
  semantic_provider: null,
  semantic_model: null,
  degraded_reason: null,
  candidates_considered: 0,
  candidates_returned: 0,
  candidates_suppressed: 0,
  latency_ms: 0.4,
  results: [],
  suppressed: [],
};

const compressionRecord = {
  original_text: "",
  compressed_text: "",
  applied: [],
  original_tokens: 10,
  compressed_tokens: 10,
  tokens_saved: 0,
  truncated: false,
  retained_segments: [],
};

export const compile: ContextCompileResponse = {
  query: "python backend language memforge",
  token_budget: 160,
  estimated_tokens: 58,
  remaining_tokens: 102,
  within_budget: true,
  utilization: 0.3625,
  candidates_considered: 5,
  selected_count: 4,
  suppressed_count: 1,
  requested_retrieval_mode: "lexical",
  retrieval_mode: "lexical",
  semantic_available: false,
  degraded_reason: null,
  context_text: `PROFILE
-------
- User prefers Python for backend work.

FACTS
-----
- preferred backend language = Python
- preferred backend language = TypeScript

RECENT EVENTS
-------------
- User recently started the MemForge context compiler.`,
  blocks: [
    {
      memory_id: PROFILE,
      memory_type: "PROFILE",
      status: "ACTIVE",
      source: "user:demo",
      confidence: 0.95,
      importance: 0.9,
      content: "User prefers Python for backend work.",
      rendered_content: "User prefers Python for backend work.",
      section: "PROFILE",
      section_rank: 0,
      order: 0,
      relevance: 0.413955,
      retrieval_score: 0.699178,
      utility_score: 0.707233,
      estimated_tokens: 10,
      compression: { ...compressionRecord, original_text: "User prefers Python for backend work.", compressed_text: "User prefers Python for backend work." },
      conflicts: [],
    },
    {
      memory_id: FACT_PYTHON,
      memory_type: "FACT",
      status: "ACTIVE",
      source: null,
      confidence: 0.7,
      importance: 0.5,
      content: "preferred backend language = Python",
      rendered_content: "preferred backend language = Python",
      section: "FACTS",
      section_rank: 1,
      order: 1,
      relevance: 0.162112,
      retrieval_score: 0.162112,
      utility_score: 0.710261,
      estimated_tokens: 10,
      compression: { ...compressionRecord, original_text: "preferred backend language = Python", compressed_text: "preferred backend language = Python" },
      conflicts: [],
    },
    {
      memory_id: FACT_TS,
      memory_type: "FACT",
      status: "ACTIVE",
      source: null,
      confidence: 0.5,
      importance: 0.9,
      content: "preferred backend language = TypeScript",
      rendered_content: "preferred backend language = TypeScript",
      section: "FACTS",
      section_rank: 1,
      order: 2,
      relevance: 0.445187,
      retrieval_score: 0.445187,
      utility_score: 0.596071,
      estimated_tokens: 11,
      compression: { ...compressionRecord, original_text: "preferred backend language = TypeScript", compressed_text: "preferred backend language = TypeScript" },
      conflicts: [],
    },
    {
      memory_id: EVENT,
      memory_type: "EVENT",
      status: "ACTIVE",
      source: "agent:planner",
      confidence: 0.9,
      importance: 0.7,
      content: "User recently started the MemForge context compiler.",
      rendered_content: "User recently started the MemForge context compiler.",
      section: "RECENT_EVENTS",
      section_rank: 2,
      order: 3,
      relevance: 0.568099,
      retrieval_score: 0.568099,
      utility_score: 0.558234,
      estimated_tokens: 14,
      compression: { ...compressionRecord, original_text: "User recently started the MemForge context compiler.", compressed_text: "User recently started the MemForge context compiler." },
      conflicts: [],
    },
  ],
  selection_trace: [
    {
      memory_id: FACT_PYTHON,
      memory_type: "FACT",
      selected: true,
      reason: "selected",
      detail: "highest utility (0.7103) that fits the remaining budget of 160 tokens",
      utility_score: 0.710261,
      value_score: 0.731599,
      estimated_tokens: 10,
      remaining_tokens: 150,
      related_memory_id: null,
    },
    {
      memory_id: PROFILE,
      memory_type: "PROFILE",
      selected: true,
      reason: "selected",
      detail: "highest utility (0.7072) that fits the remaining budget of 150 tokens",
      utility_score: 0.707233,
      value_score: 0.72848,
      estimated_tokens: 10,
      remaining_tokens: 140,
      related_memory_id: null,
    },
    {
      memory_id: FACT_TS,
      memory_type: "FACT",
      selected: true,
      reason: "selected",
      detail: "highest utility (0.5961) that fits the remaining budget of 140 tokens",
      utility_score: 0.596071,
      value_score: 0.615829,
      estimated_tokens: 11,
      remaining_tokens: 129,
      related_memory_id: null,
    },
    {
      memory_id: EVENT,
      memory_type: "EVENT",
      selected: true,
      reason: "selected",
      detail: "highest utility (0.5582) that fits the remaining budget of 129 tokens",
      utility_score: 0.558234,
      value_score: 0.581999,
      estimated_tokens: 14,
      remaining_tokens: 115,
      related_memory_id: null,
    },
    {
      memory_id: NEAR_DUPLICATE,
      memory_type: "OBSERVATION",
      selected: false,
      reason: "duplicate_of",
      detail: `near-duplicate of ${PROFILE}, which represents this duplicate group in the context`,
      utility_score: 0.464027,
      value_score: 0.483781,
      estimated_tokens: 14,
      remaining_tokens: 160,
      related_memory_id: PROFILE,
    },
  ],
  suppressed_candidates: [
    {
      memory_id: NEAR_DUPLICATE,
      memory_type: "OBSERVATION",
      selected: false,
      reason: "duplicate_of",
      detail: `near-duplicate of ${PROFILE}, which represents this duplicate group in the context`,
      utility_score: 0.464027,
      value_score: 0.483781,
      estimated_tokens: 14,
      remaining_tokens: 160,
      related_memory_id: PROFILE,
    },
  ],
  explanation: {
    estimator: "memforge-char-word/v1",
    estimator_formula:
      "memforge-char-word/v1: estimated = max(1, ceil(max(ascii_chars/chars_per_token(4), words/words_per_token(1.3)) + non_ascii_chars*non_ascii_tokens_per_char(1)))",
    utility_formula:
      "utility = value * (1 - 0.35 * min(1, tokens/120)) where value = 0.4*relevance + 0.2*effective_priority + 0.15*confidence + 0.1*freshness + 0.15*type_preference",
    utility_weights: {
      relevance: 0.4,
      effective_priority: 0.2,
      confidence: 0.15,
      freshness: 0.1,
      type_preference: 0.15,
    },
    ordering_policy: "section-then-utility/v1",
    section_order: ["PROFILE", "FACTS", "RECENT_EVENTS", "TASKS", "OBSERVATIONS"],
    selection_counts: {
      selected: 4,
      duplicate_of: 1,
      insufficient_budget: 0,
      conflict_superseded: 0,
      below_minimum_score: 0,
      excluded_by_id: 0,
      not_in_include_ids: 0,
      memory_type_filtered: 0,
    },
    compressed_blocks: 0,
    tokens_saved: 0,
    notes: [
      "token counts are MemForge estimates from a documented character/word formula, not provider tokenisations and not billing data",
      "selection is a deterministic, configurable heuristic -- it is not a learned ranking and makes no claim about optimality",
      "context is compiled with no LLM: content is never paraphrased or summarised",
      "compression never rewrites memory content; it only normalises whitespace and may drop whole sentences, which does not guarantee semantic equivalence",
    ],
  },
};

const ANSWER = `[mock provider: deterministic offline stand-in, no model was called]
operation: respond
messages: 2
characters: 566
structured_output: False
last_user_message: PROFILE
-------
- User prefers Python for backend work.

FACTS
-----
- preferred backend language = Python
- preferred backend language = TypeScript

RECENT EVENTS
-------------
- User recently started the MemForge context compiler.

---

User request: Which backend language should I use for the MemForge service?
remember: [FACT] MemForge uses Python and FastAPI.`;

export const agent: AgentRespondResponse & { traceId: string | null } = {
  request:
    "Which backend language should I use for the MemForge service?\nremember: [FACT] MemForge uses Python and FastAPI.",
  answer: ANSWER,
  memory_used: true,
  memory_gate: { use: "used", reason: "the request may depend on stored memory" },
  provider: "mock",
  model: "mock-echo-v1",
  usage: null,
  context_text: `PROFILE
-------
- User prefers Python for backend work.

FACTS
-----
- preferred backend language = Python
- preferred backend language = TypeScript

RECENT EVENTS
-------------
- User recently started the MemForge context compiler.`,
  context_estimated_tokens: 58,
  context_token_budget: 300,
  context_blocks: [
    { memory_id: PROFILE, memory_type: "PROFILE", section: "PROFILE", order: 0, utility_score: 0.707233, retrieval_score: 0.699178, estimated_tokens: 10 },
    { memory_id: FACT_PYTHON, memory_type: "FACT", section: "FACTS", order: 1, utility_score: 0.710261, retrieval_score: 0.162112, estimated_tokens: 10 },
    { memory_id: FACT_TS, memory_type: "FACT", section: "FACTS", order: 2, utility_score: 0.596071, retrieval_score: 0.445187, estimated_tokens: 11 },
    { memory_id: EVENT, memory_type: "EVENT", section: "RECENT_EVENTS", order: 3, utility_score: 0.558234, retrieval_score: 0.568099, estimated_tokens: 14 },
  ],
  candidate_memories: [
    {
      content: "MemForge uses Python and FastAPI.",
      memory_type: "FACT",
      confidence: 0.7,
      importance: 0.5,
      reason: "explicit 'remember:' instruction in the request",
      rejection: null,
    },
  ],
  stored_memory_ids: [],
  operations: [
    {
      operation: "compile_context",
      success: true,
      latency_ms: 0.0,
      provider: "mock",
      model: "mock-echo-v1",
      usage: null,
      context_estimated_tokens: 58,
      memories_recalled: 3,
      memories_stored: null,
      error_kind: null,
    },
    {
      operation: "respond",
      success: true,
      latency_ms: 22.35,
      provider: "mock",
      model: "mock-echo-v1",
      usage: null,
      context_estimated_tokens: null,
      memories_recalled: null,
      memories_stored: null,
      error_kind: null,
    },
    {
      operation: "extract_candidates",
      success: true,
      latency_ms: 21.367,
      provider: "mock",
      model: "mock-echo-v1",
      usage: null,
      context_estimated_tokens: null,
      memories_recalled: null,
      memories_stored: 0,
      error_kind: null,
    },
  ],
  messages: [
    { role: "system", content: "You are answering with compiled memory context." },
    { role: "user", content: "Which backend language should I use for the MemForge service?" },
  ],
  traceId: IDS.TRACE,
};

/** The same turn with `context_only: true` — no provider call, no answer. */
export const agentContextOnly: AgentRespondResponse & { traceId: string | null } = {
  ...agent,
  answer: "",
  usage: null,
  operations: [agent.operations[0]!],
  messages: [],
};

/** Small talk: the gate fires and nothing is recalled. */
export const agentSmallTalk: AgentRespondResponse & { traceId: string | null } = {
  ...agent,
  request: "hi",
  answer: "[mock provider: deterministic offline stand-in, no model was called]\noperation: respond\nmessages: 2\ncharacters: 12\nstructured_output: False\nlast_user_message: ",
  memory_used: false,
  memory_gate: { use: "skipped", reason: "the request does not look like it needs stored memory" },
  context_text: "",
  context_estimated_tokens: 0,
  context_blocks: [],
  candidate_memories: [],
  operations: [],
  traceId: null,
};

/** A turn against a configured custom provider, as the API reports it. */
export const agentCustomProvider: AgentRespondResponse & { traceId: string | null } = {
  ...agent,
  provider: "custom",
  model: "memforge-oss-120b",
  usage: { prompt_tokens: 1204, completion_tokens: 88, total_tokens: 1292, reported_by: "provider" },
};

/** The real `TraceListResponse` envelope: an object, not a bare array. */
export const traceList: TraceListResponse = {
  count: 2,
  traces: [
    {
      trace_id: IDS.TRACE,
      kind: "agent_request",
      operation: "agent.respond",
      started_at: "2026-10-14T07:05:17.086562+00:00",
      attributes: {
        context_only: false,
        extract_candidates: true,
        model: "mock-echo-v1",
        provider: "mock",
        request_characters: 112,
        retrieval_mode: "lexical",
        store_candidates: false,
        token_budget: 300,
        top_k: 10,
        use_memory: true,
      },
    },
    {
      trace_id: "trace-1f0c4d2a-6b31-4c8e-9a77-2bd4c5e6f708",
      kind: "api_request",
      operation: "memories.remember",
      started_at: "2026-10-14T07:02:44.911430+00:00",
      attributes: { check_duplicates: true },
    },
  ],
};

export const emptyTraceList: TraceListResponse = { count: 0, traces: [] };

export const traces: Trace[] = traceList.traces;

export const traceEvents: TraceEvent[] = [
  {
    event_id: `${IDS.TRACE}-e0000`,
    trace_id: IDS.TRACE,
    sequence: 0,
    kind: "request",
    operation: "agent.request",
    status: "ok",
    timestamp: "2026-10-14T07:05:17.098827+00:00",
    parent_id: null,
    duration_ms: null,
    metadata: {
      context_only: false,
      extract_candidates: true,
      model: "mock-echo-v1",
      provider: "mock",
      request_characters: 112,
      retrieval_mode: "lexical",
      store_candidates: false,
      token_budget: 300,
      top_k: 10,
      use_memory: true,
    },
    detail: null,
  },
  {
    event_id: `${IDS.TRACE}-e0001`,
    trace_id: IDS.TRACE,
    sequence: 1,
    kind: "retrieval_start",
    operation: "memory.retrieval",
    status: "ok",
    timestamp: "2026-10-14T07:05:17.109836+00:00",
    parent_id: null,
    duration_ms: null,
    metadata: {
      query_characters: 112,
      requested_mode: "lexical",
      semantic_available: false,
      top_k: 10,
    },
    detail: null,
  },
  {
    event_id: `${IDS.TRACE}-e0002`,
    trace_id: IDS.TRACE,
    sequence: 2,
    kind: "retrieval_complete",
    operation: "memory.retrieval",
    status: "ok",
    timestamp: "2026-10-14T07:05:17.149126+00:00",
    parent_id: `${IDS.TRACE}-e0001`,
    duration_ms: 31.061,
    metadata: {
      candidates: 5,
      degraded: false,
      duplicate_candidates: 2,
      latency_ms: 31.008,
      query_characters: 112,
      recalled: 3,
      requested_mode: "lexical",
      retrieval_mode: "lexical",
      semantic_available: false,
      top_k: 10,
    },
    detail: null,
  },
  {
    event_id: `${IDS.TRACE}-e0003`,
    trace_id: IDS.TRACE,
    sequence: 3,
    kind: "context_compile_start",
    operation: "context.compile",
    status: "ok",
    timestamp: "2026-10-14T07:05:17.165312+00:00",
    parent_id: null,
    duration_ms: null,
    metadata: {
      compress: null,
      estimator: "memforge-char-word/v1",
      requested_mode: "lexical",
      token_budget: 300,
      top_k: 10,
    },
    detail: null,
  },
  {
    event_id: `${IDS.TRACE}-e0004`,
    trace_id: IDS.TRACE,
    sequence: 4,
    kind: "context_compile_complete",
    operation: "context.compile",
    status: "ok",
    timestamp: "2026-10-14T07:05:17.218897+00:00",
    parent_id: `${IDS.TRACE}-e0003`,
    duration_ms: 44.038,
    metadata: {
      candidates: 5,
      compress: null,
      estimated_tokens: 58,
      estimator: "memforge-char-word/v1",
      over_budget: false,
      remaining_tokens: 242,
      requested_mode: "lexical",
      retrieval_mode: "lexical",
      selected: 4,
      suppressed: 1,
      suppressed_duplicate: 1,
      token_budget: 300,
      top_k: 10,
    },
    detail: null,
  },
  {
    event_id: `${IDS.TRACE}-e0005`,
    trace_id: IDS.TRACE,
    sequence: 5,
    kind: "llm_request",
    operation: "llm.generate",
    status: "ok",
    timestamp: "2026-10-14T07:05:17.229025+00:00",
    parent_id: null,
    duration_ms: null,
    metadata: {
      input_characters: 568,
      model: "mock-echo-v1",
      operation: "respond",
      provider: "mock",
      structured: false,
    },
    detail: null,
  },
  {
    event_id: `${IDS.TRACE}-e0006`,
    trace_id: IDS.TRACE,
    sequence: 6,
    kind: "llm_response",
    operation: "llm.generate",
    status: "ok",
    timestamp: "2026-10-14T07:05:17.242536+00:00",
    parent_id: `${IDS.TRACE}-e0005`,
    duration_ms: 0.105,
    metadata: {
      input_characters: 568,
      model: "mock-echo-v1",
      operation: "respond",
      output_characters: 525,
      provider: "mock",
      structured: false,
      usage_reported: false,
    },
    detail: null,
  },
  {
    event_id: `${IDS.TRACE}-e0009`,
    trace_id: IDS.TRACE,
    sequence: 9,
    kind: "memory_gate",
    operation: "agent.memory_gate",
    status: "ok",
    timestamp: "2026-10-14T07:05:17.272680+00:00",
    parent_id: null,
    duration_ms: null,
    metadata: {
      reason: "the request may depend on stored memory",
      requested: true,
      should_recall: true,
      use_memory: "used",
    },
    detail: null,
  },
  {
    event_id: `${IDS.TRACE}-e0010`,
    trace_id: IDS.TRACE,
    sequence: 10,
    kind: "memory_candidate",
    operation: "agent.memory_candidate",
    status: "ok",
    timestamp: "2026-10-14T07:05:17.285311+00:00",
    parent_id: null,
    duration_ms: null,
    metadata: { candidate_count: 1, extracted: true, rejected: 0, store_permitted: false },
    detail: null,
  },
  {
    event_id: `${IDS.TRACE}-e0011`,
    trace_id: IDS.TRACE,
    sequence: 11,
    kind: "memory_write",
    operation: "memory.write",
    status: "skipped",
    timestamp: "2026-10-14T07:05:17.297446+00:00",
    parent_id: null,
    duration_ms: null,
    metadata: {
      memory_ids: [PROFILE, FACT_PYTHON],
      reason: "storage was not permitted or no valid candidate was proposed",
    },
    detail: null,
  },
  {
    event_id: `${IDS.TRACE}-e0012`,
    trace_id: IDS.TRACE,
    sequence: 12,
    kind: "response",
    operation: "agent.response",
    status: "ok",
    timestamp: "2026-10-14T07:05:17.305786+00:00",
    parent_id: null,
    duration_ms: null,
    metadata: { answer_characters: 525, context_blocks: 4, context_characters: 232, used_memory: true },
    detail: null,
  },
];

export const traceDetail: TraceDetail = {
  trace: traces[0]!,
  summary: {
    trace_id: IDS.TRACE,
    kind: "agent_request",
    operation: "agent.respond",
    started_at: "2026-10-14T07:05:17.086562+00:00",
    event_count: traceEvents.length,
    operations: {
      "agent.memory_gate": "ok",
      "agent.request": "ok",
      "agent.response": "ok",
      "context.compile": "ok",
      "llm.generate": "ok",
      "memory.retrieval": "ok",
      "memory.write": "skipped",
    },
    durations_ms: { "context.compile": 44.038, "llm.generate": 0.237, "memory.retrieval": 31.061 },
    total_duration_ms: 75.336,
    errors: [],
    memory_ids: [PROFILE, FACT_PYTHON],
    succeeded: true,
  },
  event_count: traceEvents.length,
};

export const metrics: Metrics = {
  traces: { agent_request: 1, api_request: 1 },
  operations: {
    "agent.memory_gate": 1,
    "agent.request": 1,
    "agent.response": 1,
    "context.compile": 1,
    "llm.generate": 2,
    "memory.retrieval": 1,
  },
  statuses: { ok: 12, skipped: 1 },
  memory: {
    candidates: 1,
    created: 1,
    duplicate_candidates: 2,
    recalled: 3,
    writes_skipped: 1,
    written: 1,
  },
  context: { candidates: 5, conflicts_suppressed: 0, over_budget: 0, selected: 4, suppressed: 1 },
  provider: { calls: 2, failures: 0 },
  usage_reported: 0,
  failed_traces: 0,
  event_count: 13,
};

export const evaluation: Evaluation = {
  dataset: "memory-core",
  version: "v1",
  reference: "memory-core@v1",
  total_cases: 36,
  passed_cases: 36,
  failed_cases: 0,
  case_score: 1.0,
  metric_score: 0.982759,
  overall: 0.99138,
  threshold: 1.0,
  passed: true,
  categories: {
    agent: { passed: 6, total: 6 },
    conflict: { passed: 6, total: 6 },
    context: { passed: 7, total: 7 },
    duplicate: { passed: 5, total: 5 },
    lifecycle: { passed: 5, total: 5 },
    retrieval: { passed: 7, total: 7 },
  },
  metrics: [
    {
      name: "conflict_resolution_accuracy",
      definition:
        "share of conflict cases where the expected winner was the one selected; numerator=5; denominator=5; score=1.0",
      numerator: 5,
      denominator: 5,
      score: 1.0,
      failures: [],
    },
    {
      name: "context_budget_compliance",
      definition:
        "share of compilations whose estimated tokens were <= the requested budget; numerator=13; denominator=13; score=1.0",
      numerator: 13,
      denominator: 13,
      score: 1.0,
      failures: [],
    },
    {
      name: "context_provenance_completeness",
      definition:
        "share of selected blocks carrying complete source provenance; numerator=28; denominator=28; score=1.0",
      numerator: 28,
      denominator: 28,
      score: 1.0,
      failures: [],
    },
    {
      name: "context_selection_accuracy",
      definition:
        "share of required memories selected into the compiled context; numerator=4; denominator=4; score=1.0",
      numerator: 4,
      denominator: 4,
      score: 1.0,
      failures: [],
    },
    {
      name: "duplicate_suppression_accuracy",
      definition:
        "share of expected redundant memories that were actually suppressed; numerator=5; denominator=5; score=1.0",
      numerator: 5,
      denominator: 5,
      score: 1.0,
      failures: [],
    },
    {
      name: "history_preservation",
      definition:
        "share of lifecycle operations that preserved the required history; numerator=5; denominator=5; score=1.0",
      numerator: 5,
      denominator: 5,
      score: 1.0,
      failures: [],
    },
    {
      name: "memory_write_policy_compliance",
      definition:
        "share of agent turns that stored candidates only when explicitly permitted; numerator=6; denominator=6; score=1.0",
      numerator: 6,
      denominator: 6,
      score: 1.0,
      failures: [],
    },
    {
      name: "recall_relevance",
      definition:
        "share of cases that returned no declared-irrelevant memory (and, when a case declares required memories, returned every one of them); numerator=8; denominator=8; score=1.0",
      numerator: 8,
      denominator: 8,
      score: 1.0,
      failures: [],
    },
    {
      name: "retrieval_precision",
      definition:
        "share of returned memories that were declared relevant; a case that returned nothing is not counted (recall covers that failure); numerator=23; denominator=29; score=0.793103",
      numerator: 23,
      denominator: 29,
      score: 0.7931034482758621,
      failures: [],
    },
    {
      name: "retrieval_recall",
      definition:
        "share of required target memories that appeared in the retrieval results; numerator=23; denominator=23; score=1.0",
      numerator: 23,
      denominator: 23,
      score: 1.0,
      failures: [],
    },
    {
      name: "stale_memory_avoidance",
      definition:
        "share of cases where a superseded memory did not enter the final context; numerator=5; denominator=5; score=1.0",
      numerator: 5,
      denominator: 5,
      score: 1.0,
      failures: [],
    },
    {
      name: "top_k_hit_rate",
      definition:
        "share of cases where at least one required memory appeared within the first 3 ranked results; numerator=16; denominator=16; score=1.0",
      numerator: 16,
      denominator: 16,
      score: 1.0,
      failures: [],
    },
  ],
  cases: [],
};

/** A run with a real failure, to prove failures are never hidden. */
export const failingEvaluation: Evaluation = {
  ...evaluation,
  passed: false,
  passed_cases: 35,
  failed_cases: 1,
  case_score: 0.972222,
  overall: 0.964603,
  cases: [
    {
      case_id: "context-budget-sweep",
      category: "context",
      passed: false,
      failures: ["measurement:context_budget_compliance: 12/13 compilations respected the budget"],
      numerators: { context_budget_compliance: 12 },
      denominators: { context_budget_compliance: 13 },
      measurements: { estimated_tokens: 161 },
    },
  ],
};

/** The error envelope the backend returns for a failure. */
export function errorEnvelope(code: string, message: string): { error: { code: string; message: string } } {
  return { error: { code, message } };
}
