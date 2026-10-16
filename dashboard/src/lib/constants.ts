/**
 * Domain vocabulary shared by the dashboard.
 *
 * These are the labels for values the *backend* enumerates (`MemoryType`,
 * `MemoryStatus`, `RetrievalMode`, `ContextSection`, `TraceKind`). The
 * dashboard invents none of them: it renders exactly the strings the API sends,
 * and only supplies the human caption shown next to them.
 */

import type {
  ContextSection,
  EventStatus,
  MemoryStatus,
  MemoryType,
  RetrievalMode,
  TraceKind,
} from "../api/types";

export const MEMORY_TYPES: readonly MemoryType[] = [
  "PROFILE",
  "FACT",
  "PREFERENCE",
  "EVENT",
  "TASK",
  "OBSERVATION",
] as const;

export const MEMORY_STATUSES: readonly MemoryStatus[] = ["ACTIVE", "ARCHIVED", "DELETED"] as const;

export const RETRIEVAL_MODES: readonly RetrievalMode[] = ["lexical", "semantic", "hybrid"] as const;

export const CONTEXT_SECTIONS: readonly ContextSection[] = [
  "PROFILE",
  "FACTS",
  "RECENT_EVENTS",
  "TASKS",
  "OBSERVATIONS",
] as const;

export const TRACE_KINDS: readonly TraceKind[] = [
  "agent_request",
  "mcp_request",
  "api_request",
  "evaluation_case",
  "cli",
] as const;

/** One sentence per memory type, mirroring the M1/M4 type→section mapping. */
export const MEMORY_TYPE_CAPTIONS: Record<MemoryType, string> = {
  PROFILE: "Stable facts about the user or the system.",
  FACT: "A checkable statement.",
  PREFERENCE: "A stated preference.",
  EVENT: "Something that happened.",
  TASK: "Open or completed work.",
  OBSERVATION: "A note, usually lower confidence.",
};

/**
 * What the backend's own reason codes mean. The API returns the code; this is
 * only the caption, and it is never used to *classify* anything.
 */
export const SELECTION_REASON_CAPTIONS: Record<string, string> = {
  selected: "Admitted into the context.",
  selected_last_fit: "Admitted as the last block that fits the budget.",
  insufficient_budget: "Rejected: does not fit the remaining token budget.",
  duplicate_of: "Suppressed: a near-duplicate of a selected memory.",
  conflict_superseded: "Suppressed: lost a recorded conflict resolution.",
  conflict_unresolved: "Kept, but demoted by an unresolved conflict.",
  below_utility_floor: "Rejected: utility below the configured floor.",
  not_retrieved: "Never a candidate for this query.",
};

export const EVENT_STATUS_TONES: Record<string, "ok" | "warn" | "bad" | "muted"> = {
  ok: "ok",
  skipped: "warn",
  error: "bad",
};

export const STATUS_TONES: Record<MemoryStatus, "ok" | "warn" | "bad"> = {
  ACTIVE: "ok",
  ARCHIVED: "warn",
  DELETED: "bad",
};

/** The lifecycle an M1 history row records. */
export const VERSION_CHANGE_CAPTIONS: Record<string, { label: string; tone: "ok" | "warn" | "bad" }> = {
  CREATED: { label: "v CREATED", tone: "ok" },
  UPDATED: { label: "v UPDATED", tone: "warn" },
  FORGOTTEN: { label: "v FORGOTTEN", tone: "bad" },
};

/** The milestone each capability belongs to — used only for labelling. */
export const CAPABILITY_MILESTONES: readonly { title: string; body: string }[] = [
  {
    title: "MEMORY",
    body: "Persistent + versioned. Every write appends an immutable snapshot.",
  },
  {
    title: "INTELLIGENCE",
    body: "Importance + confidence + decay + conflicts, all deterministic.",
  },
  {
    title: "RETRIEVAL",
    body: "Memory-aware ranking: relevance blended with priority and freshness.",
  },
  {
    title: "CONTEXT",
    body: "Token-budgeted context compilation with a hard ceiling.",
  },
  {
    title: "AGENT",
    body: "A bounded, memory-aware workflow. No planner, no autonomy.",
  },
  {
    title: "MCP",
    body: "The same memory layer exposed to external agents as tools.",
  },
  {
    title: "EVALUATION",
    body: "A 36-case deterministic benchmark over the shipped fixtures.",
  },
  {
    title: "OBSERVABILITY",
    body: "A traceable operational lifecycle, correlated by one trace id.",
  },
];

export function isEventStatus(value: string): value is EventStatus {
  return value === "ok" || value === "skipped" || value === "error";
}
