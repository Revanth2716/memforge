/**
 * TypeScript mirrors of the MemForge API response contracts.
 *
 * These types are *projections of the wire shape* defined in
 * `app/api/schemas.py` (FastAPI + Pydantic). Nothing here re-implements a
 * backend rule: the dashboard only reads what the API returns and renders it.
 *
 * Every field is optional-tolerant at the parsing boundary (`src/api/parse.ts`
 * validates), because the API contract is the single source of truth and a
 * malformed response must be reported, never guessed at.
 */

/** Memory categories (M1). */
export type MemoryType =
  | "PROFILE"
  | "FACT"
  | "PREFERENCE"
  | "EVENT"
  | "TASK"
  | "OBSERVATION";

/** Memory lifecycle status (M1). */
export type MemoryStatus = "ACTIVE" | "ARCHIVED" | "DELETED";

/** Retrieval strategy (M3). */
export type RetrievalMode = "lexical" | "semantic" | "hybrid";

/** What a compiled block is *for*, in prompt order (M4). */
export type ContextSection =
  | "PROFILE"
  | "FACTS"
  | "RECENT_EVENTS"
  | "TASKS"
  | "OBSERVATIONS";

/** What started a trace (M6). */
export type TraceKind = "agent_request" | "mcp_request" | "api_request" | "evaluation_case" | "cli";

/** Outcome of one trace event (M6). */
export type EventStatus = "ok" | "skipped" | "error";

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  storage: string;
}

export interface Memory {
  id: string;
  content: string;
  memory_type: MemoryType;
  source: string | null;
  confidence: number;
  importance: number;
  status: MemoryStatus;
  created_at: string;
  updated_at: string;
  version: number;
}

export interface MemoryPage {
  items: Memory[];
  total: number;
  limit: number;
  offset: number;
}

export interface MemoryVersion {
  memory_id: string;
  version: number;
  change: "CREATED" | "UPDATED" | "FORGOTTEN" | string;
  content: string;
  memory_type: MemoryType;
  source: string | null;
  confidence: number;
  importance: number;
  status: MemoryStatus;
  recorded_at: string;
}

export interface MemoryHistory {
  memory_id: string;
  current_version: number;
  total: number;
  items: MemoryVersion[];
}

export interface IntelligenceFactor {
  name: string;
  weight: number;
  raw_value: number;
  contribution: number;
  explanation: string;
}

export interface Intelligence {
  memory_id: string;
  memory_type: MemoryType;
  status: MemoryStatus;
  version: number;
  confidence: number;
  stored_importance: number;
  importance_score: number;
  importance_factors: IntelligenceFactor[];
  importance_explanation: string;
  effective_priority: number;
  base_priority: number;
  decay_multiplier: number;
  age_days: number;
  priority_explanation: string;
  access_count: number;
  last_accessed_at: string | null;
  decay_half_life_days: number;
  similarity_threshold: number;
  computed_at: string;
}

export interface DuplicateMatch {
  memory_id: string;
  similarity: number;
  content: string;
  memory_type: MemoryType;
  shared_tokens: string[];
  explanation: string;
}

export interface Duplicates {
  memory_id: string;
  threshold: number;
  total: number;
  items: DuplicateMatch[];
}

export interface ConflictCandidate {
  target_memory_id: string;
  attribute: string;
  source_value: string;
  target_value: string;
  value_similarity: number;
  source_content: string;
  target_content: string;
  explanation: string;
}

export interface Conflicts {
  memory_id: string;
  total: number;
  items: ConflictCandidate[];
}

export interface ConflictResolution {
  winner_memory_id: string;
  loser_memory_id: string;
  decided_by: string;
  explanation: string;
  steps: { factor: string; left_value: string; right_value: string; winner_id: string }[];
  loser_archived: boolean;
  conflict: ConflictCandidate;
}

export interface Relationship {
  source_id: string;
  target_id: string;
  relationship: "DUPLICATE_OF" | "CONFLICTS_WITH" | "SUPERSEDES" | string;
  similarity: number | null;
  created_at: string;
}

export interface Relationships {
  memory_id: string;
  total: number;
  items: Relationship[];
}

/** One ranked memory, fully explained (M3). */
export interface RetrievalResult {
  memory_id: string;
  content: string;
  memory_type: MemoryType;
  status: MemoryStatus;
  confidence: number;
  importance: number;
  effective_priority: number;
  freshness: number;
  lexical_score: number;
  semantic_score: number | null;
  relevance: number;
  final_score: number;
  factors: RankingFactor[];
  explanation: string;
  duplicate_of: string | null;
  superseded_by: string | null;
  conflicts: ConflictInfo[];
}

export interface RankingFactor {
  name: string;
  weight: number;
  value: number;
  contribution: number;
  explanation: string;
}

export interface ConflictInfo {
  target_memory_id: string;
  relationship: string;
  attribute: string;
  source_value: string;
  target_value: string;
  resolution: string;
  resolution_reason: string | null;
}

export interface RetrievalSearchResponse {
  query: string;
  top_k: number;
  requested_mode: string;
  retrieval_mode: string;
  semantic_available: boolean;
  semantic_provider: string | null;
  semantic_model: string | null;
  degraded_reason: string | null;
  candidates_considered: number;
  candidates_returned: number;
  candidates_suppressed: number;
  latency_ms: number;
  results: RetrievalResult[];
  suppressed: { memory_id: string; final_score: number; reason: string }[];
}

export interface Compression {
  original_text: string;
  compressed_text: string;
  applied: string[];
  original_tokens: number;
  compressed_tokens: number;
  tokens_saved: number;
  truncated: boolean;
  retained_segments: number[];
}

export interface ContextBlock {
  memory_id: string;
  memory_type: MemoryType;
  status: MemoryStatus;
  source: string | null;
  confidence: number;
  importance: number;
  content: string;
  rendered_content: string;
  section: ContextSection;
  section_rank: number;
  order: number;
  relevance: number;
  retrieval_score: number;
  utility_score: number;
  estimated_tokens: number;
  compression: Compression;
  conflicts: ConflictInfo[];
}

/** The explained decision taken for one candidate (M4). */
export interface ContextSelection {
  memory_id: string;
  memory_type: MemoryType;
  selected: boolean;
  reason: string;
  detail: string;
  utility_score: number;
  value_score: number;
  estimated_tokens: number;
  remaining_tokens: number;
  related_memory_id: string | null;
}

export interface ContextExplanation {
  estimator: string;
  estimator_formula: string;
  utility_formula: string;
  utility_weights: Record<string, number>;
  ordering_policy: string;
  section_order: ContextSection[];
  selection_counts: Record<string, number>;
  compressed_blocks: number;
  tokens_saved: number;
  notes: string[];
}

export interface ContextCompileResponse {
  query: string;
  token_budget: number;
  estimated_tokens: number;
  remaining_tokens: number;
  within_budget: boolean;
  utilization: number;
  candidates_considered: number;
  selected_count: number;
  suppressed_count: number;
  requested_retrieval_mode: string;
  retrieval_mode: string;
  semantic_available: boolean;
  degraded_reason: string | null;
  context_text: string;
  blocks: ContextBlock[];
  selection_trace: ContextSelection[];
  suppressed_candidates: ContextSelection[];
  explanation: ContextExplanation;
}

/** Usage **as reported by the provider**; `null` when the provider sent none. */
export interface LLMUsage {
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  reported_by: string;
}

export interface CandidateMemory {
  content: string;
  memory_type: MemoryType;
  confidence: number;
  importance: number;
  reason: string;
  rejection: string | null;
}

export interface AgentOperationRecord {
  operation: string;
  success: boolean;
  latency_ms: number;
  provider: string | null;
  model: string | null;
  usage: LLMUsage | null;
  context_estimated_tokens: number | null;
  memories_recalled: number | null;
  memories_stored: number | null;
  error_kind: string | null;
}

export interface AgentContextBlock {
  memory_id: string;
  memory_type: MemoryType;
  section: ContextSection;
  order: number;
  utility_score: number;
  retrieval_score: number;
  estimated_tokens: number;
}

export interface AgentRespondResponse {
  request: string;
  answer: string;
  memory_used: boolean;
  memory_gate: Record<string, string>;
  provider: string;
  model: string;
  usage: LLMUsage | null;
  context_text: string;
  context_estimated_tokens: number;
  context_token_budget: number;
  context_blocks: AgentContextBlock[];
  candidate_memories: CandidateMemory[];
  stored_memory_ids: string[];
  operations: AgentOperationRecord[];
  messages: { role: string; content: string }[];
}

export interface Trace {
  trace_id: string;
  kind: TraceKind | string;
  operation: string;
  started_at: string;
  attributes: Record<string, unknown>;
}

export interface TraceEvent {
  event_id: string;
  trace_id: string;
  sequence: number;
  kind: string;
  operation: string;
  status: EventStatus | string;
  timestamp: string;
  parent_id: string | null;
  duration_ms: number | null;
  metadata: Record<string, unknown>;
  detail: string | null;
}

export interface TraceListResponse {
  count: number;
  traces: Trace[];
}

export interface TraceSummary {
  trace_id: string;
  kind: string;
  operation: string;
  started_at: string;
  event_count: number;
  operations: Record<string, string>;
  durations_ms: Record<string, number>;
  total_duration_ms: number;
  errors: string[];
  memory_ids: string[];
  succeeded: boolean;
}

export interface TraceDetail {
  trace: Trace;
  summary: TraceSummary;
  event_count: number;
}

export interface Metrics {
  traces: Record<string, number>;
  operations: Record<string, number>;
  statuses: Record<string, number>;
  memory: Record<string, number>;
  context: Record<string, number>;
  provider: Record<string, number>;
  usage_reported: number;
  failed_traces: number;
  event_count: number;
}

export interface MetricResult {
  name: string;
  definition: string;
  numerator: number;
  denominator: number;
  score: number;
  failures: string[];
}

export interface EvaluationCase {
  case_id: string;
  category: string;
  passed: boolean;
  failures: string[];
  numerators: Record<string, number>;
  denominators: Record<string, number>;
  measurements: Record<string, number>;
}

export interface Evaluation {
  dataset: string;
  version: string;
  reference: string;
  total_cases: number;
  passed_cases: number;
  failed_cases: number;
  case_score: number;
  metric_score: number;
  overall: number;
  threshold: number;
  passed: boolean;
  categories: Record<string, { passed: number; total: number }>;
  metrics: MetricResult[];
  cases: EvaluationCase[];
}

/** Every response the dashboard can render, keyed by endpoint. */
export interface MemforgeApi {
  health(): Promise<HealthResponse>;
  listMemories(params: MemoryListParams): Promise<MemoryPage>;
  getMemory(id: string): Promise<Memory>;
  getHistory(id: string): Promise<MemoryHistory>;
  getIntelligence(id: string): Promise<Intelligence>;
  getDuplicates(id: string): Promise<Duplicates>;
  getConflicts(id: string): Promise<Conflicts>;
  getRelationships(id: string): Promise<Relationships>;
  resolveConflict(id: string, targetId: string): Promise<ConflictResolution>;
  search(body: SearchBody): Promise<RetrievalSearchResponse>;
  compile(body: CompileBody): Promise<ContextCompileResponse>;
  respond(body: AgentBody): Promise<AgentRespondResponse & { traceId: string | null }>;
  listTraces(limit: number, kind?: string): Promise<Trace[]>;
  getTrace(traceId: string): Promise<TraceDetail>;
  getTraceEvents(traceId: string): Promise<TraceEvent[]>;
  getMetrics(): Promise<Metrics>;
  getLatestEvaluation(): Promise<Evaluation>;
}

export interface MemoryListParams {
  status?: MemoryStatus;
  memory_type?: MemoryType;
  limit: number;
  offset: number;
}

export interface SearchBody {
  query: string;
  top_k: number;
  retrieval_mode: RetrievalMode;
  semantic_weight?: number | null;
  memory_type?: MemoryType | null;
  status?: MemoryStatus;
  min_confidence?: number | null;
  min_importance?: number | null;
}

export interface CompileBody {
  query: string;
  token_budget?: number | null;
  top_k: number;
  retrieval_mode: RetrievalMode;
  semantic_weight?: number | null;
  preferred_memory_types?: MemoryType[] | null;
  min_confidence?: number | null;
  min_importance?: number | null;
  compress?: boolean | null;
  max_block_tokens?: number | null;
  status?: MemoryStatus;
}

export interface AgentBody {
  request: string;
  token_budget: number;
  top_k: number;
  retrieval_mode: RetrievalMode;
  use_memory: boolean;
  context_only: boolean;
  extract_candidates: boolean;
  store_candidates: boolean;
  source: string;
}
