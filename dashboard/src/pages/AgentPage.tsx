/**
 * Agent Playground.
 *
 * One memory-aware agent turn through `POST /api/v1/agent/respond`: the memory
 * gate, the recalled memories, the compiled context, the provider that
 * answered, and the trace id that records the whole thing.
 *
 * Two rules the page keeps deliberately:
 *
 * 1. **No chain-of-thought, no prompt.** MemForge records none, and the
 *    dashboard renders none. What is shown is operational: which step ran, what
 *    it returned, how long it took.
 * 2. **The provider is whatever the API says it is.** With the default mock the
 *    page says "mock" and shows the deterministic answer; a custom
 *    OpenAI-compatible provider is reported by name and model, never
 *    configured here.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { AgentBody, AgentRespondResponse } from "../api/types";
import { useAction } from "../hooks/useResource";
import { Badge, Card, CodePanel, KeyValues, NumberField, ScoreBar } from "../components/primitives";
import { ErrorState } from "../components/StateBlock";
import { TokenBudget } from "../components/TokenBudget";
import { RETRIEVAL_MODES } from "../lib/constants";
import { duration, integer, num, truncateMiddle } from "../lib/format";

const DEFAULT_REQUEST =
  "Which backend language should I use for the MemForge service?\nremember: [FACT] MemForge uses Python and FastAPI.";
const SMALL_TALK = "hi";

export function AgentPage() {
  const [request, setRequest] = useState(DEFAULT_REQUEST);
  const [budget, setBudget] = useState(300);
  const [topK, setTopK] = useState(10);
  const [mode, setMode] = useState<"lexical" | "semantic" | "hybrid">("lexical");
  const [useMemory, setUseMemory] = useState(true);
  const [contextOnly, setContextOnly] = useState(false);
  const [extractCandidates, setExtractCandidates] = useState(true);
  const [storeCandidates, setStoreCandidates] = useState(false);
  const [source, setSource] = useState("agent:dashboard");

  const respond = useAction<AgentBody, AgentRespondResponse & { traceId: string | null }>((body) =>
    api.respond(body),
  );

  const run = (): void => {
    void respond.run({
      request,
      token_budget: budget,
      top_k: topK,
      retrieval_mode: mode,
      use_memory: useMemory,
      context_only: contextOnly,
      extract_candidates: extractCandidates,
      store_candidates: storeCandidates,
      source,
    });
  };

  const result = respond.state.result;

  return (
    <>
      <div className="page-intro">
        <h2>Agent Playground</h2>
        <p>
          One bounded, memory-aware turn: memory gate → recall (M3) → context compilation (M4) →
          provider call → answer, plus optional candidate extraction. Storing a candidate is
          opt-in, and nothing is written without it.
        </p>
      </div>

      <div className="filter-bar">
        <div className="field filter-bar__field filter-bar__field--grow">
          <label className="field__label" htmlFor="agent-request">
            User request
          </label>
          <textarea
            id="agent-request"
            className="textarea"
            value={request}
            onChange={(event) => setRequest(event.target.value)}
            rows={3}
          />
        </div>
        <NumberField
          id="agent-budget"
          label="Context token budget"
          value={budget}
          min={1}
          onChange={setBudget}
        />
        <NumberField id="agent-topk" label="Top-k" value={topK} min={1} max={100} onChange={setTopK} />
        <div className="field filter-bar__field">
          <label className="field__label" htmlFor="agent-mode">
            Retrieval mode
          </label>
          <select
            id="agent-mode"
            className="select"
            value={mode}
            onChange={(event) => setMode(event.target.value as typeof mode)}
          >
            {RETRIEVAL_MODES.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </div>
        <div className="field filter-bar__field">
          <label className="field__label" htmlFor="agent-source">
            Candidate source
          </label>
          <input
            id="agent-source"
            className="input"
            value={source}
            onChange={(event) => setSource(event.target.value)}
          />
        </div>
      </div>

      <div className="filter-bar">
        <label className="checkbox">
          <input
            type="checkbox"
            checked={useMemory}
            onChange={(event) => setUseMemory(event.target.checked)}
          />
          use memory
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={contextOnly}
            onChange={(event) => setContextOnly(event.target.checked)}
          />
          context only (never call the provider)
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={extractCandidates}
            onChange={(event) => setExtractCandidates(event.target.checked)}
          />
          extract candidates
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={storeCandidates}
            onChange={(event) => setStoreCandidates(event.target.checked)}
          />
          store candidates (a real write)
        </label>
        <div className="filter-bar__actions">
          <button
            type="button"
            className="btn"
            onClick={() => {
              setRequest(SMALL_TALK);
              setExtractCandidates(false);
            }}
          >
            Try small talk
          </button>
          <button
            type="button"
            className="btn btn--primary"
            onClick={run}
            disabled={respond.state.running || request.trim().length === 0}
          >
            {respond.state.running ? "Running…" : "Run agent turn"}
          </button>
        </div>
      </div>

      {respond.state.error !== null && (
        <ErrorState error={respond.state.error} onRetry={run} retryLabel="Run again" />
      )}

      {result === null && !respond.state.running && respond.state.error === null && (
        <div className="state">
          <span className="state__title">No turn run yet</span>
          <span className="state__detail">
            The default request asks a question the seeded corpus can answer, then proposes a memory
            to store — with <strong>store candidates off</strong>, so the candidate is reported and
            discarded.
          </span>
        </div>
      )}

      {respond.state.running && (
        <div className="state">
          <span className="row" style={{ gap: 9 }}>
            <span className="spinner" aria-hidden="true" />
            <span className="state__title">Gate → recall → compile → provider…</span>
          </span>
        </div>
      )}

      {result !== null && (
        <>
          <div className="grid grid--stats">
            <div className="stat">
              <span className="stat__label">Memory gate</span>
              <span className="stat__value stat__value--text">
                <Badge
                  tone={
                    result.memory_gate.use === "used"
                      ? "ok"
                      : result.memory_gate.use === "disabled"
                        ? "bad"
                        : "warn"
                  }
                >
                  {result.memory_gate.use}
                </Badge>
              </span>
              <span className="stat__hint">{result.memory_gate.reason ?? "—"}</span>
            </div>
            <div className="stat">
              <span className="stat__label">Provider</span>
              <span className="stat__value stat__value--text">
                <Badge tone={result.provider === "mock" ? "accent" : "ok"} mono>
                  {result.provider}
                </Badge>
              </span>
              <span className="stat__hint">model {result.model}</span>
            </div>
            <div className="stat">
              <span className="stat__label">Recalled memories</span>
              <span className="stat__value">{integer(result.context_blocks.length)}</span>
              <span className="stat__hint">blocks compiled into the context</span>
            </div>
            <div className="stat">
              <span className="stat__label">Candidates stored</span>
              <span className="stat__value">{integer(result.stored_memory_ids.length)}</span>
              <span className="stat__hint">
                {result.candidate_memories.length} proposed · storage is opt-in
              </span>
            </div>
            <div className="stat">
              <span className="stat__label">Usage reported</span>
              <span className="stat__value stat__value--text">
                {result.usage === null ? (
                  <Badge tone="warn">none</Badge>
                ) : (
                  <Badge tone="ok">{integer(result.usage.total_tokens)} tokens</Badge>
                )}
              </span>
              <span className="stat__hint">
                {result.usage === null
                  ? "the provider sent no usage block; none is derived"
                  : `reported by ${result.usage.reported_by}`}
              </span>
            </div>
            <div className="stat">
              <span className="stat__label">Trace id</span>
              <span className="stat__value stat__value--text mono">
                {result.traceId === null ? "—" : truncateMiddle(result.traceId, 16)}
              </span>
              <span className="stat__hint">
                {result.traceId === null ? (
                  "no trace id was returned"
                ) : (
                  <Link to={`/traces/${encodeURIComponent(result.traceId)}`}>open the trace</Link>
                )}
              </span>
            </div>
          </div>

          <Card
            title="Answer"
            subtitle={`${result.provider}${result.model === "" ? "" : ` · ${result.model}`}${contextOnly ? " · context-only run, no provider call" : ""}`}
          >
            {result.answer.trim().length === 0 ? (
              <div className="state">
                <span className="state__title">No answer was produced</span>
                <span className="state__detail">
                  This turn was run with <code>context_only</code>, so the provider was never called.
                  The compiled context is below.
                </span>
              </div>
            ) : (
              <p style={{ fontSize: "var(--step-2)", lineHeight: 1.5, whiteSpace: "pre-wrap" }}>
                {result.answer}
              </p>
            )}
            <p className="small faint" style={{ marginTop: 10 }}>
              MemForge never substitutes its own text for an answer a provider did not give, and no
              chain-of-thought is recorded or rendered anywhere in this product.
            </p>
          </Card>

          <Card
            title="Compiled context"
            subtitle="M4's own estimate; the hard budget is enforced by the backend"
          >
            <TokenBudget
              budget={result.context_token_budget}
              estimated={result.context_estimated_tokens}
              remaining={Math.max(0, result.context_token_budget - result.context_estimated_tokens)}
              withinBudget={result.context_estimated_tokens <= result.context_token_budget}
              utilization={
                result.context_token_budget === 0
                  ? 0
                  : result.context_estimated_tokens / result.context_token_budget
              }
            />
            <div style={{ marginTop: 12 }}>
              <CodePanel
                label={`context_text · ${integer(result.context_estimated_tokens)} estimated tokens`}
                tall
              >
                {result.context_text.trim().length === 0
                  ? "(memory was not used for this turn, so no context was compiled)"
                  : result.context_text}
              </CodePanel>
            </div>
          </Card>

          <div className="grid grid--split">
            <Card title="Recalled memories" subtitle="The blocks the compiler admitted">
              {result.context_blocks.length === 0 ? (
                <p className="muted small">No memory was recalled for this turn.</p>
              ) : (
                <div className="stack stack--tight">
                  {result.context_blocks.map((block) => (
                    <div className="factor" key={`${block.memory_id}-${block.order}`}>
                      <span className="factor__name">
                        {block.order + 1}. {block.section}
                      </span>
                      <ScoreBar
                        value={block.utility_score}
                        label={`utility ${block.utility_score}`}
                        tone="ok"
                      />
                      <span className="factor__meta">
                        {integer(block.estimated_tokens)} tokens · retrieval {num(block.retrieval_score)}{" "}
                        · type {block.memory_type}
                      </span>
                      <Link className="factor__why" to={`/memories/${block.memory_id}`}>
                        {truncateMiddle(block.memory_id, 22)}
                      </Link>
                    </div>
                  ))}
                </div>
              )}
            </Card>

            <Card title="Candidate memories" subtitle="Proposed by the provider, stored only with permission">
              {result.candidate_memories.length === 0 ? (
                <p className="muted small">No candidate was proposed for this turn.</p>
              ) : (
                <div className="stack stack--tight">
                  {result.candidate_memories.map((candidate, index) => (
                    <div className="factor" key={`${candidate.content}-${index}`}>
                      <span className="factor__name">{candidate.memory_type}</span>
                      <span className="factor__why">{candidate.content}</span>
                      <span className="factor__meta">
                        confidence {num(candidate.confidence)} · importance {num(candidate.importance)}
                        {candidate.reason === "" ? "" : ` · ${candidate.reason}`}
                      </span>
                      {candidate.rejection !== null && (
                        <span className="factor__meta">rejected: {candidate.rejection}</span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </Card>
          </div>

          <Card
            title="Operations"
            subtitle="Safe metadata about each step of the turn — durations and counts only"
          >
            {result.operations.length === 0 ? (
              <p className="muted small">
                No operation record was returned. A context-only turn does not call the provider, and
                MemForge records nothing it did not measure.
              </p>
            ) : (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th scope="col">Operation</th>
                      <th scope="col">Status</th>
                      <th scope="col" className="num">
                        Latency
                      </th>
                      <th scope="col">Provider / model</th>
                      <th scope="col">Figures</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.operations.map((record, index) => (
                      <tr key={`${record.operation}-${index}`}>
                        <td className="mono">{record.operation}</td>
                        <td>
                          <Badge tone={record.success ? "ok" : "bad"}>
                            {record.success ? "ok" : (record.error_kind ?? "error")}
                          </Badge>
                        </td>
                        <td className="num">{duration(record.latency_ms)}</td>
                        <td className="mono small">
                          {record.provider ?? "—"}
                          {record.model === null || record.model === undefined ? "" : ` · ${record.model}`}
                        </td>
                        <td className="small muted">
                          {[
                            record.memories_recalled === null || record.memories_recalled === undefined
                              ? null
                              : `${record.memories_recalled} recalled`,
                            record.memories_stored === null || record.memories_stored === undefined
                              ? null
                              : `${record.memories_stored} stored`,
                            record.context_estimated_tokens === null ||
                            record.context_estimated_tokens === undefined
                              ? null
                              : `${record.context_estimated_tokens} est. context tokens`,
                          ]
                            .filter((part) => part !== null)
                            .join(" · ") || "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          {result.stored_memory_ids.length > 0 && (
            <Card title="Stored" subtitle="Candidate memories persisted with explicit permission">
              <div className="stack stack--tight">
                {result.stored_memory_ids.map((id) => (
                  <div className="row row--between" key={id}>
                    <span className="mono">{id}</span>
                    <Link className="btn btn--ghost" to={`/memories/${id}`}>
                      open
                    </Link>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {result.traceId !== null && (
            <Card title="Trace" subtitle="The whole turn, as recorded">
              <KeyValues
                items={[
                  ["Trace id", <span className="mono">{result.traceId}</span>],
                  ["Kind", "agent_request"],
                  ["Operations recorded", integer(result.operations.length)],
                ]}
              />
              <div style={{ marginTop: 10 }}>
                <Link className="btn btn--primary" to={`/traces/${encodeURIComponent(result.traceId)}`}>
                  Open the trace timeline
                </Link>
              </div>
            </Card>
          )}
        </>
      )}
    </>
  );
}
