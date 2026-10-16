/**
 * Retrieval Lab.
 *
 * Runs `POST /api/v1/memories/search` — the real M3 retrieval service — and
 * renders exactly what it returns: the ranked results with every ranking factor,
 * the query-level provenance, and the candidates the service deliberately
 * suppressed.
 *
 * The ranking explanation is the point of this page: for each result the
 * backend sends the weighted factors, their raw values and a sentence explaining
 * each one. Nothing is re-scored here.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { RetrievalSearchResponse, RetrievalMode, MemoryType } from "../api/types";
import { useAction } from "../hooks/useResource";
import { Badge, Card, NumberField, OptionalNumberField, ScoreBar } from "../components/primitives";
import { ErrorState } from "../components/StateBlock";
import { MEMORY_TYPES, RETRIEVAL_MODES, SELECTION_REASON_CAPTIONS } from "../lib/constants";
import { duration, integer, num, truncateMiddle } from "../lib/format";

const DEFAULT_QUERY = "preferred backend language memforge";

function FactorList({ result }: { result: RetrievalSearchResponse["results"][number] }) {
  return (
    <div>
      <h3>
        Why it ranked — <span className="mono">{num(result.final_score)}</span>
      </h3>
      {result.factors.map((factor) => (
        <div className="factor" key={factor.name}>
          <span className="factor__name">{factor.name}</span>
          <ScoreBar value={factor.value} label={`${factor.name} ${factor.value}`} />
          <span className="factor__meta">
            weight {num(factor.weight, 2)} · contribution {num(factor.contribution)}
          </span>
          <span className="factor__why">{factor.explanation}</span>
        </div>
      ))}
      <p className="small muted" style={{ marginTop: 8 }}>
        {result.explanation}
      </p>
    </div>
  );
}

export function RetrievalPage() {
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [topK, setTopK] = useState(5);
  const [mode, setMode] = useState<RetrievalMode>("lexical");
  const [semanticWeight, setSemanticWeight] = useState<number | null>(null);
  const [minConfidence, setMinConfidence] = useState<number | null>(null);
  const [minImportance, setMinImportance] = useState<number | null>(null);
  const [memoryType, setMemoryType] = useState<MemoryType | "">("");

  const search = useAction<typeof query, RetrievalSearchResponse>((text) =>
    api.search({
      query: text,
      top_k: topK,
      retrieval_mode: mode,
      semantic_weight: semanticWeight,
      memory_type: memoryType === "" ? null : memoryType,
      status: "ACTIVE",
      min_confidence: minConfidence,
      min_importance: minImportance,
    }),
  );

  const response = search.state.result;
  const top = response?.results[0];

  return (
    <>
      <div className="page-intro">
        <h2>Retrieval Lab</h2>
        <p>
          Memory-aware ranking: query relevance is blended with the stored confidence, the computed
          effective priority, freshness, and the penalties from M2's duplicate and conflict graph.
          Every number below is the backend's own output.
        </p>
      </div>

      <div className="filter-bar">
        <div className="field filter-bar__field filter-bar__field--grow">
          <label className="field__label" htmlFor="retrieval-query">
            Query
          </label>
          <input
            id="retrieval-query"
            className="input"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="what should the agent remember?"
          />
        </div>
        <NumberField
          id="retrieval-topk"
          label="Top-k"
          value={topK}
          min={1}
          max={100}
          onChange={setTopK}
        />
        <div className="field filter-bar__field">
          <label className="field__label" htmlFor="retrieval-mode">
            Retrieval mode
          </label>
          <select
            id="retrieval-mode"
            className="select"
            value={mode}
            onChange={(event) => setMode(event.target.value as RetrievalMode)}
          >
            {RETRIEVAL_MODES.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </div>
        <OptionalNumberField
          id="retrieval-semantic-weight"
          label="Semantic weight"
          value={semanticWeight}
          min={0}
          max={1}
          step={0.05}
          placeholder="default"
          onChange={setSemanticWeight}
        />
        <OptionalNumberField
          id="retrieval-confidence"
          label="Min confidence"
          value={minConfidence}
          min={0}
          max={1}
          placeholder="any"
          onChange={setMinConfidence}
        />
        <OptionalNumberField
          id="retrieval-importance"
          label="Min importance"
          value={minImportance}
          min={0}
          max={1}
          placeholder="any"
          onChange={setMinImportance}
        />
        <div className="field filter-bar__field">
          <label className="field__label" htmlFor="retrieval-type">
            Memory type
          </label>
          <select
            id="retrieval-type"
            className="select"
            value={memoryType}
            onChange={(event) => setMemoryType(event.target.value as MemoryType | "")}
          >
            <option value="">any type</option>
            {MEMORY_TYPES.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </div>
        <div className="filter-bar__actions">
          <button
            type="button"
            className="btn btn--primary"
            onClick={() => void search.run(query)}
            disabled={search.state.running || query.trim().length === 0}
          >
            {search.state.running ? "Searching…" : "Run retrieval"}
          </button>
        </div>
      </div>

      {search.state.error !== null && (
        <ErrorState error={search.state.error} onRetry={() => void search.run(query)} retryLabel="Run again" />
      )}

      {response === null && !search.state.running && search.state.error === null && (
        <div className="state">
          <span className="state__title">No query run yet</span>
          <span className="state__detail">
            Press <strong>Run retrieval</strong>. The default query works against the corpus the demo
            seeds, so the page shows a real ranking as soon as the backend is up.
          </span>
        </div>
      )}

      {search.state.running && (
        <div className="state">
          <span className="row" style={{ gap: 9 }}>
            <span className="spinner" aria-hidden="true" />
            <span className="state__title">Running the retrieval service…</span>
          </span>
        </div>
      )}

      {response !== null && (
        <>
          <Card title="Query provenance" subtitle="What the retrieval service actually did">
            <div className="grid grid--stats">
              <div className="stat">
                <span className="stat__label">Retrieval mode</span>
                <span className="stat__value stat__value--text">
                  <Badge tone="accent">{response.retrieval_mode}</Badge>
                </span>
                <span className="stat__hint">requested {response.requested_mode}</span>
              </div>
              <div className="stat">
                <span className="stat__label">Semantic availability</span>
                <span className="stat__value stat__value--text">
                  <Badge tone={response.semantic_available ? "ok" : "warn"}>
                    {response.semantic_available ? "available" : "unavailable"}
                  </Badge>
                </span>
                <span className="stat__hint">
                  {response.semantic_provider ?? "no embedding provider"}
                  {response.semantic_model === null ? "" : ` · ${response.semantic_model}`}
                </span>
              </div>
              <div className="stat">
                <span className="stat__label">Candidates considered</span>
                <span className="stat__value">{integer(response.candidates_considered)}</span>
                <span className="stat__hint">after filters</span>
              </div>
              <div className="stat">
                <span className="stat__label">Returned</span>
                <span className="stat__value">{integer(response.candidates_returned)}</span>
                <span className="stat__hint">top_k {integer(response.top_k)}</span>
              </div>
              <div className="stat">
                <span className="stat__label">Suppressed duplicates</span>
                <span className="stat__value">{integer(response.candidates_suppressed)}</span>
                <span className="stat__hint">near-duplicates dropped from the top-k</span>
              </div>
              <div className="stat">
                <span className="stat__label">Latency</span>
                <span className="stat__value stat__value--text">{duration(response.latency_ms)}</span>
                <span className="stat__hint">measured at runtime, not a benchmark</span>
              </div>
            </div>
            {response.degraded_reason !== null && (
              <p className="note" style={{ marginTop: 12 }}>
                <strong>Degraded, and says so:</strong> {response.degraded_reason}
              </p>
            )}
          </Card>

          {response.results.length === 0 ? (
            <div className="state">
              <span className="state__title">No memory passed the filters</span>
              <span className="state__detail">
                The service considered {integer(response.candidates_considered)} candidate(s) and
                returned none. Lower the thresholds, widen the memory type, or switch status.
              </span>
            </div>
          ) : (
            <div className="stack">
              {top !== undefined && (
                <Card
                  title="Why rank #1 ranked first"
                  subtitle={`${top.memory_type} · relevance ${num(top.relevance)} · priority ${num(top.effective_priority)}`}
                >
                  <p style={{ fontSize: "var(--step-1)" }}>{top.content}</p>
                  <p className="small muted" style={{ marginTop: 8 }}>
                    {top.explanation}
                  </p>
                  <div className="row" style={{ gap: 8, marginTop: 10 }}>
                    <Link className="btn" to={`/memories/${top.memory_id}`}>
                      Open this memory
                    </Link>
                    <Link className="btn" to={`/traces?operation=memory.retrieval`}>
                      Find retrieval traces
                    </Link>
                  </div>
                </Card>
              )}

              <ol className="ranked">
                {response.results.map((result, index) => (
                  <li className="ranked__item" data-top={index === 0} key={result.memory_id}>
                    <div className="ranked__head">
                      <span className="ranked__rank">#{index + 1}</span>
                      <span>
                        <span className="ranked__score-label">final score</span>
                        <br />
                        <span className="ranked__score">{num(result.final_score)}</span>
                      </span>
                      <span className="row" style={{ gap: 8 }}>
                        <Badge tone="accent">{result.memory_type}</Badge>
                        <Badge>{result.status}</Badge>
                        {result.duplicate_of !== null && (
                          <Badge tone="warn" title={result.duplicate_of}>
                            duplicate of {truncateMiddle(result.duplicate_of, 10)}
                          </Badge>
                        )}
                        {result.superseded_by !== null && (
                          <Badge tone="bad" title={result.superseded_by}>
                            superseded by {truncateMiddle(result.superseded_by, 10)}
                          </Badge>
                        )}
                        {result.conflicts.map((conflict) => (
                          <Badge
                            key={`${conflict.target_memory_id}-${conflict.attribute}`}
                            tone={conflict.resolution === "supersedes" ? "ok" : "warn"}
                            title={conflict.resolution_reason ?? conflict.resolution}
                          >
                            conflict: {conflict.attribute}
                          </Badge>
                        ))}
                      </span>
                      <span style={{ marginLeft: "auto" }}>
                        <Link className="btn btn--ghost" to={`/memories/${result.memory_id}`}>
                          open
                        </Link>
                      </span>
                    </div>
                    <div className="ranked__body">
                      <div className="ranked__content">
                        <p>{result.content}</p>
                        <dl className="kv">
                          <dt>Relevance</dt>
                          <dd>
                            <ScoreBar value={result.relevance} label={`relevance ${result.relevance}`} />
                          </dd>
                          <dt>Effective priority</dt>
                          <dd>
                            <ScoreBar
                              value={result.effective_priority}
                              label={`effective priority ${result.effective_priority}`}
                            />
                          </dd>
                          <dt>Freshness</dt>
                          <dd>
                            <ScoreBar value={result.freshness} label={`freshness ${result.freshness}`} />
                          </dd>
                          <dt>Confidence (stored)</dt>
                          <dd>{num(result.confidence)}</dd>
                          <dt>Importance (stored)</dt>
                          <dd>{num(result.importance)}</dd>
                          <dt>Lexical score</dt>
                          <dd>{num(result.lexical_score)}</dd>
                          <dt>Semantic score</dt>
                          <dd>{result.semantic_score === null ? "—" : num(result.semantic_score)}</dd>
                        </dl>
                      </div>
                      <div className="ranked__factors">
                        <FactorList result={result} />
                      </div>
                    </div>
                    <div className="ranked__foot">
                      <span>
                        memory <b>{truncateMiddle(result.memory_id, 18)}</b>
                      </span>
                      <span>
                        final <b>{num(result.final_score)}</b>
                      </span>
                      <span>
                        relevance <b>{num(result.relevance)}</b>
                      </span>
                      <span>
                        importance <b>{num(result.importance)}</b>
                      </span>
                      <span>
                        confidence <b>{num(result.confidence)}</b>
                      </span>
                      <span>
                        freshness <b>{num(result.freshness)}</b>
                      </span>
                      <span>
                        effective priority <b>{num(result.effective_priority)}</b>
                      </span>
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {response.suppressed.length > 0 && (
            <Card
              title="Suppressed candidates"
              subtitle="Found, scored and deliberately left out. Suppression is a selection rule, never a deletion."
            >
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th scope="col">Memory</th>
                      <th scope="col" className="num">
                        Penalised final score
                      </th>
                      <th scope="col">Reason</th>
                      <th scope="col">Meaning</th>
                      <th scope="col" />
                    </tr>
                  </thead>
                  <tbody>
                    {response.suppressed.map((candidate) => (
                      <tr key={candidate.memory_id}>
                        <td className="mono">{truncateMiddle(candidate.memory_id, 22)}</td>
                        <td className="num">{num(candidate.final_score)}</td>
                        <td>
                          <Badge tone="warn">{candidate.reason}</Badge>
                        </td>
                        <td className="muted small">
                          {SELECTION_REASON_CAPTIONS[candidate.reason] ?? "Reported by the retrieval service."}
                        </td>
                        <td className="cell-actions">
                          <Link className="btn btn--ghost" to={`/memories/${candidate.memory_id}`}>
                            open
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </>
      )}
    </>
  );
}
