/**
 * Context Compiler.
 *
 * Calls `POST /api/v1/context/compile` and draws the pipeline it describes:
 *
 *   candidate memories → selected → suppressed → final context
 *
 * Every figure — the budget, the estimated tokens, the remaining tokens, the
 * counts, the reasons — is the backend's. The dashboard does not run a token
 * estimator of its own: a second count could disagree with the one the hard
 * guarantee is measured against, and the guarantee is the feature.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { CompileBody, ContextCompileResponse, ContextSelection, RetrievalMode, MemoryType } from "../api/types";
import { useAction } from "../hooks/useResource";
import { Badge, Card, CodePanel, NumberField, OptionalNumberField, ScoreBar } from "../components/primitives";
import { ErrorState } from "../components/StateBlock";
import { TokenBudget } from "../components/TokenBudget";
import { CONTEXT_SECTIONS, MEMORY_TYPES, RETRIEVAL_MODES, SELECTION_REASON_CAPTIONS } from "../lib/constants";
import { integer, num, truncateMiddle } from "../lib/format";

const DEFAULT_QUERY = "preferred backend language memforge";
const DEFAULT_BUDGET = 160;

function reasonTone(reason: string): "ok" | "warn" | "bad" {
  if (reason.startsWith("selected")) {
    return "ok";
  }
  if (reason === "conflict_superseded") {
    return "bad";
  }
  if (reason === "conflict_unresolved") {
    return "warn";
  }
  return "warn";
}

function SelectionRow({ entry }: { entry: ContextSelection }) {
  return (
    <tr>
      <td className="mono">{truncateMiddle(entry.memory_id, 22)}</td>
      <td>
        <Badge tone="accent">{entry.memory_type}</Badge>
      </td>
      <td>
        <Badge tone={entry.selected ? "ok" : reasonTone(entry.reason)}>{entry.reason}</Badge>
      </td>
      <td className="num">{num(entry.utility_score)}</td>
      <td className="num">{num(entry.value_score)}</td>
      <td className="num">{integer(entry.estimated_tokens)}</td>
      <td className="num">{integer(entry.remaining_tokens)}</td>
      <td className="muted small">{entry.detail}</td>
    </tr>
  );
}

export function ContextPage() {
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [budget, setBudget] = useState<number>(DEFAULT_BUDGET);
  const [topK, setTopK] = useState(20);
  const [mode, setMode] = useState<RetrievalMode>("lexical");
  const [memoryType, setMemoryType] = useState<MemoryType | "">("");
  const [compress, setCompress] = useState(true);
  const [maxBlockTokens, setMaxBlockTokens] = useState<number | null>(null);

  const compile = useAction<CompileBody, ContextCompileResponse>((body) => api.compile(body));

  const run = (): void => {
    void compile.run({
      query,
      token_budget: budget,
      top_k: topK,
      retrieval_mode: mode,
      preferred_memory_types: memoryType === "" ? null : [memoryType],
      compress,
      max_block_tokens: maxBlockTokens,
    });
  };

  const result = compile.state.result;

  return (
    <>
      <div className="page-intro">
        <h2>Context Compiler</h2>
        <p>
          M4 turns retrieved memories into the best context that fits a hard token budget: greedy
          admission by utility, conflict and duplicate suppression, deterministic ordering, and a
          decision recorded for every candidate.
        </p>
      </div>

      <div className="filter-bar">
        <div className="field filter-bar__field filter-bar__field--grow">
          <label className="field__label" htmlFor="context-query">
            Agent query
          </label>
          <input
            id="context-query"
            className="input"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
        <NumberField
          id="context-budget"
          label="Token budget"
          value={budget}
          min={1}
          onChange={setBudget}
        />
        <NumberField
          id="context-topk"
          label="Top-k candidates"
          value={topK}
          min={1}
          max={100}
          onChange={setTopK}
        />
        <div className="field filter-bar__field">
          <label className="field__label" htmlFor="context-mode">
            Retrieval mode
          </label>
          <select
            id="context-mode"
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
        <div className="field filter-bar__field">
          <label className="field__label" htmlFor="context-type">
            Preferred type
          </label>
          <select
            id="context-type"
            className="select"
            value={memoryType}
            onChange={(event) => setMemoryType(event.target.value as MemoryType | "")}
          >
            <option value="">no preference</option>
            {MEMORY_TYPES.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </div>
        <OptionalNumberField
          id="context-max-block"
          label="Max block tokens"
          value={maxBlockTokens}
          min={1}
          step={1}
          placeholder="no ceiling"
          onChange={setMaxBlockTokens}
        />
        <div className="field field--inline filter-bar__field">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={compress}
              onChange={(event) => setCompress(event.target.checked)}
            />
            deterministic compression
          </label>
        </div>
        <div className="filter-bar__actions">
          <button
            type="button"
            className="btn btn--primary"
            onClick={run}
            disabled={compile.state.running || query.trim().length === 0}
          >
            {compile.state.running ? "Compiling…" : "Compile context"}
          </button>
        </div>
      </div>

      {compile.state.error !== null && (
        <ErrorState error={compile.state.error} onRetry={run} retryLabel="Compile again" />
      )}

      {result === null && !compile.state.running && compile.state.error === null && (
        <div className="state">
          <span className="state__title">No context compiled yet</span>
          <span className="state__detail">
            The default budget of <strong>{DEFAULT_BUDGET}</strong> estimated tokens is deliberately
            small, so the budget has something to say. Press <strong>Compile context</strong>.
          </span>
        </div>
      )}

      {compile.state.running && (
        <div className="state">
          <span className="row" style={{ gap: 9 }}>
            <span className="spinner" aria-hidden="true" />
            <span className="state__title">Retrieving, scoring and packing…</span>
          </span>
        </div>
      )}

      {result !== null && (
        <>
          <Card
            title="Budget"
            subtitle="The guarantee is measured against these backend figures"
          >
            <TokenBudget
              budget={result.token_budget}
              estimated={result.estimated_tokens}
              remaining={result.remaining_tokens}
              withinBudget={result.within_budget}
              utilization={result.utilization}
              estimator={result.explanation.estimator}
            />
          </Card>

          <Card title="The pipeline" subtitle="What the compiler did, in order">
            <div className="flow">
              <div className="flow__step">
                <span className="flow__label">Candidate memories</span>
                <span className="flow__count">{integer(result.candidates_considered)}</span>
                <span className="flow__detail">retrieved by M3, then scored by M4</span>
              </div>
              <span className="flow__arrow" aria-hidden="true">
                ↓
              </span>
              <div className="flow__step" data-emphasis="selected">
                <span className="flow__label">Selected</span>
                <span className="flow__count">{integer(result.selected_count)}</span>
                <span className="flow__detail">
                  {integer(result.explanation.compressed_blocks)} compressed ·{" "}
                  {integer(result.explanation.tokens_saved)} tokens saved
                </span>
              </div>
              <span className="flow__arrow" aria-hidden="true">
                ↓
              </span>
              <div className="flow__step" data-emphasis="suppressed">
                <span className="flow__label">Suppressed</span>
                <span className="flow__count">{integer(result.suppressed_count)}</span>
                <span className="flow__detail">
                  {integer(result.explanation.selection_counts.duplicate_of ?? 0)} duplicate ·{" "}
                  {integer(result.explanation.selection_counts.conflict_superseded ?? 0)} conflict ·{" "}
                  {integer(result.explanation.selection_counts.insufficient_budget ?? 0)} budget
                </span>
              </div>
              <span className="flow__arrow" aria-hidden="true">
                ↓
              </span>
              <div className="flow__step">
                <span className="flow__label">Final context</span>
                <span className="flow__count">{integer(result.estimated_tokens)}</span>
                <span className="flow__detail">
                  of {integer(result.token_budget)} estimated tokens
                </span>
              </div>
            </div>
          </Card>

          <div className="grid grid--split">
            <Card title="Selected blocks" subtitle="In the order the agent receives them">
              <div className="stack stack--tight">
                {result.blocks.map((block) => (
                  <div className="factor" key={block.memory_id}>
                    <span className="factor__name">
                      {block.order + 1}. {block.section}
                    </span>
                    <ScoreBar
                      value={block.utility_score}
                      label={`utility ${block.utility_score}`}
                      tone="ok"
                    />
                    <span className="factor__meta">
                      {integer(block.estimated_tokens)} tokens · relevance {num(block.relevance)} ·
                      retrieval {num(block.retrieval_score)} · type {block.memory_type}
                    </span>
                    <span className="factor__why">{block.rendered_content}</span>
                    <span className="factor__meta">
                      source {block.source ?? "—"} · confidence {num(block.confidence)} · importance{" "}
                      {num(block.importance)}
                      {block.compression.applied.length > 0 &&
                        ` · compression: ${block.compression.applied.join(", ")}`}
                      {block.conflicts.length > 0 && ` · ${block.conflicts.length} conflict link(s)`}
                    </span>
                  </div>
                ))}
              </div>
            </Card>

            <Card
              title="Selection reasons"
              subtitle="One decision per candidate, from the backend's own trace"
            >
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th scope="col">Memory</th>
                      <th scope="col">Reason</th>
                      <th scope="col" className="num">
                        Utility
                      </th>
                      <th scope="col" className="num">
                        Tokens
                      </th>
                      <th scope="col">Detail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.selection_trace.map((entry) => (
                      <SelectionRow key={entry.memory_id} entry={entry} />
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>

          <Card
            title="Suppressed candidates"
            subtitle="Rejected, with the reason and the memory that beat them"
          >
            {result.suppressed_candidates.length === 0 ? (
              <p className="muted small">Nothing was suppressed for this budget.</p>
            ) : (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th scope="col">Memory</th>
                      <th scope="col">Type</th>
                      <th scope="col">Reason</th>
                      <th scope="col" className="num">
                        Utility
                      </th>
                      <th scope="col" className="num">
                        Tokens
                      </th>
                      <th scope="col">Related</th>
                      <th scope="col">Detail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.suppressed_candidates.map((entry) => (
                      <tr key={entry.memory_id}>
                        <td className="mono">{truncateMiddle(entry.memory_id, 22)}</td>
                        <td>
                          <Badge tone="accent">{entry.memory_type}</Badge>
                        </td>
                        <td>
                          <Badge tone={reasonTone(entry.reason)}>{entry.reason}</Badge>
                        </td>
                        <td className="num">{num(entry.utility_score)}</td>
                        <td className="num">{integer(entry.estimated_tokens)}</td>
                        <td className="mono">
                          {entry.related_memory_id === null
                            ? "—"
                            : truncateMiddle(entry.related_memory_id, 18)}
                        </td>
                        <td className="muted small">{entry.detail}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          <Card
            title="Final rendered context"
            subtitle="Exactly the text the agent would receive"
            actions={
              <Link className="btn" to="/agent">
                Send this to the agent
              </Link>
            }
          >
            <CodePanel
              label={`context_text · ${integer(result.estimated_tokens)} estimated tokens`}
              tall
            >
              {result.context_text}
            </CodePanel>
          </Card>

          <Card title="Provenance and explanation" subtitle="Estimator, utility model, ordering">
            <div className="stack">
              <div className="grid grid--halves">
                <div className="stack stack--tight">
                  <h3>Estimator</h3>
                  <p className="small mono">{result.explanation.estimator_formula}</p>
                  <h3>Utility</h3>
                  <p className="small mono">{result.explanation.utility_formula}</p>
                  <div className="row" style={{ gap: 8 }}>
                    {Object.entries(result.explanation.utility_weights).map(([name, weight]) => (
                      <Badge key={name} mono>
                        {name} {num(weight, 2)}
                      </Badge>
                    ))}
                  </div>
                  <h3>Ordering</h3>
                  <p className="small mono">{result.explanation.ordering_policy}</p>
                  <div className="row" style={{ gap: 6 }}>
                    {result.explanation.section_order.map((section, index) => (
                      <span key={section} className="row" style={{ gap: 6 }}>
                        {index > 0 && <span className="faint">›</span>}
                        <Badge>{section}</Badge>
                      </span>
                    ))}
                  </div>
                </div>
                <div className="stack stack--tight">
                  <h3>Selection counts</h3>
                  <dl className="kv">
                    {Object.entries(result.explanation.selection_counts).map(([reason, count]) => (
                      <div key={reason} style={{ display: "contents" }}>
                        <dt>
                          <Badge tone={reasonTone(reason)}>{reason}</Badge>
                        </dt>
                        <dd>
                          {integer(count)} · {SELECTION_REASON_CAPTIONS[reason] ?? ""}
                        </dd>
                      </div>
                    ))}
                  </dl>
                  <h3>Honest caveats, from the backend</h3>
                  <ul className="stack stack--tight">
                    {result.explanation.notes.map((note) => (
                      <li key={note} className="small muted">
                        · {note}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              <p className="small faint">
                Sections are drawn from the backend's own section order (
                {CONTEXT_SECTIONS.join(" → ")} in the default policy). Retrieval mode:{" "}
                {result.retrieval_mode}
                {result.degraded_reason === null ? "" : ` (${result.degraded_reason})`}.
              </p>
            </div>
          </Card>
        </>
      )}

      {result !== null && (
        <p className="small faint">
          The token figures above are the backend's deterministic estimate (
          {result.explanation.estimator}), never a provider tokenisation, and the dashboard never
          recomputes them. Latency for a compilation is recorded per-trace on the Traces page rather
          than stated here, so it is never confused with a benchmark.
        </p>
      )}
    </>
  );
}
