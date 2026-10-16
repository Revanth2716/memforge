/**
 * Trace list.
 *
 * `GET /api/v1/traces` — recent trace headers, newest first. The list is a
 * navigation surface: every row opens the recorded lifecycle of one request.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { TraceKind, TraceListResponse } from "../api/types";
import { useResource } from "../hooks/useResource";
import { Badge, Card } from "../components/primitives";
import { ResourceView } from "../components/StateBlock";
import { TRACE_KINDS } from "../lib/constants";
import { integer, relativeTime, timestamp, truncateMiddle } from "../lib/format";

const LIMITS = [10, 25, 50, 100] as const;

function attributesPreview(attributes: Record<string, unknown>): string {
  const entries = Object.entries(attributes).slice(0, 3);
  if (entries.length === 0) {
    return "—";
  }
  return entries
    .map(([key, value]) => `${key}=${typeof value === "object" ? JSON.stringify(value) : String(value)}`)
    .join("  ");
}

export function TracesPage() {
  const [limit, setLimit] = useState(25);
  const [kind, setKind] = useState<TraceKind | "">("");

  const { state, reload } = useResource(
    `traces-${limit}-${kind}`,
    () => api.listTraces(limit, kind === "" ? undefined : kind),
    { ttlMs: 3_000 },
  );

  return (
    <>
      <div className="page-intro">
        <h2>Traces</h2>
        <p>
          An append-only, correlated record of what a request actually did. Traces are never updated
          and never deleted, and they carry no prompt, answer or credential — only counts, ids,
          modes, sizes and measured durations.
        </p>
      </div>

      <div className="filter-bar">
        <div className="field filter-bar__field">
          <label className="field__label" htmlFor="trace-kind">
            Kind
          </label>
          <select
            id="trace-kind"
            className="select"
            value={kind}
            onChange={(event) => setKind(event.target.value as TraceKind | "")}
          >
            <option value="">any kind</option>
            {TRACE_KINDS.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </div>
        <div className="field filter-bar__field">
          <label className="field__label" htmlFor="trace-limit">
            Limit
          </label>
          <select
            id="trace-limit"
            className="select"
            value={limit}
            onChange={(event) => setLimit(Number(event.target.value))}
          >
            {LIMITS.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </div>
        <div className="filter-bar__actions">
          <button type="button" className="btn" onClick={reload}>
            Reload
          </button>
        </div>
      </div>

      <ResourceView
        state={state}
        onRetry={reload}
        emptyTitle="No trace has been recorded yet"
        emptyDetail="Run an agent turn, a memory write or a retrieval; the trace appears here immediately."
        isEmpty={(data) => data.traces.length === 0}
      >
        {(listing: TraceListResponse) => (
          <>
            <div className="table-wrap">
              <table className="table">
                <caption>
                  {integer(listing.count)} trace{listing.count === 1 ? "" : "s"} returned by{" "}
                  <code>GET /api/v1/traces</code>, newest first.
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Trace id</th>
                    <th scope="col">Kind</th>
                    <th scope="col">Operation</th>
                    <th scope="col">Started</th>
                    <th scope="col">Age</th>
                    <th scope="col">Attributes</th>
                    <th scope="col" />
                  </tr>
                </thead>
                <tbody>
                  {listing.traces.map((trace) => (
                    <tr key={trace.trace_id}>
                      <td>
                        <Link className="mono" to={`/traces/${encodeURIComponent(trace.trace_id)}`}>
                          {trace.trace_id}
                        </Link>
                      </td>
                      <td>
                        <Badge mono>{trace.kind}</Badge>
                      </td>
                      <td className="mono">{trace.operation}</td>
                      <td className="mono">{timestamp(trace.started_at)}</td>
                      <td className="small faint">{relativeTime(trace.started_at)}</td>
                      <td className="small faint mono">{attributesPreview(trace.attributes)}</td>
                      <td className="cell-actions">
                        <Link className="btn btn--ghost" to={`/traces/${encodeURIComponent(trace.trace_id)}`}>
                          open
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Card title="What a trace holds" subtitle="And what it deliberately does not">
              <div className="differentiators">
                <div className="differentiator">
                  <div className="differentiator__title">RECORDED</div>
                  <div className="differentiator__body">
                    Operation, status, sequence, parent, measured duration, counts, ids, modes and
                    provider-reported usage.
                  </div>
                </div>
                <div className="differentiator" style={{ borderLeftColor: "var(--warn)" }}>
                  <div className="differentiator__title" style={{ color: "var(--warn)" }}>
                    NEVER RECORDED
                  </div>
                  <div className="differentiator__body">
                    API keys, Authorization headers, prompts, answers, message bodies and
                    chain-of-thought. There is no field to hold them.
                  </div>
                </div>
              </div>
            </Card>
          </>
        )}
      </ResourceView>

      <p className="small faint">
        Showing at most {integer(limit)} traces ({truncateMiddle(kind === "" ? "any kind" : kind, 24)}
        ). The list endpoint caps at 200; the traces themselves live in the same SQLite file as the
        memories.
      </p>
    </>
  );
}
