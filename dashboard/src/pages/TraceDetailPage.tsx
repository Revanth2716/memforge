/**
 * One trace, drawn as a tree.
 *
 * `GET /api/v1/traces/{id}` gives the header and the derived summary;
 * `GET /api/v1/traces/{id}/events` gives the ordered, redacted events. The
 * nesting is the backend's `parent_id`, the order is its `sequence`, and the
 * durations are its measurements.
 */

import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import type { TraceDetail, TraceEvent } from "../api/types";
import { useResource } from "../hooks/useResource";
import { Badge, Card, KeyValues } from "../components/primitives";
import { ResourceView } from "../components/StateBlock";
import { LifecycleRail, TraceTimeline } from "../components/TraceTimeline";
import { AGENT_LIFECYCLE } from "../lib/traceTree";
import { humanize, integer, num, duration, timestamp } from "../lib/format";

function SummaryPanel({ traceId, summary }: { traceId: string; summary: TraceDetail["summary"] }) {
  return (
    <Card title="Summary" subtitle="Counters derived by walking the recorded events">
      <div className="stack">
        <div className="grid grid--stats">
          <div className="stat">
            <span className="stat__label">Events</span>
            <span className="stat__value">{integer(summary.event_count)}</span>
            <span className="stat__hint">append-only rows</span>
          </div>
          <div className="stat">
            <span className="stat__label">Total duration</span>
            <span className="stat__value stat__value--text">{duration(summary.total_duration_ms)}</span>
            <span className="stat__hint">measured at runtime, not a benchmark</span>
          </div>
          <div className="stat">
            <span className="stat__label">Outcome</span>
            <span className="stat__value stat__value--text">
              <Badge tone={summary.succeeded ? "ok" : "bad"}>
                {summary.succeeded ? "succeeded" : "failed"}
              </Badge>
            </span>
            <span className="stat__hint">{integer(summary.errors.length)} error event(s)</span>
          </div>
          <div className="stat">
            <span className="stat__label">Memories touched</span>
            <span className="stat__value">{integer(summary.memory_ids.length)}</span>
            <span className="stat__hint">distinct ids in the recorded metadata</span>
          </div>
        </div>

        <KeyValues
          items={[
            ["Trace id", <span className="mono">{traceId}</span>],
            ["Kind", <Badge mono>{summary.kind}</Badge>],
            ["Operation", <span className="mono">{summary.operation}</span>],
            ["Started", timestamp(summary.started_at)],
            ...Object.entries(summary.operations).map(
              ([operation, status]) =>
                [
                  <span className="mono">{operation}</span>,
                  <Badge tone={status === "ok" ? "ok" : status === "skipped" ? "warn" : "bad"}>
                    {status}
                  </Badge>,
                ] as const,
            ),
          ]}
        />

        {Object.keys(summary.durations_ms).length > 0 && (
          <div>
            <h3>Measured duration by operation</h3>
            {Object.entries(summary.durations_ms).map(([operation, ms]) => (
              <div className="factor" key={operation}>
                <span className="factor__name">{operation}</span>
                <span className="mono">{duration(ms)}</span>
                <span className="factor__meta">sum of the events of this operation</span>
              </div>
            ))}
          </div>
        )}

        {summary.errors.length > 0 && (
          <div className="state state--error">
            <span className="state__title">This trace contains a failure</span>
            <ul className="stack stack--tight">
              {summary.errors.map((error) => (
                <li key={error} className="small mono">
                  {error}
                </li>
              ))}
            </ul>
            <span className="state__hint">
              A failed step is recorded as carefully as a successful one; nothing is retried or
              silently dropped.
            </span>
          </div>
        )}
      </div>
    </Card>
  );
}

function LifecyclePanel({ events }: { events: readonly TraceEvent[] }) {
  const stages = AGENT_LIFECYCLE.map((stage) => ({
    stage,
    events: events.filter((event) => event.operation === stage.operation),
  })).filter((entry) => entry.events.length > 0);

  if (stages.length === 0) {
    return null;
  }

  return (
    <Card
      title="Lifecycle"
      subtitle="The stages the backend recorded, in the order it recorded them"
    >
      <div className="stack">
        <LifecycleRail events={events as TraceEvent[]} />
        <ol className="stack stack--tight">
          {stages.map(({ stage, events: stageEvents }) => (
            <li className="factor" key={stage.operation}>
              <span className="factor__name">{stage.label}</span>
              <span className="mono">{stage.operation}</span>
              <span className="factor__meta">
                {integer(stageEvents.length)} event
                {stageEvents.length === 1 ? "" : "s"} ·{" "}
                {stageEvents
                  .map((event) => `${event.kind}(${event.status})`)
                  .join(", ")}
              </span>
              {stageEvents.some((event) => event.kind === "retrieval_complete") && (
                <span className="factor__meta">
                  recalled{" "}
                  {num(
                    Number(
                      stageEvents.find((event) => event.kind === "retrieval_complete")?.metadata
                        .recalled ?? 0,
                    ),
                    0,
                  )}{" "}
                  · candidates{" "}
                  {num(
                    Number(
                      stageEvents.find((event) => event.kind === "retrieval_complete")?.metadata
                        .candidates ?? 0,
                    ),
                    0,
                  )}
                </span>
              )}
            </li>
          ))}
        </ol>
      </div>
    </Card>
  );
}

export function TraceDetailPage() {
  const { traceId = "" } = useParams();
  const detail = useResource<TraceDetail>(
    `trace-${traceId}`,
    () => api.getTrace(traceId),
    { ttlMs: 5_000, enabled: traceId.length > 0 },
  );
  const events = useResource<TraceEvent[]>(
    `trace-events-${traceId}`,
    () => api.getTraceEvents(traceId),
    { ttlMs: 5_000, enabled: traceId.length > 0 },
  );

  return (
    <>
      <div className="row row--between">
        <div className="page-intro">
          <h2>Trace</h2>
          <p>
            One request, every recorded step. Order is the backend's <code>sequence</code>, nesting is
            its <code>parent_id</code>, and the timeline below is a rendering of those two fields —
            nothing is inferred.
          </p>
        </div>
        <Link className="btn" to="/traces">
          ← All traces
        </Link>
      </div>

      {traceId.length > 0 && (
        <Card title={<span className="mono">{traceId}</span>} subtitle="Trace id">
          <ResourceView state={detail.state}>
            {(data: TraceDetail) => (
              <KeyValues
                items={[
                  ["Kind", <Badge mono>{data.trace.kind}</Badge>],
                  ["Operation", <span className="mono">{data.trace.operation}</span>],
                  ["Started", timestamp(data.trace.started_at)],
                  ["Events recorded", integer(data.event_count)],
                  [
                    "Attributes",
                    <span className="mono small">
                      {Object.entries(data.trace.attributes).length === 0
                        ? "—"
                        : Object.entries(data.trace.attributes)
                            .map(([key, value]) => `${humanize(key)}=${String(value)}`)
                            .join(" · ")}
                    </span>,
                  ],
                ]}
              />
            )}
          </ResourceView>
        </Card>
      )}

      <ResourceView state={detail.state}>
        {(data: TraceDetail) => <SummaryPanel traceId={data.trace.trace_id} summary={data.summary} />}
      </ResourceView>

      <ResourceView
        state={events.state}
        emptyTitle="This trace recorded no event"
        isEmpty={(data) => data.length === 0}
      >
        {(data: TraceEvent[]) => <LifecyclePanel events={data} />}
      </ResourceView>

      <Card
        title="Event timeline"
        subtitle="Append-only events, with safe metadata only"
      >
        <ResourceView
          state={events.state}
          emptyTitle="This trace recorded no event"
          isEmpty={(data) => data.length === 0}
        >
          {(data: TraceEvent[]) => (
            <>
              <p className="small faint" style={{ marginBottom: 12 }}>
                {integer(data.length)} event{data.length === 1 ? "" : "s"}. A credential-shaped key or
                value would be dropped here even if the backend had recorded one; the backend records
                none.
              </p>
              <TraceTimeline events={data} />
            </>
          )}
        </ResourceView>
      </Card>
    </>
  );
}
