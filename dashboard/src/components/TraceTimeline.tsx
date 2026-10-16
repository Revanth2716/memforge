/**
 * The trace timeline: an append-only, correlated event list drawn as a tree.
 *
 * The order is the backend's recorded `sequence`, and the nesting is its
 * `parent_id`. The dashboard never re-orders, re-times or filters a row.
 *
 * Everything rendered here passes through the redaction filter first: a
 * credential-shaped key or value is dropped, not restyled.
 */

import { useMemo } from "react";
import type { TraceEvent } from "../api/types";
import { buildTraceTree, eventCaption, flattenTree, AGENT_LIFECYCLE, type TraceNode } from "../lib/traceTree";
import { safeEntries } from "../lib/redact";
import { humanize, duration, timestamp, truncateMiddle } from "../lib/format";
import { Badge } from "./primitives";

/** A readable label for an operation name the tracer emits. */
function operationLabel(operation: string): string {
  const known = AGENT_LIFECYCLE.find((stage) => stage.operation === operation);
  return known === undefined ? humanize(operation.replace(/\./g, " ")) : known.label;
}

function statusBadge(status: string) {
  const tone = status === "ok" ? "ok" : status === "skipped" ? "warn" : "bad";
  return <Badge tone={tone}>{status}</Badge>;
}

function memoryIds(metadata: Record<string, unknown>): string[] {
  const ids = metadata.memory_ids;
  return Array.isArray(ids) ? ids.filter((id): id is string => typeof id === "string") : [];
}

function providerModel(metadata: Record<string, unknown>): [string, string] | null {
  const provider = metadata.provider;
  const model = metadata.model;
  if (typeof provider !== "string" && typeof model !== "string") {
    return null;
  }
  return [typeof provider === "string" ? provider : "—", typeof model === "string" ? model : "—"];
}

function EventNode({ node, isLast }: { node: TraceNode; isLast: boolean }) {
  const { event, depth } = node;
  const entries = safeEntries(event.metadata);
  const ids = memoryIds(event.metadata);
  const provider = providerModel(event.metadata);
  return (
    <li className="timeline__node" data-depth={depth}>
      <div className="timeline__gutter">
        <span className="timeline__dot" data-status={event.status} aria-hidden="true" />
        {!isLast && <span className="timeline__line" aria-hidden="true" />}
      </div>
      <div className="timeline__card" style={{ marginLeft: depth * 14 }}>
        <div className="timeline__title">
          <span className="timeline__operation">{eventCaption(event.kind)}</span>
          <span className="mono faint">{event.operation}</span>
          {statusBadge(event.status)}
        </div>
        <div className="timeline__meta">
          <span title="Recorded position in the trace">#{event.sequence}</span>
          {event.duration_ms !== null && <span>{duration(event.duration_ms)}</span>}
          <span>{timestamp(event.timestamp)}</span>
          {event.parent_id !== null && (
            <span title="The opening event this one closes">parent {truncateMiddle(event.parent_id, 16)}</span>
          )}
        </div>
        {event.detail !== null && <div className="small muted">{event.detail}</div>}
        {entries.length > 0 && (
          <dl className="timeline__metadata">
            {entries.map(([key, value]) => (
              <div key={key}>
                <dt>{humanize(key)}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
        )}
        {(ids.length > 0 || provider !== null) && (
          <div className="timeline__ids">
            {ids.map((id) => (
              <Badge key={id} mono title={id}>
                {truncateMiddle(id, 14)}
              </Badge>
            ))}
            {provider !== null && (
              <Badge tone="accent" mono>
                {provider[0]}
                {provider[1] !== "—" ? ` · ${provider[1]}` : ""}
              </Badge>
            )}
          </div>
        )}
      </div>
    </li>
  );
}

/** A compact rail of the stages an agent turn went through. */
export function LifecycleRail({
  events,
}: {
  events: readonly TraceEvent[];
}) {
  const stages = useMemo(() => {
    const byOperation = new Map<string, string>();
    for (const event of events) {
      if (!byOperation.has(event.operation)) {
        byOperation.set(event.operation, event.status);
      }
    }
    return byOperation;
  }, [events]);

  if (stages.size === 0) {
    return null;
  }

  return (
    <ol className="rail">
      {[...stages.entries()].map(([operation, status], index) => (
        <li key={operation} style={{ display: "contents" }}>
          {index > 0 && (
            <span className="rail__separator" aria-hidden="true">
              →
            </span>
          )}
          <span className="rail__step" data-status={status}>
            {operationLabel(operation)}
          </span>
        </li>
      ))}
    </ol>
  );
}

export function TraceTimeline({ events }: { events: readonly TraceEvent[] }) {
  const ordered = useMemo(() => flattenTree(buildTraceTree(events)), [events]);
  return (
    <ol className="timeline">
      {ordered.map((node, index) => (
        <EventNode key={node.event.event_id} node={node} isLast={index === ordered.length - 1} />
      ))}
    </ol>
  );
}
