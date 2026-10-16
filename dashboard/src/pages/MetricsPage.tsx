/**
 * Metrics.
 *
 * `GET /api/v1/metrics` returns **aggregate counters** derived by counting
 * recorded events. There is no time series in the backend, so this page draws
 * no time series: a chart of something MemForge does not measure would be
 * decoration pretending to be data.
 *
 * What is shown instead: the counters themselves, each with the population it
 * was derived from (`event_count`), and the per-trace durations the trace store
 * does hold.
 */

import { useMemo } from "react";
import { api } from "../api/client";
import type { Metrics, TraceListResponse } from "../api/types";
import { useResource } from "../hooks/useResource";
import { Badge, Card, KeyValues, ScoreBar } from "../components/primitives";
import { ResourceView } from "../components/StateBlock";
import { integer, percent } from "../lib/format";

interface CounterGroup {
  title: string;
  caption: string;
  counters: Record<string, number>;
}

function group(title: string, caption: string, counters: Record<string, number> | undefined): CounterGroup {
  return { title, caption, counters: counters ?? {} };
}

export function MetricsPage() {
  const { state, reload } = useResource<Metrics>("metrics", () => api.getMetrics(), { ttlMs: 3_000 });
  const traces = useResource<TraceListResponse>("traces-200", () => api.listTraces(200), { ttlMs: 10_000 });

  const groups = useMemo<CounterGroup[]>(() => {
    if (state.status !== "ready") {
      return [];
    }
    const metrics = state.data;
    return [
      group("Memory", "Writes, updates, forgets and recalls recorded by the memory lifecycle.", {
        written: metrics.memory.written ?? 0,
        created: metrics.memory.created ?? 0,
        updated: metrics.memory.updated ?? 0,
        forgotten: metrics.memory.forgotten ?? 0,
        writes_skipped: metrics.memory.writes_skipped ?? 0,
        recalled: metrics.memory.recalled ?? 0,
        duplicate_candidates: metrics.memory.duplicate_candidates ?? 0,
        candidates: metrics.memory.candidates ?? 0,
      }),
      group("Retrieval", "M3 spans completed.", {
        retrievals: metrics.operations["memory.retrieval"] ?? 0,
      }),
      group("Context", "M4 compilation outcomes.", {
        compilations: metrics.operations["context.compile"] ?? 0,
        candidates: metrics.context.candidates ?? 0,
        selected: metrics.context.selected ?? 0,
        suppressed: metrics.context.suppressed ?? 0,
        conflicts_suppressed: metrics.context.conflicts_suppressed ?? 0,
        over_budget: metrics.context.over_budget ?? 0,
      }),
      group("Agent", "M5 turns and the steps inside them.", {
        agent_requests: metrics.traces.agent_request ?? 0,
        memory_gates: metrics.operations["agent.memory_gate"] ?? 0,
        candidate_extractions: metrics.operations["agent.memory_candidate"] ?? 0,
        responses: metrics.operations["agent.response"] ?? 0,
      }),
      group("MCP", "Calls made through the MCP transport.", {
        mcp_operations: metrics.operations["mcp.request"] ?? 0,
        mcp_calls: metrics.provider.mcp_calls ?? 0,
        mcp_failures: metrics.provider.mcp_failures ?? 0,
      }),
      group("Provider", "LLM calls, as recorded — usage is only ever what a provider sent.", {
        calls: metrics.provider.calls ?? 0,
        failures: metrics.provider.failures ?? 0,
        usage_reported: metrics.provider.usage_reported ?? 0,
      }),
    ];
  }, [state]);

  return (
    <>
      <div className="page-intro">
        <h2>Metrics</h2>
        <p>
          Counters derived by counting recorded trace events. Nothing here is estimated, and a counter
          that was never incremented is reported as zero rather than as a measurement.
        </p>
      </div>

      <ResourceView state={state} onRetry={reload}>
        {(metrics: Metrics) => (
          <>
            <div className="grid grid--stats">
              <div className="stat">
                <span className="stat__label">Events recorded</span>
                <span className="stat__value">{integer(metrics.event_count)}</span>
                <span className="stat__hint">the population every counter below is derived from</span>
              </div>
              <div className="stat">
                <span className="stat__label">Traces</span>
                <span className="stat__value">
                  {integer(Object.values(metrics.traces).reduce((sum, value) => sum + value, 0))}
                </span>
                <span className="stat__hint">
                  {Object.entries(metrics.traces).map(([kind, count]) => `${kind} ${count}`).join(" · ") ||
                    "no trace recorded"}
                </span>
              </div>
              <div className="stat">
                <span className="stat__label">Event statuses</span>
                <span className="stat__value">
                  {integer(Object.values(metrics.statuses).reduce((sum, value) => sum + value, 0))}
                </span>
                <span className="stat__hint">
                  {Object.entries(metrics.statuses)
                    .map(([status, count]) => `${status} ${count}`)
                    .join(" · ") || "—"}
                </span>
              </div>
              <div className="stat">
                <span className="stat__label">Failed traces</span>
                <span className="stat__value">{integer(metrics.failed_traces)}</span>
                <span className="stat__hint">traces holding at least one error event</span>
              </div>
              <div className="stat">
                <span className="stat__label">Usage reported</span>
                <span className="stat__value">{integer(metrics.usage_reported)}</span>
                <span className="stat__hint">
                  provider calls that actually sent a usage block — never derived
                </span>
              </div>
            </div>

            <div className="grid grid--cards">
              {groups.map((entry) => (
                <Card key={entry.title} title={entry.title} subtitle={entry.caption}>
                  <dl className="kv">
                    {Object.entries(entry.counters).map(([key, value]) => (
                      <div key={key} style={{ display: "contents" }}>
                        <dt className="mono">{key}</dt>
                        <dd>
                          <span className="row" style={{ gap: 10 }}>
                            <b className="mono">{integer(value)}</b>
                            <ScoreBar
                              value={
                                metrics.event_count === 0
                                  ? null
                                  : value / Math.max(1, metrics.event_count)
                              }
                              label={`${key} share of recorded events`}
                            />
                            <span className="small faint">
                              {metrics.event_count === 0
                                ? "—"
                                : percent(value / Math.max(1, metrics.event_count), 1)}
                            </span>
                          </span>
                        </dd>
                      </div>
                    ))}
                  </dl>
                </Card>
              ))}
            </div>

            <Card
              title="Latency, where the backend measured it"
              subtitle="Per-trace totals from the trace store — real measurements, not a benchmark"
            >
              <ResourceView
                state={traces.state}
                emptyTitle="No trace recorded yet"
                isEmpty={(data) => data.traces.length === 0}
              >
                {(data: TraceListResponse) => (
                  <>
                    <KeyValues
                      items={[
                        ["Traces listed", integer(data.count)],
                        [
                          "Kinds",
                          Object.entries(
                            data.traces.reduce<Record<string, number>>((accumulator, trace) => {
                              accumulator[trace.kind] = (accumulator[trace.kind] ?? 0) + 1;
                              return accumulator;
                            }, {}),
                          )
                            .map(([kind, count]) => `${kind} ${count}`)
                            .join(" · ") || "—",
                        ],
                        [
                          "Note",
                          "The metrics endpoint reports aggregate counters only. A per-trace duration is read from the trace itself, on the trace page.",
                        ],
                      ]}
                    />
                    <div className="row" style={{ gap: 8, marginTop: 12 }}>
                      {data.traces.slice(0, 8).map((trace) => (
                        <Badge key={trace.trace_id} mono>
                          {trace.operation}
                        </Badge>
                      ))}
                    </div>
                  </>
                )}
              </ResourceView>
            </Card>

            <div className="note">
              <strong>Why there is no chart.</strong> MemForge stores counters and append-only traces,
              not a time series. The metrics endpoint has no historical buckets to plot, so this page
              shows the aggregate counters it does return. Plotting a fabricated series would be
              inventing data; a per-trace latency figure is one click away on any trace.
            </div>
          </>
        )}
      </ResourceView>
    </>
  );
}
