/**
 * Dashboard home.
 *
 * Every figure on this page is read from the running backend: memory counts
 * come from `GET /api/v1/memories`, operational counters from
 * `GET /api/v1/metrics`, the score from `GET /api/v1/evaluations/latest` and
 * the activity lists from `GET /api/v1/traces` and `GET /api/v1/memories`.
 *
 * Nothing is hard-coded. When a section's endpoint is unavailable it says so,
 * on its own, without blanking the rest of the page.
 */

import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Evaluation, Memory, MemoryPage, Metrics, TraceListResponse } from "../api/types";
import { useResource, type ResourceState } from "../hooks/useResource";
import { Badge, Card, KeyValues } from "../components/primitives";
import { LoadingState, NotAvailable, ResourceView } from "../components/StateBlock";
import { integer, relativeTime, timestamp, truncateMiddle } from "../lib/format";
import { CAPABILITY_MILESTONES } from "../lib/constants";

/** One dashboard section, isolated so a failure cannot take the page down. */
function Section<T>({
  title,
  subtitle,
  state,
  children,
  emptyTitle,
  isEmpty,
  action,
}: {
  title: string;
  subtitle?: string;
  state: ResourceState<T>;
  children: (data: T) => ReactNode;
  emptyTitle?: string;
  isEmpty?: (data: T) => boolean;
  action?: ReactNode;
}) {
  return (
    <Card title={title} subtitle={subtitle} actions={action}>
      <ResourceView state={state} emptyTitle={emptyTitle} isEmpty={isEmpty}>
        {children}
      </ResourceView>
    </Card>
  );
}

function MemoryCountCard({ status, label }: { status: "ACTIVE" | "ARCHIVED" | "DELETED"; label: string }) {
  const { state } = useResource(
    `memories-total-${status}`,
    () => api.listMemories({ status, limit: 1, offset: 0 }),
    { ttlMs: 5_000 },
  );
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      {state.status === "loading" && <LoadingState label="counting…" />}
      {state.status === "error" && <NotAvailable what="the memory counter endpoint did not answer" />}
      {state.status === "ready" && (
        <>
          <span className="stat__value">{integer(state.data.total)}</span>
          <span className="stat__hint">
            <code>GET /memories?status={status}</code>
          </span>
        </>
      )}
    </div>
  );
}

function StatCard({
  label,
  value,
  hint,
  text,
}: {
  label: string;
  value: number | null;
  hint: ReactNode;
  text?: boolean;
}) {
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      {value === null ? (
        <>
          <span className="stat__value stat__value--text">not available</span>
          <span className="stat__hint">{hint}</span>
        </>
      ) : (
        <>
          <span className={text ? "stat__value stat__value--text" : "stat__value"}>
            {integer(value)}
          </span>
          <span className="stat__hint">{hint}</span>
        </>
      )}
    </div>
  );
}

const QUICK_ACTIONS: readonly { to: string; title: string; body: string }[] = [
  {
    to: "/retrieval",
    title: "Run a retrieval",
    body: "Search the corpus and read why rank #1 ranked first.",
  },
  {
    to: "/context",
    title: "Compile a context",
    body: "Set a small token budget and watch selection and suppression.",
  },
  {
    to: "/agent",
    title: "Ask the agent",
    body: "One memory-aware turn, with the gate, recall and provider shown.",
  },
  { to: "/traces", title: "Open a trace", body: "Follow the recorded lifecycle of a real request." },
  {
    to: "/evaluation",
    title: "Read the evaluation",
    body: "The measured score of the shipped 36-case dataset.",
  },
  { to: "/metrics", title: "Inspect metrics", body: "Counters derived from recorded events, never estimated." },
];

function TraceRow({ trace }: { trace: TraceListResponse["traces"][number] }) {
  return (
    <li
      className="row row--between"
      style={{ padding: "8px 0", borderBottom: "1px solid var(--border)" }}
    >
      <span className="stack stack--tight" style={{ gap: 2 }}>
        <Link className="mono" to={`/traces/${encodeURIComponent(trace.trace_id)}`}>
          {truncateMiddle(trace.trace_id, 26)}
        </Link>
        <span className="small faint">
          {trace.operation} · {timestamp(trace.started_at)}
        </span>
      </span>
      <span className="row" style={{ gap: 8 }}>
        <Badge mono>{trace.kind}</Badge>
        <span className="small faint">{relativeTime(trace.started_at)}</span>
      </span>
    </li>
  );
}

function MemoryRow({ memory }: { memory: Memory }) {
  return (
    <li
      className="row row--between"
      style={{ padding: "8px 0", borderBottom: "1px solid var(--border)" }}
    >
      <span className="cell-content">
        <Link to={`/memories/${memory.id}`}>{memory.content}</Link>
      </span>
      <span className="row" style={{ gap: 8 }}>
        <Badge tone="accent">{memory.memory_type}</Badge>
        <span className="mono faint">v{memory.version}</span>
      </span>
    </li>
  );
}

export function DashboardPage() {
  const metrics = useResource("metrics", () => api.getMetrics(), { ttlMs: 5_000 });
  const evaluation = useResource<Evaluation>("evaluation-latest", () => api.getLatestEvaluation(), {
    ttlMs: 60_000,
  });
  const traces = useResource<TraceListResponse>("traces", () => api.listTraces(6), { ttlMs: 5_000 });
  const memories = useResource<MemoryPage>(
    "memories-recent",
    () => api.listMemories({ limit: 6, offset: 0 }),
    { ttlMs: 5_000 },
  );

  const metricsData = metrics.state.status === "ready" ? metrics.state.data : null;
  const evaluationData = evaluation.state.status === "ready" ? evaluation.state.data : null;

  return (
    <>
      <div className="page-intro">
        <h2>MemForge control plane</h2>
        <p>
          This dashboard is a <strong>visualisation layer</strong> over the real memory, retrieval,
          context, agent, evaluation and observability APIs. It holds no memory of its own, no second
          storage, and no model: every figure below was returned by the backend for the corpus that
          is currently in SQLite.
        </p>
      </div>

      <Card title="30–60 second demo path" subtitle="Each step runs a real backend operation.">
        <div className="quick-actions">
          {QUICK_ACTIONS.map((action) => (
            <Link key={action.to} to={action.to} className="quick-action">
              <span className="quick-action__title">{action.title}</span>
              <span className="quick-action__body">{action.body}</span>
            </Link>
          ))}
        </div>
      </Card>

      <div className="grid grid--stats">
        <MemoryCountCard status="ACTIVE" label="Active memories" />
        <MemoryCountCard status="ARCHIVED" label="Archived" />
        <MemoryCountCard status="DELETED" label="Forgotten" />
        <StatCard
          label="Retrieval operations"
          value={metricsData === null ? null : (metricsData.operations["memory.retrieval"] ?? 0)}
          hint="memory.retrieval spans completed"
        />
        <StatCard
          label="Context compilations"
          value={metricsData === null ? null : (metricsData.operations["context.compile"] ?? 0)}
          hint="context.compile spans completed"
        />
        <StatCard
          label="Agent requests"
          value={metricsData === null ? null : (metricsData.traces.agent_request ?? 0)}
          hint="traces of kind agent_request"
        />
        <StatCard
          label="Provider calls"
          value={metricsData === null ? null : (metricsData.provider.calls ?? 0)}
          hint="llm_request events recorded"
        />
        <div className="stat">
          <span className="stat__label">Evaluation score</span>
          {evaluation.state.status === "loading" && <LoadingState label="running dataset…" />}
          {evaluation.state.status === "error" && (
            <NotAvailable what="the evaluation endpoint did not answer" />
          )}
          {evaluationData !== null && (
            <>
              <span className="stat__value">{evaluationData.overall.toFixed(3)}</span>
              <span className="stat__hint">
                {evaluationData.reference} · {integer(evaluationData.passed_cases)}/
                {integer(evaluationData.total_cases)} cases passed
              </span>
            </>
          )}
        </div>
        <div className="stat">
          <span className="stat__label">Latest evaluation</span>
          {evaluation.state.status === "loading" && <LoadingState label="running dataset…" />}
          {evaluation.state.status === "error" && (
            <NotAvailable what="the evaluation endpoint did not answer" />
          )}
          {evaluationData !== null && (
            <>
              <span className="stat__value stat__value--text">
                <Badge tone={evaluationData.passed ? "ok" : "bad"}>
                  {evaluationData.passed ? "passed" : "failed"}
                </Badge>
              </span>
              <span className="stat__hint">
                threshold {evaluationData.threshold.toFixed(2)} ·{" "}
                {integer(evaluationData.failed_cases)} failed case
                {evaluationData.failed_cases === 1 ? "" : "s"}
              </span>
            </>
          )}
        </div>
      </div>

      <div className="grid grid--split">
        <Section
          title="Latest traces"
          subtitle="Newest first, straight from GET /api/v1/traces"
          state={traces.state}
          emptyTitle="No traces recorded yet"
          isEmpty={(data) => data.traces.length === 0}
        >
          {(data) => (
            <ul>
              {data.traces.map((trace) => (
                <TraceRow key={trace.trace_id} trace={trace} />
              ))}
            </ul>
          )}
        </Section>

        <Section
          title="Latest evaluation summary"
          subtitle="Deterministic MemForge evaluation of the shipped dataset"
          state={evaluation.state}
        >
          {(data) => (
            <div className="stack">
              <KeyValues
                items={[
                  ["Dataset", `${data.dataset}@${data.version}`],
                  [
                    "Cases",
                    `${integer(data.passed_cases)} passed / ${integer(data.failed_cases)} failed of ${integer(data.total_cases)}`,
                  ],
                  ["Overall score", data.overall.toFixed(3)],
                  ["Metric score", data.metric_score.toFixed(3)],
                  ["Case score", data.case_score.toFixed(3)],
                ]}
              />
              <div className="row" style={{ gap: 8 }}>
                {Object.entries(data.categories).map(([name, counts]) => (
                  <Badge key={name} mono>
                    {name} {counts.passed}/{counts.total}
                  </Badge>
                ))}
              </div>
              <Link className="btn" to="/evaluation">
                Open the evaluation
              </Link>
            </div>
          )}
        </Section>
      </div>

      <div className="grid grid--split">
        <Section
          title="Recent memory activity"
          subtitle="Newest active memories from GET /api/v1/memories"
          state={memories.state}
          emptyTitle="No memories stored yet"
          isEmpty={(data) => data.items.length === 0}
          action={
            <Link className="btn" to="/memories">
              Open explorer
            </Link>
          }
        >
          {(data) => (
            <ul>
              {data.items.map((memory) => (
                <MemoryRow key={memory.id} memory={memory} />
              ))}
            </ul>
          )}
        </Section>

        <Section
          title="Operational counters"
          subtitle="Counters derived by counting recorded events"
          state={metrics.state}
        >
          {(data: Metrics) => (
            <div className="stack">
              <KeyValues
                items={[
                  ["Events recorded", integer(data.event_count)],
                  ["Memories written", integer(data.memory.written ?? 0)],
                  ["Memories updated", integer(data.memory.updated ?? 0)],
                  ["Memories forgotten", integer(data.memory.forgotten ?? 0)],
                  ["Memories recalled", integer(data.memory.recalled ?? 0)],
                  ["Context selections", integer(data.context.selected ?? 0)],
                  ["Duplicate candidates", integer(data.memory.duplicate_candidates ?? 0)],
                  ["Conflicts suppressed", integer(data.context.conflicts_suppressed ?? 0)],
                  ["Provider usage reported", integer(data.usage_reported)],
                  ["Failed traces", integer(data.failed_traces)],
                ]}
              />
              <Link className="btn" to="/metrics">
                Open metrics
              </Link>
            </div>
          )}
        </Section>
      </div>

      <Card title="What the system does" subtitle="The differentiators, in the milestones that ship them">
        <div className="differentiators">
          {CAPABILITY_MILESTONES.map((item) => (
            <div className="differentiator" key={item.title}>
              <div className="differentiator__title">{item.title}</div>
              <div className="differentiator__body">{item.body}</div>
            </div>
          ))}
        </div>
      </Card>
    </>
  );
}
