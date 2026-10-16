/**
 * Memory detail.
 *
 * Six endpoints, all read-only, all for one memory: the memory itself, its
 * append-only history, its computed intelligence, its near-duplicates, its
 * unresolved conflicts and its recorded relationships. Every panel fails
 * independently, so a missing relationship never hides the version history.
 */

import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type {
  ConflictResolution,
  Duplicates,
  Conflicts,
  Intelligence,
  Memory,
  MemoryHistory,
  Relationships,
} from "../api/types";
import { useAction, useResource, type ResourceState } from "../hooks/useResource";
import { Badge, Card, KeyValues, ScoreBar } from "../components/primitives";
import { ErrorState, ResourceView } from "../components/StateBlock";
import { VERSION_CHANGE_CAPTIONS } from "../lib/constants";
import { age, integer, num, percent, timestamp } from "../lib/format";

function changeTone(change: string): "ok" | "warn" | "bad" {
  const entry = VERSION_CHANGE_CAPTIONS[change];
  return entry === undefined ? "ok" : entry.tone;
}

function HistoryPanel({ memoryId }: { memoryId: string }) {
  const { state, reload } = useResource<MemoryHistory>(
    `history-${memoryId}`,
    () => api.getHistory(memoryId),
    { ttlMs: 10_000 },
  );
  return (
    <Card
      title="Version history"
      subtitle="Append-only snapshots, oldest first. Nothing is ever overwritten."
    >
      <ResourceView state={state} onRetry={reload}>
        {(history) => (
          <ol className="version-list">
            {history.items.map((version, index) => {
              const caption = VERSION_CHANGE_CAPTIONS[version.change];
              const previous = index > 0 ? history.items[index - 1] : undefined;
              const changedFields: string[] = [];
              if (previous !== undefined) {
                if (previous.content !== version.content) changedFields.push("content");
                if (previous.memory_type !== version.memory_type) changedFields.push("type");
                if (previous.status !== version.status) changedFields.push("status");
                if (previous.confidence !== version.confidence) changedFields.push("confidence");
                if (previous.importance !== version.importance) changedFields.push("importance");
                if (previous.source !== version.source) changedFields.push("source");
              }
              return (
                <li className="version" key={`${version.memory_id}-${version.version}`}>
                  <div className="version__tag">
                    <span className="version__number">v{version.version}</span>
                    <span className="version__change" data-tone={changeTone(version.change)}>
                      {version.change}
                    </span>
                    {index === history.items.length - 1 && (
                      <span className="version__change" data-tone="ok" title="Current version">
                        current
                      </span>
                    )}
                  </div>
                  <div className="version__body">
                    <span className="version__content">{version.content}</span>
                    {changedFields.length > 0 && (
                      <span className="small faint">
                        changed: {changedFields.join(", ")} (from v{previous?.version})
                      </span>
                    )}
                    <div className="version__meta">
                      <span>{timestamp(version.recorded_at)}</span>
                      <span>type {version.memory_type}</span>
                      <span>status {version.status}</span>
                      <span>confidence {num(version.confidence)}</span>
                      <span>importance {num(version.importance)}</span>
                      <span>source {version.source ?? "—"}</span>
                    </div>
                    {caption === undefined && (
                      <span className="small faint">
                        Unrecognised change code reported by the API.
                      </span>
                    )}
                  </div>
                </li>
              );
            })}
          </ol>
        )}
      </ResourceView>
    </Card>
  );
}

function IntelligencePanel({ memoryId }: { memoryId: string }) {
  const { state, reload } = useResource<Intelligence>(
    `intelligence-${memoryId}`,
    () => api.getIntelligence(memoryId),
    { ttlMs: 5_000 },
  );
  return (
    <Card title="Computed intelligence" subtitle="Recomputed on every request; never stored.">
      <ResourceView state={state} onRetry={reload}>
        {(data) => (
          <div className="stack">
            <div className="grid grid--stats">
              <div className="stat">
                <span className="stat__label">Importance score</span>
                <span className="stat__value">{num(data.importance_score)}</span>
                <span className="stat__hint">stored importance {num(data.stored_importance)}</span>
              </div>
              <div className="stat">
                <span className="stat__label">Effective priority</span>
                <span className="stat__value">{num(data.effective_priority)}</span>
                <span className="stat__hint">base {num(data.base_priority)}</span>
              </div>
              <div className="stat">
                <span className="stat__label">Decay multiplier</span>
                <span className="stat__value">{num(data.decay_multiplier)}</span>
                <span className="stat__hint">
                  age {age(data.age_days)} · half-life {num(data.decay_half_life_days, 1)} d
                </span>
              </div>
              <div className="stat">
                <span className="stat__label">Access count</span>
                <span className="stat__value">{integer(data.access_count)}</span>
                <span className="stat__hint">{timestamp(data.last_accessed_at)}</span>
              </div>
            </div>
            <KeyValues
              items={[
                ["Confidence (stored)", num(data.confidence)],
                ["Importance (stored)", num(data.stored_importance)],
                ["Importance (computed)", num(data.importance_score)],
                ["Effective priority", num(data.effective_priority)],
                ["Base priority", num(data.base_priority)],
                ["Decay multiplier", num(data.decay_multiplier)],
                ["Age (days)", num(data.age_days, 2)],
                ["Duplicate threshold", num(data.similarity_threshold, 2)],
                ["Computed at", timestamp(data.computed_at)],
              ]}
            />
            <div>
              <h3>Importance factors</h3>
              {data.importance_factors.map((factor) => (
                <div className="factor" key={factor.name}>
                  <span className="factor__name">{factor.name}</span>
                  <ScoreBar value={factor.raw_value} label={`${factor.name} ${factor.raw_value}`} />
                  <span className="factor__meta">
                    weight {num(factor.weight, 2)} · contribution {num(factor.contribution)}
                  </span>
                  <span className="factor__why">{factor.explanation}</span>
                </div>
              ))}
            </div>
            <p className="small muted">{data.importance_explanation}</p>
            <p className="small muted">{data.priority_explanation}</p>
          </div>
        )}
      </ResourceView>
    </Card>
  );
}

function DuplicatesPanel({ memoryId }: { memoryId: string }) {
  const { state, reload } = useResource<Duplicates>(
    `duplicates-${memoryId}`,
    () => api.getDuplicates(memoryId),
    { ttlMs: 10_000 },
  );
  return (
    <Card title="Near-duplicates" subtitle="Lexical Dice similarity over normalised token sets.">
      <ResourceView
        state={state}
        onRetry={reload}
        emptyTitle="No near-duplicate detected"
        isEmpty={(data) => data.items.length === 0}
      >
        {(data) => (
          <div className="stack stack--tight">
            <span className="small faint">
              threshold {num(data.threshold, 2)} · {integer(data.total)} match
              {data.total === 1 ? "" : "es"}
            </span>
            {data.items.map((match) => (
              <div className="factor" key={match.memory_id}>
                <Link className="factor__name" to={`/memories/${match.memory_id}`}>
                  {match.memory_id.slice(0, 8)}
                </Link>
                <ScoreBar value={match.similarity} label={`similarity ${match.similarity}`} tone="warn" />
                <span className="factor__meta">type {match.memory_type}</span>
                <span className="factor__why">{match.content}</span>
                <span className="factor__meta">shared: {match.shared_tokens.join(", ") || "—"}</span>
                <span className="factor__meta">{match.explanation}</span>
              </div>
            ))}
          </div>
        )}
      </ResourceView>
    </Card>
  );
}

function ConflictsPanel({ memoryId }: { memoryId: string }) {
  const { state, reload } = useResource<Conflicts>(
    `conflicts-${memoryId}`,
    () => api.getConflicts(memoryId),
    { ttlMs: 5_000 },
  );
  const [resolution, setResolution] = useState<ConflictResolution | null>(null);
  const action = useAction<string, ConflictResolution>((targetId) =>
    api.resolveConflict(memoryId, targetId),
  );

  const resolve = (targetId: string): void => {
    void action.run(targetId).then((result) => {
      if (result !== null) {
        setResolution(result);
        reload();
      }
    });
  };

  return (
    <Card
      title="Conflicts"
      subtitle="Same attribute, different value. Resolution is a documented total order."
    >
      <div className="stack">
        <ResourceView
          state={state}
          onRetry={reload}
          emptyTitle="No unresolved conflict"
          isEmpty={(data) => data.items.length === 0}
        >
          {(data) => (
            <div className="stack stack--tight">
              {data.items.map((conflict) => (
                <div className="factor" key={`${conflict.attribute}-${conflict.target_memory_id}`}>
                  <span className="factor__name">{conflict.attribute}</span>
                  <span className="mono">
                    {conflict.source_value} vs {conflict.target_value}
                  </span>
                  <span className="factor__meta">
                    value similarity {num(conflict.value_similarity)}
                  </span>
                  <span className="factor__why">{conflict.explanation}</span>
                  <button
                    type="button"
                    className="btn"
                    onClick={() => resolve(conflict.target_memory_id)}
                    disabled={action.state.running}
                  >
                    {action.state.running ? "Resolving…" : "Resolve deterministically"}
                  </button>
                </div>
              ))}
            </div>
          )}
        </ResourceView>

        {action.state.error !== null && <ErrorState error={action.state.error} onRetry={reload} />}

        {resolution !== null && (
          <div className="stack stack--tight">
            <span className="row" style={{ gap: 8 }}>
              <Badge tone="ok">resolved</Badge>
              <span className="mono">winner {resolution.winner_memory_id.slice(0, 8)}</span>
              <span className="mono">loser {resolution.loser_memory_id.slice(0, 8)}</span>
            </span>
            <ol className="stack stack--tight">
              {resolution.steps.map((step, index) => (
                <li key={`${step.factor}-${index}`} className="small mono faint">
                  {index + 1}. {step.factor}: {step.left_value} vs {step.right_value} →{" "}
                  {step.winner_id.slice(0, 8)}
                </li>
              ))}
            </ol>
            <p className="small muted">{resolution.explanation}</p>
            {resolution.loser_archived && (
              <p className="small muted">
                The loser was archived, which appended a history version. Nothing was deleted.
              </p>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}

function RelationshipsPanel({ memoryId }: { memoryId: string }) {
  const { state, reload } = useResource<Relationships>(
    `relationships-${memoryId}`,
    () => api.getRelationships(memoryId),
    { ttlMs: 10_000 },
  );
  return (
    <Card title="Relationships" subtitle="DUPLICATE_OF / CONFLICTS_WITH / SUPERSEDES links.">
      <ResourceView
        state={state}
        onRetry={reload}
        emptyTitle="No relationship recorded"
        isEmpty={(data) => data.items.length === 0}
      >
        {(data) => (
          <div className="stack stack--tight">
            {data.items.map((relationship) => (
              <div className="row row--between" key={`${relationship.source_id}-${relationship.relationship}-${relationship.target_id}`}>
                <span className="row" style={{ gap: 8 }}>
                  <Badge tone="accent">{relationship.relationship}</Badge>
                  <Link className="mono" to={`/memories/${relationship.target_id}`}>
                    {relationship.target_id.slice(0, 8)}
                  </Link>
                </span>
                <span className="small faint">
                  {relationship.similarity === null ? "—" : `similarity ${num(relationship.similarity)}`} ·{" "}
                  {timestamp(relationship.created_at)}
                </span>
              </div>
            ))}
          </div>
        )}
      </ResourceView>
    </Card>
  );
}

export function MemoryDetailPage() {
  const { memoryId = "" } = useParams();
  const { state, reload } = useResource<Memory>(
    `memory-${memoryId}`,
    () => api.getMemory(memoryId),
    { ttlMs: 3_000, enabled: memoryId.length > 0 },
  );
  // `GET /memories/{id}` answers 404 once a memory is forgotten, but its
  // history survives by design (M1 soft delete). The page therefore still loads
  // history on a 404 rather than showing a dead end.
  const forgotten =
    state.status === "error" &&
    state.error instanceof ApiError &&
    state.error.kind === "not_found";

  // A 404 is ambiguous on its own: M1 answers it both for a memory that was
  // forgotten and for one that never existed. History settles it, because a
  // forgotten memory ends with a FORGOTTEN version and a nonexistent one has no
  // history at all.
  const history = useResource<MemoryHistory>(
    `history-${memoryId}`,
    () => api.getHistory(memoryId),
    { ttlMs: 10_000, enabled: forgotten && memoryId.length > 0 },
  );
  const wasForgotten =
    forgotten &&
    history.state.status === "ready" &&
    history.state.data.items.some((version) => version.change === "FORGOTTEN");
  const viewable = (
    wasForgotten
      ? ({ status: "ready", data: undefined } as unknown as ResourceState<Memory | undefined>)
      : (state as ResourceState<Memory | undefined>)
  );

  return (
    <>
      <div className="row row--between">
        <div className="page-intro">
          <h2>Memory</h2>
          <p>
            One record, all of its recorded state. Panels load independently; a missing endpoint is
            reported, never guessed at.
          </p>
        </div>
        <Link className="btn" to="/memories">
          ← Back to the explorer
        </Link>
      </div>

      {wasForgotten && (
        <Card
          title="This memory was forgotten"
          subtitle="Nothing was destroyed: the memory is soft-deleted and its history is still readable."
        >
          <p className="muted small">
            <code>GET /api/v1/memories/{memoryId}</code> answers{" "}
            <strong>404 MEMORY_NOT_FOUND</strong> because the memory no longer takes part in reads.
            Its append-only versions survive below, which is the whole point of the M1 soft delete.
          </p>
        </Card>
      )}

      <ResourceView state={viewable} onRetry={reload}>
        {(memory) =>
          memory === undefined ? null : (
            <div className="memory-detail">
              <div className="stack">
                <Card
                  title={<span className="mono">{memory.id}</span>}
                  subtitle={`stored at ${timestamp(memory.created_at)}`}
                >
                  <div className="stack">
                    <p style={{ fontSize: "var(--step-2)", lineHeight: 1.45 }}>{memory.content}</p>
                    <KeyValues
                      items={[
                        ["Memory type", <Badge tone="accent">{memory.memory_type}</Badge>],
                        [
                          "Status",
                          <Badge
                            tone={
                              memory.status === "ACTIVE"
                                ? "ok"
                                : memory.status === "ARCHIVED"
                                  ? "warn"
                                  : "bad"
                            }
                          >
                            {memory.status}
                          </Badge>,
                        ],
                        ["Confidence (stored)", num(memory.confidence)],
                        ["Importance (stored)", num(memory.importance)],
                        ["Version", `v${memory.version}`],
                        ["Source", memory.source ?? "—"],
                        ["Created", timestamp(memory.created_at)],
                        ["Updated", timestamp(memory.updated_at)],
                        [
                          "Stored fields",
                          `${percent(memory.confidence)} confidence · ${percent(memory.importance)} importance`,
                        ],
                      ]}
                    />
                  </div>
                </Card>
                <HistoryPanel memoryId={memory.id} />
                <ConflictsPanel memoryId={memory.id} />
              </div>
              <div className="stack">
                <IntelligencePanel memoryId={memory.id} />
                <DuplicatesPanel memoryId={memory.id} />
                <RelationshipsPanel memoryId={memory.id} />
              </div>
            </div>
          )
        }
      </ResourceView>

      {wasForgotten && memoryId.length > 0 && <HistoryPanel memoryId={memoryId} />}
    </>
  );
}
