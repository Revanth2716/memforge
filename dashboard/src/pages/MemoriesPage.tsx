/**
 * Memory explorer.
 *
 * Pagination, status and memory type are handed to the backend, because
 * `GET /api/v1/memories` supports exactly those. The confidence/importance
 * thresholds, the text filter and the sort order are applied to the **loaded
 * page** and labelled as such: the list endpoint has no query parameter for
 * them, and presenting a client-side filter as a server query would be a lie
 * about what the API can do. Full-corpus ranked search lives in the Retrieval
 * Lab, which calls `POST /api/v1/memories/search`.
 */

import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Memory, MemoryStatus, MemoryType } from "../api/types";
import { useResource } from "../hooks/useResource";
import { Badge, OptionalNumberField, ScoreBar } from "../components/primitives";
import { ResourceView } from "../components/StateBlock";
import { Pagination } from "../components/Pagination";
import { MEMORY_STATUSES, MEMORY_TYPES, STATUS_TONES } from "../lib/constants";
import { integer, timestamp, truncateMiddle } from "../lib/format";

const PAGE_SIZES = [10, 25, 50, 100] as const;
type SortKey = "newest" | "oldest" | "confidence" | "importance" | "version";

const SORT_LABELS: Record<SortKey, string> = {
  newest: "Newest updated first (the API's order)",
  oldest: "Oldest updated first",
  confidence: "Confidence, high to low",
  importance: "Importance, high to low",
  version: "Version, high to low",
};

function memoryTone(status: MemoryStatus): "ok" | "warn" | "bad" {
  return STATUS_TONES[status];
}

export function MemoriesPage() {
  const [status, setStatus] = useState<MemoryStatus | "">("ACTIVE");
  const [memoryType, setMemoryType] = useState<MemoryType | "">("");
  const [pageSize, setPageSize] = useState<number>(25);
  const [offset, setOffset] = useState(0);
  const [text, setText] = useState("");
  const [minConfidence, setMinConfidence] = useState<number | null>(null);
  const [minImportance, setMinImportance] = useState<number | null>(null);
  const [sort, setSort] = useState<SortKey>("newest");

  const { state, reload } = useResource(
    `memories-${status}-${memoryType}-${pageSize}-${offset}`,
    () =>
      api.listMemories({
        status: status === "" ? undefined : status,
        memory_type: memoryType === "" ? undefined : memoryType,
        limit: pageSize,
        offset,
      }),
    { ttlMs: 2_000 },
  );

  const visible = useMemo<Memory[]>(() => {
    if (state.status !== "ready") {
      return [];
    }
    const confidenceFloor = minConfidence;
    const importanceFloor = minImportance;
    const needle = text.trim().toLowerCase();
    const filtered = state.data.items.filter((memory) => {
      if (needle.length > 0 && !memory.content.toLowerCase().includes(needle)) {
        return false;
      }
      if (confidenceFloor !== null && memory.confidence < confidenceFloor) {
        return false;
      }
      if (importanceFloor !== null && memory.importance < importanceFloor) {
        return false;
      }
      return true;
    });
    const sorted = [...filtered];
    switch (sort) {
      case "oldest":
        sorted.sort((left, right) => left.updated_at.localeCompare(right.updated_at));
        break;
      case "confidence":
        sorted.sort((left, right) => right.confidence - left.confidence);
        break;
      case "importance":
        sorted.sort((left, right) => right.importance - left.importance);
        break;
      case "version":
        sorted.sort((left, right) => right.version - left.version);
        break;
      default:
        break;
    }
    return sorted;
  }, [state, text, minConfidence, minImportance, sort]);

  const filtering = text.trim().length > 0 || minConfidence !== null || minImportance !== null;

  return (
    <>
      <div className="page-intro">
        <h2>Memory explorer</h2>
        <p>
          The stored, versioned corpus. Status, memory type and pagination are backend filters; the
          remaining controls narrow the page that is already loaded.
        </p>
      </div>

      <div className="filter-bar">
        <div className="field filter-bar__field filter-bar__field--grow">
          <label className="field__label" htmlFor="memory-filter-text">
            Filter loaded page
          </label>
          <input
            id="memory-filter-text"
            className="input"
            type="search"
            placeholder="substring of the memory content"
            value={text}
            onChange={(event) => setText(event.target.value)}
          />
        </div>
        <div className="field filter-bar__field">
          <label className="field__label" htmlFor="memory-filter-status">
            Status
          </label>
          <select
            id="memory-filter-status"
            className="select"
            value={status}
            onChange={(event) => {
              setStatus(event.target.value as MemoryStatus | "");
              setOffset(0);
            }}
          >
            <option value="">any (defaults to ACTIVE)</option>
            {MEMORY_STATUSES.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </div>
        <div className="field filter-bar__field">
          <label className="field__label" htmlFor="memory-filter-type">
            Memory type
          </label>
          <select
            id="memory-filter-type"
            className="select"
            value={memoryType}
            onChange={(event) => {
              setMemoryType(event.target.value as MemoryType | "");
              setOffset(0);
            }}
          >
            <option value="">any type</option>
            {MEMORY_TYPES.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </div>
        <OptionalNumberField
          id="memory-filter-confidence"
          label="Min confidence"
          value={minConfidence}
          min={0}
          max={1}
          placeholder="0.00"
          onChange={setMinConfidence}
        />
        <OptionalNumberField
          id="memory-filter-importance"
          label="Min importance"
          value={minImportance}
          min={0}
          max={1}
          placeholder="0.00"
          onChange={setMinImportance}
        />
        <div className="field filter-bar__field">
          <label className="field__label" htmlFor="memory-sort">
            Sort (page)
          </label>
          <select
            id="memory-sort"
            className="select"
            value={sort}
            onChange={(event) => setSort(event.target.value as SortKey)}
          >
            {(Object.keys(SORT_LABELS) as SortKey[]).map((key) => (
              <option key={key} value={key}>
                {SORT_LABELS[key]}
              </option>
            ))}
          </select>
        </div>
        <div className="field filter-bar__field">
          <label className="field__label" htmlFor="memory-page-size">
            Page size
          </label>
          <select
            id="memory-page-size"
            className="select"
            value={pageSize}
            onChange={(event) => {
              setPageSize(Number(event.target.value));
              setOffset(0);
            }}
          >
            {PAGE_SIZES.map((size) => (
              <option key={size} value={size}>
                {size}
              </option>
            ))}
          </select>
        </div>
        <div className="filter-bar__actions">
          <button
            type="button"
            className="btn"
            onClick={() => {
              setText("");
              setMinConfidence(null);
              setMinImportance(null);
            }}
            disabled={!filtering}
          >
            Reset filters
          </button>
          <button type="button" className="btn" onClick={reload}>
            Reload
          </button>
        </div>
      </div>

      <ResourceView
        state={state}
        onRetry={reload}
        emptyTitle="No memories match this page"
        isEmpty={(data) => data.items.length === 0}
      >
        {(data) => (
          <div className="stack">
            <div className="table-wrap">
              <table className="table">
                <caption>
                  {integer(data.total)} memor{data.total === 1 ? "y" : "ies"} in the backend for
                  this filter · showing {integer(visible.length)} on this page
                  {filtering ? " after page-level filters" : ""}
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Content</th>
                    <th scope="col">Type</th>
                    <th scope="col">Status</th>
                    <th scope="col" className="num">
                      Confidence
                    </th>
                    <th scope="col" className="num">
                      Importance
                    </th>
                    <th scope="col" className="num">
                      V
                    </th>
                    <th scope="col">Updated</th>
                    <th scope="col">Id</th>
                  </tr>
                </thead>
                <tbody>
                  {visible.map((memory) => (
                    <tr key={memory.id}>
                      <td className="cell-content">
                        <Link to={`/memories/${memory.id}`}>{memory.content}</Link>
                      </td>
                      <td>
                        <Badge tone="accent">{memory.memory_type}</Badge>
                      </td>
                      <td>
                        <Badge tone={memoryTone(memory.status)}>{memory.status}</Badge>
                      </td>
                      <td className="num">
                        <ScoreBar value={memory.confidence} label={`confidence ${memory.confidence}`} />
                      </td>
                      <td className="num">
                        <ScoreBar
                          value={memory.importance}
                          label={`importance ${memory.importance}`}
                          tone="accent"
                        />
                      </td>
                      <td className="num">v{memory.version}</td>
                      <td className="mono">{timestamp(memory.updated_at)}</td>
                      <td>
                        <span className="mono faint" title={memory.id}>
                          {truncateMiddle(memory.id, 14)}
                        </span>
                      </td>
                    </tr>
                  ))}
                  {visible.length === 0 && (
                    <tr>
                      <td className="table__empty" colSpan={8}>
                        Nothing on this page passes the page-level filters. Raise the thresholds,
                        clear the text filter, or use the Retrieval Lab for a ranked search.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            <Pagination
              total={data.total}
              limit={data.limit}
              offset={data.offset}
              onChange={(next) => setOffset(next)}
            />
            <p className="small faint">
              Confidence and importance are stored fields (accuracy, and the caller-supplied
              relevance). The computed importance score, temporal decay and effective priority are
              on the memory detail page, which calls{" "}
              <code>GET /memories/{"{id}"}/intelligence</code>. A backend-ranked answer to a query
              lives in the Retrieval Lab.
            </p>
          </div>
        )}
      </ResourceView>
    </>
  );
}
