/**
 * Turning a flat, append-only event list into the tree the trace viewer draws.
 *
 * The backend stores one row per event with a total `sequence` and an optional
 * `parent_id` (the opening event of the operation the closing event belongs to).
 * The dashboard does not re-derive or re-order anything: it groups by
 * `parent_id`, keeps `sequence` order, and treats an event whose parent is
 * missing as a root so the timeline can never silently drop a row.
 */

import type { TraceEvent } from "../api/types";
import { humanize } from "./format";

export interface TraceNode {
  event: TraceEvent;
  children: TraceNode[];
  /** Depth in the rendered tree (0 for a root). */
  depth: number;
}

/** Group events under their parent, preserving the recorded sequence. */
export function buildTraceTree(events: readonly TraceEvent[]): TraceNode[] {
  const ordered = [...events].sort((left, right) => left.sequence - right.sequence);
  const byId = new Map<string, TraceNode>();
  for (const event of ordered) {
    byId.set(event.event_id, { event, children: [], depth: 0 });
  }

  const roots: TraceNode[] = [];
  for (const node of byId.values()) {
    const parentId = node.event.parent_id;
    const parent = parentId === null ? undefined : byId.get(parentId);
    if (parent === undefined || parent === node) {
      roots.push(node);
    } else {
      parent.children.push(node);
    }
  }
  assignDepths(roots, 0);
  return roots;
}

function assignDepths(nodes: TraceNode[], depth: number): void {
  for (const node of nodes) {
    node.depth = depth;
    assignDepths(node.children, depth + 1);
  }
}

/**
 * The top-level stages of an agent turn, in the order the backend records them.
 *
 * Used to lay out the "lifecycle rail" at the top of the trace detail view. The
 * labels come from the operation names the tracer emits; nothing here decides
 * whether a stage happened.
 */
export const AGENT_LIFECYCLE: readonly { operation: string; label: string }[] = [
  { operation: "agent.request", label: "Agent Request" },
  { operation: "agent.memory_gate", label: "Memory Gate" },
  { operation: "memory.retrieval", label: "Retrieval" },
  { operation: "context.compile", label: "Context Compilation" },
  { operation: "llm.generate", label: "LLM Call" },
  { operation: "agent.memory_candidate", label: "Candidate Extraction" },
  { operation: "memory.write", label: "Memory Write" },
  { operation: "agent.response", label: "Response" },
];

/** Flatten a tree back into a render-ordered list, children right after parents. */
export function flattenTree(nodes: readonly TraceNode[]): TraceNode[] {
  const output: TraceNode[] = [];
  const walk = (items: readonly TraceNode[]): void => {
    for (const node of items) {
      output.push(node);
      walk(node.children);
    }
  };
  walk(nodes);
  return output;
}

/** A short caption for an event kind, e.g. `retrieval_complete` -> `Retrieval complete`. */
export function eventCaption(kind: string): string {
  return humanize(kind);
}
