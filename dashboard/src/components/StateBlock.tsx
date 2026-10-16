/**
 * The four states every page has to render: loading, empty, failed and
 * unavailable. Keeping them in one place is what stops a page from shipping a
 * blank screen when the API is down or the contract has drifted.
 */

import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import type { ResourceState } from "../hooks/useResource";
import { MalformedResponseError } from "../api/parse";
import { ApiError } from "../api/client";
import { Badge } from "./primitives";

function title(error: unknown): string {
  if (error instanceof MalformedResponseError) {
    return "Malformed API response";
  }
  if (error instanceof ApiError) {
    switch (error.kind) {
      case "unreachable":
        return "MemForge API unreachable";
      case "unavailable":
        return "Endpoint not available";
      case "validation":
        return "Request rejected by the API";
      case "not_found":
        return "Not found";
      default:
        return "API request failed";
    }
  }
  return "Unexpected error";
}

function hint(error: unknown): ReactNode {
  if (error instanceof ApiError) {
    if (error.kind === "unreachable") {
      return (
        <>
          Start the backend with <code>python -m uvicorn app.main:app --reload</code> and reload this
          page.
        </>
      );
    }
    if (error.kind === "unavailable") {
      return (
        <>
          The route returned 404/405 and is not part of this deployment. Only the endpoints MemForge
          ships are rendered here.
        </>
      );
    }
    if (error.kind === "validation") {
      return "The request the dashboard sent did not satisfy the backend contract.";
    }
  }
  if (error instanceof MalformedResponseError) {
    return "The dashboard only renders values the backend actually returned; nothing is guessed.";
  }
  return "No further detail is available.";
}

export function ErrorState({
  error,
  onRetry,
  retryLabel = "Retry",
}: {
  error: unknown;
  onRetry?: () => void;
  retryLabel?: string;
}) {
  const code = error instanceof ApiError ? error.code : "";
  return (
    <div className="state state--error" role="alert">
      <span className="row" style={{ gap: 8 }}>
        <span className="state__title">{title(error)}</span>
        {code.length > 0 && <Badge tone="bad" mono>{code}</Badge>}
      </span>
      <span className="state__detail">
        {error instanceof Error ? error.message : String(error)}
      </span>
      <span className="state__hint">{hint(error)}</span>
      {onRetry !== undefined && (
        <button type="button" className="btn" onClick={onRetry}>
          {retryLabel}
        </button>
      )}
    </div>
  );
}

export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="state" role="status" aria-live="polite">
      <span className="row" style={{ gap: 9 }}>
        <span className="spinner" aria-hidden="true" />
        <span className="state__title">{label}</span>
      </span>
    </div>
  );
}

/**
 * Render one resource through all four states.
 *
 * `isEmpty` decides whether ready data counts as empty; when it is omitted the
 * data is always rendered.
 */
export function ResourceView<T>({
  state,
  isEmpty,
  emptyTitle = "Nothing to show yet",
  emptyDetail,
  children,
  onRetry,
}: {
  state: ResourceState<T>;
  isEmpty?: (data: T) => boolean;
  emptyTitle?: string;
  emptyDetail?: ReactNode;
  children: (data: T) => ReactNode;
  onRetry?: () => void;
}) {
  if (state.status === "loading") {
    return <LoadingState />;
  }
  if (state.status === "error") {
    return <ErrorState error={state.error} onRetry={onRetry} />;
  }
  if (isEmpty !== undefined && isEmpty(state.data)) {
    return (
      <div className="state">
        <span className="state__title">{emptyTitle}</span>
        {emptyDetail !== undefined && <span className="state__detail">{emptyDetail}</span>}
      </div>
    );
  }
  return <>{children(state.data)}</>;
}

/** An inline "not available" marker for a single value inside a page. */
export function NotAvailable({ what }: { what: string }) {
  return (
    <span className="row" style={{ gap: 6 }}>
      <Badge tone="warn">not available</Badge>
      <span className="muted small">{what}</span>
    </span>
  );
}

/** Where to go when a page has nothing to show. */
export function EmptyActionLink({ to, label }: { to: string; label: string }) {
  return (
    <Link className="btn" to={to}>
      {label}
    </Link>
  );
}
