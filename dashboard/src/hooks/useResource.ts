/**
 * Request-state hooks.
 *
 * The dashboard is read-mostly and its data changes slowly, so a module-level
 * in-flight promise map plus a short TTL is enough to stop a page from
 * re-issuing the same GET on every render — without pulling in a data-fetching
 * framework or adding a cache invalidation problem the project does not have.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export type ResourceState<T> =
  | { status: "loading" }
  | { status: "error"; error: unknown }
  | { status: "ready"; data: T };

interface CacheEntry {
  at: number;
  value: unknown;
}

const cache = new Map<string, CacheEntry>();
const inFlight = new Map<string, Promise<unknown>>();

/** Drop every cached response (used by tests and by a manual refresh). */
export function clearResourceCache(): void {
  cache.clear();
  inFlight.clear();
}

export interface ResourceOptions {
  /** Skip the request entirely (e.g. a detail view without an id yet). */
  enabled?: boolean;
  /** How long a cached response may be reused, in ms. `0` disables caching. */
  ttlMs?: number;
  /** Re-run when any of these change. */
  deps?: readonly unknown[];
}

const DEFAULT_TTL_MS = 3_000;

/**
 * Load one read-only resource and expose the three states every page renders:
 * loading, error (API failure / malformed response / unavailable endpoint) and
 * ready.
 */
export function useResource<T>(
  key: string,
  loader: () => Promise<T>,
  options: ResourceOptions = {},
): { state: ResourceState<T>; reload: () => void } {
  const { enabled = true, ttlMs = DEFAULT_TTL_MS, deps = [] } = options;
  const [state, setState] = useState<ResourceState<T>>(() =>
    enabled && ttlMs > 0 && cache.has(key)
      ? { status: "ready", data: cache.get(key)!.value as T }
      : { status: "loading" },
  );
  const [nonce, setNonce] = useState(0);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;

  const reload = useCallback(() => {
    cache.delete(key);
    setNonce((value) => value + 1);
  }, [key]);

  useEffect(() => {
    if (!enabled) {
      setState({ status: "loading" });
      return;
    }
    const controller = new AbortController();
    let active = true;
    const cached = ttlMs > 0 ? cache.get(key) : undefined;
    if (cached && Date.now() - cached.at < ttlMs) {
      setState({ status: "ready", data: cached.value as T });
      return () => controller.abort();
    }

    setState((current) =>
      current.status === "ready" && !cached ? { status: "loading" } : current,
    );
    const existing = inFlight.get(key) as Promise<T> | undefined;
    const pending =
      existing ??
      (loaderRef.current() as Promise<T>).finally(() => {
        inFlight.delete(key);
      });
    if (!existing) {
      inFlight.set(key, pending);
    }

    pending
      .then((data) => {
        if (ttlMs > 0) {
          cache.set(key, { at: Date.now(), value: data });
        }
        if (active) {
          setState({ status: "ready", data });
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setState({ status: "error", error });
        }
      });

    return () => {
      active = false;
      controller.abort();
    };
    // `deps` is the caller's explicit dependency list; `key` already captures
    // everything the loader reads.

  }, [key, enabled, ttlMs, nonce, JSON.stringify(deps)]);

  return { state, reload };
}

export interface ActionState<TResult> {
  running: boolean;
  result: TResult | null;
  error: unknown;
}

/**
 * One caller-driven mutation (search, compile, respond, resolve).
 *
 * The result is never cached: a POST is a real operation against the backend,
 * and re-running it is the point.
 */
export function useAction<TInput, TResult>(
  action: (input: TInput) => Promise<TResult>,
): {
  state: ActionState<TResult>;
  run: (input: TInput) => Promise<TResult | null>;
  reset: () => void;
} {
  const [state, setState] = useState<ActionState<TResult>>({
    running: false,
    result: null,
    error: null,
  });
  const actionRef = useRef(action);
  actionRef.current = action;
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const run = useCallback(async (input: TInput): Promise<TResult | null> => {
    setState({ running: true, result: null, error: null });
    try {
      const result = await actionRef.current(input);
      if (mounted.current) {
        setState({ running: false, result, error: null });
      }
      return result;
    } catch (error) {
      if (mounted.current) {
        setState({ running: false, result: null, error });
      }
      return null;
    }
  }, []);

  const reset = useCallback(() => {
    setState({ running: false, result: null, error: null });
  }, []);

  return { state, run, reset };
}
