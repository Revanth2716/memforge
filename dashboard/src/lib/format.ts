/**
 * Display formatting.
 *
 * These are *formatting* helpers only. Nothing here recomputes a backend
 * value: a score, a token figure or a count always comes straight from the API
 * response and is only rounded for display.
 */

/** A fixed-width number, so columns of figures line up. */
export function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(digits);
}

/** A percentage, from a 0..1 ratio reported by the API. */
export function percent(ratio: number | null | undefined, digits = 1): string {
  if (ratio === null || ratio === undefined || !Number.isFinite(ratio)) {
    return "—";
  }
  return `${(ratio * 100).toFixed(digits)}%`;
}

/** An integer with thousands separators. */
export function integer(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  return Math.round(value).toLocaleString("en-US");
}

/** Milliseconds, as reported by the backend (never derived here). */
export function duration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) {
    return "—";
  }
  if (ms < 1) {
    return `${ms.toFixed(2)} ms`;
  }
  if (ms < 1000) {
    return `${ms.toFixed(1)} ms`;
  }
  return `${(ms / 1000).toFixed(2)} s`;
}

/** An ISO-8601 timestamp as a short, local, human-readable string. */
export function timestamp(iso: string | null | undefined): string {
  if (!iso) {
    return "—";
  }
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) {
    return String(iso);
  }
  return parsed.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/** A day/hour/age string derived from the API's own `*_days` / timestamp. */
export function age(days: number | null | undefined): string {
  if (days === null || days === undefined || !Number.isFinite(days)) {
    return "—";
  }
  if (days < 1) {
    const hours = days * 24;
    return `${hours.toFixed(1)} h`;
  }
  if (days < 30) {
    return `${days.toFixed(1)} d`;
  }
  return `${(days / 30).toFixed(1)} mo`;
}

/** "just now" / "3 min ago" from an ISO timestamp. */
export function relativeTime(iso: string | null | undefined): string {
  if (!iso) {
    return "—";
  }
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) {
    return String(iso);
  }
  const deltaMs = Date.now() - parsed.getTime();
  const seconds = Math.round(deltaMs / 1000);
  if (seconds < 45) {
    return "just now";
  }
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) {
    return `${minutes} min ago`;
  }
  const hours = Math.round(minutes / 60);
  if (hours < 24) {
    return `${hours} h ago`;
  }
  return `${Math.round(hours / 24)} d ago`;
}

/** Truncate in the middle, so both the start and the end stay recognisable. */
export function truncateMiddle(text: string, max = 48): string {
  if (text.length <= max) {
    return text;
  }
  const half = Math.floor((max - 1) / 2);
  return `${text.slice(0, half)}…${text.slice(text.length - half)}`;
}

/** The first line only, for one-line table cells. */
export function firstLine(text: string): string {
  const line = text.split("\n").find((candidate) => candidate.trim().length > 0);
  return line ?? "";
}

/**
 * Sentence case for an UPPER_SNAKE or dotted identifier:
 * `near_duplicate` -> `Near duplicate`, `llm.generate` -> `Llm generate`.
 */
export function humanize(value: string): string {
  const spaced = value.replace(/[._-]+/g, " ").trim().toLowerCase();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** A `?a=b` query string, or "" when there is nothing to add. */
export function queryString(params: Record<string, string | number | undefined | null>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && String(value).length > 0) {
      search.set(key, String(value));
    }
  }
  const rendered = search.toString();
  return rendered.length > 0 ? `?${rendered}` : "";
}
