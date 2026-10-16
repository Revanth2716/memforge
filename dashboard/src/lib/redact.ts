/**
 * Defensive redaction for anything the dashboard renders from a trace.
 *
 * The backend already records no credential, prompt or answer
 * (`app/observability/service.py` drops those keys before persisting). This
 * module is the second line of defence, and it is deliberately *not* the only
 * one: the dashboard never trusts a payload to be safe just because it is
 * supposed to be.
 *
 * Nothing here rewrites a value to make it "presentable" — a key that cannot be
 * shown safely is dropped, and a value that looks like a credential is replaced
 * by a placeholder rather than masked into something that still leaks its shape.
 */

/** Keys that are never rendered, whatever their value. */
const DENIED_KEYS: readonly string[] = [
  "api_key",
  "apikey",
  "api-key",
  "authorization",
  "auth",
  "token",
  "access_token",
  "refresh_token",
  "secret",
  "client_secret",
  "password",
  "passwd",
  "credential",
  "credentials",
  "bearer",
  "session",
  "cookie",
  "prompt",
  "system_prompt",
  "user_prompt",
  "answer",
  "content",
  "message",
  "messages",
  "chain_of_thought",
  "reasoning",
  "completion",
];

/** Substrings that make a *value* unsafe to render even under a benign key. */
const DENIED_VALUE_MARKERS: readonly string[] = [
  "authorization:",
  "bearer ",
  "api_key=",
  "api-key=",
  "sk-",
  "ghp_",
  "xoxb-",
  "-----begin",
];

export const REDACTED = "[redacted]";

function normaliseKey(key: string): string {
  return key.toLowerCase().replace(/[\s-]+/g, "_");
}

/** True when a key must not be rendered at all. */
export function isDeniedKey(key: string): boolean {
  const normalised = normaliseKey(key);
  return DENIED_KEYS.some((denied) => normalised === denied || normalised.endsWith(`_${denied}`));
}

/** True when a value looks like a credential. */
export function looksLikeSecret(value: string): boolean {
  const lowered = value.toLowerCase();
  return DENIED_VALUE_MARKERS.some((marker) => lowered.includes(marker));
}

/** Render a value for display, or `null` when it must not be shown. */
export function safeValue(value: unknown): string | null {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "string") {
    return looksLikeSecret(value) ? REDACTED : value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    // Ids, counts and sizes are safe; anything string-like that looks like a
    // credential is dropped from the list rather than redacted in place.
    const parts = value.map((item) => safeValue(item)).filter((part): part is string => part !== null);
    return parts.join(", ");
  }
  if (typeof value === "object") {
    return JSON.stringify(safeMapping(value as Record<string, unknown>));
  }
  return String(value);
}

/** Copy a mapping with every unsafe key removed. */
export function safeMapping(source: Record<string, unknown>): Record<string, unknown> {
  const output: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(source)) {
    if (isDeniedKey(key)) {
      continue;
    }
    output[key] = value;
  }
  return output;
}

/** `[key, renderedValue]` pairs ready for a definition list, safe ones only. */
export function safeEntries(
  source: Record<string, unknown> | undefined | null,
): [string, string][] {
  if (!source) {
    return [];
  }
  const output: [string, string][] = [];
  for (const [key, value] of Object.entries(source)) {
    if (isDeniedKey(key)) {
      continue;
    }
    const rendered = safeValue(value);
    if (rendered === null) {
      continue;
    }
    output.push([key, rendered]);
  }
  return output;
}
