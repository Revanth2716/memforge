/**
 * Response-shape validation.
 *
 * The backend contract lives in `app/api/schemas.py`. The dashboard trusts it,
 * but it does not *assume* it: every response is checked against a declared
 * shape before it reaches a component, so a malformed or drifted payload becomes
 * an explicit "malformed API response" state instead of a blank screen or a
 * silently invented value.
 *
 * The validator is deliberately tiny (no dependency) and reports the exact JSON
 * path of the first violation.
 */

/** A declared shape for one piece of a response. */
export type Shape =
  | "string"
  | "number"
  | "boolean"
  | "unknown"
  | { enum: readonly string[] }
  | { array: Shape }
  | { record: Shape }
  | { object: Record<string, Shape> }
  | { nullable: Shape }
  | { optional: Shape };

/** A response that parsed as JSON but does not match the documented contract. */
export class MalformedResponseError extends Error {
  readonly label: string;
  readonly path: string;

  constructor(label: string, path: string, expectation: string) {
    super(
      `${label}: malformed API response at ${path || "<root>"} (expected ${expectation}). ` +
        "The backend returned JSON that does not match its documented contract.",
    );
    this.name = "MalformedResponseError";
    this.label = label;
    this.path = path;
  }
}

function child(path: string, key: string | number): string {
  if (typeof key === "number") {
    return `${path}[${key}]`;
  }
  return path === "" ? key : `${path}.${key}`;
}

function describe(shape: Shape): string {
  if (typeof shape === "string") {
    return shape;
  }
  if ("enum" in shape) {
    return `one of ${shape.enum.map((value) => JSON.stringify(value)).join(", ")}`;
  }
  if ("array" in shape) {
    return `array of ${describe(shape.array)}`;
  }
  if ("record" in shape) {
    return `object of ${describe(shape.record)}`;
  }
  if ("object" in shape) {
    return `object with keys ${Object.keys(shape.object).join(", ")}`;
  }
  if ("nullable" in shape) {
    return `${describe(shape.nullable)} or null`;
  }
  return `${describe(shape.optional)} or absent`;
}

function visit(value: unknown, shape: Shape, path: string, label: string): void {
  if (typeof shape !== "string") {
    if ("enum" in shape) {
      if (typeof value !== "string" || !shape.enum.includes(value)) {
        throw new MalformedResponseError(label, path, describe(shape));
      }
      return;
    }
    if ("array" in shape) {
      if (!Array.isArray(value)) {
        throw new MalformedResponseError(label, path, describe(shape));
      }
      value.forEach((item, index) => visit(item, shape.array, child(path, index), label));
      return;
    }
    if ("record" in shape) {
      if (value === null || typeof value !== "object" || Array.isArray(value)) {
        throw new MalformedResponseError(label, path, describe(shape));
      }
      for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
        visit(item, shape.record, child(path, key), label);
      }
      return;
    }
    if ("object" in shape) {
      if (value === null || typeof value !== "object" || Array.isArray(value)) {
        throw new MalformedResponseError(label, path, describe(shape));
      }
      const source = value as Record<string, unknown>;
      for (const [key, fieldShape] of Object.entries(shape.object)) {
        const optional = typeof fieldShape !== "string" && "optional" in fieldShape;
        const inner = optional && typeof fieldShape !== "string" && "optional" in fieldShape
          ? (fieldShape as { optional: Shape }).optional
          : fieldShape;
        if (!(key in source) || source[key] === undefined) {
          if (optional) {
            continue;
          }
          throw new MalformedResponseError(label, child(path, key), describe(inner));
        }
        visit(source[key], inner, child(path, key), label);
      }
      return;
    }
    if ("nullable" in shape) {
      if (value === null) {
        return;
      }
      visit(value, shape.nullable, path, label);
      return;
    }
    // { optional } at a non-object position: treat as a plain optional field.
    if (value !== undefined) {
      visit(value, shape.optional, path, label);
    }
    return;
  }

  switch (shape) {
    case "unknown":
      return;
    case "string":
      if (typeof value !== "string") {
        throw new MalformedResponseError(label, path, "string");
      }
      return;
    case "number":
      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new MalformedResponseError(label, path, "number");
      }
      return;
    case "boolean":
      if (typeof value !== "boolean") {
        throw new MalformedResponseError(label, path, "boolean");
      }
      return;
  }
}

/**
 * Validate `raw` against `shape` and return it typed as `T`.
 *
 * @throws {MalformedResponseError} when the payload does not match.
 */
export function parseResponse<T>(raw: unknown, shape: Shape, label: string): T {
  visit(raw, shape, "", label);
  return raw as T;
}

/** Convenience: `string | null`. */
export const nullableString: Shape = { nullable: "string" };
/** Convenience: `number | null`. */
export const nullableNumber: Shape = { nullable: "number" };
/** Convenience: `Record<string, number>`. */
export const numberRecord: Shape = { record: "number" };
/** Convenience: a possibly-absent field. */
export function optional(shape: Shape): Shape {
  return { optional: shape };
}
