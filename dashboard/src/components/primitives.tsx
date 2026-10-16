/**
 * Small presentational primitives shared by every page.
 *
 * None of these know anything about MemForge: they take values and render them.
 * Every score, count and identifier passed in comes from an API response.
 */

import { type ReactNode, useEffect, useRef, useState } from "react";
import { num } from "../lib/format";

export type Tone = "ok" | "warn" | "bad" | "accent" | "neutral";

export function Card({
  title,
  subtitle,
  actions,
  children,
  footer,
  tight,
  id,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  tight?: boolean;
  id?: string;
}) {
  return (
    <section className="card" id={id}>
      {(title !== undefined || actions !== undefined) && (
        <header className="card__head">
          <div>
            {title !== undefined && <h2 className="card__title">{title}</h2>}
            {subtitle !== undefined && <p className="card__subtitle">{subtitle}</p>}
          </div>
          {actions !== undefined && <div className="btn-row">{actions}</div>}
        </header>
      )}
      <div className={tight ? "card__body card__body--tight" : "card__body"}>{children}</div>
      {footer !== undefined && <footer className="card__footer">{footer}</footer>}
    </section>
  );
}

export function Badge({
  children,
  tone = "neutral",
  mono,
  title,
}: {
  children: ReactNode;
  tone?: Tone;
  mono?: boolean;
  title?: string;
}) {
  const suffix = tone === "neutral" ? "" : ` badge--${tone}`;
  return (
    <span className={`badge${suffix}${mono ? " badge--mono" : ""}`} title={title}>
      {children}
    </span>
  );
}

/** A 0..1 value rendered as a bar plus its figure, straight from the API. */
export function ScoreBar({
  value,
  tone,
  label,
  digits = 3,
}: {
  value: number | null | undefined;
  tone?: Tone;
  label?: string;
  digits?: number;
}) {
  const clamped = value === null || value === undefined ? 0 : Math.max(0, Math.min(1, value));
  const dataTone = tone === undefined || tone === "neutral" ? undefined : tone;
  return (
    <div className="bar">
      <div className="bar__track" role="img" aria-label={label ?? `score ${num(value, digits)}`}>
        <div
          className="bar__fill"
          data-tone={dataTone}
          style={{ width: `${clamped * 100}%` }}
        />
      </div>
      <span className="bar__value">{num(value, digits)}</span>
    </div>
  );
}

/** A read-only panel for code, JSON or rendered context. */
export function CodePanel({
  label,
  actions,
  children,
  tall,
}: {
  label: string;
  actions?: ReactNode;
  children: ReactNode;
  tall?: boolean;
}) {
  return (
    <div className={tall ? "code-panel code-panel--tall" : "code-panel"}>
      <div className="code-panel__head">
        <span>{label}</span>
        {actions}
      </div>
      <pre>{children as string}</pre>
    </div>
  );
}

/** A description list that survives narrow screens. */
export function KeyValues({
  items,
}: {
  items: ReadonlyArray<readonly [ReactNode, ReactNode]>;
}) {
  return (
    <dl className="kv">
      {items.map(([key, value], index) => (
        <div key={typeof key === "string" ? key : index} style={{ display: "contents" }}>
          <dt>{key}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="row" style={{ gap: 8 }}>
      <span className="spinner" aria-hidden="true" />
      {label !== undefined && <span className="muted small">{label}</span>}
    </span>
  );
}

export function EmptyState({ title, detail }: { title: string; detail?: ReactNode }) {
  return (
    <div className="state">
      <span className="state__title">{title}</span>
      {detail !== undefined && <span className="state__detail">{detail}</span>}
    </div>
  );
}

/** A short monospace identifier, safe to break in the middle. */
export function Mono({ children }: { children: string }) {
  return <span className="mono">{children}</span>;
}

/**
 * A labelled number input.
 *
 * The typed text is held locally and the parsed value is what the parent sees,
 * so clearing a field and typing a new number replaces it instead of appending
 * to the clamped value. Bounds are enforced on every keystroke, and the raw
 * text is what the user sees.
 */
export function NumberField({
  id,
  label,
  value,
  onChange,
  min,
  max,
  step,
  placeholder,
  hint,
}: {
  id: string;
  label: string;
  value: number | null;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  hint?: string;
}) {
  const [raw, setRaw] = useState(value === null ? "" : String(value));
  const emitted = useRef(value);
  useEffect(() => {
    // Only mirror the prop back when it changed *outside* this input, so
    // clearing the field and typing a new value replaces it instead of
    // snapping back to the clamped minimum.
    if (value !== emitted.current) {
      setRaw(value === null ? "" : String(value));
      emitted.current = value;
    }
  }, [value]);

  return (
    <div className="field">
      <label className="field__label" htmlFor={id}>
        {label}
      </label>
      <input
        id={id}
        className="input"
        type="number"
        min={min}
        max={max}
        step={step}
        placeholder={placeholder}
        value={raw}
        onChange={(event) => {
          const text = event.target.value;
          setRaw(text);
          if (text.trim() === "") {
            emitted.current = min ?? 0;
            onChange(min ?? 0);
            return;
          }
          const parsed = Number(text);
          if (!Number.isFinite(parsed)) {
            return;
          }
          let next = parsed;
          if (min !== undefined) {
            next = Math.max(min, next);
          }
          if (max !== undefined) {
            next = Math.min(max, next);
          }
          emitted.current = next;
          onChange(next);
        }}
      />
      {hint !== undefined && <span className="field__hint">{hint}</span>}
    </div>
  );
}

/** An optional number input: an empty field means "no value". */
export function OptionalNumberField({
  id,
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.05,
  placeholder,
}: {
  id: string;
  label: string;
  value: number | null;
  onChange: (value: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
}) {
  const [raw, setRaw] = useState(value === null ? "" : String(value));
  const emitted = useRef(value);
  useEffect(() => {
    if (value !== emitted.current) {
      setRaw(value === null ? "" : String(value));
      emitted.current = value;
    }
  }, [value]);

  return (
    <div className="field">
      <label className="field__label" htmlFor={id}>
        {label}
      </label>
      <input
        id={id}
        className="input"
        type="number"
        min={min}
        max={max}
        step={step}
        placeholder={placeholder}
        value={raw}
        onChange={(event) => {
          const text = event.target.value;
          setRaw(text);
          if (text.trim() === "") {
            emitted.current = null;
            onChange(null);
            return;
          }
          const parsed = Number(text);
          const next = Number.isFinite(parsed) ? Math.min(max, Math.max(min, parsed)) : null;
          emitted.current = next;
          onChange(next);
        }}
      />
    </div>
  );
}
