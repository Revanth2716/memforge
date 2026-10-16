/**
 * The hard-budget indicator for M4's compiler output.
 *
 * Every figure shown here is the backend's own `token_budget`,
 * `estimated_tokens` and `remaining_tokens` from `POST /context/compile`. The
 * dashboard deliberately does **not** count tokens itself: a second estimator
 * would be able to disagree with the one the guarantee is measured against.
 */

import { integer, num } from "../lib/format";
import { Badge } from "./primitives";

export function TokenBudget({
  budget,
  estimated,
  remaining,
  withinBudget,
  utilization,
  estimator,
}: {
  budget: number;
  estimated: number;
  remaining: number;
  withinBudget: boolean;
  utilization: number;
  estimator?: string;
}) {
  const pct = budget > 0 ? Math.min(100, Math.max(0, (estimated / budget) * 100)) : 0;
  return (
    <div className="budget">
      <div className="budget__headline">
        <span className="budget__figure">{integer(estimated)}</span>
        <span className="budget__of">/ {integer(budget)} estimated tokens</span>
        <Badge tone={withinBudget ? "ok" : "bad"}>
          {withinBudget ? "within budget" : "over budget"}
        </Badge>
        <span className="budget__guarantee">
          hard guarantee: estimated tokens ≤ token budget
          {estimator !== undefined && <> · estimator <code>{estimator}</code></>}
        </span>
      </div>
      <div
        className="budget__track"
        role="img"
        aria-label={`${estimated} of ${budget} estimated tokens used`}
      >
        <div className="budget__fill" data-within={withinBudget} style={{ width: `${pct}%` }} />
      </div>
      <div className="budget__legend">
        <span>
          used <b>{integer(estimated)}</b>
        </span>
        <span>
          budget <b>{integer(budget)}</b>
        </span>
        <span>
          remaining <b>{integer(remaining)}</b>
        </span>
        <span>
          utilisation <b>{num(utilization, 3)}</b>
        </span>
      </div>
    </div>
  );
}
