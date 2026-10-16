/** Offset pagination for the memory explorer, driven by the API's own totals. */

import { integer } from "../lib/format";

export function Pagination({
  total,
  limit,
  offset,
  onChange,
}: {
  total: number;
  limit: number;
  offset: number;
  onChange: (offset: number) => void;
}) {
  const from = total === 0 ? 0 : offset + 1;
  const to = Math.min(offset + limit, total);
  const previousDisabled = offset <= 0;
  const nextDisabled = offset + limit >= total;
  return (
    <div className="row row--between">
      <span className="pager">
        <strong>
          {integer(from)}–{integer(to)}
        </strong>{" "}
        of <strong>{integer(total)}</strong> memories
      </span>
      <div className="btn-row">
        <button
          type="button"
          className="btn"
          onClick={() => onChange(Math.max(0, offset - limit))}
          disabled={previousDisabled}
        >
          ← Previous
        </button>
        <button
          type="button"
          className="btn"
          onClick={() => onChange(offset + limit)}
          disabled={nextDisabled}
        >
          Next →
        </button>
      </div>
    </div>
  );
}
