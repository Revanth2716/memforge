/**
 * Evaluation.
 *
 * `GET /api/v1/evaluations/latest` — a real, measured run of the shipped
 * dataset against the real services. The page shows the backend's own numbers
 * and nothing else: no score is rounded up, no failure is hidden, and the
 * precision figure is printed exactly as the evaluation computed it.
 */

import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Evaluation } from "../api/types";
import { useResource } from "../hooks/useResource";
import { Badge, Card, CodePanel, KeyValues, ScoreBar } from "../components/primitives";
import { ResourceView } from "../components/StateBlock";
import { integer, num } from "../lib/format";

function scoreTone(score: number): "ok" | "warn" | "bad" {
  if (score >= 0.95) {
    return "ok";
  }
  if (score >= 0.7) {
    return "warn";
  }
  return "bad";
}

function metricTone(name: string, score: number): "ok" | "warn" | "bad" {
  // Precision is a graded quality signal, not a pass/fail rule: the shipped
  // dataset deliberately leaves it below 1.0 (see M6 design note 6). Nothing
  // here "corrects" it.
  if (name === "retrieval_precision") {
    return score >= 0.75 ? "ok" : score >= 0.5 ? "warn" : "bad";
  }
  return scoreTone(score);
}

export function EvaluationPage() {
  const { state, reload } = useResource<Evaluation>(
    "evaluation-latest",
    () => api.getLatestEvaluation(),
    { ttlMs: 60_000 },
  );

  return (
    <>
      <div className="page-intro">
        <h2>Deterministic MemForge evaluation</h2>
        <p>
          The dataset is executed against the real services and counted. No judge model, no reference
          answers, no network. Every score below is a ratio of two integers the runner produced.
        </p>
      </div>

      <ResourceView state={state} onRetry={reload}>
        {(data: Evaluation) => (
          <>
            <div className="grid grid--stats">
              <div className="stat">
                <span className="stat__label">Dataset</span>
                <span className="stat__value stat__value--text">
                  <Badge tone="accent" mono>
                    {data.reference}
                  </Badge>
                </span>
                <span className="stat__hint">
                  version {data.version} · threshold {num(data.threshold, 2)}
                </span>
              </div>
              <div className="stat">
                <span className="stat__label">Cases</span>
                <span className="stat__value">{integer(data.total_cases)}</span>
                <span className="stat__hint">isolated corpus + clock + call per case</span>
              </div>
              <div className="stat">
                <span className="stat__label">Passed</span>
                <span className="stat__value">{integer(data.passed_cases)}</span>
                <span className="stat__hint">
                  case score {num(data.case_score)}
                </span>
              </div>
              <div className="stat">
                <span className="stat__label">Failed</span>
                <span className="stat__value">{integer(data.failed_cases)}</span>
                <span className="stat__hint">
                  {data.failed_cases === 0 ? "no failing case in this run" : "listed in full below"}
                </span>
              </div>
              <div className="stat">
                <span className="stat__label">Overall score</span>
                <span className="stat__value">{num(data.overall)}</span>
                <span className="stat__hint">metric score {num(data.metric_score)}</span>
              </div>
              <div className="stat">
                <span className="stat__label">Gate</span>
                <span className="stat__value stat__value--text">
                  <Badge tone={data.passed ? "ok" : "bad"}>{data.passed ? "passed" : "failed"}</Badge>
                </span>
                <span className="stat__hint">
                  the threshold gates the case pass rate, not the blended score
                </span>
              </div>
            </div>

            <Card
              title="Categories"
              subtitle="Pass/total per category of the dataset"
            >
              <div className="row" style={{ gap: 10 }}>
                {Object.entries(data.categories).map(([name, counts]) => (
                  <div className="stat" key={name} style={{ minWidth: 150 }}>
                    <span className="stat__label">{name}</span>
                    <span className="stat__value">
                      {integer(counts.passed)}
                      <span className="muted" style={{ fontSize: "var(--step-1)" }}>
                        /{integer(counts.total)}
                      </span>
                    </span>
                    <ScoreBar
                      value={counts.total === 0 ? null : counts.passed / counts.total}
                      tone={counts.passed === counts.total ? "ok" : "warn"}
                      label={`${name} pass rate`}
                    />
                  </div>
                ))}
              </div>
            </Card>

            <Card
              title="Metrics"
              subtitle="Each metric carries the exact formula the runner applied"
            >
              <div>
                {data.metrics.map((metric) => (
                  <div className="metric-row" key={metric.name}>
                    <div>
                      <div className="metric-row__name">{metric.name}</div>
                      <div className="metric-row__definition">{metric.definition}</div>
                    </div>
                    <div>
                      <div className="metric-row__figure">
                        <ScoreBar
                          value={metric.score}
                          tone={metricTone(metric.name, metric.score)}
                          label={`${metric.name} ${metric.score}`}
                        />
                        <span className="metric-row__counts">
                          {integer(metric.numerator)}/{integer(metric.denominator)}
                        </span>
                      </div>
                      {metric.failures.length > 0 && (
                        <ul className="stack stack--tight" style={{ marginTop: 6 }}>
                          {metric.failures.map((failure) => (
                            <li key={failure} className="small mono" style={{ color: "var(--bad)" }}>
                              {failure}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            <Card
              title={`Failing cases (${integer(data.cases.length)})`}
              subtitle="Every case the run did not pass, with the checks that did not hold"
            >
              {data.cases.length === 0 ? (
                <p className="muted small">
                  No failing case in this run. The list would appear here, in full, if a later
                  evaluation failed one.
                </p>
              ) : (
                <div className="stack stack--tight">
                  {data.cases.map((caseResult) => (
                    <div className="factor" key={caseResult.case_id}>
                      <span className="factor__name">{caseResult.case_id}</span>
                      <span className="mono">{caseResult.category}</span>
                      <ul className="stack stack--tight" style={{ gridColumn: "1 / -1" }}>
                        {caseResult.failures.map((failure) => (
                          <li key={failure} className="small mono" style={{ color: "var(--bad)" }}>
                            {failure}
                          </li>
                        ))}
                      </ul>
                      {Object.keys(caseResult.numerators).length > 0 && (
                        <span className="factor__meta" style={{ gridColumn: "1 / -1" }}>
                          {Object.entries(caseResult.numerators)
                            .map(
                              ([key, value]) =>
                                `${key} ${value}/${caseResult.denominators[key] ?? "?"}`,
                            )
                            .join(" · ")}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </Card>

            <div className="grid grid--split">
              <Card title="What this proves" subtitle="And what it does not">
                <KeyValues
                  items={[
                    ["Dataset", `${data.dataset}@${data.version}`],
                    ["Cases run", integer(data.total_cases)],
                    ["Case pass rate", num(data.case_score)],
                    ["Metric score", num(data.metric_score)],
                    ["Overall", num(data.overall)],
                    ["Threshold", num(data.threshold, 2)],
                    ["Gate result", data.passed ? "passed" : "failed"],
                  ]}
                />
              </Card>
              <Card title="Limitations" subtitle="Stated on every run of the real evaluation">
                <div className="note">
                  <strong>This evaluation measures the supplied fixtures.</strong> It is a
                  deterministic regression gate over the cases the dataset encodes, not proof of
                  universal AI memory quality. A score here says the system made the decisions its
                  own documented rules say it should make, for those corpora and those queries —
                  nothing about unseen data, other languages, or a different domain.
                </div>
              </Card>
            </div>

            <Card title="Re-check from the command line" subtitle="The same run, outside the browser">
              <CodePanel label="regression gate">
                {`python -m app.evaluation.run          # ships ${data.reference}
python -m app.evaluation.run --verbose   # per-case detail
python scripts/smoke_test.py             # live HTTP, incl. this endpoint`}
              </CodePanel>
              <p className="small faint" style={{ marginTop: 10 }}>
                The evaluation endpoint is read-only on purpose: re-running is the CLI's job, and a
                browser cannot trigger a benchmark on demand.{" "}
                <Link to="/metrics">Operational counters</Link> count what this process actually did.
              </p>
            </Card>
          </>
        )}
      </ResourceView>
    </>
  );
}
