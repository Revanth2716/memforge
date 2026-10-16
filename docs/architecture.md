# MemForge architecture notes (M1 → M7)

This document explains *why* the code is shaped the way it is. The README covers
usage; this one covers decisions.

## Layering

```
app/api        transport (HTTP, status codes, Pydantic DTOs, error envelope)
app/services   use cases: lifecycle, intelligence, retrieval, context, agent
app/mcp        MCP transport: the same services, different protocol
app/evaluation M6: dataset + strict parsing, metric aggregation, the case
               runner (which drives real services), the report, the CLI gate
app/observability M6: the trace lifecycle service, the decorators that emit
               events into it, the persistence decorator, the demo
app/storage    persistence behind ABCs (MemoryRepository, EmbeddingStore,
               TraceStore) + SQLite implementations
app/domain     entities, value objects, enums, errors, ports, algorithms
               (similarity, attributes, intelligence, retrieval, tokens, context,
                llm, agent, observability, evaluation)
```

`app/evaluation` and `app/observability` sit **beside** the services, not inside
them: both are driven by real service calls, and the M1-M5 code is unaware that
either exists. The evaluation imports no transport and the observability layer
imports no evaluation -- they are joined only by describing the same system.

Dependencies point **inwards**: the domain knows nothing about SQLite, FastAPI or
MCP, storage knows the domain but not the transports, and the transports know the
services. Practical consequences:

- The memory lifecycle and every M2 algorithm can be tested with zero HTTP.
- Swapping the storage backend (or adding an Ollama-backed provider later) is a
  new `MemoryRepository` implementation; nothing above it changes.
- The REST API can be re-pointed at another transport because the services
  return domain objects, not HTTP constructs — which is exactly what M5 did
  with MCP.
- The retrieval layer never imports a web framework and never knows which
  embedding provider is configured: it talks to an ``EmbeddingProvider``
  protocol and an ``EmbeddingStore`` ABC, so swapping SQLite for another local
  store — or adding a different embedding backend — is a single new
  implementation.
- **The context compiler has no storage of its own.** `ContextCompilerService`
  depends on `MemoryRetrievalService`, never on `MemoryRepository`. Everything
  M4 decides must be derivable from what M3 returned; if a decision needs a
  lookup, that lookup belongs to M3 (which is why the compiler re-scores M3's
  suppressed candidates through M3's public `explain()`, rather than reading the
  database itself).
- **The MCP layer holds no logic at all.** `app/mcp/server.py` binds typed
  functions to `MemoryOperationsService`. A tool that could answer a question
  without M1-M4 would be a bug, not a feature.

## Key design decisions

### 1. Immutable domain entities, explicit updates

`Memory` is a frozen dataclass that validates its own invariants
(`__post_init__`): non-empty content, scores in `[0, 1]`, `version >= 1`,
`updated_at >= created_at`, `access_count >= 0`, `last_accessed_at >= created_at`.
Updates therefore *construct a new value* rather than mutating in place, which is
what makes versioning hard to get wrong.

`MemoryUpdate` carries an `UNSET` sentinel so a patch can distinguish
"field absent" from "field explicitly null" (only `source` is nullable). This is
the difference between `PATCH {}` (no-op, rejected) and `PATCH {"source": null}`
(clear provenance).

### 2. Append-only history

`memories.version` is the current version number; every write appends a row to
`memory_versions` containing a **complete snapshot** plus the reason
(`CREATED` / `UPDATED` / `FORGOTTEN`) and `recorded_at`.

Snapshots (rather than diffs) were chosen deliberately:

- History queries need no replay logic;
- A snapshot is self-describing and cheap to serialise;
- Corruption of one version cannot cascade.

The repository rejects non-consecutive version numbers (`RepositoryError`), so the
chain cannot silently fork.

M2 keeps this guarantee: importance/confidence changes are ordinary updates and
therefore versioned; access telemetry is *not* versioned (documented below);
conflict resolution archives the loser, which appends a version.

### 3. Soft delete

"Forgetting" sets `status = DELETED` and appends a `FORGOTTEN` version. Rows are
never physically deleted, so `GET /{id}/history` keeps working after a memory has
been forgotten — the whole point of keeping history. The trade-off (storage
growth, no pruning) is listed in the README limitations.

### 4. Injected ports for time and identity

`Clock` and `IdGenerator` are protocols with two implementations each
(`SystemClock` / `ManualClock`, `UuidIdGenerator` / `SequentialIdGenerator`).
The CLI demo uses the deterministic pair, which is why its output is reproducible
byte-for-byte (`tests/test_cli_demo.py` asserts exactly that).

### 5. M2: everything is a pure function of its inputs

The M2 algorithms (`app/domain/similarity.py`, `app/domain/attributes.py`,
`app/domain/intelligence.py`) take the memory, an `IntelligenceConfig` and a
reference `datetime`. None of them:

- reads the wall clock,
- touches the repository,
- calls a model or the network.

That is what makes importance, decay, duplicate detection, conflict detection and
conflict resolution unit-testable without fixtures or databases, and it is why the
same memory can be scored at several reference times in one test.

### 6. M2: importance, confidence and decay are different things

| Concept | Stored | Computed | Meaning |
| --- | --- | --- | --- |
| `confidence` | yes | no | How accurate the memory is (caller-supplied). |
| `importance` (field) | yes | no | How much the caller says it matters. |
| importance **score** | no | yes | Weighted blend of type, explicit importance, recurrence, recency and access. |
| effective **priority** | no | yes | `(0.7 * importance_score + 0.3 * confidence) * decay`. |

Reporting all four separately in `GET /memories/{id}/intelligence` is deliberate:
conflating accuracy with relevance is the classic mistake in memory systems.

### 7. M2: duplicate detection is lexical by design

Without embeddings, similarity has to be transparent. The pipeline
(tokenise → drop 1-char tokens → stop words → suffix stripping → Dice) is
inspectable end to end, and `similarity_details()` returns the token sets so a
score can always be justified.

Two deliberate rules keep the feature honest:

- **A conflicting pair is never a duplicate.** If two memories state the same
  attribute with different values, that is a conflict, and reporting it as a
  duplicate would hide the disagreement.
- **Only ACTIVE memories are compared**, so archived/forgotten facts do not
  generate noise.

### 8. M2: conflict detection uses an explicit grammar

Rather than pretending to understand prose, `app/domain/attributes.py` recognises
three constructions (`=`, `:`, copula `is`), normalises both sides with the same
lexical pipeline, and compares values with a documented similarity threshold. The
false-positive/false-negative profile of that grammar is stated in the README.

### 9. M2: resolution is a documented total order

The policy compares explicit importance → confidence → recency → effective
priority → identifier. Every factor is reported in the response, and the final
identifier rule guarantees that identical inputs always resolve the same way,
regardless of iteration order.

Resolution never deletes: it records `SUPERSEDES`, and by default archives the
loser (a versioned, visible status change).

### 10. SQLite usage

- `PRAGMA foreign_keys = ON` — version rows and relationship rows cannot outlive
  their memories.
- One connection per operation, so the repository instance is safe to share
  between the API, CLI and future background jobs.
- `transaction()` wraps each write in `BEGIN IMMEDIATE`, so a memory row and its
  history row are always consistent.
- Timestamps are stored as fixed-width ISO-8601 UTC strings
  (`isoformat(timespec="microseconds")`); the fixed width keeps
  `ORDER BY created_at` correct lexicographically. Ties fall back to `id DESC`,
  so pagination order is stable.
- **Parameterised SQL only.** Dynamic parts of a statement are static SQL
  fragments (`WHERE` / `ORDER BY`); all values are bound. A test stores
  `'; DROP TABLE memories; --` and verifies the table survives.
- Migrations are an ordered tuple of id + SQL statements, applied once each and
  recorded in `_memforge_migrations`:

  | id | contents |
  | --- | --- |
  | `0001_initial_memory_core` | `memories`, `memory_versions`, three indexes. |
  | `0002_memory_intelligence` | `access_count` + `last_accessed_at` columns, `memory_relationships` table, two indexes. |

  `initialize()` is idempotent and cheap enough to call on every repository
  construction, which is what makes a freshly constructed repository immediately
  usable (and upgrades an M1 database in place).

### 11. API surface

- Strict Pydantic models (`extra="forbid"`, `str_strip_whitespace=True`) for every
  request and response body.
- One error envelope: `{"error": {"code", "message", "details"}}`, produced by a
  single `register_exception_handlers(app)` call. Domain errors are translated to
  HTTP status codes in one place (`app/api/errors.py`), never in the domain.
- `create_app(settings)` is a factory: tests build the app against a temp
  database, production uses `app.main:app`. Intelligence parameters come from
  `Settings`, so the same knobs are available through `MEMFORGE_*` env vars.
- One endpoint per milestone use case. M4 added exactly one
  (`POST /api/v1/context/compile`) and deliberately no inspection endpoint: the
  compile response already carries the rendered context, the blocks, the trace
  and the explanation, so a second "show me the same thing" route would only
  duplicate the contract.

## Test strategy

| Layer | What is proved |
| --- | --- |
| Domain | Invariants, enum stability, update semantics, stemming/Dice values, attribute grammar, scoring formulas, decay, resolution tie-breaks. |
| Storage | Round-trips, ordering, pagination, constraints, injection safety, telemetry, relationship uniqueness/cascade, error translation. |
| Service | Versioning rules, state machine, idempotent forget, history survival, all M2 operations, M1 compatibility. |
| API | Every endpoint, every documented error code, envelope shape, OpenAPI paths. |
| Lifecycle | Data outliving repositories, apps and restarts; M1 → M2 schema upgrade. |
| Demo | Deterministic, byte-identical output across runs. |
| Live HTTP | `scripts/smoke_test.py` drives a real uvicorn server end to end (see below). |
| MCP | `scripts/mcp_selftest.py` drives the real MCP server over its seven tools (see below). |

Coverage of `app/` is tracked with `pytest-cov`
(`python -m pytest --cov=app --cov-report=term-missing`).

The smoke test is deliberately **not** a pytest case. It starts a real uvicorn
process, talks to it with `httpx` over a socket and fails with a non-zero exit
code, so it can be run in a deployment check without the test suite. The MCP
self-test is the same idea for a different transport, and both stay out of the
suite for the same reason: a deployment check should not need a test runner.
\n

## M3 design decisions

### 1. Retrieval is memory-aware, not document RAG

The ranking formula mixes query relevance with the M2 intelligence signals
(priority, confidence, freshness) and with *relationship* signals (conflicts,
duplicates). A generic vector search cannot express "demote the memory that lost
a resolution" or "do not return three copies of the same fact" — those rules are
only possible because M2 recorded the relationships.

### 2. Two candidate-generation strategies, one ranking formula

| Mode | Candidates | Cost |
| --- | --- | --- |
| `lexical` | postings from the in-memory inverted index | O(candidates) |
| `semantic` / `hybrid` | the whole filtered searchable corpus | O(corpus) |

The second is unavoidable: embeddings can match texts that share no tokens.
Both paths funnel into the *same* scoring, selection and explanation code, so the
mode only changes where candidates come from and how relevance is computed.

### 3. The inverted index is rebuilt, not hooked into writes

Maintaining an incremental index would mean touching `MemoryService.create_memory`
/ `update_memory` / `forget_memory` (M1 code) and keeping two sources of truth in
sync. Instead the retriever keeps an in-memory index and rebuilds it when the
corpus signature (`COUNT` + `MAX(updated_at)` over non-forgotten memories)
changes:

- no writes are added to the M1/M2 paths,
- the index can never drift silently,
- a query after a write pays one O(n) reload; a query without writes pays
  nothing.

The signature query is answered from an index, so it is cheap. For much larger
corpora the natural next step is a persistent posting table; that is a deliberate
M4+ decision, not an oversight.

### 4. Embeddings are optional at every layer

`build_provider(None)` (the default) returns `None`, and the retrieval service
then reports `retrieval_mode: lexical`. A provider that exists but is unreachable
(Ollama not running) is probed with a short timeout and a cached answer, so a
missing server degrades a search instead of failing it. Embeddings are cached in
SQLite keyed by `(memory_id, provider, model)`; the stored vector is a packed
little-endian float array, and the provenance columns make a cached row
self-describing.

The bundled `hashed-lexical` provider is a **model-free** implementation: it
hashes normalised tokens into an L2-normalised vector so that the whole semantic
code path can be exercised (and tested) on a laptop with no download. It is
documented as a lexical proxy — it cannot capture synonymy.

### 5. Ranking is a documented, validated, bounded formula

```
base   = sum(component_weight * component)      # weights must sum to 1.0
final  = base * (1 - w_conflict * penalty) * (1 - w_redundancy * penalty)
```

Because `base` is a convex combination and both penalty factors are in `[0, 1]`,
the final score is always in `[0, 1]` — no score can be invented by a
misconfiguration. `RankingConfig.__post_init__` rejects weights that do not sum
to 1.0, out-of-range penalties and non-positive half-lives, so a bad
configuration fails at construction rather than producing nonsense rankings.

### 6. Conflict awareness reads the M2 graph, never re-derives it

`_conflicts_for` reads the `CONFLICTS_WITH` / `SUPERSEDES` rows and turns them
into a penalty:

- the winner of a resolution gets `0.0` (no penalty) — its metadata says so;
- the loser gets `1.0` and reports `superseded_by`, but stays retrievable when
  the caller asks for non-active memories;
- an *unresolved* conflict gets `0.5`, which is the honest "we do not know yet"
  answer.

### 7. Suppression is a selection rule, reported as data

Duplicates are suppressed during selection (never deleted), and the suppressed
candidates are returned with their penalised score and the reason. The
representative is the highest-scored member, so the choice is deterministic and
auditable.

### 8. Test strategy for M3

| Layer | What is proved |
| --- | --- |
| Domain | Weight validation, idf-weighted coverage, the blend formula, penalties, tie-breaks. |
| Embeddings | Determinism, dimensionality, cosine bounds, unavailable-provider behaviour. |
| Storage | Embedding round-trip, provenance, cascade, corrupted-blob detection. |
| Service | Candidate generation per mode, ranking order, filters, suppression, conflict demotion, degradation, index rebuild timing, explain(). |
| API | Search/explain contracts, validation, degradation reporting, M1/M2 compatibility. |

## M4 design decisions

### 1. Retrieval and context compilation answer different questions

Retrieval asks *"which memories are relevant?"*. Compilation asks *"given
everything relevant, what is the best context an agent can be given under a hard
budget, in what order, and why?"*

That is why M4 does not extend the M3 ranking formula. Extending it would have
made the M3 contract mean two things at once: `final_score` is a relevance rank
and would also have to be a context decision, which depends on the budget, on
what else is in the set, and on what has already been admitted. Those are
context-time quantities, so they get their own score (`utility`) with their own
documented formula, and M3's score stays exactly what it was.

### 2. Token estimation is a documented estimate, not a tokenizer

Using a provider tokenizer would tie the compiler to one model family, require a
download, and make the "deterministic and offline" promise conditional on a
cache being warm. So the estimator is a standard-library function of the text
(`app/domain/tokens.py`), with:

- an id reported next to every figure (`memforge-char-word/v1`),
- `exact: false` in the payload, so an estimate can never be read as a fact,
- validation of every ratio at construction,
- a form in which replacing it does not touch the compiler.

Two deliberate choices inside the formula:

- **`max(char_cost, word_cost)`, not a sum.** Character density and word count
  correlate but not perfectly; taking the larger one means the estimate follows
  whichever signal dominates instead of drifting high on text where they
  disagree.
- **A non-ASCII code point costs a full token.** Real BPE tokenizers need *more*
  than one token per non-ASCII character, never fewer, so this is the
  conservative direction.

Whitespace runs are normalised *before* counting. That is not a cosmetic detail:
it means a reformatted memory costs exactly what it cost before, so the compiler
can never "save" tokens by claiming a whitespace trim was a compression win.

### 3. Utility is a bounded convex blend plus a bounded cost penalty

```
value = sum(weight_i * factor_i)        # weights sum to 1.0, factors in [0, 1]
cost  = min(1, tokens / cost_reference_tokens)
utility = value * (1 - cost_weight * cost)
```

The convex-combination structure (already used by M2 importance and M3 ranking)
buys a hard guarantee: `utility ∈ [0, 1]` no matter how the configuration is
written, and `ContextPolicy.__post_init__` refuses weights that do not sum to
`1.0`. The cost penalty is the interesting half — it is what stops the compiler
from preferring long memories (they crowd out everything else) *and* from
preferring trivially short ones (they carry little information per token). A
raw `value / tokens` density was rejected because it is unbounded, unstable at
`tokens → 0`, and would have made two candidates with a utility of 0.9 and a cost
of 10 vs 100 incomparable in a bounded way.

### 4. Greedy admission with an explicit "keep going" rule

```
for candidate in sorted(candidates, key=(-utility, memory_id)):
    if candidate.cost <= remaining: admit
    else:                      record insufficient_budget; CONTINUE
```

The naive alternative — stop at the first candidate that does not fit — wastes the
remaining budget whenever a large high-utility memory is followed by small ones.
Continuing is safe because the remaining budget only shrinks, so a rejected
candidate provably cannot become affordable later; there is no need for a
backtrack or a knapsack solver. The result is not globally optimal (a large
memory can still crowd out several smaller ones) and the README says so.

The ordering key is a **total** order (`-utility` then `memory_id`). Every ordering
in M4 is total — section, then utility, then id — which is what makes "same
corpus + same clock + same request ⇒ same result" a property rather than a hope.

### 5. The rendered text is what the budget is enforced against

Per-block costs cannot see the section headings, and the headings are only known
once the surviving blocks are known. Rather than either (a) ignoring the framing
and risking an overflow, or (b) reserving a worst-case header budget up front and
wasting it, the compiler measures the *actual rendered string* and, if it still
overflows, drops the lowest-priority block until it fits.

That trim is a last resort, it normally does not run, and it deliberately does
not re-admit anything afterwards: re-packing after a trim would make the reported
`remaining_tokens` a function of a second greedy pass, which is harder to explain
than leaving the slack. The affected block keeps a single trace entry with
`reason: insufficient_budget`, so the "one decision per candidate" invariant
holds even on this path.

The reported `estimated_tokens` is always the estimate of `context_text` itself,
never a sum of per-block estimates, because the estimator is not additive.

### 6. Conflict handling reads M2's decision; it never re-decides

M2 owns the resolution policy. M4 only honours its outcome:

- `SUPERSEDES` recorded → the loser is suppressed from the context and the winner
  is named, unconditionally. Re-deciding would mean two policies disagreeing about
  the same fact, which is worse than either being right alone.
- conflict still `unresolved` → both stay candidates. M3's `0.5` penalty already
  demotes them, and inventing a tie-break here would be M2's job.

Suppressing unconditionally means a context can be *silent* about an attribute
when the winner was not retrieved. That is the deliberate trade: an out-of-date
fact in a prompt is worse than a missing one, and the suppression is reported
with its reason so the consumer can decide to widen the retrieval.

### 7. Duplicate groups are resolved transitively, and only what is visible is resolved

A `DUPLICATE_OF` link points at *one* partner, so chains (`C → B → A`) are
possible. Resolving pairs independently would keep `C` while suppressing `B` —
exactly the redundancy the step exists to remove. The compiler therefore maps
every candidate id to the *surviving representative* of its group, so a duplicate
of a duplicate resolves to the same representative.

The opposite limit matters too: a recorded duplicate whose partner is **not** in
the candidate set is kept. Suppressing a memory because of a relationship to
something that is not in the context would drop information for no reason — the
compiler only suppresses redundancy it can actually see.

M3 already suppresses near-duplicates from its result set, so on the default path
M4 would never see them. Rather than let that work disappear from the report, the
compiler re-scores M3's suppressed candidates through M3's own `explain()` API
(one extra scored candidate per suppressed duplicate) and runs them through the
same pipeline. The alternative — inventing a second suppression list — would have
made "why is this memory missing from my context?" answerable in two places.

### 8. Ordering is a policy, not an accident of the ranking

Retrieval order optimises relevance; prompt order optimises *stability of the
agent's mental model*. Stable identity first, then durable facts, then recent
events, then open work, then low-confidence notes — with utility breaking ties
inside a section and the memory id breaking ties inside that. The type → section
mapping and the section order are both part of `ContextPolicy`, so a deployment
can reorder them without touching code.

### 9. Compression is a documented, bounded transformation

M4 has no LLM, so "compression" can only mean lossless-or-decidable edits:

1. whitespace normalisation (meaning-preserving, and free — see §2);
2. dropping whole sentences when `max_block_tokens` is exceeded, ranked by query
   coverage and kept in original order.

Both keep the original text, the retained segment indices and both token figures
next to the result, and neither ever produces an empty block: when no sentence
fits the ceiling, the single most relevant one is kept rather than emptying the
memory. Compression does not guarantee semantic equivalence — a dropped sentence
can carry a negation or a condition — and that is stated in the payload, the
README and the docstrings rather than buried here.

`max_block_tokens` is a **soft** target by design. It shapes what a block looks
like; the budget decides whether the block is affordable. A hard per-block ceiling
would have produced empty blocks or mid-sentence cuts, both of which are worse
than an expensive block the budget can reject.

### 10. Configuration is validated twice, on purpose

`Settings` rejects bad context configuration at startup (weights must sum to
`1.0`, the default budget may not exceed the maximum, every ratio must be
positive) *and* `ContextPolicy` re-validates its own invariants at construction.
The duplication is intentional: the first gives an operator a startup error tied
to an environment variable, the second protects any caller that builds a policy
in code (a test, an embedding, the CLI) and gives a message that names the actual
knob. The weight tables themselves stay code-level defaults, exactly like
`IntelligenceConfig` and `RankingConfig`.

### 11. Test strategy for M4

| Layer | What is proved |
| --- | --- |
| Estimator | Determinism, boundaries (empty, single char, exact multiples), unicode and punctuation, configured ratios, validation. |
| Domain | Request/policy validation, utility monotonicity in every factor, relevance-vs-cost trade-off, compression rules, ordering total-order, admission, transitive duplicate groups, budget edge cases (exact fit, one too large, all too large, many small, identical utilities). |
| Service | Hard-budget compliance across a sweep of budgets, conflicts, duplicates, provenance, filters, ageing through the injected clock, determinism across service instances, semantic degradation. |
| API | Response contract, budget validation, malformed requests, error envelope, service wiring and sharing. |
| Demo | The M4 section's content and byte-for-byte determinism. |
| Live HTTP | A real uvicorn run: 24 mandatory checks (see `scripts/smoke_test.py`). |

## M5 design decisions

### 1. A provider interface, not a client

`LLMProvider` is an abstract class with two operations (`generate`,
`generate_structured`) and a configuration flag. That is the entire contract
MemForge depends on, and it is why the mock provider is a *first-class*
implementation rather than a test double:

- the agent, the API and the demo are written once against the interface;
- the mandatory reproducible mode and the optional real mode take exactly the
  same code path, so the offline path cannot rot;
- a third provider (a local runtime, a queue, a different wire format) is a new
  class, not a patch.

The provider is deliberately **not** an HTTP client abstraction. `httpx` appears
only inside `CustomLLMProvider`, so nothing above that layer knows that a network
exists.

### 2. Usage is reported, never derived

`LLMUsage` is optional and its presence means "the provider said so". The
alternative — filling in zeros or running the local estimator — was rejected for a
specific reason: the two numbers would then be indistinguishable downstream, and
a cost report that mixes an estimate with a measurement is worse than one that
says "unknown". So:

- a provider that sends no usage block yields `usage = None`;
- observability records `usage: null`, not `usage: 0`;
- MemForge's own context token figure travels separately as
  `context_estimated_tokens` and is always labelled as an estimate.

### 3. The credential is structurally absent

Four mechanisms, in order of importance:

1. **It is never copied.** The key lives on `CustomLLMConfig` with `repr=False`
   and is read only in `headers()`. No response, error, log record or evidence
   payload is built from it.
2. **It cannot reach an error.** Every provider-generated string passes through
   `redact_text`, so a server that echoes the `Authorization` header back in its
   error body still cannot leak the key through an exception message.
3. **It cannot reach a header dump.** `redact_mapping` *drops* auth-bearing
   header names rather than masking their values; masking would still advertise
   that a credential exists.
4. **A misconfigured URL is defused.** `safe_endpoint` strips `user:password@`
   userinfo, so a base URL with credentials embedded in it cannot turn a log line
   into a leak. (This was found by a test, not by inspection.)

Selecting `llm_provider=custom` without a base URL is a **startup error**, not a
silent fallback to the mock. A silent fallback would let an operator believe a
model answered when none was called — the failure mode the README lists as the
one to avoid.

### 4. The agent is a pipeline, not a framework

One bounded sequence with no planner, no tool loop, no queue and no autonomy.
The decision to keep it that way is the substance of M5: an agent framework would
have been a second system with its own state, its own retries and its own ideas
about memory — the exact divergence M1-M4 exist to prevent.

Two choices inside the pipeline are worth stating:

- **The memory gate runs before retrieval.** A cheap, deterministic check
  ("could this turn possibly benefit from stored memory?") avoids an entire
  retrieval + compilation cycle for small talk, and the decision is reported
  (`used` / `skipped` / `disabled` / `no_matches`) so the caller can see *why*
  memory was or was not used. When recall finds nothing with non-zero relevance,
  compilation is skipped rather than re-run to produce the same empty result.
- **The model may propose; only the caller may dispose.** Candidate extraction
  is a separate, optional step, and `store_candidates` defaults to `False`.
  Provider output is treated as untrusted input: a malformed proposal is returned
  with a reason (`empty_content`, `unknown_type`) and never written. The
  alternative — letting the model write whenever it judges a fact durable — would
  make the memory layer an append-only log of model opinions.

### 5. Provider failure is explicit, and never back-filled

A failed provider call raises a typed `LLMProviderError`. The API maps `kind` to a
status code and returns the standard envelope. Nothing in the chain produces a
plausible substitute answer:

```
custom API timeout
  -> LLMTimeoutError (kind="timeout", retryable=True)
  -> POST /api/v1/agent/respond returns 504 {"error": {"code": "LLM_TIMEOUT", ...}}
  -> no answer, no usage, no fabricated text
```

The observability record for the failed step is still emitted (with
`success: false` and the error kind), so a failure is as auditable as a success.

### 6. Retries are bounded and selective

`max_retries` caps the attempts, backoff is linear and injectable (tests pass a
no-op sleep), and only *retryable* kinds are retried: timeouts, transport
failures, 429 and 5xx. A 4xx is a client mistake and repeating it only wastes the
budget. The bounds are validated at configuration time, and a test asserts the
exact number of attempts (`max_retries=2` means exactly three calls).

### 7. MCP is a transport, not a subsystem

The server registers seven typed functions over one `MemoryOperationsService`
instance. Three properties follow from doing it that way:

- **Schemas are published, not described.** Flat annotated parameters let the
  SDK derive a real JSON input schema (required fields, numeric bounds), and a
  pydantic return type produces a validated output schema. A client can check a
  call before making it.
- **Errors are data.** A domain exception becomes the same envelope the REST API
  uses, with `isError` set, so a client sees a structured failure instead of a
  stack trace. The SDK checks for a `CallToolResult` before validating a result
  against the output schema, which is how the two shapes coexist; the SDK refuses
  a union mentioning `CallToolResult`, so that one documented cast lives in a
  single helper rather than in seven tools.
- **There is no inference path.** No MCP tool touches a provider, so no MCP
  surface can leak a credential. The MCP client already is the model.

### 8. The operations facade translates, it does not re-implement

`MemoryOperationsService` exposes six verbs and holds no business logic. Its one
substantive translation is `None` → `UNSET`: M1 distinguishes "field omitted"
from "field explicitly null", and a facade that forwarded `None` directly would
turn every partial update into a validation error. That bug was caught by the
MCP self-test, which is the argument for keeping a real protocol-level check
alongside the unit tests.

### 9. Test strategy for M5

| Layer | What is proved |
| --- | --- |
| Domain | Provider interface, messages, requests, usage semantics, JSON parsing, the error hierarchy, redaction. |
| Providers | Request formation, bearer auth, timeouts, HTTP failures, malformed bodies, structured output, bounded retries, usage passthrough, factory selection. |
| Facade | All six verbs, the `UNSET` translation, recall gating, budget forwarding. |
| Agent | Gate, recall → compile → provider, candidate handling, opt-in storage, observability, determinism. |
| API | REST contracts, candidate opt-in, budget validation, failure mapping, secret hygiene. |
| MCP | Tool schemas, annotations, validation, error normalisation, integration, determinism. |
| Demo | The M5 section, the storage flag, and byte-for-byte determinism. |
| MCP self-test | A real server over all seven tools plus the error paths, offline. |
| Live HTTP | A real uvicorn run: 24 mandatory checks, offline. |
| Live API | **Opt-in only** (`scripts/llm_smoke_test.py`); refuses to start without credentials and never prints the key. |

## M6 design decisions

M6 answers two questions that M1-M5 deliberately did not answer: *is the memory
system making good decisions?* and *what exactly happened?* Everything below
exists to keep those two answers honest.

### 1. The evaluation is MemForge-specific, and it is not EvalForge

A generic LLM evaluation scores generated text against a reference or a judge.
MemForge does not generate prose for an end user; it *decides* which memories to
recall, which to suppress and what to compile into a context. Scoring that text
would require a judge model -- making the evaluation non-deterministic,
network-dependent, costly, and, worst of all, uninformative about the memory
layer.

So the question is narrowed to something a count can settle:

> Given this corpus and this query, did the memory system make the decision its
> own documented rules say it should make?

No judge model. No reference answers. No network. Every number is a ratio of two
integers the evaluation produced by running the real services.

### 2. Cases describe a corpus and a call, never a score

An `EvaluationCase` holds an id, a category, an action, a query, a fixture (the
corpus), expected behaviour expressed against **fixture labels**, and the metrics
it applies to. It never says "precision should be 0.9".

Two consequences:

- **Expectations are readable.** A reviewer reads
  `"duplicates_suppressed": ["restated"]`, not a threshold.
- **Expectations are stable.** Ids are generated at execution time; the labels
  are not. A case keeps meaning the same thing across runs and across refactors
  of the id generator.

### 3. Nothing is re-implemented

The single most important structural rule. A case asks M3 to search, M2 to detect
and resolve, M4 to compile, M5 to run a turn, and then *counts what came back*.
It never re-implements scoring, ranking, suppression or budget selection.

The alternative -- computing metrics from a second copy of the logic -- would let
the evaluation agree with itself while disagreeing with the product, which is the
one thing a regression gate must never do. Because the case runner calls the
same services, any behavioural change in M1-M5 shows up as a named failing case.

### 4. Cases are isolated by construction

Each case gets:

- a fresh SQLite file (so a case cannot see another's corpus),
- a fresh `ManualClock` (so "recent vs old" is reproducible, not a function of
  when the suite ran),
- a private `SequentialIdGenerator` namespace per case id,
- a fixture-level `age_minutes` so a memory can be "a month old" deterministically.

Isolation by construction rather than by convention means a case cannot become
order-dependent, and two runs produce byte-identical results.

### 5. A metric never appears without a count

Each case *declares* the metrics it applies to. If the runner cannot measure one
of them, the case **fails** with a `measurement:` reason. A metric that no case
exercised is reported with a zero denominator and marked not applicable, so
"never measured" can never be mistaken for "measured and perfect".

This is the same discipline M4 uses for token counts: distinguish *estimated*,
*reported* and *absent*.

### 6. Pass/fail and graded metrics are different kinds of thing

- **Recall, top-k, suppression, compliance, write policy** are expectations. The
  case states what must happen; missing it fails the case. They are the sensitive
  part of a regression gate.
- **Precision** is graded. An extra returned memory lowers the metric but does
  not fail the build, because "how much of what we returned was useful" is a
  quality signal, not a binary rule.

The threshold therefore gates the **case pass rate** (default `1.0`), not the
blended `overall`. A healthy build can show `retrieval_precision = 0.79` -- which
it does -- without the gate flapping.

### 7. The report states its own limits

The report prints, on every run, that each case proves only what its own fixture
encodes, that no LLM-as-a-judge was involved, and that scores are not comparable
across datasets. A score without its boundary is easy to over-read, so the
boundary ships with the number.

### 8. Traces are an audit trail, not a log

Two separate models:

- `Trace` / `TraceEvent` -- *what happened*: an append-only, deterministic tree
  correlated by one `trace_id`.
- `TraceSummary` / `OperationMetrics` -- *how well*: counters derived by walking
  those events.

`sequence` makes the order total, because two events can share an instant and a
timestamp comparison is not an order. `parent_id` turns a flat list into a tree.
Events are never updated and never deleted: an audit that can be rewritten is not
an audit.

### 9. Instrumentation is decorators, never edits

M1-M5 were written before traces existed and were deliberately left alone. They
import nothing about observability. Instead the decorators in
`app/observability/instrumentation.py` wrap the services and delegate *first*:

```
TracedMemoryService          TracedRetrievalService
TracedContextCompilerService TracedLLMProvider
AgentTracer (one trace per turn)   McpTracer / RequestTracer (one per call)
```

A decorator that changed a result would make the trace a description of something
other than what happened. The test suite compares a traced turn against an
untraced one and asserts they are identical.

`TracedRetrievalService.explain` is deliberately *not* instrumented: M4 calls it
to re-score memories M3 already suppressed, and counting it would double-count
the work. For the same reason the compiler keeps the undecorated retrieval
service -- M4's internal re-run of retrieval is part of compiling a context, and
its work is accounted for inside the `context.compile` span.

### 10. Correlation is ambient and scoped

`ObservabilityService.use_trace(trace_id)` binds the id through a `ContextVar`.
Every decorator below it records into whatever trace is in scope, and records
nothing when none is. A `ContextVar` rather than instance state, because two
concurrent requests must not capture each other's events; the binding is restored
on exit, including when the block raises.

This is why the MCP server needs almost no code to be traceable: the tool body is
unchanged, and the traced services it calls pick up the id automatically.

### 11. Safety is structural, not procedural

Three independent mechanisms, because a single one is a single point of failure:

1. **Deny-list.** `safe_metadata()` drops credential- and content-shaped keys
   (`api_key`, `authorization`, `token`, `secret`, `password`, `prompt`,
   `answer`, `content`, `chain_of_thought`, ...) *before* inspecting the value, so
   an accidental `metadata={"api_key": ...}` cannot be persisted.
2. **Registered secrets.** `register_secret(key)` makes the service redact that
   literal out of any string it records, wherever it came from. Both the REST app
   and the MCP server register `MEMFORGE_LLM_API_KEY` at startup.
3. **No content at all.** Payloads are not truncated, they are absent: only sizes
   (`input_characters`, `output_characters`), counts, ids, modes and durations are
   recorded. There is no redaction heuristic that could be fooled, because there
   is no text to fool.

Auth-bearing headers additionally pass through M5's `redact_mapping`.

### 12. Nothing measured is invented

- **Latency** is measured with an injectable monotonic timer and reported on the
  closing event. It is not a benchmark: no test compares a duration to a number,
  and the HTTP smoke test asserts only that a duration is present and
  non-negative. The demo injects a deterministic stepper so its output is
  reproducible -- which is a *measurement of a known clock*, not a hard-coded
  value.
- **Provider usage** is echoed only when the provider sent it. With the offline
  mock, `usage_reported` is `false` and no token figure is written at all.
  Characters are never converted into tokens; no cost is ever derived.
- **Token figures in a trace** are M4's own estimate and are named
  `estimated_tokens`, never mixed with usage.
- **Counters that were never incremented are absent**, not zero: a zero that was
  never measured is not a measurement.

### 13. The trace store is dumb on purpose

`SqliteTraceStore` stores what it is given and re-reads it faithfully. Redaction
is the *service's* job, because a store that silently rewrites its input is a
store you cannot trust when auditing.

It does validate what the schema should: a `CHECK` constraint on trace kind (so
an unknown kind cannot be written at all), `UNIQUE (trace_id, sequence)` (so the
order cannot fork), and a corrupt row is reported as
`ObservabilityError` rather than silently dropped.

Reads lazily migrate, so a store handed a brand-new path works without a separate
initialisation step -- and it is the *same* migration runner the rest of MemForge
uses. There is no second schema mechanism.

### 14. Evaluation and observability are independent

The evaluation runs against real services and produces counts. The observability
layer records what operations did. Neither depends on the other:

- the evaluation can run with tracing disabled (it uses `NullTraceStore`-free
  plain stacks), and
- the REST API serves traces without ever running an evaluation.

They are joined only by the fact that both describe the same system, and by the
evaluation endpoint being a read-only projection of the last run.

### 15. Dashboard preparation, not a dashboard

M6 builds no UI. It fixes the response shapes a dashboard will consume:
evaluation summaries with per-metric series carrying their own definitions and
raw counts, per-case results with figures, traces, trace events, memory/context/
provider counters, and failing cases with the checks that did not hold.

All of them are stable, sorted and credential-free, which is the actual
deliverable: M7 is a presentation layer over data that already exists.

### 16. Test strategy for M6

| Area | What is asserted |
|---|---|
| Domain models | validation, the metric definitions, score arithmetic at its edges |
| Dataset | strict parsing: every malformed field names itself, references resolve by name or version |
| Evaluation service | every metric reads a real count; **every failure branch is reachable** (a gate that cannot fail is not a gate) |
| Shipped dataset | all 36 cases pass, every category covered, every metric measured, two runs identical |
| Trace store | round-trip, ordering, corruption, idempotence, restart persistence |
| Observability service | sequence totality, span parenting, duration measurement, redaction, counters |
| Instrumentation | the traced lifecycle, `usage_reported` only when supplied, tracing is transparent |
| API | shapes, error envelopes, and that no payload can carry a secret |
| Scripts | HTTP smoke test (30 checks), MCP self-test with trace correlation (11 checks) |

The deliberate negative tests matter most: the evaluation service has tests that
encode *wrong* expectations to prove each failure branch is reachable, and the
observability tests feed a registered secret through every emitter to prove it
cannot be persisted.

## M7 design decisions

M7 adds a presentation layer. Everything below is about keeping it one.

### 1. The dashboard renders decisions; it never makes them

The single rule the whole frontend is organised around. A score, a token figure,
a ranking or an evaluation result is rendered exactly as the API reported it,
rounded for display and never recomputed.

The specific trap was the context compiler. It would have been easy — and
tempting — to run a second token estimator in the browser and draw a budget bar.
That would have been a **different** measurement: the hard guarantee M4 makes is
`estimated_tokens <= token_budget` with *its* estimator, and a second estimator
could disagree with it. So the Context page renders `token_budget`,
`estimated_tokens`, `remaining_tokens`, `utilization` and `within_budget`
verbatim, names `memforge-char-word/v1` next to them, and the money note in the
product brief ("do not calculate a separate client-side token count") is a design
constraint rather than a caution.

The same reasoning drives the Evaluation page: `retrieval_precision = 0.793` is
printed as 0.793. Rounding it to 1.0, or averaging it away into "overall", would
be the dashboard editing the product's result.

### 2. One client, one place to change

`dashboard/src/api/client.ts` is the only module that calls `fetch`. It owns the
base URL, the request construction, the response parsing, the error envelope and
the trace-id header. No component knows a URL.

That is not tidiness, it is the contract boundary: when a response shape changes,
there is exactly one file to update, and the shapes themselves are declared once
per endpoint next to the code that reads them.

### 3. Every response is validated before it reaches a component

`src/api/parse.ts` declares a shape per endpoint and checks the payload against
it, reporting the exact JSON path of the first violation. A drifted backend
therefore produces an explicit **"malformed API response"** state instead of a
blank panel or, worse, a confidently rendered wrong number.

This is the frontend's version of M6's "a metric never appears without a count".
The validator is ~150 lines and has no dependency, because a validation library
would have been a larger addition than the thing it validates.

### 4. Four states per page, not one

Loading, ready, empty and failed are separate components (`ResourceView`), and
the failure state itself classifies *why* it failed:

| Condition | What the user is told |
| --- | --- |
| API unreachable | How to start the backend, and which base URL was tried. |
| Endpoint not available (bare 404/405) | That this deployment does not expose it. |
| Request rejected (422) | The backend's own error code and message. |
| Malformed response | That the contract did not hold, and that nothing was guessed. |
| Empty result | What would populate it. |

The distinction between "not found" and "not available" matters: a typed
`MEMORY_NOT_FOUND` is a real answer from the contract, while a bare 404 means the
route is not there at all. Collapsing them would hide a misconfigured deployment.

### 5. Filters the API does not support are labelled, not disguised

`GET /memories` takes `status`, `memory_type`, `limit` and `offset`. The explorer
also offers a text filter, confidence/importance thresholds and a sort order —
applied to the **loaded page**, with the caption saying so and the count
reporting "after page-level filters".

The alternative was to drop the controls, which would have made the page less
useful, or to present them as server queries, which would have been a lie about
the API. A ranked search across the whole corpus is the Retrieval Lab's job, and
the explorer says where to go.

### 6. Defence in depth on the trace viewer

M6 already records no credential, prompt or answer. The dashboard does not rely
on that: `src/lib/redact.ts` drops credential-shaped **keys** and replaces
credential-shaped **values** before anything from a trace is rendered.

It is deliberately not the only mechanism, and the reasons differ. The backend's
filter is the real one: a credential that was never persisted cannot leak. The
frontend filter covers the case where the backend is replaced or a proxy is added
in front of it. A test feeds a hostile event payload carrying `api_key`,
`authorization`, `prompt`, `answer` and `chain_of_thought` and asserts that none
of it reaches the DOM.

### 7. A forgotten memory is a state, not an error

`GET /memories/{id}` answers `404 MEMORY_NOT_FOUND` once a memory is soft-deleted
(M1 design, §3 of these notes). A naive detail page therefore shows a dead end
for exactly the memories M1 cares most about preserving.

The page treats it as a *distinguishable* state: on a 404 it loads the history,
and if the history ends in a `FORGOTTEN` version it renders the versions and
explains that nothing was destroyed. If the history has no `FORGOTTEN` version,
the id simply does not exist and the normal not-found state is shown. The
distinction is made from the backend's own data rather than from a guess.

### 8. No credential can reach a browser bundle, and it is checked

Structurally: the API is unauthenticated, the client sends `credentials: "omit"`,
builds no `Authorization` header, and the only configuration is
`VITE_MEMFORGE_API_BASE` / `VITE_MEMFORGE_API_TARGET` — URLs. Vite's `envPrefix`
is never widened.

Then checked, three ways: a frontend test asserts no request carries an
authorization header, `scripts/dashboard_smoke.py` greps the **built bundle** for
header-shaped markers and for a hard-coded backend origin, and the same script
sweeps every endpoint the dashboard can read for credential-shaped text.

The bundle check is the interesting one. A naive "does the word `authorization`
appear" scan fails, because the dashboard's own redaction deny-list contains that
word — a false positive that would train a reader to ignore the check. The
markers are therefore shaped like a credential being *used* (`"authorization":`
as an object key, `x-api-key`, a PEM header) rather than like the bare word.

### 9. No second opinion on latency, and no chart where there is no data

`GET /metrics` returns aggregate counters; the trace store holds per-trace
durations. The Metrics page renders both, and states plainly that there is no
time series, because there isn't one. Plotting a series would require inventing
the historical points, and a chart of invented points is worse than a counter.

Every duration shown is labelled as measured at runtime and compared to nothing.

### 10. Request state without a data-fetching framework

`useResource(key, loader, { ttlMs })` gives the three states, dedupes in-flight
requests by key and caches a response for a few seconds. `useAction` runs a POST
and is never cached.

The TTL is deliberately short (2–10 s) because the corpus can change under the
user's feet: a stale count on a dashboard is a lie. The win is real though — the
dashboard fires one `GET /memories?status=...` per status card and one
`/traces?limit=N` shared across pages, which is the "no uncontrolled rerenders"
requirement without adding react-query and its cache-invalidation problems to a
project that has no state to invalidate.

### 11. The real-data check is HTTP-backed on purpose

`scripts/dashboard_smoke.py` starts a real uvicorn, serves the **production
build** with an SPA fallback and a proxy, seeds a corpus through the public API
and walks Dashboard -> Memories -> Retrieval -> Context -> Agent -> Trace ->
Evaluation.

A browser stack (Playwright) would have proved more about pixels and less about
the contract, at the cost of a ~100 MB dependency in a repository whose entire
argument is that it is local-first and offline. The project's existing pattern
already verifies real integration at the transport level (`smoke_test.py`,
`mcp_selftest.py`); this extends the same idea to the presentation layer:

* the build is servable and every deep link gets the shell,
* the bundle carries no credential and no baked origin,
* every endpoint each page consumes answers with the exact bodies it sends,
* and the demo flow produces real data at each step.

What it does **not** prove is the rendered result in a real browser. That
limitation is stated in the README rather than glossed over.

### 12. Test strategy for M7

| Layer | What is proved |
| --- | --- |
| `parse.ts` | The declared shapes accept real payloads and reject drifted ones, naming the exact path. |
| `client.ts` | Base URL, query construction, the error envelope, malformed bodies, unreachable hosts, and that no request carries a credential. |
| `lib/` | Redaction filters; the trace tree keeps `sequence` order and loses no row. |
| Pages | Loading, ready, empty, error, unavailable, unreachable and malformed, per page, with real payloads. |
| Navigation | All eight routes render from their own URL; the sidebar, the 404 page and the theme toggle behave. |
| Responsive | The shell regions, the landmarks, the scroll containers and the label coverage that the CSS collapse depends on. |
| Smoke | The production build, against a real backend, over the whole demo flow. |

Fixtures are transcriptions of responses captured from a live uvicorn run — the
real ranking explanation, the real `0.793` precision, a real 13-event trace and a
real `CREATED -> UPDATED -> FORGOTTEN` history. A fixture written from
imagination would have tested the imagination.

The deliberate negative tests matter most: a malformed metrics payload must
produce an error state and *not* render a counter; a hostile trace payload must
produce no visible secret; a later evaluation that fails must show the failing
case with its checks.
