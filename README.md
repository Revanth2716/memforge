# MemForge

**Local-first persistent memory & context engineering platform for AI agents.**

Milestone 1 — *Memory Core*: deterministic, offline memory storage with a
versioned REST API backed by SQLite.
Milestone 2 — *Memory Intelligence*: deterministic importance scoring, temporal
decay, duplicate detection, conflict detection and conflict resolution, all
computed from the stored data with no LLM and no embeddings.
Milestone 3 — *Memory Retrieval*: memory-aware hybrid retrieval with a
documented ranking formula, conflict-aware results, duplicate suppression and
full explainability — deterministic and offline by default.
Milestone 4 — *Context Engineering*: a deterministic **context compiler** that
turns retrieved memories into a budgeted, ordered, fully explained agent
context.
Milestone 5 — *Agent Integration*: a bounded, memory-aware agent, a pluggable
LLM provider (a deterministic **mock by default**, an opt-in custom
OpenAI-compatible API), and an **MCP server** exposing the memory layer as tools.
Milestone 6 — *Evaluation and Observability*: a deterministic **36-case
evaluation** of the memory decisions themselves, and an append-only **trace
lifecycle** you can read back operation by operation.
Milestone 7 — *Control Plane*: a **React + TypeScript dashboard** over the real
memory, retrieval, context, agent, evaluation and observability APIs.

> M2 and M3 ship **deterministic lexical/text heuristics**. M3 has a pluggable
> embedding interface (a model-free local provider by default, Ollama
> optionally). M4 adds a **documented token estimator**, budgeted selection,
> prompt ordering and deterministic (never paraphrasing) compression. M5 adds the
> agent loop, the provider interface and MCP. M7 adds the dashboard that renders
> all of it over the existing API — see [M7](#m7-control-plane).

---

> **The mock provider is the mandatory, reproducible mode.** No API key, no
> network, no model. Every test, the demo and the smoke test run offline by
> default. A real API is opt-in through environment variables only — see
> [M5](#m5-agent-integration--custom-api).

## The problem

AI agents are stateless. Every new session starts from zero, so anything the agent
"learned" about a user, a project or a task is lost unless an application
re-implements memory from scratch — usually as a bag of strings in a table with no
versioning, no lifecycle rules and no way to answer *"what did we know last week,
and why did we believe it?"*.

Managed memory services solve this by sending the user's data to someone else's
API. MemForge takes the opposite approach: **the memory layer is a plain local
library** — a SQLite file plus typed Python — that an agent can embed directly,
with no network access, no API keys and no per-call cost.

## What MemForge is

MemForge is a reusable **memory infrastructure layer**.

| Primitive | Meaning |
| --- | --- |
| **Store** | Create a typed memory (`FACT`, `PREFERENCE`, `PROFILE`, `EVENT`, `TASK`, `OBSERVATION`) with `confidence` and `importance`. |
| **Retrieve** | Fetch by id, or list with `status` / `memory_type` filters and pagination. |
| **Update** | Partially patch a memory; a new history version is appended. |
| **Forget** | Soft-delete a memory. It disappears from reads, but **nothing is destroyed**. |
| **Inspect history** | Every memory keeps an append-only list of full snapshots (`CREATED` → `UPDATED` → `FORGOTTEN`). |
| **Score** *(M2)* | A bounded, explainable importance score and a decayed effective priority. |
| **Detect** *(M2)* | Lexical near-duplicates and attribute conflicts, with the evidence. |
| **Resolve** *(M2)* | A deterministic conflict policy that records `SUPERSEDES` and explains itself. |
| **Retrieve** *(M3)* | Hybrid search over the corpus with memory-aware ranking, duplicate suppression, conflict metadata and a per-result explanation. |
| **Compile** *(M4)* | Turn retrieved memories into a token-budgeted, ordered agent context with provenance and a decision for every candidate. |
| **Respond** *(M5)* | Answer a request using compiled memory context, with the memory decision and every step explained. |
| **Integrate** *(M5)* | The same memory layer over REST and over MCP (`memory.store`, `memory.search`, `memory.recall`, `memory.update`, `memory.forget`, `memory.compile_context`, `memory.explain`). |

Everything is deterministic and offline: no LLM, no embedding model and no network
call sits on the default path.

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                         app/api (FastAPI)                      │
│  routes/memories.py   routes/retrieval.py   routes/context.py  │
│  routes/agent.py  (strict models + one error envelope)        │
└───────────────┬───────────────────────────────────────────────┘
                │ depends on the interface only
┌───────────────▼───────────────────────────────────────────────┐
│                    app/services (use cases)                    │
│  MemoryService          MemoryRetrievalService                 │
│    M1 lifecycle           M3 retrieval                         │
│    M2 intelligence   ──▶  ContextCompilerService  (M4)         │
│  MemoryOperationsService (M5 facade over M1-M4)                │
│    ──▶ AgentService (M5, bounded memory-aware workflow)        │
│    ──▶ LLMProvider: mock (default) | custom (opt-in)           │
└───────────────┬──────────────┬────────────────────────────────┘
                │              │ composes M3, never re-implements it
┌───────────────▼──────────────▼────────────────────────────────┐
│              app/storage (MemoryRepository ABC)                │
│  app/storage/sqlite/repository.py   ← the M1/M2 implementation │
│  (swap-in point: another local store, or an Ollama-backed one) │
└───────────────┬───────────────────────────────────────────────┘
                │
        ┌───────▼────────┐        ┌─────────────────────────┐
        │  SQLite file   │        │  memory_versions table  │
        │  (memories)    │──1:N──▶│  append-only snapshots  │
        └───────┬────────┘        └─────────────────────────┘
                │1:N
        ┌───────▼─────────────────────────────────────────┐
        │  memory_relationships (M2)                      │
        │  DUPLICATE_OF / CONFLICTS_WITH / SUPERSEDES     │
        └─────────────────────────────────────────────────┘

app/domain  → pure standard-library entities, enums, errors, clock/id ports,
              similarity, attribute extraction, scoring/decay, resolution policy,
              retrieval ranking, token estimation, context compilation,
              LLM contracts, agent workflow data
app/mcp     → MCP server exposing the M1-M4 services as tools
app/cli     → deterministic demo walkthrough
scripts/    → live HTTP smoke test, MCP self-test, optional live-API smoke test
```

Design rules that the code follows:

- **Layered dependencies point one way.** The domain imports nothing but the
  standard library; storage imports the domain; services import storage
  *interfaces*; the API imports services. No layer reaches back up.
- **Providers sit behind interfaces.** `MemoryRepository` (storage) is an ABC, and
  `Clock` / `IdGenerator` are protocols, so a local Ollama-backed store or a
  different time source can be added without touching the service or API code.
- **Injected clocks and id generators.** That is what makes the tests and the CLI
  demo byte-for-byte deterministic, and it is why "now" is never read implicitly
  inside a scoring function.
- **Parameterised SQL only.** Values are always bound as parameters; only static
  SQL fragments (`WHERE`/`ORDER BY` clauses) are assembled (there is a test with
  `'; DROP TABLE memories; --`).
- **One connection per operation**, one transaction per write, foreign keys on.

### Data model

| Table | Notes |
| --- | --- |
| `memories` | `id` (PK), `content`, `memory_type`, `source`, `confidence`, `importance`, `status`, `created_at`, `updated_at`, `version`, plus M2 telemetry `access_count`, `last_accessed_at`. |
| `memory_versions` | `id` (PK), `memory_id` (FK → `memories`, `ON DELETE CASCADE`), `version`, `change`, full snapshot, `recorded_at`; `UNIQUE (memory_id, version)`. |
| `memory_relationships` *(M2)* | `id` (PK), `source_id` (FK), `target_id` (FK), `relationship`, `similarity`, `created_at`; `UNIQUE (source_id, target_id, relationship)`, `CHECK (source_id <> target_id)`, `CHECK (relationship IN (...))`, `CHECK (similarity IS NULL OR 0 <= similarity <= 1)`. |
| `_memforge_migrations` | Bookkeeping for the tiny migration runner. |

Indexes support the common lookups: `(status, memory_type)`, `(created_at DESC, id DESC)`
for newest-first pagination, `(memory_id, version)` for history, and
`(source_id, relationship)` / `(target_id, relationship)` for relationships.

## API endpoints

Base URL: `http://127.0.0.1:8000`

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness + storage check. |
| `POST` | `/api/v1/memories` | Create a memory → `201`. |
| `GET` | `/api/v1/memories` | List `ACTIVE` memories (newest first). Filters: `status`, `memory_type`, `limit` (1–100, default 50), `offset`. |
| `GET` | `/api/v1/memories/{id}` | Fetch one memory. Forgotten memories → `404`. |
| `GET` | `/api/v1/memories/{id}/history` | Full version history, oldest first. |
| `PATCH` | `/api/v1/memories/{id}` | Partial update → new version. |
| `DELETE` | `/api/v1/memories/{id}` | Forget (soft delete) → `204`, idempotent. |
| `GET` | `/api/v1/memories/{id}/intelligence` *(M2)* | Importance breakdown, confidence, decayed priority, access telemetry. |
| `GET` | `/api/v1/memories/{id}/duplicates` *(M2)* | Near-duplicates. Query: `threshold`, `limit`. |
| `GET` | `/api/v1/memories/{id}/conflicts` *(M2)* | Unresolved conflicts (same attribute, different value). |
| `POST` | `/api/v1/memories/{id}/conflicts/{target_id}/resolve` *(M2)* | Deterministic resolution + `SUPERSEDES` link. |
| `GET` | `/api/v1/memories/{id}/relationships` *(M2)* | Every recorded relationship for a memory. |
| `POST` | `/api/v1/memories/search` *(M3)* | Search with `query`, `top_k`, `retrieval_mode`, `semantic_weight` and filters. |
| `GET` | `/api/v1/memories/{id}/explain?query=...` *(M3)* | How one memory would rank for a query. |
| `POST` | `/api/v1/context/compile` *(M4)* | Compile retrieved memories into a budgeted context: `query`, `token_budget`, `top_k`, filters and compression options. |
| `POST` | `/api/v1/agent/respond` *(M5)* | Answer a request using compiled memory context. |
| `POST` | `/api/v1/memories/remember` *(M5)* | Store a memory and report its near-duplicates. |

### Example session

```bash
# create
curl -s -X POST http://127.0.0.1:8000/api/v1/memories \
  -H 'content-type: application/json' \
  -d '{"content": "User prefers dark mode.", "memory_type": "PREFERENCE",
       "source": "user:revanth", "confidence": 0.9, "importance": 0.8}'
# {"id": "...", "content": "User prefers dark mode.", "memory_type": "PREFERENCE", ...,
#  "status": "ACTIVE", "created_at": "...", "updated_at": "...", "version": 1}

# update (only the provided fields change)
curl -s -X PATCH http://127.0.0.1:8000/api/v1/memories/<id> \
  -H 'content-type: application/json' -d '{"importance": 1.0}'

# importance + decayed priority (M2)
curl -s http://127.0.0.1:8000/api/v1/memories/<id>/intelligence

# near-duplicates (M2)
curl -s "http://127.0.0.1:8000/api/v1/memories/<id>/duplicates?threshold=0.6"

# forget
curl -s -X DELETE http://127.0.0.1:8000/api/v1/memories/<id>   # 204

# compile a budgeted context (M4)
curl -s -X POST http://127.0.0.1:8000/api/v1/context/compile \
  -H 'content-type: application/json' \
  -d '{"query": "preferred backend language", "token_budget": 1200, "top_k": 20}'
```

### Errors

Every error uses the same envelope:

```json
{
  "error": {
    "code": "MEMORY_NOT_FOUND",
    "message": "Memory '00000000-...' does not exist",
    "details": { "errors": [] }
  }
}
```

| HTTP | Code | When |
| --- | --- | --- |
| 400 | `MEMORY_VALIDATION_ERROR` | A domain invariant was violated. |
| 400 | `INVALID_RELATIONSHIP` | A relationship is invalid (self-link, unknown kind, bad similarity). |
| 400 | `CONTEXT_VALIDATION_ERROR` *(M4)* | A context request is invalid (e.g. a budget above the configured maximum). |
| 400 | `AGENT_VALIDATION_ERROR` *(M5)* | An agent request is invalid (e.g. a budget above the configured maximum). |
| 429 | `LLM_RATE_LIMIT` *(M5)* | The provider rate limited the request. |
| 502 | `LLM_AUTHENTICATION` / `LLM_TRANSPORT` / `LLM_PROTOCOL` / `LLM_PROVIDER_ERROR` *(M5)* | The provider failed; MemForge returns an explicit failure and never invents an answer. |
| 503 | `LLM_NOT_CONFIGURED` *(M5)* | The selected provider is missing configuration. |
| 504 | `LLM_TIMEOUT` *(M5)* | The provider did not answer within the timeout. |
| 404 | `MEMORY_NOT_FOUND` | Unknown id, or forgotten (soft-deleted) memory. |
| 409 | `MEMORY_ALREADY_EXISTS` | Duplicate identifier. |
| 409 | `INVALID_MEMORY_STATE` | Invalid transition (e.g. `PATCH {"status": "DELETED"}` — use `DELETE`). |
| 409 | `NO_CONFLICT` | A conflict resolution was requested but nothing is unresolved. |
| 422 | `VALIDATION_ERROR` | Request body or query params failed validation. |
| 500 | `INTERNAL_ERROR` / `REPOSITORY_ERROR` | Unexpected failure (logged server-side). |
| 500 | `INVALID_CONFIGURATION` | Intelligence configuration is inconsistent. |
| 503 | `STORAGE_UNAVAILABLE` | Storage did not answer the health query. |

## M2 algorithms

All M2 algorithms are pure functions of their inputs (memory, configuration,
reference time). None of them reads a clock, touches storage or calls a model
implicitly, which is why every behaviour below is reproducible in a unit test.

### 1. Importance scoring

```
importance = 0.20 * type_weight(memory_type)          # fixed, documented table
          + 0.40 * stored_importance                  # caller supplied
          + 0.15 * recurrence                        # min(1, near-duplicates / 3)
          + 0.15 * recency                            # 2 ** (-age_days / 30)
          + 0.10 * access_frequency                  # min(1, accesses / 5)
```

- **Bounded**: the weights sum to 1.0 and every factor is in `[0, 1]`, so the
  score is a convex combination and can never leave `[0, 1]`.
- **Explainable**: `GET /memories/{id}/intelligence` returns each factor with its
  weight, raw value, contribution and a sentence of explanation.
- **Never stored**: the score is recomputed on request, so it tracks ageing,
  duplicates and accesses automatically. `stored_importance` (the caller's input)
  and `confidence` (accuracy) are reported separately and are never conflated.

Documented type weights: `PROFILE 1.0`, `FACT 0.8`, `PREFERENCE 0.7`, `TASK 0.7`,
`EVENT 0.5`, `OBSERVATION 0.3`. These are explicit heuristics, not measurements.

### 2. Temporal decay

```
age_days = (now - updated_at) in days          # from the injected clock
decay    = 2 ** (-age_days / half_life_days)   # half-life defaults to 30 days
priority = (0.70 * importance_score + 0.30 * confidence) * decay
```

- Priority is **computed, never stored** — the stored memory is untouched by the
  passage of time, and the same memory yields different priorities at different
  reference times.
- Access telemetry (`access_count`, `last_accessed_at`) is written only by an
  explicit `record_access` call. It deliberately does **not** create a history
  version and does **not** touch `updated_at`: reading a memory must never look
  like editing it.

### 3. Duplicate detection (lexical, no embeddings)

Text is normalised through a documented pipeline, then compared with the Dice
coefficient over token sets:

```
tokenise ([a-z0-9]+) -> drop 1-char tokens -> remove stop words -> strip suffixes
similarity = 2 * |A n B| / (|A| + |B|)      # bounded [0, 1], symmetric
```

Effects worth knowing: `user's` → `user`, `prefers` / `preferred` / `preferring`
all → `prefer`, and `'; DROP TABLE memories; --` is just ordinary text. With the
default threshold of `0.6`, *"User prefers Python"* and *"Python is the user's
preferred language"* score **0.857** and are reported as near-duplicates, while
*"User prefers Python"* and *"The build server is in Hyderabad"* score 0.33.

Rules: only `ACTIVE` memories are compared, matches are ranked by similarity
descending then id, a pair that **conflicts** on an attribute is never reported as
a duplicate, and detection records `DUPLICATE_OF` links idempotently (re-running
detection never duplicates a row, and no memory is modified).

### 4. Conflict detection

Two memories conflict when they state the **same normalised attribute** with
**different normalised values**. Attributes are extracted with an explicit,
documented grammar — unrestricted language understanding is deliberately out of
scope:

```
<attribute> = <value>     preferred backend language = Python
<attribute>: <value>      preferred backend language: Python
<attribute> is <value>    Python is the user's preferred language
```

Values are compared with the same similarity measure: if `similarity >= 0.5` the
values are treated as the *same* (so `Python` and `Python 3` are not a conflict);
below that they are different values, and the pair becomes a conflict candidate.
Conflicts are visible from both sides, exclude archived/forgotten memories, and
disappear from the unresolved list once a `SUPERSEDES` link exists.

### 5. Resolution policy

`POST /memories/{id}/conflicts/{target_id}/resolve` applies a deterministic
policy and stops at the first decisive factor:

1. **explicit importance** — the caller's stated importance;
2. **confidence** — the more accurate memory;
3. **recency** — the more recently updated memory;
4. **effective priority** — the decayed composite (also captures recurrence and
   access counts);
5. **identifier** — the lexicographically smaller id, which guarantees a total
   order so the outcome never depends on iteration order.

The response names the winner, the loser, the deciding factor and every
comparison. The loser is archived by default — which appends a history version, so
the decision is auditable — and a `SUPERSEDES` relationship is recorded. Nothing is
overwritten or deleted.

## M3 retrieval

`POST /api/v1/memories/search` returns the most useful memories for a query. The
default path is **lexical-only** and requires no embedding provider at all.

### Retrieval modes

| Mode | Meaning |
| --- | --- |
| `lexical` (default) | Deterministic inverted-index search. Always available. |
| `semantic` | Embedding similarity only. Requires an available provider. |
| `hybrid` | Documented blend of both. Requires an available provider. |

When `semantic` or `hybrid` is requested without an available provider the
response says so explicitly — `retrieval_mode: "lexical"`, `semantic_available:
false`, `degraded_reason: "..."` — instead of pretending embeddings were used.

### Ranking formula

```
base = 0.45 * relevance        # lexical, semantic or hybrid blend
     + 0.20 * effective_priority   # M2 priority (importance × confidence × decay)
     + 0.15 * confidence           # stored accuracy
     + 0.10 * freshness            # M2 decay factor
     + 0.10 * type_preference      # documented type table

final = base
      * (1 - 0.50 * conflict_penalty)     # 1.0 superseded, 0.5 unresolved, 0.0 winner
      * (1 - 0.70 * redundancy_penalty)   # 1.0 duplicate of a higher-ranked memory
```

- The five component weights must sum to `1.0` (validated), so `base` is a convex
  combination and the score is bounded in `[0, 1]`.
- `relevance` is `lexical` for lexical mode, `semantic` for semantic mode, and
  `(1 - w) * lexical + w * semantic` for hybrid mode, with `w` defaulting to
  `0.5` and configurable per request.
- The conflict penalty is derived from the M2 relationship graph, so a memory
  that lost a resolution is demoted but never hidden (and never deleted).
- All weights live in `RankingConfig` and are injectable per service instance.

### Lexical relevance

```
relevance = sum(idf(t) for matched query terms t) / sum(idf(t) for scored query terms)
idf(t)    = log((N + 1) / (df(t) + 1)) + 1
```

Bounded in `[0, 1]`: it is the share of the query's information mass (weighted by
rarity) that the memory covers. Terms that no memory contains are excluded from
both sides. The normalisation pipeline (tokenise → drop 1-character tokens →
stop words → suffix stripping) is the same one M2 uses.

### Candidate generation

- `lexical` mode queries an **in-memory inverted index** built from the
  searchable corpus, so only matching memories are scored.
- `semantic` / `hybrid` modes must score the whole (filtered) searchable corpus,
  because embeddings can match without shared tokens.
- The index is rebuilt when the corpus changes; the change check is a cheap
  indexed aggregate (`COUNT` + `MAX(updated_at)`), so queries without writes cost
  nothing extra. This trade-off (versus hooking every write path) is documented
  in `docs/architecture.md`.

### Duplicate suppression

During selection, a candidate that has a recorded `DUPLICATE_OF` link to an
already-selected memory is **suppressed, not deleted**: it is reported in the
`suppressed` array with its penalised score and a reason naming the
representative. Nothing is modified.

### Embedding providers (all optional)

| Provider | Behaviour |
| --- | --- |
| *(none, default)* | Lexical retrieval only. No embeddings, no server. |
| `hashed` | Deterministic, model-free, offline: normalised tokens hashed into an L2-normalised vector. Captures token overlap (including the morphological variants the normaliser handles); **cannot** capture synonymy or paraphrase. |
| `ollama` | Optional local embeddings from an Ollama server (`MEMFORGE_OLLAMA_EMBEDDING_MODEL`, `MEMFORGE_OLLAMA_ENDPOINT`, `MEMFORGE_OLLAMA_TIMEOUT_SECONDS`). Probed with a short timeout, cached briefly, and never required. |

Embeddings are cached in SQLite (`memory_embeddings`, one row per
`(memory_id, provider, model)` with dimensions and timestamps). No external
vector database is used.

### Example

```powershell
curl -s -X POST http://127.0.0.1:8000/api/v1/memories/search `
  -H 'content-type: application/json' `
  -d '{"query": "preferred backend language", "top_k": 5, "retrieval_mode": "hybrid"}'
```

```json
{
  "query": "preferred backend language",
  "retrieval_mode": "hybrid",
  "semantic_available": true,
  "candidates_considered": 4,
  "candidates_returned": 2,
  "candidates_suppressed": 1,
  "latency_ms": 3.1,
  "results": [
    {
      "memory_id": "...",
      "lexical_score": 1.0,
      "semantic_score": 0.61,
      "relevance": 0.81,
      "final_score": 0.75,
      "explanation": "final 0.7458 for query ... = relevance=0.81 (x0.45 -> ...) + ...",
      "conflicts": [
        {
          "target_memory_id": "...",
          "attribute": "preferred backend language",
          "resolution": "supersedes_other"
        }
      ]
    }
  ],
  "suppressed": [
    {"memory_id": "...", "final_score": 0.21, "reason": "duplicate_of ..., which ranks higher in this result set"}
  ]
}
```

## M4 context engineering

Retrieval answers **"which memories are relevant?"**. Context compilation answers a
different question: **"given everything that is relevant, what is the best context
an agent can actually be given, under a hard budget, and why?"**

```
MEMORY RETRIEVAL (M3)
        |
        v
CONTEXT COMPILER (M4)   ── budget, utility, suppression, ordering, provenance
        |
        v
BUDGETED AGENT CONTEXT
```

The compiler is **not** summarisation, **not** a chatbot and **not** another RAG
pipeline. It contains no LLM, no network call and no provider tokenizer. It is a
pure function of (retrieved candidates, policy, request, budget):

```
API  ->  ContextCompilerService  ->  MemoryRetrievalService  ->  repository / embeddings
```

The dependency arrow points one way. Retrieval stays M3's job: candidate
generation, relevance, ranking, conflict and duplicate metadata all come from
there. M4 adds only what retrieval cannot decide — *what fits*, *in what order*,
and *with what justification*.

### 1. Token estimation (`memforge-char-word/v1`)

Token counts are **estimates**, never provider tokenisations and never billing
data. The default estimator is standard-library-only:

```
A = ASCII code points      N = non-ASCII code points      W = whitespace-separated words

estimated = max(1, ceil(max(A / 4.0, W / 1.3) + N * 1.0))
```

- Both a character term and a word term are computed and the **larger** one is
  used, so the estimate follows whichever signal dominates (dense code vs prose).
- A non-ASCII code point is charged a full token — deliberately conservative for
  CJK, accented text and symbol-heavy prose, where real BPE tokenizers need *more*
  than one token per character, never fewer.
- Empty content costs `0`.
- Every estimate carries `exact: false` and the id of the estimator that produced
  it. The ratios are configurable (`TokenEstimatorConfig`), validated, and
  replacing the estimator does not change any other part of the compiler.

**The contract is about MemForge's own estimator**, not about any model:
`estimated_tokens <= token_budget` always holds.

### 2. Utility model

```
value = 0.40 * relevance            # M3 lexical/semantic/hybrid blend
      + 0.20 * effective_priority   # M2 importance × confidence × decay
      + 0.15 * confidence           # stored accuracy
      + 0.10 * freshness            # M2 decay factor
      + 0.15 * type_preference      # documented type table

cost_penalty = min(1, estimated_tokens / 120)
utility      = value * (1 - 0.35 * cost_penalty)
```

- The five weights are validated to sum to `1.0`, so `value` is a convex
  combination bounded in `[0, 1]`; `cost_penalty` and `cost_weight` are in
  `[0, 1]`, so `utility` is bounded too. A misconfiguration cannot inflate a
  score.
- Utility is **not** M3's `final_score`. M3 asks "how relevant is this memory?";
  M4 asks "how much of the remaining budget is this worth?". This is what stops
  the compiler from preferring the longest memory, or the shortest one.
- All weights are heuristic, global, documented and injectable
  (`ContextPolicy` / `MEMFORGE_CONTEXT_*`). They are **not** measured values and
  make no optimality claim.

### 3. Budget algorithm

1. Score every retrieval candidate into a `ContextCandidate` (value, cost,
   utility, rendered token cost, compression record).
2. **Admission filters** — `exclude_ids`, `include_ids`, memory-type preference
   and the utility floor. A single preferred type is pushed down into the M3 SQL
   filter; several are applied by the compiler.
3. **Redundancy suppression** — conflict losers and duplicate-group members.
4. **Order** by `(-utility, memory_id)`: a *total* order, so ties never depend on
   iteration order.
5. **Admit greedily** while `estimated_tokens <= remaining`.
6. A candidate that does not fit is **reported and skipped, never fatal** — the
   loop continues, which is exactly what lets a smaller, lower-utility memory use
   the leftover budget instead of wasting it.
7. **Last-resort render trim** — the rendered context (section headings
   included) is measured exactly; if the framing still overflows, the
   lowest-priority block is dropped until it fits. This step normally does not
   run and deliberately does *not* re-admit anything.

Worked example (`token_budget = 1000`, candidates 420 / 260 / 240 / 210 tokens):
the highest-utility candidate that fits is admitted, the next is tried against
the reduced remainder, and so on. Nothing is ever returned beyond the budget, and
every rejection says how many tokens it needed and how many were left.

### 4. Conflicts and duplicates

| Situation | Decision | Reason code |
| --- | --- | --- |
| M2 recorded `SUPERSEDES` | The loser never enters the context; the winner is named | `conflict_superseded` |
| Conflict still `unresolved` | Both stay candidates; M3's `0.5` penalty already demoted them | *(none)* |
| Recorded `DUPLICATE_OF` inside the candidate set | Highest utility (ties by id) represents the whole group | `duplicate_of` |
| Recorded duplicate whose partner is **not** a candidate | Kept — the compiler only suppresses redundancy it can see | *(none)* |

Nothing is ever deleted. A suppressed memory stays stored, stays readable through
the API and keeps its history. Because M3 already drops near-duplicates from its
result set, the compiler re-scores them through M3's own `explain()` entry point
— so `suppressed_candidates` is complete rather than "whatever happened to reach
M4", and every candidate has exactly one explained decision.

### 5. Ordering

Retrieval rank is *not* prompt order. Blocks are grouped into documented sections
and ordered by `(section_rank, -utility, memory_id)`:

| Rank | Section | Memory types | Rationale |
| --- | --- | --- | --- |
| 0 | `PROFILE` | `PROFILE`, `PREFERENCE` | stable identity and stated preferences first |
| 1 | `FACTS` | `FACT` | durable, authoritative statements |
| 2 | `RECENT EVENTS` | `EVENT` | what happened lately |
| 3 | `TASKS` | `TASK` | open work |
| 4 | `OBSERVATIONS` | `OBSERVATION` | lowest-confidence notes |

The type → section mapping and the section order are both configurable. The
rendered context is exactly what the budget is enforced against:

```
PROFILE
-------
- User prefers Python for backend work.

FACTS
-----
- preferred backend language = Python
```

### 6. Compression (deterministic, no LLM)

M4 never paraphrases. It only:

- **normalises whitespace** (runs collapse to a single space — meaning-preserving,
  and free, because the estimator already normalises before counting);
- **keeps salient sentences** of an over-long memory when `max_block_tokens` is
  set: segments are ranked by query-term coverage (ties: earliest position), kept
  while they fit the soft ceiling, then re-sorted into their original order.

Both are recorded next to the result: `original_text`, `compressed_text`,
`retained_segments`, both token figures and the steps applied. Compression
**does not guarantee semantic equivalence**, and `max_block_tokens` is a soft
target — a block that cannot be reduced is simply more expensive, and the budget
decides whether it fits.

### 7. Provenance and explanation

Every block carries `memory_id`, `memory_type`, `source`, `confidence`,
`importance`, `relevance`, `retrieval_score`, `utility_score`,
`estimated_tokens`, the compression record and the conflicts it participates in.
Every candidate — selected or not — appears once in `selection_trace`:

```json
{
  "memory_id": "…",
  "selected": false,
  "reason": "insufficient_budget",
  "detail": "estimated cost 14 tokens exceeds the remaining budget of 3 tokens",
  "utility_score": 0.5777,
  "estimated_tokens": 14,
  "remaining_tokens": 3
}
```

Reasons: `selected`, `duplicate_of`, `conflict_superseded`,
`below_minimum_score`, `memory_type_filtered`, `excluded_by_id`,
`not_in_include_ids`, `insufficient_budget`. No explanation is LLM-generated —
every `detail` is built from the numbers that produced the decision.

### 8. Example

```powershell
curl -s -X POST http://127.0.0.1:8000/api/v1/context/compile `
  -H 'content-type: application/json' `
  -d '{"query": "preferred backend language", "token_budget": 1200, "top_k": 20}'
```

```json
{
  "query": "preferred backend language",
  "token_budget": 1200,
  "estimated_tokens": 1047,
  "remaining_tokens": 153,
  "within_budget": true,
  "utilization": 0.8725,
  "candidates_considered": 31,
  "selected_count": 9,
  "suppressed_count": 8,
  "retrieval_mode": "hybrid",
  "context_text": "PROFILE\n-------\n- User prefers Python for backend work.\n\nFACTS\n-----\n- …",
  "blocks": [
    {
      "memory_id": "…",
      "memory_type": "PROFILE",
      "source": "user:revanth",
      "confidence": 0.9,
      "importance": 0.8,
      "content": "User prefers Python for backend work.",
      "rendered_content": "User prefers Python for backend work.",
      "section": "PROFILE",
      "order": 0,
      "relevance": 0.91,
      "retrieval_score": 0.78,
      "utility_score": 0.87,
      "estimated_tokens": 124,
      "compression": {
        "original_text": "User prefers Python for backend work.",
        "compressed_text": "User prefers Python for backend work.",
        "applied": [],
        "original_tokens": 124,
        "compressed_tokens": 124,
        "tokens_saved": 0,
        "truncated": false,
        "retained_segments": []
      },
      "conflicts": []
    }
  ],
  "selection_trace": [],
  "suppressed_candidates": [],
  "explanation": {
    "estimator": "memforge-char-word/v1",
    "estimator_formula": "memforge-char-word/v1: estimated = max(1, ceil(max(…)))",
    "utility_formula": "utility = value * (1 - 0.35 * min(1, tokens/120)) where value = …",
    "ordering_policy": "section-then-utility/v1",
    "section_order": ["PROFILE", "FACTS", "RECENT_EVENTS", "TASKS", "OBSERVATIONS"],
    "selection_counts": {"selected": 9, "insufficient_budget": 8},
    "notes": [
      "token counts are MemForge estimates …, not provider tokenisations …",
      "context is compiled with no LLM: content is never paraphrased or summarised",
      "…"
    ]
  }
}
```

(The figures above illustrate the shape of the response; the demo prints a real
run with real numbers — see [Demo](#demo).)

## M5: agent integration + custom API

M5 adds three things on top of the compiled context: a **provider interface**, a
**bounded memory-aware agent**, and an **MCP server**.

> **The mock provider is the mandatory reproducible mode.** Everything below —
> the test suite, the demo, the HTTP smoke test and the MCP self-test — runs with
> `mock`, offline, with no API key. A real API is opt-in and configured only
> through environment variables.

### 1. The provider interface

```
app/domain/llm.py            LLMProvider (ABC), LLMRequest, LLMResponse,
                             LLMUsage, redaction helpers, error kinds
app/services/llm/mock_provider.py    MockLLMProvider  (default, offline)
app/services/llm/custom_provider.py  CustomLLMProvider + CustomLLMConfig
app/services/llm/factory.py          build_llm_provider(settings)
```

| Operation | Returns |
| --- | --- |
| `generate(request)` | `LLMResponse(text, provider, model, usage, latency_ms, finish_reason, response_id, request_metadata)` |
| `generate_structured(request)` | the same, plus `structured` — a parsed JSON object validated against `request.response_schema` |

Three rules are structural, not conventions:

- **Token counts are never invented.** `LLMUsage` is `None` unless the provider
  sent a usage block. An absent block is recorded as "unknown", never as `0`.
- **Secrets are structurally absent.** The key lives on `CustomLLMConfig` with
  `repr=False` and is never copied into a response, an error, a log record or a
  header dump.
- **Failures are typed.** Every failure is an `LLMProviderError` subclass
  carrying `kind` and `retryable`; the API layer maps `kind` → HTTP status.

### 2. The custom API contract

Configuration (environment only — never a file, never the database):

| Variable | Default | Meaning |
| --- | --- | --- |
| `MEMFORGE_LLM_PROVIDER` | `mock` | `mock` or `custom` |
| `MEMFORGE_LLM_BASE_URL` | *(empty)* | Base URL of an OpenAI-compatible API, e.g. `https://host/v1` |
| `MEMFORGE_LLM_API_KEY` | *(empty)* | Bearer credential. **Never logged, never persisted, never returned** |
| `MEMFORGE_LLM_MODEL` | *(empty)* | Model identifier sent in the request |
| `MEMFORGE_LLM_TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `MEMFORGE_LLM_MAX_RETRIES` | `2` | Extra attempts after the first (`0` disables retrying) |
| `MEMFORGE_LLM_RETRY_BACKOFF_SECONDS` | `0.5` | Linear backoff base |
| `MEMFORGE_LLM_CHAT_PATH` | `/chat/completions` | Path of the compatible contract |
| `MEMFORGE_LLM_SEND_AUTHORIZATION` | `true` | Set `false` for an endpoint needing no credential |

Wire format — the widely adopted "OpenAI-compatible" chat-completions shape. No
vendor is named or hard-coded anywhere; the base URL, path and model are all
configuration.

```http
POST {base_url}{chat_path}
Authorization: Bearer <api key>
Content-Type: application/json

{"model": "...", "messages": [{"role": "user", "content": "..."}],
 "temperature": 0.0, "max_tokens": 512,
 "response_format": {"type": "json_schema",
                     "json_schema": {"name": "...", "schema": {...}, "strict": true}}}
```

The first choice's `message.content` is the answer; `usage` is used only when
present. Failure normalisation:

| Condition | Error | HTTP | Retryable |
| --- | --- | --- | --- |
| missing base URL / model / key | `LLMNotConfiguredError` | `503` | no |
| timeout | `LLMTimeoutError` | `504` | yes |
| 401 / 403 | `LLMAuthenticationError` | `502` | no |
| 429 | `LLMRateLimitError` | `429` | yes |
| 5xx | `LLMProviderError` | `502` | yes |
| other 4xx | `LLMProviderError` | `502` | no |
| connection failure | `LLMTransportError` | `502` | yes |
| non-JSON body, no choices, no content, bad JSON | `LLMProtocolError` | `502` | no |

Retries are **bounded** by `max_retries` and only cover retryable kinds. Nothing
is retried forever, and a 4xx is never retried.

**Safety.** The key is redacted from every provider-generated string
(`redact_text`); auth headers are dropped rather than masked; `user:password@`
embedded in a base URL is stripped from anything reported. Selecting
`llm_provider=custom` without a base URL is a **startup error**, not a silent
fallback to the mock — MemForge never claims a model answered when it did not.

### 3. The agent workflow

One bounded pipeline. No planner, no tool loop, no autonomous behaviour:

```
user request
  → memory gate            deterministic heuristic (app/domain/agent.py)
  → recall                 M3, through the operations facade
  → compile context        M4, hard token budget
  → provider call          mock or custom
  → answer
  → candidate extraction   provider, structured output
  → store                  M1, only when explicitly permitted
```

The **memory gate** is a documented heuristic, not a classifier: short greetings
skip memory, and `use_memory=false` disables it outright. If recall finds
nothing with a non-zero relevance, the turn reports `no_matches` and skips
compilation rather than re-running retrieval to produce the same empty result.

**Writes are opt-in.** `store_candidates` defaults to `false`. Model output is
treated as *untrusted input*: the provider may propose memories, MemForge reports
them, and nothing is written until the caller explicitly permits it. A malformed
proposal is reported with a reason (`empty_content`, `unknown_type`) instead of
being stored or silently dropped.

### 4. API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/v1/agent/respond` *(M5)* | One memory-aware turn: `request`, `token_budget`, `top_k`, `use_memory`, `context_only`, `extract_candidates`, `store_candidates`. |
| `POST` | `/api/v1/memories/remember` *(M5)* | Store a memory and report its M2 near-duplicates. |
| `GET` | `/api/v1/traces` *(M6)* | Recent trace headers. Filters: `kind`, `limit` (1-200, default 20). |
| `GET` | `/api/v1/traces/{id}` *(M6)* | One trace with its attributes and derived summary. |
| `GET` | `/api/v1/traces/{id}/events` *(M6)* | That trace's events, in recorded sequence order. |
| `GET` | `/api/v1/metrics` *(M6)* | Operational counters derived from recorded events. |
| `GET` | `/api/v1/evaluations/latest` *(M6)* | The latest run of the shipped evaluation dataset. |
| `GET` | `/api/v1/evaluations/{name@version}` *(M6)* | One evaluation run by reference (also accepts `memory-core`). |

```powershell
curl -s -X POST http://127.0.0.1:8000/api/v1/agent/respond `
  -H 'content-type: application/json' `
  -d '{"request": "Which backend language should I use?", "token_budget": 800}'
```

The response carries the answer, the memory-gate decision, the compiled context,
block provenance, candidate memories (reported, not stored), and an
`operations` array with one record per step: `operation`, `success`, `latency_ms`,
`provider`, `model`, `usage` (`null` unless reported), `context_estimated_tokens`,
`memories_recalled`, `memories_stored`, `error_kind`. **No response field can
hold a credential** — the schema simply has no place for one.

### 5. MCP server

Built on the official Python SDK (`mcp`). Every tool is a thin adapter over
`MemoryOperationsService`; none of them touch a language-model provider.

| Tool | Purpose | MCP annotation |
| --- | --- | --- |
| `memory.store` | store a memory (M1) | write |
| `memory.search` | ranked search (M3) | read-only |
| `memory.recall` | agent recall with a relevance gate (M3) | read-only |
| `memory.update` | partial update (M1) | write |
| `memory.forget` | soft delete (M1) | destructive, idempotent |
| `memory.compile_context` | budgeted context (M4) | read-only |
| `memory.explain` | per-memory ranking explanation (M3) | read-only |

- **Input**: flat typed parameters, so the SDK derives a strict JSON schema
  (required fields, bounds, enums) a client can validate before calling.
- **Output**: a typed model, so an output schema is published *and* validated.
- **Errors**: a MemForge exception becomes the same envelope the REST API uses —
  `{"error": {"code", "message", "details"}}` — returned with `isError` set.
  Schema violations are refused by the SDK before the tool body runs.
- Tool names use dots as specified; the SDK accepts them and logs a conformance
  warning.

### 6. Optional live API testing

`scripts/llm_smoke_test.py` is **never** part of the test suite and refuses to
start without real credentials:

```powershell
$env:MEMFORGE_LLM_BASE_URL = "https://your-host/v1"
$env:MEMFORGE_LLM_MODEL    = "your-model"
$env:MEMFORGE_LLM_API_KEY  = "your-key"
python scripts/llm_smoke_test.py
```

It verifies connectivity, a text round-trip, a structured round-trip, a
memory-aware request, and how usage is reported — and it **never prints the
key**, not even in an error.

## M6: evaluation + observability

Milestone 5 made MemForge *work*. Milestone 6 makes it **answerable**: two
questions, two subsystems.

1. **"Is MemForge making good memory decisions?"** -> the evaluation layer.
2. **"What exactly happened during an operation?"** -> the observability layer.

Both are MemForge-specific, deterministic, offline and free. There is no judge
model, no paid API and no external observability service.

### Why MemForge has its own evaluation layer

This is **not** EvalForge and **not** a generic LLM evaluation platform. A
general-purpose LLM evaluation answers "is this output good?". MemForge never
generates prose for a user-facing answer; it *decides* which memories to recall,
which to suppress and what to compile. So the right question is much narrower
and much more testable:

> Given this corpus and this query, did the memory system make the decision its
> own documented rules say it should make?

That question is answered by counting real decisions, not by scoring text.
Evaluating the text would also require a judge model, which would make the
evaluation non-deterministic, network-dependent and costly -- and would prove
nothing about the *memory* layer.

### 1. The dataset

`app/evaluation/datasets/memory-core-v1.json` is version-controlled, human
written and parsed into strict typed models. It is plain text so a change to a
case is a reviewable diff, and **no database is involved**: a dataset is an
input, not a resource.

The shipped dataset is `memory-core@v1`: **36 cases** across six categories.

| Category | Cases | What it exercises |
|---|---|---|
| `retrieval` | 7 | relevant memory exists; irrelevant memories exist; multiple similar memories; recent vs old; confidence differences |
| `duplicate` | 5 | near-duplicates; exact duplicates; multiple duplicate candidates; unrelated memories that must survive; sub-threshold similarity |
| `conflict` | 6 | clear conflict; resolved / unresolved; newer vs older; higher vs lower confidence; different attributes must not collide |
| `context` | 7 | tight budget; large candidate pool; duplicates in context; insufficient budget; many small vs one large; provenance; conflicting pair under budget |
| `lifecycle` | 5 | create, update, archive, forget, double-forget no-op, history preservation |
| `agent` | 6 | relevant memory recalled; irrelevant memory must not dominate; storage disabled; storage explicitly enabled; malformed candidate rejected; small talk |

Each case declares:

```json
{
  "id": "dup-001",
  "title": "A near-duplicate memory is suppressed in favour of the original",
  "category": "duplicate",
  "action": "search_with_duplicates",
  "query": "editor theme preference",
  "fixture": [
    {"label": "original", "content": "The user prefers the dark theme in every code editor.",
     "memory_type": "PREFERENCE", "confidence": 0.9, "importance": 0.9},
    {"label": "restated", "content": "The user prefers the dark theme in all code editors.",
     "memory_type": "OBSERVATION", "confidence": 0.4, "importance": 0.2}
  ],
  "expected": {
    "required": ["original"],
    "duplicates_suppressed": ["restated"],
    "metrics": ["retrieval_recall", "retrieval_precision", "top_k_hit_rate",
                "duplicate_suppression_accuracy"]
  }
}
```

Expectations are written against **labels**, not generated ids, so a case stays
readable and stable across runs. `age_minutes` lets a fixture say "this memory is
a month old" without depending on when the suite happened to run.

### 2. How a case runs

```
EvaluationService
    ↓  builds a real, isolated MemForge stack per case
M1 MemoryService / M3 RetrievalService / M4 Compiler / M5 AgentService
    ↓  produces real results
a handful of counts
    ↓
CaseResult -> MetricResult -> EvaluationSummary
```

Three rules make the numbers trustworthy:

1. **Nothing is re-implemented.** A case never scores memories itself. It asks
   M3 to search, M2 to detect and resolve, M4 to compile, M5 to run a turn -- then
   it counts what came back. There is no second scoring path that could agree
   with the real one by accident.
2. **Every case is isolated.** Fresh SQLite file, manual clock and a private
   deterministic id namespace per case. Cases cannot leak state into each other
   and two runs are identical.
3. **A metric never appears without a count.** A case that declares a metric it
   could not measure *fails*, rather than contributing a silent zero.

### 3. The metrics

Every metric is a ratio of two integers the evaluation produced. The formulas are
in `app/domain/evaluation.py::METRIC_DEFINITIONS` and printed by
`--verbose`, so a reader never has to trust a name.

| Metric | Formula |
|---|---|
| `retrieval_recall` | required memories that appeared / required memories |
| `retrieval_precision` | returned memories declared relevant / returned memories (undefined when nothing was returned, so not counted) |
| `top_k_hit_rate` | cases with a required memory in the first **3** ranked results / cases |
| `duplicate_suppression_accuracy` | expected duplicates that were absent / expected duplicates |
| `conflict_resolution_accuracy` | cases where M2 chose the expected winner / conflict cases |
| `stale_memory_avoidance` | superseded memories kept out of the final context / superseded memories |
| `context_budget_compliance` | compilations with `estimated_tokens <= token_budget` / compilations |
| `context_selection_accuracy` | required memories selected into the context / required memories |
| `context_provenance_completeness` | selected blocks naming one stored memory with content, type and section / selected blocks |
| `history_preservation` | lifecycle operations that produced the expected version count / lifecycle cases |
| `memory_write_policy_compliance` | turns that stored only when permitted / agent turns |
| `recall_relevance` | cases with no declared-irrelevant result (and every required one, when declared) / cases |

Two kinds of metric, deliberately separated:

- **Pass/fail** (recall, top-k, suppression, compliance, write policy): the case
  states what must happen, and missing it fails the case.
- **Graded** (precision): a partly-useful result set lowers the metric without
  failing the build, because "how much of what we returned was useful" is a
  quality signal, not a binary rule.

The regression gate is `case_score >= threshold`, defaulting to `1.0` (every
case must hold). `overall` folds in the graded metrics and is reported for
comparison, so a healthy build can still show `retrieval_precision < 1.0`
without the gate flapping.

### 4. The report and the gate

```powershell
python -m app.evaluation.run                  # the shipped dataset
python -m app.evaluation.run --verbose        # print every metric formula
python -m app.evaluation.run --json           # deterministic JSON alongside
python -m app.evaluation.run --threshold 0.95 # tolerate a known failure
python -m app.evaluation.run --dataset mine   # another file in datasets/
```

Exit codes: `0` pass, `1` evaluation failed, `2` the dataset could not be read.

Real output of the shipped dataset (numbers are measured, not illustrative):

```
==============================================================================
                              EVALUATION REPORT
==============================================================================
  Dataset      : memory-core@v1
  Cases        : 36
  Passed       : 36
  Failed       : 0
  Case score   : 1.000
  Metric score : 0.983
  Overall      : 0.991
  Threshold    : 1.000 (minimum case pass rate)
------------------------------------------------------------------------------
  Categories:
    agent        6/6
    conflict     6/6
    context      7/7
    duplicate    5/5
    lifecycle    5/5
    retrieval    7/7

  Metrics (measured):
    conflict_resolution_accuracy       1.000  (5/5)
    context_budget_compliance          1.000  (13/13)
    context_provenance_completeness    1.000  (28/28)
    context_selection_accuracy         1.000  (4/4)
    duplicate_suppression_accuracy     1.000  (5/5)
    history_preservation               1.000  (5/5)
    memory_write_policy_compliance     1.000  (6/6)
    recall_relevance                   1.000  (8/8)
    retrieval_precision                0.793  (23/29)
    retrieval_recall                   1.000  (23/23)
    stale_memory_avoidance             1.000  (5/5)
    top_k_hit_rate                     1.000  (16/16)
```

### 5. What the evaluation can and cannot prove

The report prints this on every run, because a score without its boundary is
easy to over-read:

- Every number is a count over real MemForge services.
- Each case proves only the behaviour its own fixture encodes, on that fixture,
  with the configured defaults.
- A pass is **not** a claim about "memory intelligence" in any general sense.
- **No LLM-as-a-judge was involved** in producing any number.
- Scores are not comparable across datasets or configurations: a different
  dataset is a different measurement.

### 6. Observability: the trace model

Two questions, two models:

- **What happened?** -> `Trace` and `TraceEvent`: an append-only, deterministic
  tree of operational events correlated by one `trace_id`.
- **How well did it work?** -> `TraceSummary` and `OperationMetrics`: counters
  derived from those events by *counting*, never by guessing.

A trace has `trace_id`, `kind`, `operation`, `started_at` and safe attributes.
Every event carries `trace_id`, `event_id`, a per-trace `sequence`, `kind`,
`operation`, `status`, `timestamp`, an optional `parent_id`, an optional measured
`duration_ms`, safe `metadata` and an optional `detail`.

The lifecycle an agent turn emits, always in this order:

```
Agent Request
 |- memory gate
 |- retrieval           (start -> complete, parented)
 |- context compilation (start -> complete, parented)
 |- llm.generate        (request -> response, parented)
 |- memory.candidate
 |- memory.write
 `- response
```

`sequence` makes the order **total**: two events can share an instant, and the
sequence is what an operator can rely on. `parent_id` is what turns a flat list
into an inspectable tree. One recorded turn looks like this:

```
TRACE demo-trace-0001  (agent_request / agent.respond)
   0  request                  ok
   1  retrieval_start          ok
   2  retrieval_complete       ok        1.500 ms <- e0001
   3  context_compile_start    ok
   4  context_compile_complete ok        1.500 ms <- e0003
   5  llm_request              ok
   6  llm_response             ok        1.500 ms <- e0005
   7  llm_request              ok
   8  llm_response             ok        1.500 ms <- e0007
   9  memory_gate              ok
  10  memory_candidate         ok
  11  memory_write             skipped
  12  response                 ok
```

### 7. Safe observability: what is never recorded

Traces contain counts, identifiers, modes, token figures and payload **sizes**.
They never contain:

| Never recorded | Recorded instead |
|---|---|
| API keys, `Authorization` headers, passwords, secrets | nothing; credential-shaped keys are dropped and registered secrets are redacted |
| prompts and answers | `input_characters`, `output_characters`, `request_characters` |
| chain-of-thought or reasoning | nothing; only what operation ran, on how many items, with what outcome and how long |
| memory content | `memory_id`, `memory_type`, `status`, `version` |

Two independent mechanisms enforce this:

1. **A deny-list.** `safe_metadata()` drops any key named `api_key`,
   `authorization`, `token`, `secret`, `password`, `prompt`, `answer`,
   `content`, `chain_of_thought` ... *before* the value is even inspected, so an
   accidental `metadata={"api_key": ...}` cannot be persisted.
2. **Registered secrets.** `ObservabilityService.register_secret(key)` makes the
   service redact that literal out of any string it records, wherever it came
   from. The REST API and the MCP server both register `MEMFORGE_LLM_API_KEY`
   at startup.

Auth-bearing headers additionally go through M5's `redact_mapping`.

### 8. Latency and usage: measured, never invented

- **Latency** is *measured* with an injectable monotonic timer
  (`time.perf_counter` in production, a deterministic stepper in the demo) and
  reported on the closing event of each operation. It is **not** a benchmark and
  no test compares latency to a fixed number -- the HTTP smoke test asserts that
  a duration is present and non-negative, then moves on.
- **Provider usage** is echoed **only when the provider sent it**. The mock
  reports none, so the trace records `usage_reported: false` and no token figure
  at all. Characters are never converted into tokens and no cost is ever derived.
  The metrics endpoint exposes `usage_reported`, which is `0` unless a real
  provider actually reported usage.
- **Token figures in a trace** come from M4's own estimator and are labelled
  `estimated_tokens`; they are never confused with provider usage.

### 9. Persistence

Two tables in the **same SQLite file** as everything else (migration
`0004_observability`). No external observability dependency, no second database,
no telemetry service.

```
traces(trace_id PK, kind, operation, started_at, attributes)
trace_events(event_id PK, trace_id FK, sequence, kind, operation, status,
             timestamp, parent_id, duration_ms, metadata, detail)
  UNIQUE (trace_id, sequence)      -- the order is total and reproducible
```

Indexed by `(trace_id, sequence)`, by `(operation, status)` and by
`(started_at, trace_id)`. Events are **append-only** -- never updated, never
deleted -- so an audit survives a restart. A trace carries a `kind` CHECK
constraint, so an unknown kind cannot be written at all.

The store is deliberately dumb: it stores what it is given. Redaction is the
*service's* job, because a store that silently rewrites its input is a store you
cannot trust when auditing.

### 10. Traceability

One `trace_id` correlates the whole chain:

```
agent request -> memory gate -> retrieval -> context compiler
             -> provider -> memory candidate -> memory write -> response
```

Correlation is ambient and scoped: `ObservabilityService.use_trace(trace_id)`
binds the id through a `ContextVar`, so two concurrent requests cannot capture
each other's events and the binding is restored even when a call raises.

Instrumentation is **decorators**, not edits. M1-M5 know nothing about traces:

| Decorator | Records |
|---|---|
| `TracedMemoryService` | `MEMORY_WRITE` / `_UPDATE` / `_FORGET` |
| `TracedRetrievalService` | `RETRIEVAL_START` / `_COMPLETE` with real mode, counts and latency |
| `TracedContextCompilerService` | `CONTEXT_COMPILE_START` / `_COMPLETE` with selection and suppression counts |
| `TracedLLMProvider` | `LLM_REQUEST` / `LLM_RESPONSE` with sizes and *reported* usage |
| `AgentTracer` | opens/closes the turn's trace, reports the gate and candidate events |
| `McpTracer` / `RequestTracer` | one trace per MCP tool call or REST write |

A decorator delegates first and reports afterwards, so a traced turn returns
exactly what an untraced one does -- a property the test suite asserts by
comparing the two.

### 11. Metrics

`GET /api/v1/metrics` returns counters derived by counting recorded events:

| Group | Counters |
|---|---|
| traces | `agent_request`, `mcp_request`, `api_request`, ... |
| operations | one per completed operation |
| statuses | `ok`, `skipped`, `error` |
| memory | `written`, `created`, `updated`, `forgotten`, `candidates`, `recalled`, `duplicate_candidates`, `writes_skipped`, `memory_ids_touched` |
| context | `candidates`, `selected`, `suppressed`, `conflicts_suppressed`, `over_budget` |
| provider | `calls`, `failures`, `usage_reported`, `mcp_calls`, `mcp_failures` |
| top level | `usage_reported`, `failed_traces`, `event_count` |

A counter that was never incremented is **absent**, not zero: a zero that was
never measured is not a measurement.

### 12. API

All read-only. There is deliberately no "record a trace" endpoint -- traces come
from real operations, never from a caller asserting that one happened.

| Endpoint | Returns |
|---|---|
| `GET /api/v1/traces` | recent trace headers (filter by `kind`, paginate with `limit`) |
| `GET /api/v1/traces/{trace_id}` | one trace with its attributes and derived summary |
| `GET /api/v1/traces/{trace_id}/events` | its events, ordered by recorded sequence |
| `GET /api/v1/metrics` | the counters above, plus the event count they came from |
| `GET /api/v1/evaluations/latest` | the latest run of the shipped dataset |
| `GET /api/v1/evaluations/{name@version}` | one run by reference (also accepts `memory-core`) |

`POST /api/v1/agent/respond` and `POST /api/v1/memories/remember` return the
trace id they produced in the **`X-MemForge-Trace-Id`** header.

Unknown trace -> `404 TRACE_NOT_FOUND`. Unknown dataset -> `404
EVALUATION_NOT_FOUND`. Both use the existing error envelope. No response schema
has a field for a credential, a prompt or an answer, so no secret can leave
through this surface.

### 13. CLI

```powershell
python -m app.evaluation.run      # the evaluation gate
python -m app.observability.demo  # a real trace, event list and metric summary
python -m app.cli.demo            # the full M1-M6 walkthrough
```

`python -m app.observability.demo` prints two turns -- one that proposes a
candidate without permission (`memory.write SKIPPED`), one with permission
(`memory.write ALLOWED`) -- then the derived metrics. Both runs are
byte-identical: ids come from a sequential factory and timestamps from a manual
clock.

The main demo ends with steps 16-19: the trace of its agent turn, the evaluation
summary, the metrics summary, and the final state.

### 14. Dashboard preparation

M6 builds **no** dashboard; [M7](#m7-control-plane) is that milestone. M6 only
designed the response shapes a dashboard consumes, and they shipped unchanged:

- evaluation summaries with per-category and per-metric series, each carrying
  its own definition and raw numerator/denominator;
- per-case results with their counts and figures;
- traces, trace events and trace summaries;
- memory, context and provider counters;
- the failing cases, with the checks that did not hold.

Every one of those shapes is stable, sorted and credential-free, so M7 is a
presentation layer over data that already exists.

## M7: control plane

**MemForge's control plane is a visualisation layer over the real memory,
retrieval, context, agent, evaluation and observability APIs.**

It adds no AI capability, no memory of its own, no second storage and no
backend logic. Every number it renders was returned by MemForge for the corpus
that is currently in SQLite, and every button calls an endpoint that already
existed before the dashboard did.

### 1. Running it

Two terminals: the backend, then the dashboard.

```powershell
# terminal 1 - the API (from the repository root)
python -m uvicorn app.main:app --reload

# terminal 2 - the dashboard
cd dashboard
npm install
npm run dev            # http://127.0.0.1:5173
```

The dev server proxies `/api` and `/health` to `http://127.0.0.1:8000`, so the
browser talks to one origin and no CORS configuration is needed. Point it
somewhere else with:

```bash
VITE_MEMFORGE_API_TARGET=http://127.0.0.1:9000 npm run dev
```

For a production build:

```bash
npm run build          # -> dashboard/dist
npm run preview        # serves the build on :4173, proxying the same paths
```

Serve `dashboard/dist` with any static host as long as `/api/v1` and `/health`
reach MemForge: the client uses the same-origin relative prefix by default, so no
origin is baked into the bundle. For a backend on a different origin, build with
`VITE_MEMFORGE_API_BASE=https://memforge.example/api/v1`.

| Command | What it does |
| --- | --- |
| `npm run dev` | Vite dev server with the API proxy. |
| `npm run build` | Typecheck, then build `dashboard/dist`. |
| `npm run preview` | Serve the production build with the same proxy. |
| `npm run typecheck` | `tsc --noEmit`, strict. |
| `npm test` | The frontend suite (Vitest + Testing Library). |

### 2. Architecture

```
dashboard/src/
|-- api/
|   |-- client.ts     the only place that calls fetch: base URL, requests,
|   |                 parsing, the error envelope, the trace-id header
|   |-- types.ts      TypeScript mirrors of the Pydantic response contracts
|   `-- parse.ts      response-shape validation -> a "malformed response" state
|-- hooks/
|   |-- useResource.ts  loading / ready / error, with in-flight dedupe + short TTL
|   `-- useTheme.ts    dark/light, remembered locally
|-- lib/
|   |-- format.ts     formatting only: no backend value is recomputed here
|   |-- redact.ts     second line of defence for anything a trace carries
|   |-- traceTree.ts  events -> the tree the trace viewer draws
|   `-- constants.ts  the backend's own vocabulary, as captions
|-- components/       shell, cards, badges, tables, timeline, budget bar
|-- pages/            the eight areas
`-- styles/           hand-written CSS: design tokens, layout, components
```

Two rules the code follows:

1. **One client.** No component calls `fetch`. Base URL, request shape, response
   validation and error classification live in `src/api/client.ts`, so a change
   to the backend contract has exactly one place to be reflected in.
2. **No second opinion.** The dashboard never recomputes a score, a token count,
   a ranking or an evaluation result. A token figure that disagreed with the
   backend's estimator would make the hard-budget guarantee meaningless, so the
   compiler page renders `estimated_tokens`, `remaining_tokens` and
   `within_budget` as the backend reported them and names the estimator.

### 3. Pages

| Page | Route | Backend endpoints | What it is for |
| --- | --- | --- | --- |
| **Dashboard** | `/` | `GET /memories?status=...`, `GET /metrics`, `GET /evaluations/latest`, `GET /traces` | Live summary, the 30-60 s demo path, latest traces and evaluation. |
| **Memories** | `/memories`, `/memories/{id}` | `GET /memories`, `GET/PATCH/DELETE /memories/{id}`, `/history`, `/intelligence`, `/duplicates`, `/conflicts`, `/relationships` | The stored, versioned corpus: filters, pagination, the v1 -> v2 -> v3 history, computed intelligence, duplicate/conflict evidence. |
| **Retrieval Lab** | `/retrieval` | `POST /memories/search` | Run a query and read, factor by factor, why rank #1 ranked first. Suppression is shown, not hidden. |
| **Context Compiler** | `/context` | `POST /context/compile` | Compile under a hard budget: candidates -> selected -> suppressed -> final context, with a reason for every candidate. |
| **Agent Playground** | `/agent` | `POST /agent/respond` | One memory-aware turn: gate, recall, compiled context, provider, answer, candidates, trace id. |
| **Traces** | `/traces`, `/traces/{id}` | `GET /traces`, `GET /traces/{id}`, `GET /traces/{id}/events` | The recorded lifecycle as a tree, with safe metadata only. |
| **Evaluation** | `/evaluation` | `GET /evaluations/latest` | The deterministic 36-case benchmark, every metric with its counts, failing cases in full. |
| **Metrics** | `/metrics` | `GET /metrics`, `GET /traces?limit=200` | The operational counters the backend actually counted. |

### 4. Demo flow (30-60 seconds)

Every step is a link on the Dashboard, so the path never has to be memorised.

1. **Dashboard** - real counts for active/archived/forgotten memories, retrieval
   and compilation counters, the evaluation score and the latest traces.
2. **Retrieval Lab** - press *Run retrieval*. The default query
   (`preferred backend language memforge`) matches the corpus the smoke test and
   the demo seed, so there is a real ranking immediately.
3. Inspect **why #1 ranked first**: the five weighted factors with their raw
   values, weights, contributions and the backend's own sentence for each.
4. **Context Compiler** - the budget field defaults to `160`, small enough that
   the budget has something to say. Selected and suppressed candidates appear
   with their reason, and `estimated tokens <= token budget` is shown from the
   backend's figures.
5. **Agent Playground** - *Run agent turn*. The default request asks a question
   the corpus can answer and proposes a memory to store, with storage off.
6. **Trace** - the card in the answer links straight to the turn's timeline.
7. **Evaluation** - the real score, including `retrieval_precision = 0.793`.

### 5. Modes

- **Mock / default** - the backend runs with `MEMFORGE_LLM_PROVIDER=mock` (the
  default). The Agent page says `mock` and shows the deterministic answer, and
  the usage card reports `none` rather than a number: the mock reports no usage
  and the dashboard does not derive one.
- **Custom API mode** - set `MEMFORGE_LLM_PROVIDER=custom`,
  `MEMFORGE_LLM_BASE_URL`, `MEMFORGE_LLM_API_KEY` and `MEMFORGE_LLM_MODEL` for
  the backend. The Agent page then shows the provider and model the *API*
  returned, and the usage card shows only the token counts the provider actually
  sent. The dashboard never reads a credential.

### 6. Security

- **No secret is ever in the frontend.** The only configuration is
  `VITE_MEMFORGE_API_BASE` / `VITE_MEMFORGE_API_TARGET`, which are URLs. The
  client sends `credentials: "omit"` and builds no `Authorization` header: the
  API is unauthenticated by design, and a browser bundle is the wrong place for
  a credential. A test asserts that no request carries one, and the dashboard
  smoke test greps the built bundle for header-shaped markers.
- **No environment variable is exposed to Vite.** No `envPrefix` is widened and
  no dashboard source file names `MEMFORGE_LLM_API_KEY`.
- **Traces are re-rendered through a second filter.** The backend records no
  prompt, answer or credential; the trace viewer additionally drops
  credential-shaped keys and redacts credential-shaped values before rendering,
  so even a hostile payload cannot reach the screen.
- **No prompt and no chain-of-thought is displayed.** The agent response's
  `messages` array is never rendered; only operational metadata is.

### 7. Testing

```bash
cd dashboard
npm run typecheck      # strict tsc
npm test               # the frontend suite
npm run build          # typecheck + production build
```

| File | Focus |
| --- | --- |
| `tests/parse.test.ts` | Response-shape validation: what a drifted backend payload must do. |
| `tests/client.test.ts` | Base URL, request construction, error envelope, malformed body, no auth header. |
| `tests/format.test.ts` | Formatting only; no backend value is recomputed. |
| `tests/libs.test.ts` | Redaction filters and the trace tree. |
| `tests/navigation.test.tsx` | All eight routes, the sidebar, the 404 page, the API badge, the theme toggle. |
| `tests/dashboard.test.tsx` | Live counters, traces, evaluation score, loading, empty, unavailable, unreachable, malformed. |
| `tests/memories.test.tsx` | Explorer filters/sorting/pagination and the six detail panels, including the forgotten-memory history. |
| `tests/retrieval.test.tsx` | A real query, the ranking, the explanation, suppression, degradation, errors. |
| `tests/context.test.tsx` | The budget guarantee, the pipeline, selection reasons, suppression, provenance. |
| `tests/agent.test.tsx` | Gate, provider, mock vs custom, context, candidates, trace id, failure surfacing. |
| `tests/traces.test.tsx` | Trace list, lifecycle rail, nested timeline, safe metadata, secret redaction. |
| `tests/observability.test.tsx` | Evaluation figures (including the real 0.793 precision), failing cases, metrics counters. |
| `tests/responsive.test.tsx` | The shell structure, landmarks, scroll containers and label coverage. |

The fixtures in `tests/fixtures.ts` are transcriptions of **real** MemForge
responses captured from a live uvicorn run. They are used only inside the test
suite; the running dashboard never reads them.

A real-data smoke test drives the production build against a real backend:

```bash
python scripts/dashboard_smoke.py      # needs dashboard/dist; run npm run build first
```

It starts uvicorn on a throwaway database, serves `dashboard/dist` with an SPA
fallback and a proxy, seeds a corpus through the public API, and checks the
build, the bundle's safety, all eight routes and the whole demo flow. No browser
stack is involved: the project already verifies real integration at the transport
level, and this extends the same idea to the presentation layer.

### 8. Honest limitations

- **The dashboard adds no AI capability.** It renders decisions that M1-M6
  already made. Nothing it shows was computed in the browser.
- **Filters the API does not support are page-level, and say so.** The memory
  explorer's confidence/importance thresholds, text filter and sort order apply
  to the *loaded page*, because `GET /memories` has no parameter for them. A
  ranked search across the whole corpus is the Retrieval Lab's job.
- **No time series.** `GET /metrics` returns aggregate counters and the backend
  stores no historical buckets, so the Metrics page draws no chart. Plotting one
  would be inventing data.
- **Token figures are MemForge's estimates.** `memforge-char-word/v1`, not a
  provider tokenisation. The Context page names the estimator next to the
  figures rather than hiding it.
- **Latency is not a benchmark.** Every duration shown was measured at runtime on
  the operator's machine, and nothing is compared against a fixed number.
- **A forgotten memory answers 404.** `GET /memories/{id}` reports
  `MEMORY_NOT_FOUND` once a memory is soft-deleted. The detail page detects that,
  loads the surviving history instead, and explains why.
- **The evaluation measures the shipped fixtures.** A score of 0.991 says the
  system made the decisions its own rules prescribe, for those corpora and those
  queries - not that memory quality is solved.
- **No screenshot ships with this README.** None has been captured; add a real
  one before publishing rather than pasting a mockup.

## Local setup

Requirements: **Python 3.12+**. No database server, no API keys, no network access.

```powershell
# Windows / PowerShell (from the repository root)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

`fastapi`, `pydantic`, `pydantic-settings` and `uvicorn` are the runtime
dependencies; `pytest`, `httpx`, `mypy` and `ruff` are dev-only.

### Run the API

```bash
python -m uvicorn app.main:app --reload
```

Interactive docs: <http://127.0.0.1:8000/docs>.

### Configuration

Settings are read from the environment (prefix `MEMFORGE_`) or a local `.env`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `MEMFORGE_DATABASE_PATH` | `<repo>/data/memforge.sqlite` | SQLite file. Parent directories are created automatically. |
| `MEMFORGE_API_PREFIX` | `/api/v1` | Root path for the memory routes. |
| `MEMFORGE_APP_NAME` | `MemForge` | Service name in `/health`. |
| `MEMFORGE_LOG_LEVEL` | `INFO` | Logging verbosity. |
| `MEMFORGE_DECAY_HALF_LIFE_DAYS` | `30` | Days for the recency/decay factor to halve. |
| `MEMFORGE_SIMILARITY_THRESHOLD` | `0.6` | Dice similarity at/above which two memories are near-duplicates. |
| `MEMFORGE_VALUE_SIMILARITY_THRESHOLD` | `0.5` | Value similarity above which two values count as the same value. |
| `MEMFORGE_RECURRENCE_SATURATION` | `3` | Near-duplicates that saturate the recurrence factor. |
| `MEMFORGE_ACCESS_SATURATION` | `5` | Recorded accesses that saturate the access factor. |
| `MEMFORGE_DEFAULT_TOP_K` | `10` | Results returned by default. |
| `MEMFORGE_DEFAULT_SEMANTIC_WEIGHT` | `0.5` | Semantic blend weight in hybrid mode. |
| `MEMFORGE_EMBEDDING_PROVIDER` | *(none)* | `none`, `hashed` or `ollama`. |
| `MEMFORGE_EMBEDDING_DIMENSIONS` | `256` | Dimensions for the `hashed` provider. |
| `MEMFORGE_OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model. |
| `MEMFORGE_OLLAMA_ENDPOINT` | `http://127.0.0.1:11434` | Ollama server. |
| `MEMFORGE_OLLAMA_TIMEOUT_SECONDS` | `5` | Ollama request timeout. |
| `MEMFORGE_DEFAULT_CONTEXT_TOKEN_BUDGET` *(M4)* | `1200` | Budget used when a compile request does not name one. |
| `MEMFORGE_MAX_CONTEXT_TOKEN_BUDGET` *(M4)* | `16000` | Largest budget a single compilation may request. |
| `MEMFORGE_CONTEXT_TOP_K` *(M4)* | `20` | Retrieval candidates handed to the compiler by default. |
| `MEMFORGE_CONTEXT_RELEVANCE_WEIGHT` *(M4)* | `0.40` | Utility weight of relevance. |
| `MEMFORGE_CONTEXT_PRIORITY_WEIGHT` *(M4)* | `0.20` | Utility weight of the M2 effective priority. |
| `MEMFORGE_CONTEXT_CONFIDENCE_WEIGHT` *(M4)* | `0.15` | Utility weight of confidence. |
| `MEMFORGE_CONTEXT_FRESHNESS_WEIGHT` *(M4)* | `0.10` | Utility weight of freshness. |
| `MEMFORGE_CONTEXT_TYPE_WEIGHT` *(M4)* | `0.15` | Utility weight of the type preference. |
| `MEMFORGE_CONTEXT_COST_WEIGHT` *(M4)* | `0.35` | Strength of the token-cost penalty. |
| `MEMFORGE_CONTEXT_COST_REFERENCE_TOKENS` *(M4)* | `120` | Token cost at which that penalty saturates. |
| `MEMFORGE_CONTEXT_MIN_CANDIDATE_SCORE` *(M4)* | `0.0` | Utility floor for a candidate to be eligible. |
| `MEMFORGE_CONTEXT_MAX_BLOCK_TOKENS` *(M4)* | *(none)* | Soft per-block ceiling for sentence selection. |
| `MEMFORGE_TOKEN_CHARS_PER_TOKEN` *(M4)* | `4.0` | Estimator ratio: characters per token. |
| `MEMFORGE_TOKEN_WORDS_PER_TOKEN` *(M4)* | `1.3` | Estimator ratio: words per token. |
| `MEMFORGE_TOKEN_NON_ASCII_TOKENS_PER_CHAR` *(M4)* | `1.0` | Estimator ratio: tokens charged per non-ASCII character. |
| `MEMFORGE_LLM_PROVIDER` *(M5)* | `mock` | `mock` (default, offline) or `custom`. |
| `MEMFORGE_LLM_BASE_URL` *(M5)* | *(empty)* | Base URL of an OpenAI-compatible API. Required when the provider is `custom`. |
| `MEMFORGE_LLM_API_KEY` *(M5)* | *(empty)* | Bearer credential. Environment only; never persisted, logged or returned. |
| `MEMFORGE_LLM_MODEL` *(M5)* | *(empty)* | Model identifier sent in the request. |
| `MEMFORGE_LLM_TIMEOUT_SECONDS` *(M5)* | `30` | Per-request provider timeout. |
| `MEMFORGE_LLM_MAX_RETRIES` *(M5)* | `2` | Extra attempts after the first (`0` disables retrying). |
| `MEMFORGE_LLM_RETRY_BACKOFF_SECONDS` *(M5)* | `0.5` | Linear backoff base between attempts. |
| `MEMFORGE_LLM_CHAT_PATH` *(M5)* | `/chat/completions` | Path of the compatible contract. |
| `MEMFORGE_LLM_SEND_AUTHORIZATION` *(M5)* | `true` | Set `false` for an endpoint needing no credential. |
| `MEMFORGE_MOCK_LLM_MODEL` *(M5)* | `mock-echo-v1` | Model name reported by the offline provider. |

The M2 scoring *weights* themselves are code-level defaults in
`IntelligenceConfig` (documented above) and are injectable in code. The same is
true of the M4 section ordering (`ContextPolicy.section_order`, `type_sections`).
All settings are validated at startup: context utility weights must sum to `1.0`,
the default budget may not exceed the maximum, and every estimator ratio must be
positive.

Data lives in `data/` (SQLite files there are git-ignored).

## Testing

```bash
python -m pytest                     # full suite
python -m pytest -q --cov=app --cov-report=term-missing
python -m ruff check .               # lint
python -m ruff format --check .      # formatting
python -m mypy                       # strict static typing
python scripts/smoke_test.py         # live HTTP smoke test (real uvicorn)
python scripts/mcp_selftest.py       # local MCP self-test
python -m app.evaluation.run         # evaluation regression gate (M6)
python -m app.observability.demo     # observability demo (M6)
python scripts/llm_smoke_test.py     # OPTIONAL live API (needs MEMFORGE_LLM_*)
python scripts/dashboard_smoke.py    # M7: the real dashboard build against a real backend
```

The dashboard has its own suite, run from `dashboard/`:

```bash
npm run typecheck       # strict tsc
npm test                # frontend tests
npm run build           # typecheck + production build
```


The suite is organised by layer:

| File | Focus |
| --- | --- |
| `tests/test_domain_models.py` | Entity validation, enums, update semantics, sentinel behaviour. |
| `tests/test_ids_and_clocks.py` | Deterministic id generation and the clock port. |
| `tests/test_similarity.py` | Tokenisation, stop words, suffix rules, Dice values and boundaries. |
| `tests/test_attributes.py` | Attribute extraction grammar and value comparison. |
| `tests/test_embeddings.py` | Provider interface, deterministic hashed vectors, Ollama unavailability. |
| `tests/test_retrieval_domain.py` | Query/filter validation, ranking weights, relevance blending, penalties, tie-breaks. |
| `tests/test_embedding_store.py` | SQLite embedding cache: round-trip, provenance, cascade, corruption. |
| `tests/test_intelligence.py` | Config validation, factor functions, decay, priority, resolution policy and tie-breaks. |
| `tests/test_migrations.py` | Schema creation, migration idempotency, foreign keys, unique constraints. |
| `tests/test_sqlite_repository.py` | Data access, ordering, pagination, soft delete, SQL-injection safety, error translation. |
| `tests/test_m2_storage.py` | Access telemetry, relationship constraints/indexes/cascade, active-corpus scans. |
| `tests/test_memory_service.py` | M1 lifecycle rules: versioning, state machine, idempotent forget, history. |
| `tests/test_m2_service.py` | M2 operations: scoring, decay, duplicates, conflicts, resolution, M1 compatibility. |
| `tests/test_m2_validation.py` | Validation and error-mapping branches of the M2 algorithms. |
| `tests/test_api_memories.py` | REST happy paths for the M1 endpoints. |
| `tests/test_api_intelligence.py` | REST paths and errors for the M2 endpoints. |
| `tests/test_api_errors.py` | Validation, error envelope, status-code mapping, 500/503 paths. |
| `tests/test_persistence.py` | Data outliving repositories, applications and restarts; schema upgrade. |
| `tests/test_config.py` | Settings defaults and environment overrides. |
| `tests/test_retrieval_service.py` | Candidate generation, ranking, filters, duplicate suppression, conflict awareness, degradation, index rebuild, explain. |
| `tests/test_api_retrieval.py` | REST search/explain contracts, validation, degradation reporting. |
| `tests/test_cli_demo.py` | Demo correctness and byte-for-byte determinism. |
| `tests/test_tokens.py` *(M4)* | Token estimation: determinism, boundaries, empty content, unicode/punctuation, configured estimator, validation. |
| `tests/test_context_domain.py` *(M4)* | Request/policy validation, utility scoring, compression, ordering, admission, redundancy, budget selection, rendering. |
| `tests/test_context_service.py` *(M4)* | End-to-end compilation: budget compliance, conflicts, duplicates, provenance, filters, ageing, determinism, semantic degradation. |
| `tests/test_api_context.py` *(M4)* | REST contract, budget validation, malformed requests, error envelope, service wiring. |
| `tests/test_m4_cli_demo.py` *(M4)* | The M4 demo section and its byte-for-byte determinism. |
| `tests/test_llm_domain.py` *(M5)* | Provider interface, messages, requests, usage, JSON parsing, error hierarchy, redaction. |
| `tests/test_llm_providers.py` *(M5)* | Request formation, bearer auth, timeouts, HTTP failures, malformed bodies, structured output, bounded retries, usage passthrough. |
| `tests/test_memory_operations.py` *(M5)* | The M1-M4 facade: store/search/recall/update/forget/compile/explain and the `UNSET` translation. |
| `tests/test_agent_service.py` *(M5)* | Memory gate, recall → compile → provider, candidate handling, opt-in storage, observability, determinism. |
| `tests/test_api_agent.py` *(M5)* | REST contracts, candidate opt-in, budget validation, provider-failure mapping, secret hygiene. |
| `tests/test_mcp_server.py` *(M5)* | Tool schemas, annotations, validation, error normalisation, service integration, determinism. |
| `tests/test_m5_config.py` *(M5)* | Provider settings, environment wiring, key hygiene, startup validation. |
| `tests/test_m5_cli_demo.py` *(M5)* | The M5 demo section, the storage opt-in flag and determinism. |
| `tests/test_m5_edge_cases.py` *(M5)* | Defensive branches: error shapes, optional helpers, SDK seams. |
| `dashboard/tests/` *(M7)* | The frontend suite: the typed client, response-shape validation, redaction, the trace tree, all eight pages, navigation, responsive structure, and every loading/empty/error/unavailable/malformed state. |
| `scripts/dashboard_smoke.py` *(M7)* | The production build served in front of a real uvicorn: build, bundle safety, all routes, the whole demo flow. |

## Demo

```bash
python -m app.cli.demo                 # writes data/demo.sqlite (recreated each run)
python -m app.cli.demo --db C:\tmp\demo.sqlite
```

The demo is deterministic — same output on every run — and walks through the whole
lifecycle:

```
create → list → update → archive → history → forget → history →
score importance → detect near-duplicate → create conflicting memory →
detect conflict → resolve conflict → show explanation → show history intact →
retrieve (lexical) → retrieve (hybrid) →
build a memory pool → compile context under a token budget →
show selection, suppression, provenance and remaining budget →
compile again with a tight budget →
agent turn (memory gate → recall → context → mock provider → answer →
candidates → storage) → summary
```

The M5 section prints:

```
  USER REQUEST  ->  MEMORY GATE  ->  RECALL (M3)  ->  CONTEXT (M4)
              ->  PROVIDER  ->  ANSWER  ->  CANDIDATES  ->  STORE (M1)

  provider: mock (model=mock-echo-v1, configured=True)
  request:   'What backend language should I use for the MemForge service?'
  memory:    used (the request may depend on stored memory)
  context:   40 estimated tokens of a 200 budget

  answer:
      [mock provider: deterministic offline stand-in, no model was called]
      …

  candidate memories:
      [FACT] confidence=0.70 importance=0.50 MemForge uses Python and FastAPI for its backend.
          reason: explicit 'remember:' instruction in the request
  stored: 0 (not permitted -- nothing was written)

  observability:
      respond              success=True provider=mock usage=not reported
      extract_candidates   success=True provider=mock usage=not reported
```

Nothing is written unless you ask for it:

```bash
python -m app.cli.demo --store-candidates            # let the agent persist candidates
python -m app.cli.demo --llm-provider custom         # opt-in, needs MEMFORGE_LLM_*
```

The default command requires **no network and no API key**.

The M4 section prints:

```
  Context Compiler
  ----------------
  Query:              'python backend language fastapi memforge'
  Budget:             160
  Candidates:         7
  Selected:           5
  Suppressed:         2
  Remaining:          82

  Selected:
    [PROFILE] utility=0.5254 tokens=11 9fe307b5-…
        Python is the user's preferred language
        provenance: type=PREFERENCE source=user:revanth confidence=0.50 …
    …

  Suppressed:
    duplicate_of           3502c946-… utility=0.4717 cost=6
        near-duplicate of 9fe307b5-…, which represents this duplicate group …

  Final context:
    PROFILE
    -------
    - Python is the user's preferred language
    …

  Final context tokens: 78 / 160
  Within budget:        True
  Estimator:            memforge-char-word/v1
  Token counts are MemForge estimates, not provider tokenisations.
```

## Smoke test

`scripts/smoke_test.py` starts a **real uvicorn server** on a throwaway SQLite
file and drives it over HTTP with `httpx` — no in-process test client, no mocks.
It walks a realistic corpus from creation through retrieval to context
compilation and checks 30 mandatory properties: the response contract, selection,
duplicate suppression, conflict suppression, budget accounting, hard-budget
compliance across ten budgets, provenance, the explanation, determinism (three
identical compilations), the error envelope, honest degradation and the offline
default path.

M6 adds six more: an agent turn produces a trace, its events arrive in the
documented order, no trace payload carries a secret, traces are listable and
filterable, the counters agree with the raw events, and the evaluation endpoint
reports a real run. The traceable flow is exercised end to end:

```
agent request -> memory gate -> retrieval -> context compilation -> mock
provider -> trace created -> events persisted -> trace readable -> metrics
updated -> evaluation available
```

The trace id comes back in the `X-MemForge-Trace-Id` response header. Latency
values are read back and checked for being present and non-negative; they are
never compared against a fixed number, because they are measured at runtime and
are not a benchmark.

```bash
python scripts/smoke_test.py     # 30/30 mandatory checks passed
```

It is **zero-network**: the server binds to `127.0.0.1` and the default
configuration has no embedding provider, so no outbound call is possible.

## MCP self-test

`scripts/mcp_selftest.py` drives the real MCP server in-process and walks the
whole tool lifecycle — `tools/list`, `memory.store`, `memory.search`,
`memory.recall`, `memory.compile_context`, `memory.explain`, `memory.update`,
`memory.forget`, plus the validation and error paths — over a throwaway SQLite
file.

```bash
python scripts/mcp_selftest.py     # 11/11 mandatory MCP checks passed
```

M6 adds two checks: every call is traceable (a trace, its ordered events and the
nested write inside it) and a failed call is recorded as failed rather than
clean.

It is **zero-network**: the memory tools never reach a language-model provider,
so there is nothing to leak and nothing to reach.

## Current limitations (honest gaps)

### M1
- **No semantic search.** Retrieval is exact (`id`) or filtered (`status`,
  `memory_type`) with pagination. There is no embedding, similarity ranking or
  hybrid retrieval.
- **No automatic scoring.** `confidence` / `importance` are caller-supplied.
- **No deduplication or conflict detection in M1.**
- **Single SQLite file, no concurrency control.** Writes are serialised by SQLite
  (`BEGIN IMMEDIATE`); there is no optimistic-concurrency token in the API and no
  multi-writer coordination across machines.
- **No authentication or multi-tenancy.** The API trusts whoever can reach the
  port; run it on localhost or behind your own gateway.
- **History is append-only with no pruning**, and there is no
  "revert to version N" endpoint.
- **No context compilation or token budgets.** *(M1 only — see
  [M4](#m4-context-engineering).)*

### M2
- **M2 intelligence is lexical, not semantic.** Near-duplicate detection uses
  token-set similarity, so it misses paraphrases that share no tokens
  ("changes the engine" vs "swaps the backend"), and the suffix stripper is not a
  linguistic stemmer (`typed` stays `typed`).
- **Negation is not understood.** "likes X" and "does not like X" are scored as
  similar. Conflict detection only covers the documented attribute grammar, not
  contradictions in free prose.
- **The conflict grammar is conservative.** If a memory does not match
  `attr = value`, `attr: value` or `attr is value`, no conflict can be detected
  for it — and the grammar can also produce false positives
  ("X is fast" vs "X is slow" is reported as a conflict, which may be intended or
  not).
- **Detection scans are O(n)** over the active corpus on every call; there is no
  inverted index or ANN structure (that is M3's job).
- **Importance weights are heuristics.** The type table and the five weights are
  documented choices, not measured values, and they are global (not per-user).
- **Recurrence is inferred from near-duplicates**, so it only counts facts that
  were stored more than once.
- **The `hashed` embedding provider is a lexical proxy.** It exercises the semantic
  code path offline but cannot capture synonymy, paraphrase or negation, and its
  cosine values approximate token-set overlap.
- **The inverted index is rebuilt, not incrementally maintained.** A write forces
  one reload of the index on the next query; the reload is O(n) over the
  searchable corpus.
- **Semantic and hybrid modes scan the searchable corpus** (no ANN structure), so
  they are O(n) per query.
- **Retrieval ranks, it does not decide.** It surfaces the winner of a conflict
  and demotes the loser, but it never silently removes or rewrites anything.
- **Duplicate suppression relies on recorded M2 relationships.** Two memories that
  were never compared are both returned even if they look similar.
- **Ties are broken by identifier.** Two memories with identical scores are
  ordered by id, which is deterministic but not semantic.
- **Access telemetry is opt-in.** Nothing records accesses automatically; the
  caller (agent integration layer) must call `record_access`.
- **Relationships are append-only facts**: a `DUPLICATE_OF` link recorded when
  two memories looked similar is not removed if they later diverge.

### M4
- **Token counts are estimates.** `memforge-char-word/v1` is a documented
  character/word heuristic. It is deterministic, configurable and conservative
  for non-ASCII text, but it is **not** any provider's tokenizer and it is not
  billing data. The hard guarantee is `estimated_tokens <= token_budget` *in
  MemForge's own estimator*; a real prompt sent to a real model may tokenise
  differently, which is why `max_context_token_budget` is the only ceiling M4
  enforces.
- **No LLM summarisation.** Content is never paraphrased, rewritten or
  summarised. Compression is limited to whitespace normalisation and dropping
  whole sentences.
- **Compression does not guarantee semantic equivalence.** Dropping sentences can
  remove a negation, a caveat or the condition of a statement. The original text,
  the retained segment indices and both token figures are always reported so the
  transformation stays auditable — but the compiler cannot prove the shorter text
  means the same thing.
- **Selection is a configurable heuristic, not a learned ranker.** The utility
  weights, the section order and the type → section mapping are documented
  defaults chosen for explainability. They are global, not per-user, not
  per-tenant, and no optimality, recall or relevance claim is made about them.
- **Greedy admission is not optimal packing.** Candidates are admitted in utility
  order; a large high-utility candidate can crowd out several smaller ones whose
  combined utility is higher. The algorithm is deterministic and its decisions
  are explained, but it is not a knapsack solver.
- **The last-resort render trim does not re-admit.** When section headings push
  the rendered context over the limit, the lowest-priority block is dropped and
  the budget is left unused. Re-admitting smaller candidates afterwards is a
  documented non-goal for M4.
- **Cost is estimated per block, not per rendered byte.** Because the estimator
  is not additive, the sum of the per-block costs is not exactly the reported
  total; the authoritative figure is always the estimate of `context_text`.
- **Duplicate suppression only sees what retrieval surfaces.** Redundancy between
  two memories that were never compared, or that share no query terms, is not
  detected here — that is M2's job, and M2's similarity is lexical.
- **A superseded memory is always suppressed**, even when its conflict winner is
  not in the candidate set. That is deliberate (an out-of-date fact must not
  reach the prompt), but it means a context can be silent about an attribute
  rather than quoting the losing value.
- **No PII detection, redaction or per-field privacy policy.** The M4 roadmap item
  that mentioned privacy controls is not implemented here.
- **Ordering ignores dialogue structure.** There is no notion of turn, of a
  "system" block, of interleaving or of recency windows; the order is
  section-then-utility only.
- **One estimator, one language assumption.** The token estimator is tuned for
  English-like prose. While non-ASCII characters are charged conservatively, no
  per-language model is provided.
- **`status` is a single filter.** Compiling a mixed ACTIVE/ARCHIVED context is
  possible, but there is no per-section or per-block status policy.

### M7

- **The dashboard is a presentation layer, not a capability.** It renders
  decisions M1-M6 already made; nothing is computed in the browser, and the
  dashboard adds no memory, retrieval or generation of its own.
- **Some explorer filters are page-level.** `GET /memories` supports only
  `status`, `memory_type`, `limit` and `offset`, so the confidence/importance
  thresholds, text filter and sort order apply to the loaded page and are
  labelled as such. The Retrieval Lab performs a real corpus-wide ranked search.
- **No time series, so no chart.** The metrics endpoint returns aggregate
  counters and the backend keeps no historical buckets; the Metrics page states
  this instead of plotting something invented.
- **No authentication.** MemForge's API is unauthenticated and local-first, and
  the dashboard is meant to run beside it. Do not expose either to a network you
  do not control.
- **The smoke test is HTTP-backed, not browser-driven.** It proves the build is
  servable and that every page's endpoints answer with real data; it does not
  prove the rendered pixels of a real browser. There is no Playwright stack.

### M5
- **The mock provider is not a model.** It is a deterministic, offline script
  that renders the request back. It exists so that everything above it is
  reproducible; it is not a language model and produces no useful answer.
- **Usage is only ever echoed, never derived.** `usage` is `null` unless the
  provider sent a usage block. MemForge's own token figure (M4's estimate) is a
  *different, labelled* number and is never mixed with a provider count. When a
  provider does not report usage, MemForge has no cost or token figure for that
  call and says so.
- **No streaming, no tool-calling loop, no multi-turn memory.** The agent is one
  request → one answer. There is no conversation history, no plan, no retry of
  the agent itself and no agent-to-agent protocol.
- **The memory gate is a keyword heuristic.** Short greetings skip memory;
  everything else uses it. It does not understand intent, and a request that
  *should* have used memory but is phrased as small talk will not.
- **Candidate extraction is untrusted input.** The provider may propose
  anything; a malformed proposal is reported with a reason, never stored. But
  MemForge does not verify that a stored candidate is *true* — that judgement
  belongs to the caller, which is why storage is opt-in.
- **The MCP server exposes memory, not inference.** An MCP client already is the
  model. There is no MCP tool that calls a provider, which also means no MCP
  path can leak a credential.
- **Tool names use dots** (`memory.store`). The SDK accepts them and logs a
  conformance warning; a client that enforces the strict `[a-zA-Z0-9_-]` pattern
  would need underscore aliases.
- **The custom provider targets one wire shape.** The widely adopted
  "OpenAI-compatible" chat-completions contract. It does not cover streaming
  responses, tool/function calls, embeddings, batch endpoints, logprobs or
  vendor-specific extensions, and content that is not a plain string is
  normalised rather than interpreted.
- **Retries are naive.** Linear backoff, no jitter, no `Retry-After`
  compliance, and no circuit breaker. A failing provider is retried up to the
  configured budget on every turn.
- **No per-caller isolation.** The credential is process-wide: every request
  uses the same key, with no tenant separation, quota accounting or per-request
  overrides. Run one MemForge instance per trust boundary.
- **Prompt injection is out of scope.** Memory content and provider output are
  passed through; nothing sanitises instructions found inside them.

## Roadmap

| Milestone | Theme |
| --- | --- |
| **M1** ✅ | Memory core: domain model, SQLite persistence, versioned history, REST API. |
| **M2** ✅ | Memory intelligence: importance scoring, temporal decay, lexical duplicate/conflict detection, deterministic resolution. |
| **M3** ✅ | Semantics: embeddings behind a provider interface, hybrid retrieval ranking. |
| **M4** ✅ | Context engineering: deterministic compilation with token budgets, ordering, provenance and full explanation. |
| **M5** ✅ | Agent integration: pluggable LLM provider (mock default, custom opt-in), memory-aware agent, MCP server. |
| **M6** ✅ | Evaluation + observability: a deterministic 36-case benchmark of the memory decisions, and an append-only trace lifecycle. |
| **M7** ✅ | Control plane: a React + TypeScript dashboard over the real memory, retrieval, context, agent, evaluation and observability APIs. |
| *next* | Local models: Ollama-backed providers (summarisation, extraction, embeddings). |

Each milestone keeps the same rule as M1: nothing is claimed until it is
implemented and tested.

## Project layout

```
memforge/
├── app/
│   ├── api/            # FastAPI routes, schemas, error envelope, dependencies
│   ├── cli/            # deterministic demo (M1-M6)
│   ├── evaluation/     # M6 dataset, metrics, service, report, CLI, registry
│   └── observability/   # M6 trace service, instrumentation decorators, demo
│   ├── config.py       # pydantic-settings configuration
│   ├── domain/         # entities, enums, errors, ports, similarity, intelligence,
│   │                   # retrieval, token estimation, context compilation
│   ├── evaluation.py      # M6 dataset/case/result models, metric definitions
│   └── observability.py   # M6 trace/event models, safe metadata
│   ├── main.py         # application factory + ASGI entrypoint
│   ├── services/       # memory lifecycle, intelligence rules, retrieval, compiler
│   └── storage/        # repository ABC + SQLite implementation + migrations
├── dashboard/          # M7: React + TypeScript control plane (Vite, Vitest)
│   ├── src/
│   │   ├── api/        # typed client, contracts, response-shape validation
│   │   ├── components/ # shell, cards, tables, trace timeline, budget bar
│   │   ├── hooks/      # request state and theme
│   │   ├── lib/        # formatting, redaction, trace tree, vocabulary
│   │   ├── pages/      # the eight areas
│   │   └── styles/     # design tokens, layout, components
│   └── tests/          # frontend suite (fixtures captured from a live backend)
├── data/               # local SQLite files (git-ignored)
├── docs/               # architecture notes
├── scripts/            # HTTP smoke test, MCP self-test, dashboard smoke test,
│                       # optional live-API test
├── tests/              # pytest suite
├── pyproject.toml
└── README.md
```

## License

MIT.
