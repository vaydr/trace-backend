# TRACE backend

Python implementation of *TRACE: An Interactive Visual Paradigm for Knowledge Graph Question-Answering* (ISWC 2026 submission): WebQSP data preparation, SubgraphRAG retrieval, anchor expansion, predicate-clause parsing, and evidence-path search.

The release contains source and training scripts. Data, embeddings, graph databases, and checkpoints are generated locally and ignored by Git. No full WebQSP-trained checkpoint is included. See [validation](docs/VALIDATION.md) for measured results and verification limits.

## Code organization

All modules below are in `src/trace_backend/`.

| Module | Responsibility |
| --- | --- |
| `data.py`, `schema.py` | WebQSP normalization and typed data/API records |
| `graph.py` | SQLite graph, question metadata, entity lookup, anchor neighborhoods |
| `ml/encoder.py` | Frozen text encoder and embedding cache |
| `ml/features.py` | Entity/relation features, topic indicators, shortest-path supervision |
| `ml/model.py` | Directional distance encoding and triple-scoring MLP |
| `ml/training.py` | Feature preparation, optimization, checkpoints, retrieval evaluation |
| `ranking.py`, `retrieval.py` | Shared top-K ordering and checkpoint-backed inference |
| `pipeline.py` | Query orchestration and legacy TRACE response adapter |
| `search.py` | Algorithm 1: bounded simple-path enumeration and clause agreement |
| `prompts.py`, `llm.py` | Answer generation, clause parsing, predicate labels |
| `api.py`, `cli.py`, `config.py` | HTTP service, commands, runtime settings |
| `evaluation.py`, `smoke.py` | End-to-end evaluation and synthetic integration check |

Start with `Pipeline.query`, then follow `SubgraphRetriever.retrieve` and `search_paths`. The training entry points are `prepare`, `train`, and `evaluate_model` in `ml/training.py`.

## Installation

Python 3.11–3.13; validation used Python 3.12.

```sh
uv sync --extra ml --extra dev
uv run trace doctor
uv run pytest -q
uv run trace smoke
```

Alternatively, install with `python -m pip install -e '.[ml,dev]'`. The `ml` extra supplies PyTorch, Transformers, sentence-transformers, and datasets. The core API supports an injected retriever without these dependencies. No PyTorch Geometric or CUDA-specific extension is required.

`trace smoke` trains a small synthetic model, reloads its checkpoint, and exercises retrieval, expansion, search, JSON responses, and SSE. It uses a deterministic hash encoder and fixed provider responses. Production retrieval rejects hash-encoder checkpoints.

## Data and training

```sh
uv run trace download
uv run trace build-graph --input data/webqsp/train.jsonl data/webqsp/validation.jsonl data/webqsp/test.jsonl
uv run trace prepare --config configs/webqsp.yaml
uv run trace train --config configs/webqsp.yaml
uv run trace evaluate --split test --k 50
```

The downloader streams the pinned `ml1996/webqsp` release into normalized JSONL, recording revision, counts, and file hashes. Each record contains a question, topic entities, answer annotations, and its candidate graph. The graph database is the deduplicated union of those candidate graphs. It provides shared graph facts across question splits; this is a transductive benchmark setting, not a complete Freebase import.

Training uses train-split answers for supervision, validation selects checkpoints, and test answers are used for evaluation. Runtime generation receives retrieved triples without gold annotations.

### Retriever

`configs/webqsp.yaml` follows the released SubgraphRAG configuration:

- Frozen, normalized 1,024-dimensional GTE-large-en-v1.5 CLS embeddings.
- A shared learned vector for opaque/CVT entities and two-channel topic indicators.
- Two forward and two reverse rounds of mean propagation for directional distance encoding.
- Question/head/relation/tail features concatenated into a two-layer ReLU MLP.
- Binary cross-entropy and Adam with learning rate 0.001; no default gradient clipping.
- Validation recall at K=100, patience 10, and an upper limit of 10,000 epochs. Query defaults use K=50.

Supervision is the union of equally shortest directed routes in either orientation between topic and answer entities. Parallel predicates are retained. Training skips examples without a nonzero supervision path; validation requires usable supervision. Triple scoring is chunked, with loss normalized per graph.

Retrieval evaluation reports mean recall of supervised triples, mean recall of annotated answer entities among retrieved endpoints, and the fraction of questions with any retrieved answer entity (`answer_hit`). Triple recall excludes questions with no positive supervision; answer metrics exclude questions with no annotated answers. Reports retain selected triple IDs/scores and checkpoint, prepared-data, configuration, split, and device provenance.

Validation, evaluation, and serving rank raw logits in descending order, breaking ties by stable triple ID. Selected scores are converted to sigmoid probabilities. This preserves selection when distinct logits round to the same probability.

The encoder weights and custom model code are pinned to revisions in the configuration. Eager attention supports CPU and Apple MPS. Device selection is CUDA, then MPS, then CPU; `--device cpu`, `--device mps`, or `--device cuda` overrides it. GTE feature preparation is the principal computational cost. The answer-generating LLM is not fine-tuned.

### Small real-data check

The following prepares three questions per split with the default encoder, trains for two epochs, resumes for a third, and evaluates the checkpoint. It checks execution on real data; nine questions are insufficient for a useful retriever.

```sh
uv run trace download --limit 3 --output artifacts/small-check/data
uv run trace prepare --data artifacts/small-check/data --output artifacts/small-check/prepared --cache artifacts/small-check/embeddings.sqlite --config configs/webqsp.yaml
uv run trace train --prepared artifacts/small-check/prepared --output artifacts/small-check/model --config configs/webqsp.yaml --epochs 2
uv run trace train --prepared artifacts/small-check/prepared --output artifacts/small-check/model --config configs/webqsp.yaml --resume artifacts/small-check/model/last.pt --epochs 3
uv run trace evaluate --prepared artifacts/small-check/prepared --checkpoint artifacts/small-check/model/best.pt --output artifacts/small-check/evaluation.json --k 50
```

To check API assembly with these artifacts and fixed offline provider responses:

```sh
uv run trace build-graph --input artifacts/small-check/data/train.jsonl artifacts/small-check/data/validation.jsonl artifacts/small-check/data/test.jsonl --db artifacts/small-check/graph.sqlite
uv run python examples/offline_api_check.py --db artifacts/small-check/graph.sqlite --checkpoint artifacts/small-check/model/best.pt --cache artifacts/small-check/embeddings.sqlite --output artifacts/small-check/api-check.json
```

The script uses cached encoder files, makes no provider calls, and saves the fixed responses and complete canonical evidence. It checks canonical JSON, legacy JSON, SSE, graph membership, and a two-hop path for `WebQTrn-0`; it does not measure model answer quality.

### Artifacts and resume

| Location | Contents |
| --- | --- |
| `data/webqsp/` | Normalized splits and download manifest |
| `data/graph.sqlite` | Graph facts and benchmark question metadata |
| `artifacts/embedding-cache.sqlite` | Encoder-versioned text embeddings |
| `artifacts/prepared/` | Per-example tensors, weak labels, split fingerprints |
| `artifacts/webqsp/best.pt` | Best validation checkpoint |
| `artifacts/webqsp/last.pt` | Resumable model, optimizer, RNG, epoch, early-stop state |
| `artifacts/webqsp/history.jsonl` | Per-epoch loss and validation metrics |
| `artifacts/evaluation.json` | Held-out retrieval metrics and predictions |

```sh
uv run trace train --resume artifacts/webqsp/last.pt
```

Prepared format v2 uses immutable tensor files and an atomically published manifest. Training and evaluation verify tensor checksums and encoder identity. Resume also checks data fingerprints, run identity, the sibling `best.pt`, model configuration, seed, learning rate, clipping, evaluation K, and ranking rule. Keep `best.pt` beside a moved resumable checkpoint and use a new output directory for a new experiment. If an improving `last.pt` was saved before an interrupted `best.pt` write, resume reconstructs the best checkpoint from that state; unrelated or incompatible siblings are rejected. A run already stopped by patience performs no further epochs unless patience is increased.

Checkpoints load with `weights_only=True`. Original upstream pickle artifacts are not imported implicitly. Older prepared formats require regeneration; checkpoints without run provenance or matching ranking metadata support inference only.

### Larger graph imports

```sh
uv run trace import-graph --input path/to/triples.tsv.gz --db data/full-kg.sqlite --scope provided_kg --entities path/to/entities.jsonl
uv run trace build-graph --input data/webqsp/train.jsonl data/webqsp/validation.jsonl data/webqsp/test.jsonl --db data/full-kg.sqlite
```

Triple files use `subject<TAB>predicate<TAB>object`; `.gz` is supported. Optional entity records have the form `{"id":"m.example","label":"Example","source_urls":["https://example.org/source"]}`. Imported facts must use the candidate graphs' identifier space. The benchmark mixes names and opaque IDs without a complete reversible Freebase mapping; identifiers are preserved verbatim.

## Query pipeline

1. Score candidate triples and retain the top K. The answer generator predicts an entity from these retrieved facts.
2. Form the working graph by unioning retrieved triples with every stored incoming/outgoing triple incident to the topic or predicted answer. Expansion is one hop at those two anchors.
3. Parse the question into ranked predicate sequences using the working graph's complete predicate list. The backend prediction remains the search target; parser disagreement is returned separately. Parser failure gives empty hints.
4. Enumerate simple paths up to P=8 edges using adjacency in either direction. Rank each path by `max(LCP(path_predicates, clause_i) / i)` for one-based clause rank `i`, then by increasing hop count. Empty hints give length-only ranking. Reverse-distance pruning preserves admissible paths.
5. Return the working graph, selected path, alternatives, scores, and search status.

Algorithm 1 treats reciprocal facts with the same predicate as one predicate/neighbor step. The response retains both graph facts; path provenance prefers the forward source edge, then stable triple ID.

Question IDs select the benchmark's candidate graph and topic annotations. Free questions require an explicit topic ID or uniquely resolving label and score the supplied indexed graph by default. The optional `candidate_hops` field enables a bounded-neighborhood experiment outside the paper's default retrieval procedure.

Candidate and anchor-neighborhood limits raise errors rather than truncate facts. A search-state limit returns `search_incomplete`, including any provisional path. Other statuses are `path_found`, `no_path_within_budget`, `no_answer`, and `answer_unresolved`. Path existence establishes connectivity in the supplied evidence, not semantic correctness of the answer.

## HTTP service

Copy `.env.example` to `.env` and set `TRACE_LLM_API_KEY`, then run:

```sh
uv run trace humanize
uv run trace serve
```

The default model is `gpt-4o`, accessed through an OpenAI-compatible Chat Completions endpoint with strict JSON-schema output. Set `TRACE_LLM_BASE_URL` for another compatible provider. `trace humanize` uses `TRACE_GRAPH_DB` unless `--db` is supplied and caches predicate labels; `--token-only` uses token-derived labels instead. Query responses reuse the cache.

```sh
uv run trace query --question-id WebQTest-0
uv run trace query --question 'Who is Justin Bieber’s brother?' --topic 'Justin Bieber'
```

| Method | Path | Response |
| --- | --- | --- |
| GET | `/healthz` | Process liveness |
| GET | `/readyz` | Graph/retriever initialization and model-key configuration |
| GET | `/docs` | Interactive API documentation |
| GET | `/v1/entities?q=Justin` | Matching entity IDs and labels |
| GET | `/v1/questions/{id}` | Question and topic annotations, excluding gold answers |
| POST | `/v1/query` | Canonical evidence response |
| POST | `/v1/trace/query` | Legacy TRACE graph/clauses/answer plus canonical evidence |
| POST | `/v1/trace/stream` | SSE events using TRACE's `data-kgqa-*` names |

Example request: `{"question_id":"WebQTest-0"}` or `{"question":"...","topic_entity":"...","k":50,"max_hops":8}`. Schemas are available at `/openapi.json`. `TRACE_API_KEY` enables `X-API-Key` authentication; the default host is localhost. Readiness makes no provider call and does not verify quota.

With live credentials, `trace evaluate-pipeline --limit 10` evaluates predicted answer identity and path states. It checks input questions, topics, splits, and candidate graphs against stored records before inference, includes failures in metric denominators, and records input/checkpoint/graph fingerprints, model and provider settings, search limits, and hashes of bundled prompts and response schemas. Provider metadata retains only the endpoint origin, excluding credentials, paths, and query strings. Each successful prediction includes the full canonical response and its content hash, retaining clauses, explanations, scores, and selected/alternative paths. Custom injected LLM behavior is not fingerprinted by the bundled prompt hashes. Structural paths are not treated as manually judged semantic answers.

## Integration and limitations

[TRACE_INTEGRATION.md](docs/TRACE_INTEGRATION.md) specifies the frontend adapter, entity selection, canonical graph IDs, highlighting, alternatives, and status handling. Layout, sessions, fixtures, and interaction logs remain in the TRACE application. SPIRAL and community visualization are outside this backend.

Expansion-only edges have canonical score `null`; the legacy adapter supplies numeric `0.0` while retaining the original score under `evidence`. The parser accepts 1–5 predicates per clause because Appendix B includes a one-predicate example despite its stated minimum of two. Missing domain/range ontology types are represented as `unknown`.

The submission does not publish the original answer-generation prompt or weights. This implementation supports the specified methods; identical historical predictions and user-study results have not been established. See [implementation notes](docs/PAPER_FIDELITY.md) and [validation](docs/VALIDATION.md).

The retriever follows [Graph-COM/SubgraphRAG](https://github.com/Graph-COM/SubgraphRAG) at commit `48eb05e56d11293fa780849512310462c45b956b`, accompanying [Li, Miao, and Li, ICLR 2025](https://arxiv.org/abs/2410.20724). Attribution and licenses are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
