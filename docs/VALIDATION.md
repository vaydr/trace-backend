# Validation record

These checks ran locally on an Apple Silicon Mac with Python 3.12.13. They verify a source implementation and reproducible training workflow, not full WebQSP convergence or the paper's reported results. Generated artifacts remain ignored by Git and are not included in the source package.

## Automated checks

- **141 tests pass.** The suite covers graph identity, exact anchor expansion, graph limits, parser ambiguity/failure, disconnected/unknown answers, provider errors, authentication, JSON/SSE responses, training, artifact provenance, and replacement of temporary download files during writing.
- Twenty-five randomized multigraph cases compare path enumeration with NetworkX. Ten additional repeated-predicate cases compare against an independent direct implementation of Algorithm 1's predicate/neighbor set F; reciprocal equal-predicate triples produce one route.
- Four dense-reference tests compare the retriever's node features, logits, parameter gradients, and embedding-input gradients against independent DDE and MLP feature assembly. Cases include default/asymmetric forward/reverse rounds, topic encoding on/off, duplicate directed edges, CVTs, self-loops, and isolated vertices.
- Training regressions cover exact CPU resume with explicit or default seeds, interrupted best-checkpoint publication, restored early stopping, checksum corruption, interrupted preparation, requested/prepared encoder mismatch, foreign best-checkpoint provenance, valid weak supervision, and default unclipped BCE/Adam behavior.
- Ranking regressions verify identical evaluation/serving selection on real model ties and sigmoid saturation, including candidate permutations and K=1/2. Checkpoints record the ranking rule; resume rejects incompatible selection metadata while old weights remain usable for inference.
- Humanization tests verify that token labels upgrade to LLM labels, cached LLM labels are reused, and token fallback never overwrites existing LLM labels.
- Pipeline evaluation rejects question, topic, split, or candidate-graph mismatches before inference. Reports include input/checkpoint hashes, logical graph fingerprints (including committed WAL contents), model/provider settings, search limits, and bundled prompt/schema hashes. Tests cover graph import-order invariance, content changes, concurrent writes during fingerprinting, configuration identity, and credential redaction. Successful records preserve complete canonical evidence and a content hash. Retrieval-evaluation reports bind metrics to the loaded checkpoint and prepared data; concurrent checkpoint replacement is covered.
- Compatibility tests preserve stored predicate direction during reverse traversal. CLI tests verify environment-configured graph selection and explicit overrides.
- Provider request failures (including protocol disconnects and content decoding errors) have bounded retries and structured errors. API/SSE regressions verify that parser failures still allow length-only path search, answer failures remain errors, malformed message envelopes are rejected, and upstream details/credentials are not exposed.
- The downloader initializes lazy readers before opening its temporary output, then checks the written and published content against the bytes it serialized. A regression reproduces replacement of the temporary pathname and confirms that it raises instead of publishing a contradictory success count.
- Ruff checks/formatting, dependency-lock consistency, source/wheel builds, and source-archive content checks pass.

## Synthetic end-to-end check

`trace smoke` completes four CPU epochs using synthetic graphs and a deterministic test-only encoder, reloads its checkpoint, and exercises retrieval, anchor expansion, path search, canonical/compatibility API responses, and SSE. Its fixed provider responses make no paid model calls. Production inference rejects the synthetic hash-encoder checkpoint.

## Small real-WebQSP run with the default architecture

The download workflow streamed three actual examples per split from pinned `ml1996/webqsp` revision `54a938897b1ef1e8dbd2fddb6985fd1e688172e8`:

- Train: `WebQTrn-0`, `WebQTrn-1`, `WebQTrn-3`.
- Validation: `WebQTrn-9`, `WebQTrn-11`, `WebQTrn-15`.
- Test: `WebQTest-0`, `WebQTest-1`, `WebQTest-3`.

The original candidate graphs contain 2,069–8,761 triples per example. Their indexed union contains 14,023 entities and 39,829 unique triples.

The default `configs/webqsp.yaml` was used, with only the epoch budget reduced via `--epochs`:

- Pinned GTE-large-en-v1.5 weights and external custom-code revision, normalized 1,024-dimensional CLS embeddings.
- Frozen text encoder, shared non-text/CVT vector, topic indicators, two forward and two reverse DDE rounds, and the released two-layer MLP architecture.
- BCE plus Adam at 0.001 without gradient clipping.
- All nine examples prepared into immutable checksummed format-v2 artifacts, using Apple MPS for embeddings.
- Two training epochs over the three train questions on MPS; checkpoint reload; one additional resumed epoch on MPS.
- Mean training losses: 0.457508, 0.133400, and 0.057294. Values establish execution and finite optimization, not convergence.
- Held-out evaluation on CPU over the three test questions completed. Triple recall and answer recall at K=50 were both 0. The three-question training run did not produce a useful retriever; these metrics are not a benchmark estimate.

The recorded training/resume run preceded the ranking-consistency correction. Its unchanged weights were re-evaluated with the final shared ranking rule: held-out triple recall, answer recall, and answer hit remained 0. New training runs record that rule explicitly; older selection metadata cannot be used for resume.

The README's small-check commands reproduce this workflow in separate generated directories. Separate probes checked normalized GTE embeddings on CPU and a GTE/retriever gradient step on MPS.

## Real-data API assembly check

The default-GTE checkpoint, graph database, and structured-HTTP client were connected to a fixed **offline provider response** for the Justin Bieber sibling question. No live GPT-4o answer was generated.

The canonical query scored the annotated 8,761-triple candidate graph, retrieved 50 triples, expanded to 802 stored edges / 410 vertices, and recovered a two-hop path with agreement 2.0. Search completed after 940 examined states. Every returned edge was verified against graph storage. Canonical JSON, TRACE compatibility JSON, and SSE requests all passed with this assembled pipeline.

The included `examples/offline_api_check.py` replays this assembly check from generated artifacts; commands are in the README. An offline CPU replay with the recorded default-GTE checkpoint reproduced these counts and passed all three response contracts.

This check validates the trained model/data/path/API connection. It does not establish answer-generation accuracy: the answer and parser response were test fixtures. The generated OpenAPI schema is saved in `docs/openapi.json`.

The HTTP service was additionally started as a real local process during initial verification. Liveness, entity lookup, question metadata, and OpenAPI returned HTTP 200. Readiness correctly returned HTTP 503 with `llm_unconfigured` when no model API key was supplied.

## Explicitly unverified

- Full WebQSP embedding preparation/training convergence and full held-out benchmark quality.
- Live GPT-4o credentials/quota, answer quality, and reproduction of historical study results. Offline transport tests cover the structured response and error contracts.
- CUDA and the Docker image. CPU and MPS paths were exercised.
- A complete Freebase import. The checked graph is explicitly a benchmark union; other imports require aligned identifiers.
- Browser rendering and sessions in the existing TRACE application. Its remaining glue is documented separately: topic selection, canonical graph IDs/display labels, API/evidence event wiring, highlighting, alternatives, and status banners.

Dataset files, embeddings, graph databases, environment files, checkpoints, and temporary validation outputs are excluded from Git and the source archive.
