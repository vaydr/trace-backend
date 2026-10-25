# Implementation notes

The specification is *TRACE: An Interactive Visual Paradigm for Knowledge Graph Question-Answering*, ISWC 2026 submission. This backend implements retrieval, anchor expansion, clause parsing, and evidence search. Visualization, sessions, study fixtures, and interaction logs belong to the TRACE application; community visualization and SPIRAL are outside this implementation.

Manuscript: `KGQA_ISWC_2026_Submission.pdf`, SHA-256 `c2d2609c9de53bea17c5492c5123f1242a508b3d128254474f5cec7421520e38`. Section references below refer to this draft.

| Requirement | Implementation and evidence |
| --- | --- |
| §3.1 SubgraphRAG retriever | `ml/model.py`, `ml/features.py`, `configs/webqsp.yaml`: frozen normalized GTE CLS text vectors; shared learned non-text/CVT vector; topic/non-topic channels; independent forward/reverse mean DDE; concatenated q/head/relation/tail features; two-layer ReLU MLP and sigmoid triple scores. Independent dense-reference tests check features, logits, and gradients. |
| Released retriever training workflow | `download`, `prepare`, `train`, `evaluate` CLI commands; pinned dataset/model/code revisions, train-only weak supervision, validation checkpoint selection, BCE/Adam without default clipping. Small real-data default-GTE training/resume checks are recorded separately from full training in `VALIDATION.md`. |
| §3.1 top K over supplied graph | Question IDs use released benchmark candidate graphs, matching the public upstream workflow. Free questions default to scoring the supplied indexed G. Explicit capacity errors never truncate candidates silently. Opt-in `candidate_hops` is marked as an experiment outside paper defaults. |
| §3.2 exact anchor expansion | `GraphStore.incident` unions every real incoming/outgoing triple touching the question anchor or backend predicted-answer anchor. It does not expand all retrieved vertices or iterate a learned policy. Tests verify the exact edge set and missing-edge path recovery. |
| §3.3 question parsing | GPT-4o default, both Appendix B examples, every working-graph predicate ID, ranked predicate sequences, structured validation. Parser names never create graph facts. Parser ambiguity preserves valid hints; parser failure uses empty H. Backend a remains the target and disagreement is exposed. |
| Appendix A complete bounded search | `search.py` enumerates simple predicate/node routes within P, with bidirectional adjacency; reciprocal duplicates are collapsed according to set F. Safe reverse-distance pruning preserves the result set. Independent graph-library and repeated-predicate reference tests check completeness. |
| Appendix A ranking/output | `max(LCP(sequence, clause_i)/i)`, then fewer hops; best path and alternatives returned. Tests show hints affect ranking, not the set of paths. Default P=8 and K=50. |
| §3.5 offline predicate labels | `trace humanize` caches LLM labels; token fallbacks can be upgraded and never overwrite an LLM label. Query/render responses reuse stored labels. Tests verify cache upgrade and reuse. |
| Visualization contract | Canonical nodes/edges, original directions, selected/alternative paths, scores, expansion origin, disagreement, no-path and search-limit states; existing TRACE-shaped compatibility response plus SSE adapter. Actual force layout remains frontend glue. |
| Reproducibility and generated artifacts | Immutable checksummed prepared tensors; atomic manifest publication; encoder mismatch rejection; run/best-checkpoint provenance; deterministic CPU resume regression. `.gitignore` excludes datasets, caches, embeddings, databases, checkpoints, and environment files. |

## Specification boundaries

- The paper omits an executable training configuration in its own text; the released SubgraphRAG configuration and source supply it. Original TRACE backend weights and original predictions are not recovered.
- The paper does not specify its answer-generation prompt. The implementation grounds generation on retrieved triples and produces the required predicted entity, answer, and explanation; identical historical model outputs cannot be established.
- Public benchmark entity strings mix names and opaque IDs. IDs are preserved verbatim, and imported graph facts must use the same ID space. The benchmark union is not described as a complete Freebase dump.
- The public benchmark does not include full domain/range ontology types. Parser schema fields use explicit `unknown` placeholders while preserving all predicate identifiers. No ontology types are fabricated.
- Appendix B's prose requests 2–5 predicates but its example uses one. Validation accepts 1–5 to accommodate both the example and single-hop questions.
- The paper does not assign retrieval scores to expansion-only edges. Their canonical scores remain null; compatibility-only numeric zero is documented.
- Explicit memory/state ceilings return errors or `search_incomplete`; no incomplete enumeration is claimed complete or interpreted as a proof that an answer is false.
- Full training, live model answering, CUDA, complete Freebase imports, and browser rendering remain unverified; see `VALIDATION.md`.
