# Connecting the existing TRACE application

The `vaydr/kgqa-refactor` compatibility contract uses `{ triples: [{ node1, relationship, node2, score }] }`; clauses use `{ clauses: [{ node1, relationship, node2 }] }`; answers use `{ answer, reasoning? }`. `/v1/trace/query` supplies those three objects unchanged in shape and adds `evidence` containing canonical IDs, paths, scores, and support state.

## Minimal server-side call

In the Next.js server-side KGQA workflow, replace graph generation/reasoning with one request. Keep the backend URL/key on the server:

```typescript
const response = await fetch(`${process.env.TRACE_BACKEND_URL}/v1/trace/query`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    ...(process.env.TRACE_BACKEND_API_KEY
      ? { "X-API-Key": process.env.TRACE_BACKEND_API_KEY }
      : {}),
  },
  body: JSON.stringify({ question, topic_entity: selectedGraphEntityId }),
  signal: AbortSignal.timeout(300_000),
});
if (!response.ok) throw new Error(`TRACE backend: ${response.status}`);
const result = await response.json();
dataStream.write({ type: "data-kgqa-graph", data: result.graph });
dataStream.write({ type: "data-kgqa-clauses", data: result.clauses });
dataStream.write({ type: "data-kgqa-answer", data: result.answer });
dataStream.write({ type: "data-kgqa-evidence", data: result.evidence });
dataStream.write({ type: "data-kgqa-step", data: "complete" });
```

For benchmark questions, send `question_id` instead. Ordinary free text needs a topic anchor from the UI/entity-selection step. The paper assumes topic annotations; automatic arbitrary-question entity linking is not implemented here. Use `/v1/entities?q=...` to find exact IDs, including ambiguous names. Do not send a fixture's gold answer to `/v1/query`.

The existing `runKGQAWorkflow` creates clauses, generates a graph, adds missing clause edges, and may request bridging edges. Bypass that sequence when using this backend: all evidence edges must come from the indexed KG. The backend already returns the completed graph and a real evidence path.

## Remaining UI wiring

- Add a typed `data-kgqa-evidence` data part and retain the canonical response in provider state.
- Render and join vertices using canonical `evidence.nodes[].id`, with `label` used only for display. The legacy label-oriented compatibility shape can merge distinct entities that share a display name; the canonical graph preserves their identities.
- Highlight using `evidence.highlighted_path.steps[].triple_id` against `evidence.edges[].id`, preserving direction via each step's `reversed` flag. Legacy clauses list the selected path's facts in path order but preserve each stored edge's subject/object direction. Ordered traversal, including reverse steps, is represented in canonical evidence.
- Show alternatives from `evidence.alternative_paths` and parser disagreement from `evidence.parser_disagreement`.
- For `no_path_within_budget`, say that no path was recovered in this evidence within the hop budget. For `search_incomplete`, show a resource-limit state; a provisional path is not guaranteed to be the best-ranked path.
- Preserve graph rendering even when the highlighted path is absent. `no_answer` and `answer_unresolved` are separate conditions.
- Use canonical nullable `evidence.edges[].score` for heatmaps. Expansion-only edges are unscored. Do not interpret the compatibility adapter's `0.0` as an estimated relevance probability.
- The current study fixtures can remain a separately selected frozen-snapshot mode. Use the live backend explicitly for live retrieval; the API does not alter fixtures or study logs.

The SSE endpoint emits JSON objects with the current `data-kgqa-*` names, including the new evidence event. It is a conventional SSE stream, not the complete Vercel UI Message Stream wire protocol. A Next.js route must parse and forward those events through its existing `dataStream.write` calls. Graph/clauses/answer are emitted after pipeline completion; the implementation does not claim token-by-token LLM streaming.

The Python API does not own PostgreSQL/Drizzle sessions, study assignment, or user identity. Those stay in the existing Next.js application.
