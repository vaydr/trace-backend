from __future__ import annotations

import asyncio
import time
import uuid

from .config import Settings
from .errors import TraceError
from .graph import GraphStore
from .llm import token_label
from .schema import GraphEdge, GraphNode, QueryRequest, QueryResponse, ScoredTriple
from .search import search_paths


class Pipeline:
    def __init__(self, store: GraphStore, retriever, llm, settings: Settings):
        self.store, self.retriever, self.llm, self.settings = store, retriever, llm, settings
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_queries)

    def _input(self, request: QueryRequest):
        if request.question_id:
            ex = self.store.question(request.question_id)
            if request.question and request.question != ex.question:
                raise TraceError("question_mismatch", "question does not match question_id")
            if not ex.topic_entities:
                raise TraceError(
                    "missing_topic", "This benchmark question has no annotated topic entity"
                )
            question, topics, candidates = ex.question, ex.topic_entities, ex.triples
            topic = self.store.resolve(request.topic_entity) if request.topic_entity else topics[0]
            if topic not in topics:
                raise TraceError(
                    "topic_mismatch", "topic_entity must be one of the question's annotated topics"
                )
            source = "annotated_candidate_graph"
        else:
            question = request.question
            topic = self.store.resolve(request.topic_entity)
            if topic is None:
                raise TraceError(
                    "unknown_topic",
                    "Unknown topic_entity; use /v1/entities to find an exact graph ID",
                    404,
                )
            topics = [topic]
            if request.candidate_hops is None:
                candidates = self.store.all_triples(self.settings.candidate_max_triples)
                source = "supplied_graph"
            else:
                candidates = self.store.neighborhood(
                    topic, request.candidate_hops, self.settings.candidate_max_triples
                )
                source = f"experimental_{request.candidate_hops}_hop_candidate_neighborhood"
        if len(candidates) > self.settings.candidate_max_triples:
            raise TraceError(
                "candidate_limit",
                "Annotated candidate graph exceeds TRACE_CANDIDATE_MAX_TRIPLES",
                422,
            )
        return question, topic, topics, candidates, source

    async def query(self, request: QueryRequest) -> QueryResponse:
        async with self.semaphore:
            started = time.perf_counter()
            question, topic, topics, candidates, source = await asyncio.to_thread(
                self._input, request
            )
            retrieved = await asyncio.to_thread(
                self.retriever.retrieve, question, topics, candidates, request.k
            )
            answer = await self.llm.answer(question, retrieved)
            warnings = []
            if source.startswith("experimental_"):
                warnings.append(
                    "Candidate-neighborhood prefilter explicitly enabled; this differs from paper retrieval over supplied G"
                )
            target = answer.answer_entity
            if target is not None:
                try:
                    target = await asyncio.to_thread(self.store.resolve, target)
                except TraceError as error:
                    warnings.append(error.code)
                    target = None
                if target is None:
                    warnings.append(
                        "Predicted answer could not be resolved to a unique graph entity"
                    )
            anchors = [topic] + ([target] if target is not None else [])
            expansion = await asyncio.to_thread(
                self.store.incident, anchors, self.settings.candidate_max_triples
            )
            working = {t.id: ScoredTriple(**t.model_dump(), origin="expansion") for t in expansion}
            working.update({t.id: t for t in retrieved})
            triples = sorted(working.values(), key=lambda t: t.key)
            labels = await asyncio.to_thread(self.store.predicate_labels)
            label_sources = await asyncio.to_thread(self.store.predicate_label_sources)
            predicates = sorted({t.predicate for t in triples})
            if any(label_sources.get(p) != "llm" for p in predicates):
                warnings.append(
                    "Some predicates use token-derived labels; run trace humanize for cached LLM labels"
                )
            parser_answer, disagreement, clauses = None, None, []
            try:
                parsed = await self.llm.parse(question, predicates, labels)
                parser_answer, clauses = parsed.answer_entity, parsed.clauses
            except (TraceError, ValueError) as error:
                warnings.append(
                    "Parser failed; using length-only path ranking: "
                    + (error.code if isinstance(error, TraceError) else "invalid_output")
                )
                clauses = []
            if parser_answer is not None:
                try:
                    parser_id = await asyncio.to_thread(self.store.resolve, parser_answer)
                except TraceError:
                    # Entity ambiguity does not invalidate predicate hints (§3.3).
                    warnings.append("Parser answer is ambiguous; predicate hints retained")
                else:
                    if target is not None:
                        disagreement = parser_id != target
            path = None
            alternatives = []
            complete, states = True, 0
            if answer.answer_entity is None:
                status = "no_answer"
            elif target is None:
                status = "answer_unresolved"
            else:
                search = await asyncio.to_thread(
                    search_paths,
                    triples,
                    topic,
                    target,
                    clauses,
                    request.max_hops,
                    self.settings.search_max_states,
                )
                complete, states = search.complete, search.states_examined
                path = search.paths[0] if search.paths else None
                alternatives = search.paths[1:]
                if not complete:
                    status = "search_incomplete"
                    warnings.append(
                        "Search hit its state limit; path existence and ranking may be incomplete"
                    )
                elif path is not None:
                    status = "path_found"
                else:
                    status = "no_path_within_budget"
            highlighted_edges = {s.triple_id for s in path.steps} if path else set()
            highlighted_nodes = set(path.nodes) if path else set()
            node_ids = sorted({v for t in triples for v in (t.subject, t.object)} | set(anchors))
            entities = await asyncio.to_thread(self.store.entities, node_ids)
            nodes = [
                GraphNode(
                    **e,
                    is_topic=e["id"] == topic,
                    is_answer=e["id"] == target,
                    on_path=e["id"] in highlighted_nodes,
                )
                for e in entities
            ]
            edges = [
                GraphEdge(
                    id=t.id,
                    source=t.subject,
                    target=t.object,
                    predicate=t.predicate,
                    label=labels.get(t.predicate, token_label(t.predicate)),
                    score=t.score,
                    origin=t.origin,
                    on_path=t.id in highlighted_edges,
                )
                for t in triples
            ]
            if path and path.hops == 0:
                warnings.append(
                    "Topic and predicted answer are the same entity; the path has zero edges"
                )
            return QueryResponse(
                request_id=str(uuid.uuid4()),
                question=question,
                topic_entity=topic,
                answer_entity=target,
                answer=answer.answer,
                explanation=answer.explanation,
                parser_answer=parser_answer,
                parser_disagreement=disagreement,
                clauses=clauses,
                status=status,
                nodes=nodes,
                edges=edges,
                highlighted_path=path,
                alternative_paths=alternatives,
                search_complete=complete,
                states_examined=states,
                warnings=warnings,
                metadata={
                    "k": request.k,
                    "max_hops": request.max_hops,
                    "retrieved_triples": len(retrieved),
                    "expanded_triples": len(triples),
                    "candidate_triples": len(candidates),
                    "candidate_source": source,
                    "topic_entities": topics,
                    "graph_scope": self.store.scope(),
                    "retriever": self.retriever.metadata,
                    "llm_model": self.settings.llm_model,
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                    "predicted_answer_raw": answer.answer_entity,
                    "evidence_interpretation": "A recovered graph route is structural evidence, not a guarantee of semantic correctness",
                },
            )


def frontend_payload(response: QueryResponse) -> dict:
    """Legacy TRACE payload; canonical IDs and nullable scores remain in evidence."""
    labels = {n.id: n.label for n in response.nodes}
    edges = {e.id: e for e in response.edges}
    path = response.highlighted_path
    return {
        "clauses": {
            "clauses": [
                {
                    "node1": labels[edges[s.triple_id].source],
                    "relationship": edges[s.triple_id].label,
                    "node2": labels[edges[s.triple_id].target],
                }
                for s in path.steps
            ]
            if path
            else []
        },
        "graph": {
            "triples": [
                {
                    "node1": labels[e.source],
                    "relationship": e.label,
                    "node2": labels[e.target],
                    "score": e.score if e.score is not None else 0.0,
                }
                for e in response.edges
            ]
        },
        "answer": {"answer": response.answer, "reasoning": response.explanation},
        "evidence": response.model_dump(),
    }
