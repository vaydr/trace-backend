import json

import httpx
import pytest
from fastapi.testclient import TestClient

from trace_backend.api import create_app
from trace_backend.config import Settings
from trace_backend.errors import TraceError
from trace_backend.graph import GraphStore
from trace_backend.llm import LLMClient
from trace_backend.pipeline import Pipeline, frontend_payload
from trace_backend.schema import (
    AnswerOutput,
    Example,
    ParserOutput,
    QueryRequest,
    ScoredTriple,
    Triple,
)


class Retriever:
    metadata = {"name": "test-fixture"}

    def retrieve(self, question, topics, candidates, k):
        return [ScoredTriple(**candidates[0].model_dump(), score=0.75)]


class Provider:
    target = "B"
    parser_answer = "B"
    parse_error = False

    async def answer(self, question, triples):
        return AnswerOutput(
            answer_entity=self.target, answer=self.target or "Unavailable", explanation="test"
        )

    async def parse(self, question, predicates, labels):
        if self.parse_error:
            raise TraceError("parse_error", "test")
        return ParserOutput(answer_entity=self.parser_answer, clauses=[["p", "q"]])


@pytest.fixture
def pipeline(tmp_path):
    store = GraphStore(tmp_path / "graph.sqlite")
    store.initialize()
    triples = [
        Triple(subject="A", predicate="p", object="X"),
        Triple(subject="X", predicate="q", object="B"),
        Triple(subject="Y", predicate="r", object="Z"),
    ]
    store.add_examples(
        [
            Example(
                id="test",
                question="Question?",
                topic_entities=["A"],
                answers=["SECRET-GOLD"],
                triples=triples,
                split="test",
            )
        ]
    )
    settings = Settings(_env_file=None, llm_api_key=None, search_max_states=100)
    return Pipeline(store, Retriever(), Provider(), settings)


async def test_real_anchor_expansion_closes_gap_without_fabricating_edges(pipeline):
    result = await pipeline.query(QueryRequest(question_id="test"))
    assert result.status == "path_found"
    assert result.highlighted_path.nodes == ["A", "X", "B"]
    assert len(result.edges) == 2
    assert result.edges[1].score is None
    assert result.edges[1].origin == "expansion"
    assert "SECRET-GOLD" not in result.model_dump_json()


async def test_free_question_default_scores_entire_supplied_graph(pipeline):
    result = await pipeline.query(QueryRequest(question="Q?", topic_entity="A"))
    assert result.metadata["candidate_source"] == "supplied_graph"
    assert result.metadata["candidate_triples"] == 3  # includes disconnected Y-r-Z


async def test_candidate_prefilter_is_explicit_experiment(pipeline):
    result = await pipeline.query(QueryRequest(question="Q?", topic_entity="A", candidate_hops=1))
    assert result.metadata["candidate_triples"] == 1
    assert any("differs from paper" in w for w in result.warnings)


async def test_disagreement_retains_backend_target(pipeline):
    pipeline.llm.parser_answer = "Z"
    result = await pipeline.query(QueryRequest(question_id="test"))
    assert result.parser_disagreement
    assert result.answer_entity == "B"
    assert result.highlighted_path.nodes[-1] == "B"


async def test_ambiguous_parser_name_retains_valid_predicate_hints(pipeline):
    with pipeline.store.connection(write=True) as c:
        c.execute("UPDATE entities SET label='Shared name' WHERE id IN ('Y','Z')")
    pipeline.llm.parser_answer = "Shared name"
    result = await pipeline.query(QueryRequest(question_id="test"))
    assert result.clauses == [["p", "q"]]
    assert result.highlighted_path.agreement == 2.0
    assert result.parser_disagreement is None
    assert result.answer_entity == "B"
    assert any("ambiguous" in warning for warning in result.warnings)


async def test_cached_token_labels_still_warn_about_llm_humanization(pipeline):
    for p in pipeline.store.predicates():
        pipeline.store.save_predicate_label(p, p, "tokens")
    result = await pipeline.query(QueryRequest(question_id="test"))
    assert any("run trace humanize" in warning for warning in result.warnings)


async def test_parse_failure_falls_back_and_disconnected_target_is_explicit(pipeline):
    pipeline.llm.parse_error = True
    result = await pipeline.query(QueryRequest(question_id="test"))
    assert result.status == "path_found"
    assert result.clauses == []
    pipeline.llm.target = "Z"
    result = await pipeline.query(QueryRequest(question_id="test"))
    assert result.status == "no_path_within_budget"
    assert result.highlighted_path is None
    assert not any(e.on_path for e in result.edges)


async def test_unknown_answer_and_empty_answer(pipeline):
    pipeline.llm.target = "nonexistent"
    assert (await pipeline.query(QueryRequest(question_id="test"))).status == "answer_unresolved"
    pipeline.llm.target = None
    assert (await pipeline.query(QueryRequest(question_id="test"))).status == "no_answer"


async def test_state_limit_is_not_no_path(pipeline):
    pipeline.settings.search_max_states = 1
    result = await pipeline.query(QueryRequest(question_id="test"))
    assert result.status == "search_incomplete"
    assert not result.search_complete


async def test_reverse_path_legacy_clauses_preserve_stored_relation_direction(pipeline):
    pipeline.llm.target = "A"
    pipeline.llm.parser_answer = "A"
    result = await pipeline.query(QueryRequest(question="Q?", topic_entity="B"))
    assert result.highlighted_path.nodes == ["B", "X", "A"]
    assert all(step.reversed for step in result.highlighted_path.steps)
    payload = frontend_payload(result)
    assert payload["clauses"]["clauses"] == [
        {"node1": "X", "relationship": "q", "node2": "B"},
        {"node1": "A", "relationship": "p", "node2": "X"},
    ]
    facts = {(t["node1"], t["relationship"], t["node2"]) for t in payload["graph"]["triples"]}
    assert all(
        (t["node1"], t["relationship"], t["node2"]) in facts for t in payload["clauses"]["clauses"]
    )


def test_api_contract_auth_errors_and_stream(pipeline):
    settings = Settings(_env_file=None, api_key="test-key")
    with TestClient(create_app(settings, pipeline)) as client:
        assert client.get("/healthz").status_code == 200
        assert client.post("/v1/query", json={"question_id": "test"}).status_code == 401
        headers = {"X-API-Key": "test-key"}
        assert client.get("/readyz", headers=headers).status_code == 200
        assert client.post("/v1/query", json={}, headers=headers).status_code == 422
        assert (
            client.post("/v1/query", json={"question_id": "missing"}, headers=headers).status_code
            == 404
        )
        assert "SECRET-GOLD" not in client.get("/v1/questions/test", headers=headers).text
        result = client.post(
            "/v1/trace/query", json={"question_id": "test"}, headers=headers
        ).json()
        assert result["graph"]["triples"]
        assert result["clauses"]["clauses"][0]["relationship"] == "p"
        assert result["evidence"]["highlighted_path"]["hops"] == 2
        stream = client.post("/v1/trace/stream", json={"question_id": "test"}, headers=headers)
        assert stream.headers["content-type"].startswith("text/event-stream")
        assert '"data-kgqa-step", "data": "complete"' in stream.text


def test_missing_artifacts_explain_not_ready(tmp_path):
    settings = Settings(
        _env_file=None, graph_db=tmp_path / "absent.db", checkpoint=tmp_path / "absent.pt"
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/healthz").status_code == 200
        result = client.get("/readyz")
        assert result.status_code == 503
        assert result.json()["error"]["code"] == "not_ready"


async def test_llm_json_schema_http_contract_and_unknown_predicates():
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"answer_entity":"B","clauses":[["unknown"]]}'},
                    }
                ]
            },
        )

    settings = Settings(_env_file=None, llm_api_key="test-secret")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        llm = LLMClient(settings, client)
        with pytest.raises(TraceError, match="unknown predicates"):
            await llm.parse("Q?", ["p"], {})
    assert captured[0].headers["authorization"] == "Bearer test-secret"
    assert b'"strict":true' in captured[0].content


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_provider_errors_are_bounded_and_do_not_expose_response_bodies(status):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(status, text="sensitive upstream response")

    settings = Settings(_env_file=None, llm_api_key="test", llm_attempts=2)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(TraceError) as caught:
            await LLMClient(settings, client).answer("Q?", [])
    assert "sensitive" not in str(caught.value)
    assert len(calls) == (1 if status == 401 else 2)


@pytest.fixture
def retry_delays(monkeypatch):
    delays = []

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("trace_backend.llm.asyncio.sleep", record_sleep)
    return delays


def completion_response(content):
    return httpx.Response(
        200,
        json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}]},
    )


@pytest.mark.parametrize(
    "error_type",
    [
        httpx.RemoteProtocolError,
        httpx.DecodingError,
        httpx.ProxyError,
        httpx.ReadError,
        httpx.ReadTimeout,
    ],
)
async def test_request_error_families_exhaust_retries_as_bounded_errors(error_type, retry_delays):
    calls = []

    def handler(request):
        calls.append(request)
        raise error_type("private upstream detail test-secret", request=request)

    settings = Settings(_env_file=None, llm_api_key="test-secret", llm_attempts=3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(TraceError) as caught:
            await LLMClient(settings, client).answer("Q?", [])
    assert caught.value.code == "llm_unavailable"
    assert caught.value.status_code == 502
    assert "private upstream detail" not in str(caught.value)
    assert "test-secret" not in str(caught.value)
    assert len(calls) == 3
    assert retry_delays == [0.25, 0.5]


async def test_transient_protocol_error_recovers_within_retry_budget(retry_delays):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("connection closed", request=request)
        return completion_response({"answer_entity": "B", "answer": "B", "explanation": "test"})

    settings = Settings(_env_file=None, llm_api_key="test", llm_attempts=3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        answer = await LLMClient(settings, client).answer("Q?", [])
    assert answer.answer_entity == "B"
    assert len(calls) == 2
    assert retry_delays == [0.25]


@pytest.mark.parametrize("failure_stage", ["answer", "parser"])
async def test_protocol_failure_preserves_canonical_and_stream_contracts(pipeline, failure_stage):
    captured = []

    def handler(request):
        captured.append(request)
        schema = json.loads(request.content)["response_format"]["json_schema"]["name"]
        if failure_stage == "answer" or schema == "ParserOutput":
            raise httpx.RemoteProtocolError("private upstream detail test-secret", request=request)
        return completion_response({"answer_entity": "B", "answer": "B", "explanation": "test"})

    settings = Settings(_env_file=None, llm_api_key="test-secret", llm_attempts=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as provider_client:
        pipeline.llm = LLMClient(settings, provider_client)
        with TestClient(create_app(settings, pipeline)) as client:
            canonical = client.post("/v1/query", json={"question_id": "test"})
            stream = client.post("/v1/trace/stream", json={"question_id": "test"})
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: ")
    ]
    if failure_stage == "answer":
        assert canonical.status_code == 502
        assert canonical.json()["error"]["code"] == "llm_unavailable"
        assert [event["type"] for event in events] == [
            "data-kgqa-step",
            "data-kgqa-error",
            "data-kgqa-step",
        ]
        assert events[1]["data"]["code"] == "llm_unavailable"
        assert events[-1]["data"] == "error"
        assert len(captured) == 2
    else:
        assert canonical.status_code == 200
        evidence = canonical.json()
        assert evidence["status"] == "path_found"
        assert evidence["highlighted_path"]["nodes"] == ["A", "X", "B"]
        assert evidence["clauses"] == []
        assert any("llm_unavailable" in warning for warning in evidence["warnings"])
        streamed_evidence = next(
            event["data"] for event in events if event["type"] == "data-kgqa-evidence"
        )
        assert streamed_evidence["highlighted_path"]["hops"] == 2
        assert streamed_evidence["clauses"] == []
        assert events[-1] == {"type": "data-kgqa-step", "data": "complete"}
        assert all(event["type"] != "data-kgqa-error" for event in events)
        assert len(captured) == 4
    for secret in ("private upstream detail", "test-secret", "SECRET-GOLD"):
        assert secret not in canonical.text + stream.text
    assert all(b"SECRET-GOLD" not in request.content for request in captured)


@pytest.mark.parametrize("malformed_message", [None, "not a message object", []])
async def test_malformed_message_envelopes_are_bounded_and_parser_falls_back(
    pipeline, malformed_message
):
    parser_only_failure = False

    def handler(request):
        schema = json.loads(request.content)["response_format"]["json_schema"]["name"]
        if parser_only_failure and schema == "AnswerOutput":
            return completion_response({"answer_entity": "B", "answer": "B", "explanation": "test"})
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": malformed_message}]},
        )

    settings = Settings(_env_file=None, llm_api_key="test", llm_attempts=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        llm = LLMClient(settings, client)
        with pytest.raises(TraceError) as caught:
            await llm.answer("Q?", [])
        assert caught.value.code == "llm_invalid_output"
        assert caught.value.status_code == 502
        parser_only_failure = True
        pipeline.llm = llm
        result = await pipeline.query(QueryRequest(question_id="test"))
    assert result.status == "path_found"
    assert result.highlighted_path.nodes == ["A", "X", "B"]
    assert result.clauses == []
    assert any("llm_invalid_output" in warning for warning in result.warnings)
