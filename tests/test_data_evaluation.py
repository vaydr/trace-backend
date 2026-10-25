import hashlib
import json
from types import SimpleNamespace

import pytest

from trace_backend.config import Settings
from trace_backend.data import file_sha256, normalize_record, read_examples, write_examples
from trace_backend.errors import TraceError
from trace_backend.evaluation import evaluate_pipeline
from trace_backend.graph import GraphStore
from trace_backend.llm import LabelsOutput, LLMClient
from trace_backend.pipeline import Pipeline
from trace_backend.prompts import ANSWER_SYSTEM, HUMANIZE_SYSTEM, PARSER_SYSTEM
from trace_backend.schema import (
    AnswerOutput,
    Example,
    ParserOutput,
    QueryResponse,
    ScoredTriple,
    Triple,
)


def test_normalization_keeps_opaque_identifiers_and_deduplicates_facts():
    row = {
        "id": "q",
        "question": "Q?",
        "q_entity": ["m.001"],
        "a_entity": ["Name"],
        "graph": [["m.001", "r", "Name"], ["m.001", "r", "Name"]],
    }
    result = normalize_record(row, "test")
    assert len(result.triples) == 1
    assert result.triples[0].subject == "m.001"


def test_reader_rejects_duplicate_ids(tmp_path):
    ex = Example(
        id="q",
        question="Q?",
        topic_entities=["a"],
        answers=["b"],
        triples=[Triple(subject="a", predicate="p", object="b")],
        split="test",
    )
    path = tmp_path / "test.jsonl"
    write_examples(path, [ex, ex])
    with pytest.raises(ValueError, match="duplicate"):
        list(read_examples(path))


def test_lazy_dataset_initializes_before_output_handle_is_opened(tmp_path):
    ex = Example(
        id="q", question="Q?", topic_entities=["a"], answers=["b"], triples=[], split="test"
    )

    def lazy_rows():
        assert not list(tmp_path.glob("*.tmp"))
        yield ex

    target = tmp_path / "test.jsonl"
    assert write_examples(target, lazy_rows()) == 1
    assert list(read_examples(target)) == [ex]


def test_replaced_temporary_path_fails_instead_of_publishing_false_success(tmp_path):
    ex = Example(
        id="q", question="Q?", topic_entities=["a"], answers=["b"], triples=[], split="test"
    )
    target = tmp_path / "test.jsonl"
    write_examples(target, [ex])
    original = target.read_bytes()

    class ReplacedFile:
        def model_dump_json(self):
            [temporary] = tmp_path.glob("*.tmp")
            temporary.unlink()
            temporary.write_bytes(b"")
            return ex.model_dump_json()

    with pytest.raises(ValueError, match="integrity check failed"):
        write_examples(target, [ReplacedFile()])
    assert target.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


async def test_pipeline_evaluation_counts_provider_failures(tmp_path):
    ex = Example(
        id="q", question="Q?", topic_entities=["a"], answers=["gold"], triples=[], split="test"
    )
    path = tmp_path / "test.jsonl"
    write_examples(path, [ex])
    store = GraphStore(tmp_path / "graph.sqlite")
    store.initialize()
    store.add_examples([ex])

    class FailedPipeline:
        retriever = SimpleNamespace(metadata={"checkpoint_sha256": "test-checkpoint"})

        def __init__(self):
            self.store = store

        async def query(self, request):
            assert request.question_id == "q"
            assert request.question == "Q?"
            assert request.topic_entity is None
            assert "gold" not in request.model_dump_json()
            raise TraceError("llm_unavailable", "test")

    result = await evaluate_pipeline(FailedPipeline(), path, tmp_path / "eval.json")
    assert result["examples"] == 1
    assert result["error_rate"] == 1
    assert result["answer_entity_exact_match"] == 0
    assert json.loads((tmp_path / "eval.json").read_text())["predictions"][0] == {
        "id": "q",
        "status": "error",
        "error_code": "llm_unavailable",
    }
    provenance = json.loads((tmp_path / "eval.json").read_text())["provenance"]
    assert provenance["input_sha256"] == file_sha256(path)
    assert provenance["graph_sha256"] == store.fingerprint()
    assert provenance["graph_changed_during_evaluation"] is False
    assert provenance["checkpoint_sha256"] == "test-checkpoint"
    assert provenance["configuration"]["llm"] is None
    assert provenance["configuration"]["pipeline"] is None
    assert provenance["configuration"]["query_defaults"] == {
        "k": 50,
        "max_hops": 8,
        "candidate_hops": None,
    }


async def test_evaluation_retains_full_pipeline_evidence_and_content_digest(tmp_path):
    triples = [
        Triple(subject="X", predicate="p", object="A"),
        Triple(subject="X", predicate="q", object="B"),
        Triple(subject="A", predicate="r", object="Y"),
        Triple(subject="Y", predicate="s", object="B"),
    ]
    ex = Example(
        id="q",
        question="Where does A lead?",
        topic_entities=["A"],
        answers=["B"],
        triples=triples,
        split="test",
    )
    store = GraphStore(tmp_path / "graph.sqlite")
    store.initialize()
    store.add_examples([ex])
    input_path = tmp_path / "test.jsonl"
    write_examples(input_path, [ex])

    class Retriever:
        metadata = {"name": "offline-fixture"}

        def retrieve(self, question, topics, candidates, k):
            return [ScoredTriple(**candidates[0].model_dump(), score=0.75)]

    class Provider:
        async def answer(self, question, retrieved):
            return AnswerOutput(
                answer_entity="B", answer="B", explanation="A’s route reaches B through X."
            )

        async def parse(self, question, predicates, labels):
            return ParserOutput(answer_entity="Y", clauses=[["p", "q"], ["r", "s"]])

    pipeline = Pipeline(store, Retriever(), Provider(), Settings(_env_file=None))
    output = tmp_path / "evaluation.json"
    summary = await evaluate_pipeline(pipeline, input_path, output)
    report = json.loads(output.read_text())
    prediction = report["predictions"][0]
    response = prediction["response"]
    canonical = QueryResponse.model_validate(response)
    assert set(response) == set(QueryResponse.model_fields)
    assert summary["answer_entity_exact_match"] == 1
    assert prediction["path_hops"] == 2
    assert prediction["request_id"] == canonical.request_id
    assert canonical.answer == "B"
    assert canonical.explanation == "A’s route reaches B through X."
    assert {n.id for n in canonical.nodes} == {"A", "B", "X", "Y"}
    assert {e.id for e in canonical.edges} == {t.id for t in triples}
    assert {(e.origin, e.score) for e in canonical.edges} == {
        ("retrieved", 0.75),
        ("expansion", None),
    }
    assert canonical.clauses == [["p", "q"], ["r", "s"]]
    assert canonical.parser_answer == "Y"
    assert canonical.parser_disagreement is True
    assert canonical.search_complete is True
    assert canonical.highlighted_path.nodes == ["A", "X", "B"]
    assert canonical.highlighted_path.steps[0].reversed is True
    assert canonical.highlighted_path.steps[0].triple_id == triples[0].id
    assert [p.nodes for p in canonical.alternative_paths] == [["A", "Y", "B"]]

    def digest(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()

    # The saved-content digest is stable across JSON formatting/key order, while
    # an altered explanation changes it. Fresh requests have different IDs/timing.
    assert prediction["response_sha256"] == digest(response)
    reordered = dict(reversed(list(response.items())))
    assert prediction["response_sha256"] == digest(json.loads(json.dumps(reordered, indent=4)))
    assert prediction["response_sha256"] != digest({**response, "explanation": "Changed"})


@pytest.mark.parametrize(
    "field,value",
    [
        ("question", "Different question?"),
        ("topic_entities", ["other"]),
        ("split", "validation"),
        ("triples", [Triple(subject="a", predicate="p", object="gold")]),
    ],
)
async def test_evaluation_rejects_reused_id_with_different_metadata_before_inference(
    tmp_path, field, value
):
    ex = Example(
        id="q", question="Q?", topic_entities=["a"], answers=["gold"], triples=[], split="test"
    )
    store = GraphStore(tmp_path / "graph.sqlite")
    store.initialize()
    store.add_examples([ex])
    path = tmp_path / "test.jsonl"
    write_examples(path, [ex.model_copy(update={field: value})])

    class UnexpectedPipeline:
        def __init__(self):
            self.store = store

        async def query(self, request):
            pytest.fail("Inference must not run when evaluation metadata differs")

    with pytest.raises(ValueError, match=field):
        await evaluate_pipeline(UnexpectedPipeline(), path, tmp_path / "eval.json")
    assert not (tmp_path / "eval.json").exists()


@pytest.mark.parametrize(
    "endpoint,expected_origin",
    [
        (
            "https://user:password@models.example:8443/private-token/v1?key=query-token#fragment-token",
            "https://models.example:8443",
        ),
        ("http://user:password@[::1]:8000/private-token", "http://[::1]:8000"),
        ("https://models.example:invalid/private-token", None),
    ],
)
async def test_evaluation_records_configuration_and_prompt_provenance_without_secrets(
    tmp_path, endpoint, expected_origin
):
    ex = Example(
        id="q", question="Q?", topic_entities=["a"], answers=["gold"], triples=[], split="test"
    )
    path = tmp_path / "test.jsonl"
    write_examples(path, [ex])
    store = GraphStore(tmp_path / "graph.sqlite")
    store.initialize()
    store.add_examples([ex])
    settings = Settings(
        _env_file=None,
        llm_model="unused-pipeline-model",
        search_max_states=4321,
        candidate_max_triples=1234,
        max_concurrent_queries=1,
        api_key="api-secret",
    )
    llm_settings = Settings(
        _env_file=None,
        llm_model="research-model-v2",
        llm_base_url=endpoint,
        llm_api_key="llm-secret",
        llm_timeout=12.5,
        llm_attempts=2,
        max_prompt_chars=2345,
    )

    class FailedPipeline:
        retriever = SimpleNamespace(metadata={})

        async def query(self, request):
            assert request.k == 50
            assert request.max_hops == 8
            assert request.candidate_hops is None
            raise TraceError("llm_unavailable", "test")

    pipeline = FailedPipeline()
    pipeline.store = store
    pipeline.settings = settings
    pipeline.llm = LLMClient(llm_settings)
    report_path = tmp_path / "evaluation.json"
    await evaluate_pipeline(pipeline, path, report_path)
    serialized = report_path.read_text()
    provenance = json.loads(serialized)["provenance"]
    config = provenance["configuration"]
    assert config["llm"] == {
        "model": "research-model-v2",
        "endpoint_origin": expected_origin,
        "timeout_seconds": 12.5,
        "attempts": 2,
        "max_prompt_chars": 2345,
    }
    assert config["pipeline"] == {
        "search_max_states": 4321,
        "candidate_max_triples": 1234,
        "max_concurrent_queries": 1,
    }
    assert config["query_defaults"] == {"k": 50, "max_hops": 8, "candidate_hops": None}
    for secret in (
        "api-secret",
        "llm-secret",
        "password",
        "private-token",
        "query-token",
        "fragment-token",
        "Authorization",
        "unused-pipeline-model",
    ):
        assert secret not in serialized
    expected_config_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    assert provenance["configuration_sha256"] == expected_config_digest
    assert provenance["bundled_prompts_sha256"] == {
        name: hashlib.sha256(prompt.encode()).hexdigest()
        for name, prompt in {
            "answer": ANSWER_SYSTEM,
            "parser": PARSER_SYSTEM,
            "humanize": HUMANIZE_SYSTEM,
        }.items()
    }
    for name, model in {
        "answer": AnswerOutput,
        "parser": ParserOutput,
        "humanize": LabelsOutput,
    }.items():
        schema_digest = hashlib.sha256(
            json.dumps(
                model.model_json_schema(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        assert provenance["bundled_output_schemas_sha256"][name] == schema_digest

    await evaluate_pipeline(pipeline, path, report_path)
    repeated = json.loads(report_path.read_text())["provenance"]
    assert repeated == provenance
    pipeline.llm.settings = llm_settings.model_copy(update={"llm_model": "research-model-v3"})
    await evaluate_pipeline(pipeline, path, report_path)
    changed = json.loads(report_path.read_text())["provenance"]
    assert changed["configuration_sha256"] != expected_config_digest
    assert changed["bundled_prompts_sha256"] == provenance["bundled_prompts_sha256"]
