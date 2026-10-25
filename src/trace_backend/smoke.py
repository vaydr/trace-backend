"""Synthetic end-to-end test with a hash encoder and fixed provider responses."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from .config import Settings
from .data import write_examples
from .graph import GraphStore
from .schema import AnswerOutput, Example, ParserOutput, Triple


def smoke(output: Path) -> dict:
    from fastapi.testclient import TestClient

    from .api import create_app
    from .ml.training import evaluate_model, load_checkpoint, prepare, train
    from .pipeline import Pipeline
    from .retrieval import SubgraphRetriever

    report_path = output / "smoke-report.json"
    output = output / "runs" / uuid.uuid4().hex[:12]
    config = {
        "seed": 42,
        "device": "cpu",
        "encoder": {
            "kind": "hash",
            "name": "test-only-hash",
            "dimension": 16,
            "revision": "1",
            "batch_size": 8,
        },
        "model": {"topic_pe": True, "forward_rounds": 2, "reverse_rounds": 2},
        "training": {
            "epochs": 4,
            "patience": 4,
            "learning_rate": 0.01,
            "eval_k": 2,
            "score_batch_size": 2,
            "gradient_clip": 1.0,
        },
    }
    examples = []
    for split in ("train", "validation", "test"):
        triples = [
            Triple(subject=f"{split} Alice", predicate="person.works_at", object=f"{split} Lab"),
            Triple(subject=f"{split} Lab", predicate="organization.city", object=f"{split} Boston"),
            Triple(subject=f"{split} Alice", predicate="person.likes", object=f"{split} Music"),
            Triple(subject=f"{split} Rome", predicate="city.country", object=f"{split} Italy"),
        ]
        example = Example(
            id=f"smoke-{split}",
            question=f"In which city is {split} Alice's employer?",
            topic_entities=[f"{split} Alice"],
            answers=[f"{split} Boston"],
            triples=triples,
            split=split,
        )
        examples.append(example)
        write_examples(output / "data" / f"{split}.jsonl", [example])
    manifest = prepare(output / "data", output / "prepared", config, output / "embeddings.sqlite")
    training = train(output / "prepared", output / "model", config)
    restored, _ = load_checkpoint(output / "model" / "best.pt", "cpu")
    evaluation = evaluate_model(
        restored, output / "prepared", manifest["splits"]["test"], "cpu", 2, 2
    )
    store = GraphStore(output / "graph.sqlite")
    store.initialize("synthetic_test")
    store.add_examples(examples)
    retriever = SubgraphRetriever(
        output / "model" / "best.pt", output / "embeddings.sqlite", "cpu", allow_test_encoder=True
    )

    class FixtureLLM:
        async def answer(self, question, triples):
            return AnswerOutput(
                answer_entity="test Boston",
                answer="test Boston",
                explanation="Synthetic test response; no live model call",
            )

        async def parse(self, question, predicates, labels):
            return ParserOutput(
                answer_entity="test Boston", clauses=[["person.works_at", "organization.city"]]
            )

    settings = Settings(
        graph_db=store.path, checkpoint=output / "model" / "best.pt", llm_api_key=None
    )
    pipeline = Pipeline(store, retriever, FixtureLLM(), settings)
    with TestClient(create_app(settings, pipeline)) as client:
        response = client.post("/v1/query", json={"question_id": "smoke-test", "k": 4})
        response.raise_for_status()
        body = response.json()
        if body["status"] != "path_found" or body["highlighted_path"]["hops"] != 2:
            raise AssertionError("Synthetic end-to-end query did not recover the expected path")
        legacy = client.post("/v1/trace/query", json={"question_id": "smoke-test", "k": 4})
        legacy.raise_for_status()
        stream = client.post("/v1/trace/stream", json={"question_id": "smoke-test", "k": 4})
        if "data-kgqa-evidence" not in stream.text:
            raise AssertionError("Missing streamed evidence")
    report = {
        "kind": "synthetic_software_validation",
        "full_webqsp_training": False,
        "paid_model_calls": False,
        "training": training,
        "retrieval_evaluation": {k: v for k, v in evaluation.items() if k != "predictions"},
        "query_status": body["status"],
        "path_hops": body["highlighted_path"]["hops"],
        "api_and_frontend_adapter": "passed",
    }
    report["run_directory"] = str(output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
