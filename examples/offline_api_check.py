"""Replay real-data API assembly with fixed offline responses, not model answer evaluation."""

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from trace_backend.api import create_app
from trace_backend.config import Settings
from trace_backend.graph import GraphStore
from trace_backend.llm import LLMClient
from trace_backend.pipeline import Pipeline
from trace_backend.retrieval import SubgraphRetriever

QUESTION_ID = "WebQTrn-0"
CLAUSES = [["people.person.sibling_s", "people.sibling_relationship.sibling"]]
FIXTURES = {
    "AnswerOutput": {
        "answer_entity": "Jaxon Bieber",
        "answer": "Jaxon Bieber",
        "explanation": "Fixed offline response for API assembly; no model answer was generated.",
    },
    "ParserOutput": {"answer_entity": "Jaxon Bieber", "clauses": CLAUSES},
}


def provider_response(request):
    schema_name = json.loads(request.content)["response_format"]["json_schema"]["name"]
    message = {"content": json.dumps(FIXTURES[schema_name])}
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": message}]})


def check_evidence(body, store):
    assert body["status"] == "path_found" and body["search_complete"], body["status"]
    assert body["answer_entity"] == "Jaxon Bieber" and body["topic_entity"] == "Justin Bieber"
    assert body["clauses"] == CLAUSES and body["highlighted_path"]["hops"] == 2
    with store.connection() as connection:
        for edge in body["edges"]:
            stored = connection.execute(
                "SELECT subject,predicate,object FROM triples WHERE id=?", (edge["id"],)
            ).fetchone()
            expected = (edge["source"], edge["predicate"], edge["target"])
            assert stored is not None and tuple(stored) == expected


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("db", "checkpoint", "cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    store = GraphStore(args.db)
    example = store.question(QUESTION_ID)
    assert example.question == "what is the name of justin bieber brother"
    assert example.topic_entities == ["Justin Bieber"]
    retriever = SubgraphRetriever(args.checkpoint, args.cache, args.device)
    settings = Settings(
        _env_file=None,
        api_key=None,
        llm_api_key="offline-fixture",
        llm_base_url="https://offline.invalid/v1",
        llm_model="fixed-offline-fixture",
    )
    endpoints = ["/v1/query", "/v1/trace/query", "/v1/trace/stream"]
    request = {"question_id": QUESTION_ID, "question": example.question, "k": 50}
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider_response)) as provider:
        pipeline = Pipeline(store, retriever, LLMClient(settings, provider), settings)
        with TestClient(create_app(settings, pipeline)) as client:
            responses = [client.post(endpoint, json=request) for endpoint in endpoints]
            for response in responses:
                response.raise_for_status()
            canonical, legacy = responses[0].json(), responses[1].json()
            lines = responses[2].text.splitlines()
            events = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
            assert responses[2].headers["content-type"].startswith("text/event-stream")
            assert events[-1] == {"type": "data-kgqa-step", "data": "complete"}
            streamed = next(e["data"] for e in events if e["type"] == "data-kgqa-evidence")
            for body in (canonical, legacy["evidence"], streamed):
                check_evidence(body, store)
                assert body["edges"] == canonical["edges"]
                assert body["highlighted_path"] == canonical["highlighted_path"]
            assert legacy["answer"]["answer"] == "Jaxon Bieber"
            assert len(legacy["graph"]["triples"]) == len(canonical["edges"])
            assert len(legacy["clauses"]["clauses"]) == 2
    summary = {
        "kind": "offline_api_assembly_check",
        "model_answer_evaluation": False,
        "question_id": QUESTION_ID,
        "checked_endpoints": endpoints,
        "candidate_triples": canonical["metadata"]["candidate_triples"],
        "retrieved_triples": canonical["metadata"]["retrieved_triples"],
        "nodes": len(canonical["nodes"]),
        "edges": len(canonical["edges"]),
        "path_hops": canonical["highlighted_path"]["hops"],
        "agreement": canonical["highlighted_path"]["agreement"],
        "states_examined": canonical["states_examined"],
        "device": str(retriever.device),
        "graph_sha256": store.fingerprint(),
        "checkpoint_sha256": retriever.metadata["checkpoint_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "summary": summary,
        "fixed_provider_responses": FIXTURES,
        "canonical_response": canonical,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
