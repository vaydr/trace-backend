"""Live pipeline evaluation; gold answers are compared only after inference."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from itertools import islice
from pathlib import Path
from urllib.parse import urlsplit

from .data import file_sha256, read_examples
from .errors import TraceError
from .llm import LabelsOutput
from .prompts import ANSWER_SYSTEM, HUMANIZE_SYSTEM, PARSER_SYSTEM
from .schema import AnswerOutput, ParserOutput, QueryRequest


def _json_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _endpoint_origin(url: str) -> str | None:
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"{parsed.scheme}://{host}{port}"
    except ValueError:
        return None


def _configuration(pipeline) -> dict:
    settings = getattr(pipeline, "settings", None)
    llm_settings = getattr(getattr(pipeline, "llm", None), "settings", None)
    return {
        "llm": {
            "model": llm_settings.llm_model,
            "endpoint_origin": _endpoint_origin(llm_settings.llm_base_url),
            "timeout_seconds": llm_settings.llm_timeout,
            "attempts": llm_settings.llm_attempts,
            "max_prompt_chars": llm_settings.max_prompt_chars,
        }
        if llm_settings is not None
        else None,
        "pipeline": {
            "search_max_states": settings.search_max_states,
            "candidate_max_triples": settings.candidate_max_triples,
            "max_concurrent_queries": settings.max_concurrent_queries,
        }
        if settings is not None
        else None,
        "query_defaults": {
            name: QueryRequest.model_fields[name].default
            for name in ("k", "max_hops", "candidate_hops")
        },
    }


async def evaluate_pipeline(
    pipeline, input_path: Path, output: Path, limit: int | None = None
) -> dict:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    input_sha256 = file_sha256(input_path)
    graph_sha256 = pipeline.store.fingerprint()
    count = 0
    # Validate selected records before inference; gold labels are evaluation-only.
    for ex in islice(read_examples(input_path), limit):
        stored = pipeline.store.question(ex.id)
        mismatches = [
            field
            for field in ("question", "topic_entities", "split")
            if getattr(ex, field) != getattr(stored, field)
        ]
        if {t.key for t in ex.triples} != {t.key for t in stored.triples}:
            mismatches.append("triples")
        if mismatches:
            raise ValueError(
                f"Evaluation input {ex.id} differs from the graph database: "
                + ", ".join(mismatches)
            )
        count += 1
    if not count:
        raise ValueError("No evaluation examples")
    configuration = _configuration(pipeline)
    statuses = Counter()
    answers_correct = 0
    latencies = []
    records = []
    for ex in islice(read_examples(input_path), limit):
        started = time.perf_counter()
        try:
            result = await pipeline.query(QueryRequest(question_id=ex.id, question=ex.question))
            gold = set(ex.answers)
            correct = result.answer_entity in gold
            answers_correct += int(correct)
            statuses[result.status] += 1
            response = result.model_dump(mode="json")
            records.append(
                {
                    "id": ex.id,
                    "status": result.status,
                    "answer_entity": result.answer_entity,
                    "answer_exact_match": correct,
                    "path_hops": result.highlighted_path.hops if result.highlighted_path else None,
                    "search_complete": result.search_complete,
                    "parser_disagreement": result.parser_disagreement,
                    "request_id": result.request_id,
                    "response": response,
                    "response_sha256": _json_sha256(response),
                }
            )
        except TraceError as error:
            statuses["error"] += 1
            records.append({"id": ex.id, "status": "error", "error_code": error.code})
        latencies.append(time.perf_counter() - started)
    count = len(records)
    if file_sha256(input_path) != input_sha256:
        raise ValueError("Evaluation input changed during evaluation; report not published")
    graph_sha256_after = pipeline.store.fingerprint()
    retriever_metadata = getattr(pipeline.retriever, "metadata", {})
    provenance = {
        "input_sha256": input_sha256,
        "graph_sha256": graph_sha256,
        "graph_sha256_after": graph_sha256_after,
        "graph_changed_during_evaluation": graph_sha256 != graph_sha256_after,
        "checkpoint_sha256": retriever_metadata.get("checkpoint_sha256"),
        "checkpoint_data_fingerprint": retriever_metadata.get("data_fingerprint"),
        "retriever": retriever_metadata,
        "configuration": configuration,
        "configuration_sha256": _json_sha256(configuration),
        "bundled_prompts_sha256": {
            name: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for name, prompt in {
                "answer": ANSWER_SYSTEM,
                "parser": PARSER_SYSTEM,
                "humanize": HUMANIZE_SYSTEM,
            }.items()
        },
        "bundled_output_schemas_sha256": {
            name: _json_sha256(model.model_json_schema())
            for name, model in {
                "answer": AnswerOutput,
                "parser": ParserOutput,
                "humanize": LabelsOutput,
            }.items()
        },
    }
    summary = {
        "examples": count,
        "answer_entity_exact_match": answers_correct / count,
        "status_counts": dict(statuses),
        "mean_latency_seconds": sum(latencies) / count,
        "error_rate": statuses["error"] / count,
        "note": "Includes failures in the denominator; structural paths are not manually judged semantic validity",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "provenance": provenance, "predictions": records}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return summary
