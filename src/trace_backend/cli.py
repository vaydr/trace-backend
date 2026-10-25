from __future__ import annotations

import argparse
import asyncio
import gzip
import json
from pathlib import Path

from .config import Settings
from .data import download_webqsp, file_sha256, read_examples
from .errors import TraceError
from .graph import GraphStore
from .llm import LLMClient, token_label
from .schema import QueryRequest, Triple


def emit(value):
    print(json.dumps(value, indent=2, ensure_ascii=False))


async def humanize(store: GraphStore, client: LLMClient, batch_size: int, token_only: bool):
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    existing = store.predicate_label_sources()
    pending = [
        p
        for p in store.predicates()
        if p not in existing or (not token_only and existing[p] != "llm")
    ]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        labels = {p: token_label(p) for p in batch} if token_only else await client.humanize(batch)
        for p, label in labels.items():
            store.save_predicate_label(p, label, "tokens" if token_only else "llm")
    return {
        "new_labels": sum(p not in existing for p in pending),
        "upgraded_labels": sum(p in existing for p in pending),
        "processed_labels": len(pending),
        "label_source": "tokens" if token_only else "llm",
    }


def parser():
    p = argparse.ArgumentParser(description="TRACE knowledge graph backend")
    commands = p.add_subparsers(dest="command", required=True)
    download = commands.add_parser(
        "download", help="Download and normalize the pinned WebQSP release"
    )
    download.add_argument("--output", type=Path, default=Path("data/webqsp"))
    download.add_argument("--limit", type=int)
    graph = commands.add_parser(
        "build-graph", help="Build the indexed union of benchmark candidate graphs"
    )
    graph.add_argument("--input", type=Path, nargs="+", required=True)
    graph.add_argument("--db", type=Path, default=Path("data/graph.sqlite"))
    graph.add_argument("--scope", default="benchmark_union")
    imp = commands.add_parser(
        "import-graph", help="Add real subject/predicate/object TSV triples to an indexed KG"
    )
    imp.add_argument("--input", type=Path, required=True)
    imp.add_argument("--db", type=Path, default=Path("data/graph.sqlite"))
    imp.add_argument("--scope", default="provided_kg")
    imp.add_argument("--entities", type=Path, help="Optional JSONL records: id, label, source_urls")
    labels = commands.add_parser("humanize", help="Cache predicate labels offline")
    labels.add_argument(
        "--db", type=Path, help="Graph database (defaults to TRACE_GRAPH_DB or data/graph.sqlite)"
    )
    labels.add_argument("--batch-size", type=int, default=32)
    labels.add_argument(
        "--token-only", action="store_true", help="Use token labels without a model call"
    )
    prep = commands.add_parser(
        "prepare", help="Precompute cached text embeddings and weak training labels"
    )
    prep.add_argument("--data", type=Path, default=Path("data/webqsp"))
    prep.add_argument("--output", type=Path, default=Path("artifacts/prepared"))
    prep.add_argument("--cache", type=Path, default=Path("artifacts/embedding-cache.sqlite"))
    prep.add_argument("--config", type=Path, default=Path("configs/webqsp.yaml"))
    prep.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    training = commands.add_parser(
        "train", help="Train the SubgraphRAG retriever; save best and resumable last checkpoints"
    )
    training.add_argument("--prepared", type=Path, default=Path("artifacts/prepared"))
    training.add_argument("--output", type=Path, default=Path("artifacts/webqsp"))
    training.add_argument("--config", type=Path, default=Path("configs/webqsp.yaml"))
    training.add_argument("--resume", type=Path)
    training.add_argument("--epochs", type=int)
    training.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    evaluate = commands.add_parser("evaluate", help="Evaluate held-out retrieval without LLM calls")
    evaluate.add_argument("--prepared", type=Path, default=Path("artifacts/prepared"))
    evaluate.add_argument("--checkpoint", type=Path, default=Path("artifacts/webqsp/best.pt"))
    evaluate.add_argument("--split", choices=["train", "validation", "test"], default="test")
    evaluate.add_argument("--k", type=int, default=50)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    live_eval = commands.add_parser(
        "evaluate-pipeline",
        help="Evaluate answers and path states with live model calls (uses API quota)",
    )
    live_eval.add_argument("--input", type=Path, default=Path("data/webqsp/test.jsonl"))
    live_eval.add_argument(
        "--output", type=Path, default=Path("artifacts/pipeline-evaluation.json")
    )
    live_eval.add_argument("--limit", type=int)
    query = commands.add_parser("query", help="Query the running API")
    query.add_argument("--question")
    query.add_argument("--topic")
    query.add_argument("--question-id")
    query.add_argument("--url", default="http://127.0.0.1:8000")
    query.add_argument("--k", type=int, default=50)
    serve = commands.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    smoke = commands.add_parser(
        "smoke",
        help="Run synthetic training, checkpoint reload, and the API without downloads or paid model calls",
    )
    smoke.add_argument("--output", type=Path, default=Path("artifacts/smoke"))
    commands.add_parser(
        "doctor", help="Report available devices and artifact readiness without revealing keys"
    )
    return p


def run(args):
    if args.command == "download":
        emit(download_webqsp(args.output, args.limit))
    elif args.command == "build-graph":
        store = GraphStore(args.db)
        store.initialize(args.scope)
        for path in args.input:
            store.add_examples(read_examples(path))
        emit(store.stats())
    elif args.command == "import-graph":
        store = GraphStore(args.db)
        store.initialize(args.scope)

        def triples():
            opener = gzip.open if args.input.suffix == ".gz" else open
            with opener(args.input, "rt") as f:
                for i, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    fields = line.rstrip("\r\n").split("\t")
                    if len(fields) != 3:
                        raise ValueError(f"Expected three TSV fields on line {i}")
                    yield Triple(subject=fields[0], predicate=fields[1], object=fields[2])

        store.add_triples(triples())
        if args.entities:
            with store.connection(write=True) as c, args.entities.open() as f:
                for line in f:
                    record = json.loads(line)
                    c.execute(
                        "INSERT INTO entities(id,label,urls) VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET label=excluded.label,urls=excluded.urls",
                        (record["id"], record["label"], json.dumps(record.get("source_urls", []))),
                    )
        emit(store.stats())
    elif args.command == "humanize":
        settings = Settings()
        emit(
            asyncio.run(
                humanize(
                    GraphStore(args.db if args.db is not None else settings.graph_db),
                    LLMClient(settings),
                    args.batch_size,
                    args.token_only,
                )
            )
        )
    elif args.command in ("prepare", "train"):
        from .ml.training import load_config, prepare, train

        config = load_config(args.config)
        if args.device:
            config["device"] = args.device
        if args.command == "prepare":
            manifest = prepare(args.data, args.output, config, args.cache)
            emit(
                {
                    "fingerprint": manifest["fingerprint"],
                    "encoder": manifest["encoder"],
                    "counts": {s: len(r) for s, r in manifest["splits"].items()},
                }
            )
        else:
            if args.epochs is not None:
                if args.epochs < 1:
                    raise ValueError("epochs must be positive")
                config["training"]["epochs"] = args.epochs
            emit(train(args.prepared, args.output, config, args.resume))
    elif args.command == "evaluate":
        from .ml.training import evaluate_model, load_checkpoint, load_manifest

        if args.k < 1:
            raise ValueError("k must be positive")
        checkpoint_sha256 = file_sha256(args.checkpoint)
        model, checkpoint = load_checkpoint(args.checkpoint, args.device)
        if file_sha256(args.checkpoint) != checkpoint_sha256:
            raise ValueError("Checkpoint changed while loading; rerun evaluation")
        manifest = load_manifest(args.prepared)
        if checkpoint["data_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Checkpoint and prepared data do not match")
        evaluation_device = next(model.parameters()).device
        result = evaluate_model(
            model,
            args.prepared,
            manifest["splits"][args.split],
            evaluation_device,
            args.k,
            4096,
        )
        result["provenance"] = {
            "checkpoint_sha256": checkpoint_sha256,
            "prepared_data_fingerprint": manifest["fingerprint"],
            "split": args.split,
            "encoder": checkpoint.get("encoder"),
            "configuration": checkpoint.get("config"),
            "device": str(evaluation_device),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_run_id": checkpoint.get("run_id"),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        emit({k: v for k, v in result.items() if k != "predictions"})
    elif args.command == "evaluate-pipeline":
        from .evaluation import evaluate_pipeline
        from .pipeline import Pipeline
        from .retrieval import SubgraphRetriever

        settings = Settings()
        pipeline = Pipeline(
            GraphStore(settings.graph_db),
            SubgraphRetriever(settings.checkpoint, settings.embedding_cache, settings.device),
            LLMClient(settings),
            settings,
        )
        emit(asyncio.run(evaluate_pipeline(pipeline, args.input, args.output, args.limit)))
    elif args.command == "query":
        import httpx

        settings = Settings()
        request = QueryRequest(
            question=args.question, topic_entity=args.topic, question_id=args.question_id, k=args.k
        )
        headers = {"X-API-Key": settings.api_key.get_secret_value()} if settings.api_key else {}
        response = httpx.post(
            args.url.rstrip("/") + "/v1/query",
            json=request.model_dump(exclude_none=True),
            headers=headers,
            timeout=300,
        )
        response.raise_for_status()
        emit(response.json())
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("trace_backend.api:app", host=args.host, port=args.port)
    elif args.command == "smoke":
        from .smoke import smoke

        emit(smoke(args.output))
    elif args.command == "doctor":
        settings = Settings()
        result = {
            "graph_exists": settings.graph_db.is_file(),
            "checkpoint_exists": settings.checkpoint.is_file(),
            "model_api_key_configured": bool(
                settings.llm_api_key and settings.llm_api_key.get_secret_value()
            ),
        }
        try:
            import torch

            result.update(
                torch=torch.__version__,
                cpu=True,
                mps=torch.backends.mps.is_available(),
                cuda=torch.cuda.is_available(),
            )
        except ImportError:
            result["ml_extra_installed"] = False
        emit(result)


def main():
    try:
        run(parser().parse_args())
    except (TraceError, ValueError, FileNotFoundError, ImportError) as error:
        raise SystemExit(str(error)) from None
