from __future__ import annotations

import json
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings
from .errors import TraceError
from .graph import GraphStore
from .llm import LLMClient
from .pipeline import Pipeline, frontend_payload
from .schema import QueryRequest, QueryResponse


def create_app(settings: Settings | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline
        app.state.load_error = None
        if pipeline is None:
            try:
                from .retrieval import SubgraphRetriever

                store = GraphStore(settings.graph_db)
                store.stats()
                retriever = SubgraphRetriever(
                    settings.checkpoint, settings.embedding_cache, settings.device
                )
                app.state.pipeline = Pipeline(store, retriever, LLMClient(settings), settings)
            except (FileNotFoundError, ImportError, ValueError, TraceError) as error:
                app.state.load_error = str(error)
        yield

    app = FastAPI(title="TRACE backend", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    async def authorize(x_api_key: str | None = Header(default=None)):
        if settings.api_key and not secrets.compare_digest(
            x_api_key or "", settings.api_key.get_secret_value()
        ):
            raise TraceError("unauthorized", "Invalid or missing X-API-Key", 401)

    def ready_pipeline() -> Pipeline:
        if app.state.pipeline is None:
            raise TraceError("not_ready", app.state.load_error or "Backend is not initialized", 503)
        return app.state.pipeline

    @app.exception_handler(TraceError)
    async def handle_error(_, error):
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code, "message": str(error)}},
        )

    @app.get("/healthz")
    async def health():
        return {"status": "alive"}

    @app.get("/readyz", dependencies=[Depends(authorize)])
    async def readiness():
        p = ready_pipeline()
        if not settings.llm_api_key or not settings.llm_api_key.get_secret_value():
            # Injected pipelines are used for tests and explicit offline clients.
            if pipeline is None:
                raise TraceError("llm_unconfigured", "Set TRACE_LLM_API_KEY", 503)
        return {"status": "ready", "graph": p.store.stats(), "retriever": p.retriever.metadata}

    @app.get("/v1/entities", dependencies=[Depends(authorize)])
    async def entities(
        q: str = Query(min_length=1, max_length=200), limit: int = Query(default=20, ge=1, le=100)
    ):
        return {"items": ready_pipeline().store.find_entities(q, limit)}

    @app.get("/v1/questions/{question_id}", dependencies=[Depends(authorize)])
    async def question(question_id: str):
        ex = ready_pipeline().store.question(question_id)
        # Gold annotations stay in the evaluation path, never in runtime inputs.
        return {
            "id": ex.id,
            "question": ex.question,
            "topic_entities": ex.topic_entities,
            "split": ex.split,
        }

    @app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(authorize)])
    async def query(request: QueryRequest):
        return await ready_pipeline().query(request)

    @app.post("/v1/trace/query", dependencies=[Depends(authorize)])
    async def trace_query(request: QueryRequest):
        return frontend_payload(await ready_pipeline().query(request))

    @app.post("/v1/trace/stream", dependencies=[Depends(authorize)])
    async def trace_stream(request: QueryRequest):
        p = ready_pipeline()

        async def stream():
            def event(type_, data):
                return "data: " + json.dumps({"type": type_, "data": data}) + "\n\n"

            yield event("data-kgqa-step", "graph")
            try:
                result = await p.query(request)
                payload = frontend_payload(result)
                yield event("data-kgqa-graph", payload["graph"])
                yield event("data-kgqa-clauses", payload["clauses"])
                yield event("data-kgqa-step", "answer")
                yield event("data-kgqa-answer", payload["answer"])
                yield event("data-kgqa-evidence", payload["evidence"])
                yield event("data-kgqa-step", "complete")
            except TraceError as error:
                yield event("data-kgqa-error", {"code": error.code, "message": str(error)})
                yield event("data-kgqa-step", "error")

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
