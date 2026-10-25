from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Triple(StrictModel):
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)

    @property
    def id(self) -> str:
        raw = json.dumps(self.key, ensure_ascii=False, separators=(",", ":"))
        return "t_" + hashlib.sha256(raw.encode()).hexdigest()[:24]

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


class Example(StrictModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    topic_entities: list[str]
    answers: list[str]
    triples: list[Triple]
    split: Literal["train", "validation", "test"]


class ScoredTriple(Triple):
    score: float | None = Field(default=None, ge=0, le=1)
    origin: Literal["retrieved", "expansion"] = "retrieved"


class ParserOutput(StrictModel):
    answer_entity: str
    clauses: list[list[str]]


class AnswerOutput(StrictModel):
    answer_entity: str | None
    answer: str
    explanation: str


class QueryRequest(StrictModel):
    question: str | None = Field(default=None, min_length=1, max_length=4000)
    topic_entity: str | None = Field(default=None, min_length=1, max_length=1000)
    question_id: str | None = Field(default=None, min_length=1, max_length=1000)
    k: int = Field(default=50, ge=1, le=1000)
    max_hops: int = Field(default=8, ge=1, le=12)
    candidate_hops: int | None = Field(default=None, ge=1, le=3)

    @model_validator(mode="after")
    def require_source(self):
        if self.question_id is None and (self.question is None or self.topic_entity is None):
            raise ValueError("Supply question_id, or question and topic_entity")
        return self


class PathStep(StrictModel):
    triple_id: str
    source: str
    target: str
    predicate: str
    reversed: bool


class EvidencePath(StrictModel):
    nodes: list[str]
    steps: list[PathStep]
    agreement: float
    hops: int


class SearchResult(StrictModel):
    paths: list[EvidencePath]
    complete: bool
    states_examined: int
    reason: str | None = None


class GraphNode(StrictModel):
    id: str
    label: str
    is_topic: bool = False
    is_answer: bool = False
    on_path: bool = False
    source_urls: list[str] = Field(default_factory=list)


class GraphEdge(StrictModel):
    id: str
    source: str
    target: str
    predicate: str
    label: str
    score: float | None
    origin: Literal["retrieved", "expansion"]
    on_path: bool


class QueryResponse(StrictModel):
    request_id: str
    question: str
    topic_entity: str
    answer_entity: str | None
    answer: str
    explanation: str
    parser_answer: str | None
    parser_disagreement: bool | None
    clauses: list[list[str]]
    status: Literal[
        "path_found", "no_path_within_budget", "search_incomplete", "no_answer", "answer_unresolved"
    ]
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    highlighted_path: EvidencePath | None
    alternative_paths: list[EvidencePath]
    search_complete: bool
    states_examined: int
    warnings: list[str]
    metadata: dict = Field(default_factory=dict)
