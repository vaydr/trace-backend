from __future__ import annotations

import asyncio
import json

import httpx
from pydantic import ValidationError

from .config import Settings
from .errors import TraceError
from .prompts import ANSWER_SYSTEM, HUMANIZE_SYSTEM, PARSER_SYSTEM
from .schema import AnswerOutput, ParserOutput, ScoredTriple, StrictModel


class Label(StrictModel):
    predicate: str
    label: str


class LabelsOutput(StrictModel):
    labels: list[Label]


def token_label(predicate: str) -> str:
    return predicate.rsplit(".", 1)[-1].replace("_", " ").strip() or predicate


class LLMClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client

    async def complete(self, system: str, prompt: str, output_type):
        if len(system) + len(prompt) > self.settings.max_prompt_chars:
            raise TraceError(
                "prompt_limit", "LLM prompt exceeds the configured character budget", 422
            )
        key = self.settings.llm_api_key
        if not key or not key.get_secret_value():
            raise TraceError(
                "llm_unconfigured", "Set TRACE_LLM_API_KEY to use live answer generation", 503
            )
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_completion_tokens": 4096,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": output_type.__name__,
                    "strict": True,
                    "schema": output_type.model_json_schema(),
                },
            },
        }
        headers = {"Authorization": "Bearer " + key.get_secret_value()}
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as owned_client:
            client = self.client or owned_client
            for attempt in range(self.settings.llm_attempts):
                try:
                    response = await client.post(
                        self.settings.llm_base_url.rstrip("/") + "/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=self.settings.llm_timeout,
                    )
                    if response.status_code == 429 or response.status_code >= 500:
                        if attempt + 1 < self.settings.llm_attempts:
                            await asyncio.sleep(min(2, 0.25 * 2**attempt))
                            continue
                    if response.is_error:
                        raise TraceError(
                            "llm_http_error",
                            f"Model endpoint returned HTTP {response.status_code}",
                            502,
                        )
                    choice = response.json()["choices"][0]
                    message = choice["message"]
                    if not isinstance(message, dict):
                        raise ValueError("Model message must be an object")
                    if message.get("refusal"):
                        raise TraceError("llm_refusal", "Model declined this request", 502)
                    if choice.get("finish_reason") != "stop":
                        raise TraceError(
                            "llm_incomplete", "Model response did not finish normally", 502
                        )
                    return output_type.model_validate_json(message["content"])
                except httpx.RequestError as error:
                    if attempt + 1 >= self.settings.llm_attempts:
                        raise TraceError(
                            "llm_unavailable", "Model request failed after retries", 502
                        ) from error
                    await asyncio.sleep(min(2, 0.25 * 2**attempt))
                except (ValidationError, KeyError, IndexError, ValueError, TypeError) as error:
                    raise TraceError(
                        "llm_invalid_output",
                        "Model response did not match the expected schema",
                        502,
                    ) from error
        raise TraceError("llm_unavailable", "Model endpoint failed after retries", 502)

    async def answer(self, question: str, triples: list[ScoredTriple]) -> AnswerOutput:
        evidence = [
            {"subject": t.subject, "predicate": t.predicate, "object": t.object, "score": t.score}
            for t in triples
        ]
        return await self.complete(
            ANSWER_SYSTEM,
            json.dumps({"question": question, "triples": evidence}, ensure_ascii=False),
            AnswerOutput,
        )

    async def parse(
        self, question: str, predicates: list[str], labels: dict[str, str]
    ) -> ParserOutput:
        schema = "\n".join(
            f"{p} | {labels.get(p, token_label(p))} | unknown -> unknown" for p in predicates
        )
        result = await self.complete(
            PARSER_SYSTEM, "SCHEMA FRAGMENT\n" + schema + "\nQUESTION\n" + question, ParserOutput
        )
        allowed = set(predicates)
        # Appendix B's examples include a single-predicate clause.
        if len(result.clauses) > 10 or any(
            not 1 <= len(c) <= 5 or any(p not in allowed for p in c) for c in result.clauses
        ):
            raise TraceError(
                "parser_invalid_clauses",
                "Parser returned unknown predicates or invalid clause lengths",
                502,
            )
        return result

    async def humanize(self, predicates: list[str]) -> dict[str, str]:
        result = await self.complete(HUMANIZE_SYSTEM, json.dumps(predicates), LabelsOutput)
        labels = {r.predicate: r.label.strip() for r in result.labels}
        if (
            len(result.labels) != len(predicates)
            or set(labels) != set(predicates)
            or not all(labels.values())
        ):
            raise TraceError(
                "invalid_labels", "Model did not label every supplied predicate exactly once", 502
            )
        return labels
