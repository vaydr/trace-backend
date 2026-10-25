"""Pinned WebQSP import and normalized JSONL storage."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from .schema import Example, Triple

WEBQSP_DATASET = "ml1996/webqsp"
WEBQSP_REVISION = "54a938897b1ef1e8dbd2fddb6985fd1e688172e8"


def read_examples(path: str | Path) -> Iterator[Example]:
    seen = set()
    with Path(path).open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                ex = Example.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: invalid example") from error
            if ex.id in seen:
                raise ValueError(f"{path}:{number}: duplicate ID {ex.id}")
            seen.add(ex.id)
            yield ex


def write_examples(path: Path, examples: Iterable[Example]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Initialize lazy readers before creating the output file.
    iterator = iter(examples)
    first = next(iterator, None)
    rows = itertools.chain([first], iterator) if first is not None else ()
    count = 0
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as f:
        temp = Path(f.name)
        try:
            for ex in rows:
                payload = (ex.model_dump_json() + "\n").encode("utf-8")
                f.write(payload)
                digest.update(payload)
                count += 1
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        if file_sha256(temp) != digest.hexdigest():
            raise ValueError(
                f"Output integrity check failed for {path}; temporary file changed while writing"
            )
        temp.replace(path)
        if file_sha256(path) != digest.hexdigest():
            raise ValueError(f"Output integrity check failed for {path}; published file changed")
    finally:
        temp.unlink(missing_ok=True)
    return count


def normalize_record(row: dict, split: str) -> Example:
    """Preserve upstream identifiers: surface forms, Freebase IDs, CVTs, and literals."""
    triples = [Triple(subject=t[0], predicate=t[1], object=t[2]) for t in row["graph"]]
    return Example(
        id=row["id"],
        question=row["question"],
        topic_entities=list(dict.fromkeys(row["q_entity"])),
        answers=list(dict.fromkeys(row["a_entity"])),
        triples=list({t.id: t for t in triples}.values()),
        split=split,
    )


def download_webqsp(
    output: Path,
    limit: int | None = None,
    dataset: str = WEBQSP_DATASET,
    revision: str = WEBQSP_REVISION,
) -> dict:
    from datasets import load_dataset

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "validation", "test"):
        rows = load_dataset(dataset, revision=revision, split=split, streaming=True)
        if limit is not None:
            rows = rows.take(limit)
        counts[split] = write_examples(
            output / f"{split}.jsonl", (normalize_record(row, split) for row in rows)
        )
    manifest = {
        "dataset": dataset,
        "revision": revision,
        "limit_per_split": limit,
        "counts": counts,
        "graph_scope": "benchmark_union",
        "files": {s: file_sha256(output / f"{s}.jsonl") for s in counts},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
